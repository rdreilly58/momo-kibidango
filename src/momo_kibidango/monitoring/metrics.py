"""Lightweight metrics collection for speculative decoding."""

from __future__ import annotations

import statistics
import threading
import time
from collections import defaultdict, deque
from typing import Any


class MetricsCollector:
    """Thread-safe metrics collector.

    Tracks throughput, latency, acceptance rates, memory usage, errors,
    and inference counts using bounded deques (maxlen=1000).
    No Prometheus or Flask dependency required.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()

        # Core sample buffers
        self.throughput_samples: deque[tuple[float, float]] = deque(maxlen=1000)
        self.latency_samples: deque[tuple[float, float]] = deque(maxlen=1000)
        self.acceptance_rates: deque[tuple[float, float]] = deque(maxlen=1000)
        self.memory_samples: deque[tuple[float, float]] = deque(maxlen=1000)

        # Per-stage acceptance rates
        self.stage_rates: dict[str, deque[tuple[float, float]]] = defaultdict(
            lambda: deque(maxlen=1000)
        )

        # Counters
        self.error_counts: dict[str, int] = defaultdict(int)
        self.inference_count: int = 0
        self.total_tokens: int = 0

        # Optional Prometheus integration
        self._prom_inference_total = None
        self._prom_latency = None
        try:
            from prometheus_client import Counter, Histogram

            self._prom_inference_total = Counter(
                "momo_inference_total",
                "Total inference calls",
                ["model_mode"],
            )
            self._prom_latency = Histogram(
                "momo_latency_seconds",
                "Inference latency in seconds",
                ["model_mode"],
                buckets=(0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0),
            )
        except ImportError:
            pass

    # ------------------------------------------------------------------ #
    # Recording
    # ------------------------------------------------------------------ #

    def record_inference(
        self,
        duration_seconds: float,
        tokens_generated: int,
        model_mode: str,
        acceptance_rate: float = 0.0,
        stage_rates: dict[str, float] | None = None,
    ) -> None:
        """Record a single inference call."""
        now = time.time()
        throughput = tokens_generated / duration_seconds if duration_seconds > 0 else 0.0

        with self._lock:
            self.inference_count += 1
            self.total_tokens += tokens_generated
            self.throughput_samples.append((now, throughput))
            self.latency_samples.append((now, duration_seconds))
            self.acceptance_rates.append((now, acceptance_rate))

            if stage_rates:
                for stage, rate in stage_rates.items():
                    self.stage_rates[stage].append((now, rate))

        # Prometheus (outside lock — those objects have their own locking)
        if self._prom_inference_total is not None:
            self._prom_inference_total.labels(model_mode=model_mode).inc()
        if self._prom_latency is not None:
            self._prom_latency.labels(model_mode=model_mode).observe(duration_seconds)

    def record_error(self, error_type: str) -> None:
        """Increment error counter for *error_type*."""
        with self._lock:
            self.error_counts[error_type] += 1

    def record_memory(self, memory_gb: float) -> None:
        """Record a memory-usage sample (GB)."""
        with self._lock:
            self.memory_samples.append((time.time(), memory_gb))

    # ------------------------------------------------------------------ #
    # Summaries
    # ------------------------------------------------------------------ #

    def get_summary(self) -> dict[str, Any]:
        """Return an aggregated snapshot of all collected metrics."""
        with self._lock:
            return {
                "inference_count": self.inference_count,
                "total_tokens": self.total_tokens,
                "throughput": self._summarise_values(self.throughput_samples),
                "latency": self._summarise_values(self.latency_samples),
                "acceptance_rate": self._summarise_values(self.acceptance_rates),
                "memory_gb": self._summarise_values(self.memory_samples),
                "stage_acceptance_rates": {
                    stage: self._summarise_values(samples)
                    for stage, samples in self.stage_rates.items()
                },
                "error_counts": dict(self.error_counts),
            }

    def reset(self) -> None:
        """Clear all collected data."""
        with self._lock:
            self.throughput_samples.clear()
            self.latency_samples.clear()
            self.acceptance_rates.clear()
            self.memory_samples.clear()
            self.stage_rates.clear()
            self.error_counts.clear()
            self.inference_count = 0
            self.total_tokens = 0

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #

    @staticmethod
    def _summarise_values(
        samples: deque[tuple[float, float]],
    ) -> dict[str, float]:
        """Compute mean / min / max / p50 / p95 / count for a deque of (ts, value)."""
        if not samples:
            return {"mean": 0.0, "min": 0.0, "max": 0.0, "p50": 0.0, "p95": 0.0, "count": 0}

        values = [v for _, v in samples]
        sorted_v = sorted(values)
        n = len(sorted_v)
        return {
            "mean": round(statistics.mean(values), 6),
            "min": round(sorted_v[0], 6),
            "max": round(sorted_v[-1], 6),
            "p50": round(sorted_v[n // 2], 6),
            "p95": round(sorted_v[int(n * 0.95)], 6),
            "count": n,
        }
