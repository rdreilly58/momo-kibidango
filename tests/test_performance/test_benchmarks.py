"""Performance benchmark tests.

Marked with @pytest.mark.slow so they can be excluded from fast CI runs.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict

import pytest

from momo_kibidango.core.adaptive import AdaptiveThreshold
from momo_kibidango.core.decoder import GenerationResult
from momo_kibidango.monitoring.metrics import MetricsCollector


pytestmark = pytest.mark.slow


class TestMetricsCollectionOverhead:
    """Recording 10000 metrics should take less than 1 second."""

    def test_metrics_collection_overhead(self):
        collector = MetricsCollector()

        t_start = time.perf_counter()
        for i in range(10_000):
            collector.record_inference(
                duration_seconds=0.1,
                tokens_generated=50,
                model_mode="2model",
                acceptance_rate=0.8,
            )
        elapsed = time.perf_counter() - t_start

        assert elapsed < 1.0, (
            f"Recording 10000 metrics took {elapsed:.3f}s, expected < 1.0s"
        )

        # Verify data was actually recorded
        summary = collector.get_summary()
        # deque maxlen=1000, so inference_count tracks all 10000
        assert summary["inference_count"] == 10_000


class TestAdaptiveThresholdConvergence:
    """Simulate 1000 updates and verify the threshold stabilizes."""

    def test_adaptive_threshold_convergence(self):
        adaptive = AdaptiveThreshold(
            initial_stage2=0.10,
            target_acceptance_rate=0.70,
            ema_alpha=0.05,
            adjustment_step=0.005,
            warmup_iterations=20,
        )

        # Simulate 1000 rounds with a consistent 70% acceptance rate
        for _ in range(1000):
            adaptive.update("stage2", accepted=7, total=10)

        snapshot = adaptive.snapshot()

        # EMA should converge near the target acceptance rate
        assert abs(snapshot["stage2"]["ema"] - 0.70) < 0.05, (
            f"EMA did not converge: {snapshot['stage2']['ema']}"
        )

        # Threshold should be within min/max bounds
        threshold = snapshot["stage2"]["threshold"]
        assert 0.01 <= threshold <= 0.50

    def test_adaptive_threshold_adjusts_to_high_rate(self):
        """When acceptance rate is consistently high, threshold should increase."""
        adaptive = AdaptiveThreshold(
            initial_stage2=0.05,
            target_acceptance_rate=0.70,
            warmup_iterations=10,
        )

        initial = adaptive.stage2_threshold

        # Feed consistently high acceptance (95%)
        for _ in range(200):
            adaptive.update("stage2", accepted=95, total=100)

        # Threshold should have increased (too lenient -> raise threshold)
        assert adaptive.stage2_threshold > initial

    def test_adaptive_threshold_adjusts_to_low_rate(self):
        """When acceptance rate is consistently low, threshold should decrease."""
        adaptive = AdaptiveThreshold(
            initial_stage2=0.20,
            target_acceptance_rate=0.70,
            warmup_iterations=10,
        )

        initial = adaptive.stage2_threshold

        # Feed consistently low acceptance (30%)
        for _ in range(200):
            adaptive.update("stage2", accepted=3, total=10)

        # Threshold should have decreased (too strict -> lower threshold)
        assert adaptive.stage2_threshold < initial


class TestRequestResultSerialization:
    """Ensure GenerationResult can be serialized to JSON."""

    def test_request_result_serialization(self):
        result = GenerationResult(
            text="Hello, world!",
            tokens_generated=3,
            elapsed_seconds=0.25,
            tokens_per_second=12.0,
            acceptance_rate=0.75,
            stage_acceptance_rates={"stage1": 0.85, "stage2": 0.75},
            peak_memory_gb=2.5,
            mode="3model",
            draft_attempts=4,
            accepted_tokens=3,
        )

        # Should not raise
        serialized = json.dumps(asdict(result))
        assert isinstance(serialized, str)

        # Round-trip: deserialize and verify
        deserialized = json.loads(serialized)
        assert deserialized["text"] == "Hello, world!"
        assert deserialized["tokens_generated"] == 3
        assert deserialized["mode"] == "3model"
        assert deserialized["stage_acceptance_rates"]["stage1"] == 0.85
