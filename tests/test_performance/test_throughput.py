"""Throughput and concurrency performance tests."""

from __future__ import annotations

import threading
import time

import pytest

from momo_kibidango.monitoring.metrics import MetricsCollector


pytestmark = pytest.mark.slow


class TestBatchProcessingOverhead:
    """Measure overhead of batching vs single requests for metrics recording."""

    def test_batch_processing_overhead(self):
        collector = MetricsCollector()
        n_items = 500

        # Single recording loop
        t_single_start = time.perf_counter()
        for i in range(n_items):
            collector.record_inference(
                duration_seconds=0.05,
                tokens_generated=20,
                model_mode="2model",
            )
        t_single = time.perf_counter() - t_single_start

        collector.reset()

        # Batch-style: record same data but in groups of 10
        t_batch_start = time.perf_counter()
        batch_size = 10
        for batch_start in range(0, n_items, batch_size):
            for i in range(batch_size):
                collector.record_inference(
                    duration_seconds=0.05,
                    tokens_generated=20,
                    model_mode="2model",
                )
        t_batch = time.perf_counter() - t_batch_start

        # Batching should not add excessive overhead (less than 5x single)
        assert t_batch < t_single * 5, (
            f"Batch overhead too high: single={t_single:.4f}s, batch={t_batch:.4f}s"
        )

        # Both should be fast (< 1 second for 500 recordings)
        assert t_single < 1.0
        assert t_batch < 1.0


class TestConcurrentMetricsRecording:
    """100 threads recording metrics simultaneously."""

    def test_concurrent_metrics_recording(self):
        collector = MetricsCollector()
        n_threads = 100
        records_per_thread = 100
        errors: list[Exception] = []

        def record_metrics(thread_id: int) -> None:
            try:
                for i in range(records_per_thread):
                    collector.record_inference(
                        duration_seconds=0.01 * (thread_id % 10 + 1),
                        tokens_generated=thread_id + i,
                        model_mode=f"mode_{thread_id % 3}",
                        acceptance_rate=0.5 + (thread_id % 50) / 100.0,
                    )
            except Exception as e:
                errors.append(e)

        threads = [
            threading.Thread(target=record_metrics, args=(tid,))
            for tid in range(n_threads)
        ]

        t_start = time.perf_counter()
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        elapsed = time.perf_counter() - t_start

        # No errors from concurrent access
        assert len(errors) == 0, f"Concurrent recording errors: {errors}"

        # All recordings should be counted
        summary = collector.get_summary()
        assert summary["inference_count"] == n_threads * records_per_thread

        # Should complete reasonably fast (< 5 seconds)
        assert elapsed < 5.0, (
            f"Concurrent recording took {elapsed:.3f}s, expected < 5.0s"
        )

    def test_concurrent_read_write(self):
        """Readers and writers operating simultaneously should not deadlock."""
        collector = MetricsCollector()
        errors: list[Exception] = []
        stop_event = threading.Event()

        def writer(thread_id: int) -> None:
            try:
                for _ in range(200):
                    collector.record_inference(
                        duration_seconds=0.1,
                        tokens_generated=10,
                        model_mode="2model",
                    )
            except Exception as e:
                errors.append(e)

        def reader() -> None:
            try:
                while not stop_event.is_set():
                    collector.get_summary()
            except Exception as e:
                errors.append(e)

        writers = [threading.Thread(target=writer, args=(i,)) for i in range(10)]
        readers = [threading.Thread(target=reader) for _ in range(5)]

        for t in readers:
            t.start()
        for t in writers:
            t.start()
        for t in writers:
            t.join()

        stop_event.set()
        for t in readers:
            t.join(timeout=2.0)

        assert len(errors) == 0, f"Concurrent read/write errors: {errors}"
