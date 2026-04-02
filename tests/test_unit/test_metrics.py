"""Unit tests for MetricsCollector."""

import pytest
import threading
from momo_kibidango.monitoring.metrics import MetricsCollector


class TestRecordInference:
    """Test inference recording and summary retrieval."""

    def test_record_inference(self, metrics_collector):
        metrics_collector.record_inference(
            duration_seconds=1.0,
            tokens_generated=50,
            model_mode="2model",
            acceptance_rate=0.8,
        )
        summary = metrics_collector.get_summary()
        assert summary["inference_count"] == 1
        assert summary["total_tokens"] == 50
        assert summary["throughput"]["count"] == 1
        assert summary["throughput"]["mean"] == pytest.approx(50.0)
        assert summary["latency"]["mean"] == pytest.approx(1.0)
        assert summary["acceptance_rate"]["mean"] == pytest.approx(0.8)

    def test_record_multiple_inferences(self, metrics_collector):
        metrics_collector.record_inference(1.0, 50, "2model", 0.8)
        metrics_collector.record_inference(2.0, 100, "2model", 0.9)
        summary = metrics_collector.get_summary()
        assert summary["inference_count"] == 2
        assert summary["total_tokens"] == 150

    def test_record_inference_with_stage_rates(self, metrics_collector):
        metrics_collector.record_inference(
            duration_seconds=1.0,
            tokens_generated=50,
            model_mode="3model",
            acceptance_rate=0.8,
            stage_rates={"stage1": 0.85, "stage2": 0.75},
        )
        summary = metrics_collector.get_summary()
        assert "stage1" in summary["stage_acceptance_rates"]
        assert "stage2" in summary["stage_acceptance_rates"]
        assert summary["stage_acceptance_rates"]["stage1"]["mean"] == pytest.approx(0.85)

    def test_record_inference_zero_duration(self, metrics_collector):
        """Zero duration should produce zero throughput, not crash."""
        metrics_collector.record_inference(0.0, 10, "2model", 0.5)
        summary = metrics_collector.get_summary()
        assert summary["throughput"]["mean"] == pytest.approx(0.0)


class TestRecordError:
    """Test error counting."""

    def test_record_error(self, metrics_collector):
        metrics_collector.record_error("timeout")
        metrics_collector.record_error("timeout")
        metrics_collector.record_error("oom")
        summary = metrics_collector.get_summary()
        assert summary["error_counts"]["timeout"] == 2
        assert summary["error_counts"]["oom"] == 1

    def test_record_error_increments(self, metrics_collector):
        for _ in range(5):
            metrics_collector.record_error("test_error")
        summary = metrics_collector.get_summary()
        assert summary["error_counts"]["test_error"] == 5


class TestRecordMemory:
    """Test memory usage tracking."""

    def test_record_memory(self, metrics_collector):
        metrics_collector.record_memory(4.5)
        metrics_collector.record_memory(5.0)
        summary = metrics_collector.get_summary()
        assert summary["memory_gb"]["count"] == 2
        assert summary["memory_gb"]["min"] == pytest.approx(4.5)
        assert summary["memory_gb"]["max"] == pytest.approx(5.0)


class TestGetSummaryEmpty:
    """Test summary output when no data has been recorded."""

    def test_get_summary_empty(self, metrics_collector):
        summary = metrics_collector.get_summary()
        assert summary["inference_count"] == 0
        assert summary["total_tokens"] == 0
        assert summary["throughput"]["mean"] == 0.0
        assert summary["throughput"]["count"] == 0
        assert summary["latency"]["mean"] == 0.0
        assert summary["acceptance_rate"]["mean"] == 0.0
        assert summary["memory_gb"]["mean"] == 0.0
        assert summary["error_counts"] == {}


class TestReset:
    """Test clearing all collected data."""

    def test_reset(self, metrics_collector):
        metrics_collector.record_inference(1.0, 50, "2model", 0.8)
        metrics_collector.record_error("timeout")
        metrics_collector.record_memory(4.0)
        metrics_collector.reset()

        summary = metrics_collector.get_summary()
        assert summary["inference_count"] == 0
        assert summary["total_tokens"] == 0
        assert summary["throughput"]["count"] == 0
        assert summary["latency"]["count"] == 0
        assert summary["acceptance_rate"]["count"] == 0
        assert summary["memory_gb"]["count"] == 0
        assert summary["error_counts"] == {}


class TestThreadSafety:
    """Test concurrent recording does not crash."""

    def test_thread_safety(self, metrics_collector):
        errors = []

        def record_many(collector, thread_id):
            try:
                for i in range(100):
                    collector.record_inference(
                        duration_seconds=0.1,
                        tokens_generated=10,
                        model_mode="2model",
                        acceptance_rate=0.7,
                    )
                    collector.record_error(f"err_{thread_id}")
                    collector.record_memory(float(i))
            except Exception as e:
                errors.append(e)

        threads = []
        for t_id in range(10):
            t = threading.Thread(target=record_many, args=(metrics_collector, t_id))
            threads.append(t)
            t.start()

        for t in threads:
            t.join()

        assert len(errors) == 0
        summary = metrics_collector.get_summary()
        # 10 threads * 100 inferences each
        assert summary["inference_count"] == 1000
        assert summary["total_tokens"] == 10000
