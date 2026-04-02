"""Memory-related performance tests."""

from __future__ import annotations

import pytest

from momo_kibidango.core.kv_cache import KVCacheManager
from momo_kibidango.monitoring.metrics import MetricsCollector


pytestmark = pytest.mark.slow


class TestMetricsMemoryBounded:
    """100k recordings should not grow unbounded (deque maxlen)."""

    def test_metrics_memory_bounded(self):
        collector = MetricsCollector()

        # Record 100,000 metrics
        for i in range(100_000):
            collector.record_inference(
                duration_seconds=0.01,
                tokens_generated=10,
                model_mode="2model",
                acceptance_rate=0.7,
            )

        summary = collector.get_summary()

        # inference_count tracks total calls (100k)
        assert summary["inference_count"] == 100_000

        # But the sample deques are bounded to maxlen=1000
        assert len(collector.throughput_samples) == 1000
        assert len(collector.latency_samples) == 1000
        assert len(collector.acceptance_rates) == 1000

        # Summary counts should reflect the bounded window
        assert summary["throughput"]["count"] == 1000
        assert summary["latency"]["count"] == 1000

    def test_deque_maxlen_is_1000(self):
        """Verify the deques have the expected maxlen."""
        collector = MetricsCollector()

        assert collector.throughput_samples.maxlen == 1000
        assert collector.latency_samples.maxlen == 1000
        assert collector.acceptance_rates.maxlen == 1000
        assert collector.memory_samples.maxlen == 1000


class TestKVCacheInvalidationReclaims:
    """After invalidation, cache is None."""

    def test_kv_cache_invalidation_reclaims(self):
        cache_mgr = KVCacheManager()

        # Store something in the cache
        fake_cache = {"layer_0": "some_tensor_data", "layer_1": "more_data"}
        cache_mgr.update(fake_cache, accepted_length=100)

        assert cache_mgr.get_cache() is not None
        assert cache_mgr.cached_length == 100

        # Invalidate
        cache_mgr.invalidate()

        assert cache_mgr.get_cache() is None
        assert cache_mgr.cached_length == 0

    def test_kv_cache_auto_invalidates_on_overflow(self):
        """Cache auto-invalidates when accepted_length exceeds max_cache_tokens."""
        cache_mgr = KVCacheManager(max_cache_tokens=512)

        fake_cache = {"data": "test"}
        cache_mgr.update(fake_cache, accepted_length=600)

        # Should have auto-invalidated due to exceeding max
        assert cache_mgr.get_cache() is None
        assert cache_mgr.cached_length == 0

    def test_kv_cache_update_then_invalidate_cycle(self):
        """Multiple update/invalidate cycles work correctly."""
        cache_mgr = KVCacheManager()

        for i in range(10):
            cache_mgr.update({"round": i}, accepted_length=50 * (i + 1))
            assert cache_mgr.get_cache() is not None

            cache_mgr.invalidate()
            assert cache_mgr.get_cache() is None
            assert cache_mgr.cached_length == 0
