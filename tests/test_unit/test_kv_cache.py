"""Unit tests for KVCacheManager."""

import pytest
from momo_kibidango.core.kv_cache import KVCacheManager


class TestKVCacheManagerInit:
    """Test initial state of KVCacheManager."""

    def test_initial_state(self):
        mgr = KVCacheManager()
        assert mgr.get_cache() is None
        assert mgr.cached_length == 0

    def test_initial_state_custom_max(self):
        mgr = KVCacheManager(max_cache_tokens=512)
        assert mgr.get_cache() is None
        assert mgr.cached_length == 0


class TestKVCacheManagerUpdate:
    """Test update and retrieval of cache."""

    def test_update(self):
        mgr = KVCacheManager()
        fake_cache = {"layer_0": "some_tensor"}
        mgr.update(fake_cache, accepted_length=100)
        assert mgr.get_cache() is fake_cache
        assert mgr.cached_length == 100

    def test_update_multiple(self):
        mgr = KVCacheManager()
        cache1 = {"v": 1}
        cache2 = {"v": 2}
        mgr.update(cache1, accepted_length=50)
        assert mgr.get_cache() is cache1
        mgr.update(cache2, accepted_length=150)
        assert mgr.get_cache() is cache2
        assert mgr.cached_length == 150

    def test_update_preserves_object_identity(self):
        mgr = KVCacheManager()
        cache = [1, 2, 3]
        mgr.update(cache, accepted_length=10)
        assert mgr.get_cache() is cache


class TestKVCacheManagerInvalidate:
    """Test cache invalidation."""

    def test_invalidate(self):
        mgr = KVCacheManager()
        mgr.update({"data": True}, accepted_length=50)
        assert mgr.get_cache() is not None
        mgr.invalidate()
        assert mgr.get_cache() is None
        assert mgr.cached_length == 0

    def test_invalidate_when_empty(self):
        """Invalidating an empty cache should not raise."""
        mgr = KVCacheManager()
        mgr.invalidate()
        assert mgr.get_cache() is None
        assert mgr.cached_length == 0


class TestKVCacheManagerOverflow:
    """Test max_cache_tokens overflow behavior."""

    def test_max_tokens_overflow(self):
        mgr = KVCacheManager(max_cache_tokens=100)
        cache = {"data": True}
        mgr.update(cache, accepted_length=101)
        # Should have been invalidated due to overflow
        assert mgr.get_cache() is None
        assert mgr.cached_length == 0

    def test_max_tokens_exact_boundary(self):
        mgr = KVCacheManager(max_cache_tokens=100)
        cache = {"data": True}
        mgr.update(cache, accepted_length=100)
        # Exactly at max should be fine
        assert mgr.get_cache() is cache
        assert mgr.cached_length == 100

    def test_max_tokens_below_boundary(self):
        mgr = KVCacheManager(max_cache_tokens=100)
        cache = {"data": True}
        mgr.update(cache, accepted_length=99)
        assert mgr.get_cache() is cache
        assert mgr.cached_length == 99
