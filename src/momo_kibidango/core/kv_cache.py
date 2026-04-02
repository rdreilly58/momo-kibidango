"""KV-cache management for speculative decoding iterations."""

from __future__ import annotations
import logging
from typing import Optional, Any

logger = logging.getLogger(__name__)


class KVCacheManager:
    """Manages KV-cache reuse across speculative decoding iterations.

    After verifying N accepted tokens, the draft model's KV-cache for the
    accepted prefix is valid and can be reused in the next draft phase.
    """

    def __init__(self, max_cache_tokens: int = 2048) -> None:
        self._max_tokens = max_cache_tokens
        self._cache: Optional[Any] = None
        self._cached_length: int = 0

    @property
    def cached_length(self) -> int:
        return self._cached_length

    def get_cache(self) -> Optional[Any]:
        """Return current cache if available."""
        return self._cache

    def update(self, cache: Any, accepted_length: int) -> None:
        """Store cache and record length of accepted prefix."""
        if accepted_length > self._max_tokens:
            self.invalidate()
            return
        self._cache = cache
        self._cached_length = accepted_length
        logger.debug("KV-cache updated: %d tokens cached", accepted_length)

    def invalidate(self) -> None:
        """Clear cache (e.g., on full rejection or context switch)."""
        self._cache = None
        self._cached_length = 0
        logger.debug("KV-cache invalidated")
