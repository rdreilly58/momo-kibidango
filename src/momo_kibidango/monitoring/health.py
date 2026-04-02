"""Health checking and resource monitoring."""

from __future__ import annotations
import logging
import psutil
import torch

logger = logging.getLogger(__name__)


class HealthChecker:
    """Checks system health and resource availability."""

    def __init__(
        self,
        memory_warn_gb: float = 10.0,
        memory_critical_gb: float = 11.5,
    ) -> None:
        self._warn_gb = memory_warn_gb
        self._critical_gb = memory_critical_gb

    def check(self) -> dict:
        """Run all health checks. Returns dict with status and details."""
        memory = self._check_memory()
        device = self._check_device()
        return {
            "status": "critical" if memory["critical"] else ("warn" if memory["warning"] else "healthy"),
            "memory": memory,
            "device": device,
        }

    def _check_memory(self) -> dict:
        process = psutil.Process()
        used_gb = process.memory_info().rss / (1024**3)
        return {
            "used_gb": round(used_gb, 2),
            "warning": used_gb >= self._warn_gb,
            "critical": used_gb >= self._critical_gb,
        }

    def _check_device(self) -> dict:
        if torch.cuda.is_available():
            free, total = torch.cuda.mem_get_info()
            return {"type": "cuda", "free_gb": round(free / (1024**3), 2)}
        if torch.backends.mps.is_available():
            return {"type": "mps", "free_gb": None}  # MPS doesn't expose free mem
        return {"type": "cpu", "free_gb": None}
