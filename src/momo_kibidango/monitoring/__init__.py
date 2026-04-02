"""Monitoring and metrics collection."""

__all__ = [
    "MetricsCollector",
    "HealthChecker",
]


def __getattr__(name: str):
    if name == "MetricsCollector":
        from momo_kibidango.monitoring.metrics import MetricsCollector
        return MetricsCollector
    if name == "HealthChecker":
        from momo_kibidango.monitoring.health import HealthChecker
        return HealthChecker
    raise AttributeError(f"module 'momo_kibidango.monitoring' has no attribute {name!r}")
