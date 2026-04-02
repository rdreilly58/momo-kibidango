"""API servers for momo-kibidango."""

__all__ = [
    "InferenceServer",
    "InputValidator",
]


def __getattr__(name: str):
    if name in ("InferenceServer", "InputValidator"):
        from momo_kibidango.api.server import InferenceServer, InputValidator
        return {"InferenceServer": InferenceServer, "InputValidator": InputValidator}[name]
    raise AttributeError(f"module 'momo_kibidango.api' has no attribute {name!r}")
