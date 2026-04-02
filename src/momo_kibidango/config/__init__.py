"""Configuration management for momo-kibidango."""

__all__ = ["DecoderSettings", "ServerSettings"]


def __getattr__(name: str):
    if name in ("DecoderSettings", "ServerSettings"):
        from momo_kibidango.config.settings import DecoderSettings, ServerSettings
        return {"DecoderSettings": DecoderSettings, "ServerSettings": ServerSettings}[name]
    raise AttributeError(f"module 'momo_kibidango.config' has no attribute {name!r}")
