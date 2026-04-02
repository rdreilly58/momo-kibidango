"""Model management: registry, loading, tokenizer bridging."""

__all__ = [
    "ModelRegistry",
    "ModelSpec",
    "ModelTier",
    "ModelLoader",
    "LoadedModel",
    "TokenizerBridge",
]


def __getattr__(name: str):
    if name in ("ModelRegistry", "ModelSpec", "ModelTier"):
        from momo_kibidango.models.registry import ModelRegistry, ModelSpec, ModelTier
        return {"ModelRegistry": ModelRegistry, "ModelSpec": ModelSpec, "ModelTier": ModelTier}[name]
    if name in ("ModelLoader", "LoadedModel"):
        from momo_kibidango.models.loader import ModelLoader, LoadedModel
        return {"ModelLoader": ModelLoader, "LoadedModel": LoadedModel}[name]
    if name == "TokenizerBridge":
        from momo_kibidango.models.tokenizer_bridge import TokenizerBridge
        return TokenizerBridge
    raise AttributeError(f"module 'momo_kibidango.models' has no attribute {name!r}")
