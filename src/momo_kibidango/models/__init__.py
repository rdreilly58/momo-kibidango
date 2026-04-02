"""Model management: registry, loading, tokenizer bridging."""

__all__ = [
    "ModelRegistry",
    "ModelSpec",
    "ModelTier",
    "ModelLoader",
    "LoadedModel",
    "TokenizerBridge",
    "ClaudeClient",
    "CostTracker",
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
    if name in ("ClaudeClient", "CostTracker"):
        from momo_kibidango.models.claude_client import ClaudeClient, CostTracker
        return {"ClaudeClient": ClaudeClient, "CostTracker": CostTracker}[name]
    raise AttributeError(f"module 'momo_kibidango.models' has no attribute {name!r}")
