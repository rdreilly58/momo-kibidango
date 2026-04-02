"""momo-kibidango: A 3-tier speculative decoding framework for LLM inference acceleration."""

__version__ = "2.0.0"
__author__ = "Robert Reilly"
__email__ = "robert.reilly@reillydesignstudio.com"
__license__ = "MIT"

# Lazy imports — heavy dependencies (torch, transformers) are only loaded when
# the actual classes are accessed.  This lets the package be imported without
# GPU libraries installed (e.g. for running unit tests with mocks).

from momo_kibidango.exceptions import (
    MomoError,
    ModelLoadError,
    ModelNotLoadedError,
    TokenizerMismatchError,
    ResourceExhaustedError,
    RateLimitExceededError,
    InvalidPromptError,
    GenerationTimeoutError,
    ConfigurationError,
    CascadeError,
    APIError,
    ConfidenceError,
)

__all__ = [
    # Core
    "BaseDecoder",
    "GenerationRequest",
    "GenerationResult",
    "TwoModelDecoder",
    "ThreeModelDecoder",
    "AdaptiveThreshold",
    "CascadeDecoder",
    "ConfidenceScorer",
    "ConfidenceResult",
    # Configuration
    "DecoderSettings",
    "ServerSettings",
    # Models
    "ModelRegistry",
    "ModelSpec",
    "ModelTier",
    "ModelLoader",
    "ClaudeClient",
    "CostTracker",
    # Monitoring
    "MetricsCollector",
    # Exceptions
    "MomoError",
    "ModelLoadError",
    "ModelNotLoadedError",
    "TokenizerMismatchError",
    "ResourceExhaustedError",
    "RateLimitExceededError",
    "InvalidPromptError",
    "GenerationTimeoutError",
    "ConfigurationError",
    "CascadeError",
    "APIError",
    "ConfidenceError",
]


def __getattr__(name: str):  # noqa: C901
    """Lazily import heavy modules only when their symbols are accessed."""
    if name in ("BaseDecoder", "GenerationRequest", "GenerationResult"):
        from momo_kibidango.core.decoder import BaseDecoder, GenerationRequest, GenerationResult
        return {"BaseDecoder": BaseDecoder, "GenerationRequest": GenerationRequest, "GenerationResult": GenerationResult}[name]
    if name == "TwoModelDecoder":
        from momo_kibidango.core.two_model import TwoModelDecoder
        return TwoModelDecoder
    if name == "ThreeModelDecoder":
        from momo_kibidango.core.three_model import ThreeModelDecoder
        return ThreeModelDecoder
    if name == "AdaptiveThreshold":
        from momo_kibidango.core.adaptive import AdaptiveThreshold
        return AdaptiveThreshold
    if name == "CascadeDecoder":
        from momo_kibidango.core.cascade import CascadeDecoder
        return CascadeDecoder
    if name in ("ConfidenceScorer", "ConfidenceResult"):
        from momo_kibidango.core.confidence import ConfidenceScorer, ConfidenceResult
        return {"ConfidenceScorer": ConfidenceScorer, "ConfidenceResult": ConfidenceResult}[name]
    if name in ("DecoderSettings", "ServerSettings"):
        from momo_kibidango.config.settings import DecoderSettings, ServerSettings
        return {"DecoderSettings": DecoderSettings, "ServerSettings": ServerSettings}[name]
    if name in ("ModelRegistry", "ModelSpec", "ModelTier"):
        from momo_kibidango.models.registry import ModelRegistry, ModelSpec, ModelTier
        return {"ModelRegistry": ModelRegistry, "ModelSpec": ModelSpec, "ModelTier": ModelTier}[name]
    if name == "ModelLoader":
        from momo_kibidango.models.loader import ModelLoader
        return ModelLoader
    if name in ("ClaudeClient", "CostTracker"):
        from momo_kibidango.models.claude_client import ClaudeClient, CostTracker
        return {"ClaudeClient": ClaudeClient, "CostTracker": CostTracker}[name]
    if name == "MetricsCollector":
        from momo_kibidango.monitoring.metrics import MetricsCollector
        return MetricsCollector
    raise AttributeError(f"module 'momo_kibidango' has no attribute {name!r}")
