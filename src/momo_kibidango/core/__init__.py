"""Core speculative decoding engine."""

__all__ = [
    "BaseDecoder",
    "GenerationRequest",
    "GenerationResult",
    "TwoModelDecoder",
    "ThreeModelDecoder",
    "AdaptiveThreshold",
    "CascadeDecoder",
    "ConfidenceScorer",
    "ConfidenceResult",
]


def __getattr__(name: str):
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
    raise AttributeError(f"module 'momo_kibidango.core' has no attribute {name!r}")
