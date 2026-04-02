"""Custom exceptions for the momo-kibidango speculative decoding framework."""


class MomoError(Exception):
    """Base exception for all momo-kibidango errors."""


class ModelLoadError(MomoError):
    """Failed to load a model (OOM, missing weights, corrupted download)."""


class ModelNotLoadedError(MomoError):
    """Attempted generation before calling load()."""


class TokenizerMismatchError(MomoError):
    """Cross-tokenizer token mapping failed."""


class ResourceExhaustedError(MomoError):
    """System memory or GPU memory exceeded safe limits."""


class RateLimitExceededError(MomoError):
    """Request rate limit exceeded."""


class InvalidPromptError(MomoError):
    """Prompt failed validation (too long, injection pattern detected)."""


class GenerationTimeoutError(MomoError):
    """Generation exceeded the configured timeout."""


class ConfigurationError(MomoError):
    """Invalid or missing configuration."""


class CascadeError(MomoError):
    """Error during cascade decoding (tier escalation failure, all tiers exhausted)."""


class APIError(MomoError):
    """Error communicating with an external API (Anthropic, etc.)."""


class ConfidenceError(MomoError):
    """Error during confidence scoring."""
