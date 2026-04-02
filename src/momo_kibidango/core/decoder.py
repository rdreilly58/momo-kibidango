"""Base decoder interface for speculative decoding."""

from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Iterator


@dataclass(frozen=True)
class GenerationRequest:
    """Immutable generation request."""
    prompt: str
    max_new_tokens: int = 256
    temperature: float = 0.7
    top_p: float = 0.9
    stop_sequences: list[str] = field(default_factory=list)


@dataclass
class GenerationResult:
    """Result of a generation call with detailed metrics."""
    text: str
    tokens_generated: int
    elapsed_seconds: float
    tokens_per_second: float
    acceptance_rate: float
    stage_acceptance_rates: dict[str, float]
    peak_memory_gb: float
    mode: str  # "1model", "2model", "3model"
    draft_attempts: int
    accepted_tokens: int


class BaseDecoder(ABC):
    """Abstract base for all speculative decoder variants."""

    @abstractmethod
    def load(self) -> None:
        """Load models into memory. Raises ModelLoadError on failure."""
        ...

    @abstractmethod
    def generate(self, request: GenerationRequest) -> GenerationResult:
        """Run synchronous generation."""
        ...

    def stream(self, request: GenerationRequest) -> Iterator[str]:
        """Token-by-token streaming. Default: non-streaming fallback."""
        result = self.generate(request)
        yield result.text

    @abstractmethod
    def unload(self) -> None:
        """Release all GPU/system memory."""
        ...

    @property
    @abstractmethod
    def mode(self) -> str:
        """Return mode identifier: '1model', '2model', '3model'."""
        ...

    @property
    @abstractmethod
    def is_loaded(self) -> bool:
        ...
