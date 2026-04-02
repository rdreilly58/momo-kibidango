"""Model registry: maps tiers to configurable model specifications."""

from __future__ import annotations
import logging
from enum import Enum
from dataclasses import dataclass
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from momo_kibidango.config.settings import DecoderSettings

logger = logging.getLogger(__name__)


class ModelTier(str, Enum):
    DRAFT = "draft"
    QUALIFIER = "qualifier"
    TARGET = "target"


@dataclass(frozen=True)
class ModelSpec:
    """Immutable specification for a model in the pipeline."""
    model_id: str
    tier: ModelTier
    tokenizer_id: str | None = None
    dtype: str = "float16"
    trust_remote_code: bool = True
    revision: str | None = None

    @property
    def effective_tokenizer_id(self) -> str:
        return self.tokenizer_id or self.model_id


class ModelRegistry:
    """Maps tier names to ModelSpec instances. Single source of truth for model config."""

    def __init__(self) -> None:
        self._specs: dict[ModelTier, ModelSpec] = {}

    def register(self, spec: ModelSpec) -> None:
        logger.info("Registered %s model: %s", spec.tier.value, spec.model_id)
        self._specs[spec.tier] = spec

    def get_tier(self, tier: ModelTier | str) -> ModelSpec:
        if isinstance(tier, str):
            tier = ModelTier(tier)
        if tier not in self._specs:
            raise KeyError(f"No model registered for tier '{tier.value}'")
        return self._specs[tier]

    def has_tier(self, tier: ModelTier | str) -> bool:
        if isinstance(tier, str):
            tier = ModelTier(tier)
        return tier in self._specs

    @property
    def tiers(self) -> list[ModelTier]:
        return sorted(self._specs.keys(), key=lambda t: list(ModelTier).index(t))

    @classmethod
    def from_settings(cls, settings: DecoderSettings) -> ModelRegistry:
        registry = cls()
        registry.register(ModelSpec(
            model_id=settings.draft_model_id,
            tier=ModelTier.DRAFT,
            tokenizer_id=settings.draft_tokenizer_id,
            dtype=settings.draft_dtype,
        ))
        if settings.qualifier_model_id:
            registry.register(ModelSpec(
                model_id=settings.qualifier_model_id,
                tier=ModelTier.QUALIFIER,
                tokenizer_id=settings.qualifier_tokenizer_id,
                dtype=settings.qualifier_dtype,
            ))
        registry.register(ModelSpec(
            model_id=settings.target_model_id,
            tier=ModelTier.TARGET,
            tokenizer_id=settings.target_tokenizer_id,
            dtype=settings.target_dtype,
        ))
        return registry
