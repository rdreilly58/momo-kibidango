"""Integration tests for graceful degradation / fallback chains.

Tests that when a higher-tier model fails to load, the system falls back
to a simpler decoder configuration:
  3-model -> 2-model -> 1-model (target only).
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
import torch

from momo_kibidango.config.settings import DecoderSettings
from momo_kibidango.core.decoder import BaseDecoder, GenerationRequest, GenerationResult
from momo_kibidango.core.two_model import TwoModelDecoder
from momo_kibidango.core.three_model import ThreeModelDecoder
from momo_kibidango.exceptions import ModelLoadError
from momo_kibidango.models.loader import LoadedModel, ModelLoader
from momo_kibidango.models.registry import ModelRegistry, ModelSpec, ModelTier
from momo_kibidango.monitoring.metrics import MetricsCollector


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

VOCAB_SIZE = 128


def _make_mock_tokenizer() -> MagicMock:
    tok = MagicMock()
    tok.eos_token_id = 2
    tok.encode.return_value = torch.tensor([[1, 10, 20]])
    tok.decode.return_value = "fallback output"
    return tok


def _make_mock_model() -> MagicMock:
    model = MagicMock()

    def model_call(input_ids):
        seq_len = input_ids.shape[1]
        output = MagicMock()
        logits = torch.randn(1, seq_len, VOCAB_SIZE)
        logits[:, :, 42] = 10.0
        output.logits = logits
        return output

    model.side_effect = model_call
    return model


def _make_loaded_model(tier: ModelTier, model_id: str) -> LoadedModel:
    return LoadedModel(
        model=_make_mock_model(),
        tokenizer=_make_mock_tokenizer(),
        spec=ModelSpec(model_id=model_id, tier=tier),
    )


class SingleModelDecoder(BaseDecoder):
    """Minimal single-model decoder for fallback testing."""

    def __init__(self, target_loaded: LoadedModel, metrics: MetricsCollector) -> None:
        self._target = target_loaded
        self._metrics = metrics
        self._loaded = True

    @property
    def mode(self) -> str:
        return "1model"

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    def load(self) -> None:
        self._loaded = True

    def unload(self) -> None:
        self._loaded = False

    def generate(self, request: GenerationRequest) -> GenerationResult:
        return GenerationResult(
            text="single model fallback output",
            tokens_generated=5,
            elapsed_seconds=0.1,
            tokens_per_second=50.0,
            acceptance_rate=1.0,
            stage_acceptance_rates={},
            peak_memory_gb=1.0,
            mode="1model",
            draft_attempts=0,
            accepted_tokens=5,
        )


def _build_decoder_with_fallback(
    qualifier_fails: bool = False,
    draft_fails: bool = False,
) -> BaseDecoder:
    """Attempt to build a 3-model decoder, falling back as needed.

    This simulates the fallback chain:
      3model -> 2model -> 1model
    """
    settings = DecoderSettings(
        draft_model_id="test/draft-0.5b",
        qualifier_model_id="test/qualifier-1.5b",
        target_model_id="test/target-7b",
        device="cpu",
        max_draft_tokens=3,
    )

    registry = ModelRegistry()
    registry.register(ModelSpec(model_id="test/draft-0.5b", tier=ModelTier.DRAFT))
    registry.register(ModelSpec(model_id="test/qualifier-1.5b", tier=ModelTier.QUALIFIER))
    registry.register(ModelSpec(model_id="test/target-7b", tier=ModelTier.TARGET))

    loader = MagicMock(spec=ModelLoader)
    metrics = MetricsCollector()

    target_loaded = _make_loaded_model(ModelTier.TARGET, "test/target-7b")
    draft_loaded = _make_loaded_model(ModelTier.DRAFT, "test/draft-0.5b")
    qualifier_loaded = _make_loaded_model(ModelTier.QUALIFIER, "test/qualifier-1.5b")

    def load_side_effect(spec, device="cpu"):
        if spec.tier == ModelTier.QUALIFIER and qualifier_fails:
            raise ModelLoadError("Qualifier model failed to load")
        if spec.tier == ModelTier.DRAFT and draft_fails:
            raise ModelLoadError("Draft model failed to load")
        mapping = {
            ModelTier.DRAFT: draft_loaded,
            ModelTier.QUALIFIER: qualifier_loaded,
            ModelTier.TARGET: target_loaded,
        }
        return mapping[spec.tier]

    loader.load.side_effect = load_side_effect

    # Try 3-model
    decoder_3 = ThreeModelDecoder(
        settings=settings,
        registry=registry,
        loader=loader,
        metrics=metrics,
    )
    try:
        decoder_3.load()
        return decoder_3
    except ModelLoadError:
        pass

    # Fallback: try 2-model
    loader_2 = MagicMock(spec=ModelLoader)

    def load_2_side_effect(spec, device="cpu"):
        if spec.tier == ModelTier.DRAFT and draft_fails:
            raise ModelLoadError("Draft model failed to load")
        mapping = {
            ModelTier.DRAFT: draft_loaded,
            ModelTier.TARGET: target_loaded,
        }
        return mapping[spec.tier]

    loader_2.load.side_effect = load_2_side_effect

    decoder_2 = TwoModelDecoder(
        settings=settings,
        registry=registry,
        loader=loader_2,
        metrics=metrics,
    )
    try:
        decoder_2.load()
        return decoder_2
    except ModelLoadError:
        pass

    # Fallback: 1-model (target only)
    return SingleModelDecoder(target_loaded, metrics)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestThreeToTwoModelFallback:
    """When qualifier fails to load, fall back to 2-model."""

    def test_three_to_two_model_fallback(self):
        decoder = _build_decoder_with_fallback(qualifier_fails=True)

        assert decoder.mode == "2model"
        assert decoder.is_loaded


class TestTwoToSingleModelFallback:
    """When draft model fails, fall back to target-only."""

    def test_two_to_single_model_fallback(self):
        decoder = _build_decoder_with_fallback(
            qualifier_fails=True,
            draft_fails=True,
        )

        assert decoder.mode == "1model"
        assert decoder.is_loaded


class TestFullFallbackChain:
    """3model -> 2model -> 1model fallback chain."""

    def test_full_fallback_chain(self):
        # All models work: should get 3model
        decoder = _build_decoder_with_fallback()
        assert decoder.mode == "3model"

    def test_qualifier_fails_gives_two_model(self):
        decoder = _build_decoder_with_fallback(qualifier_fails=True)
        assert decoder.mode == "2model"

    def test_both_fail_gives_single_model(self):
        decoder = _build_decoder_with_fallback(
            qualifier_fails=True, draft_fails=True,
        )
        assert decoder.mode == "1model"

    def test_fallback_decoder_can_generate(self):
        decoder = _build_decoder_with_fallback(
            qualifier_fails=True, draft_fails=True,
        )
        request = GenerationRequest(prompt="Hello", max_new_tokens=5)
        result = decoder.generate(request)

        assert result.text
        assert result.mode == "1model"
