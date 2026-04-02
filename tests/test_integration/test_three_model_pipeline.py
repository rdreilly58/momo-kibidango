"""Integration tests for the 3-model speculative decoder pipeline."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
import torch

from momo_kibidango.config.settings import DecoderSettings
from momo_kibidango.core.adaptive import AdaptiveThreshold
from momo_kibidango.core.decoder import GenerationRequest
from momo_kibidango.core.three_model import ThreeModelDecoder
from momo_kibidango.models.loader import LoadedModel, ModelLoader
from momo_kibidango.models.registry import ModelRegistry, ModelSpec, ModelTier
from momo_kibidango.monitoring.metrics import MetricsCollector


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

VOCAB_SIZE = 128


def _make_mock_tokenizer(eos_token_id: int = 2, name: str = "default") -> MagicMock:
    tok = MagicMock()
    tok.eos_token_id = eos_token_id
    tok.encode.return_value = torch.tensor([[1, 10, 20]])
    tok.decode.return_value = "generated text"
    tok._name = name
    return tok


def _make_mock_model(accept_prob: float = 0.9) -> MagicMock:
    """Return a mock model. accept_prob controls how likely token 42 is."""
    model = MagicMock()

    def model_call(input_ids):
        seq_len = input_ids.shape[1]
        output = MagicMock()
        logits = torch.randn(1, seq_len, VOCAB_SIZE)
        logits[:, :, 42] = accept_prob * 20  # make token 42 dominant
        output.logits = logits
        return output

    model.side_effect = model_call
    return model


def _make_rejecting_model() -> MagicMock:
    """Return a model that gives uniform probabilities (low per-token prob)."""
    model = MagicMock()

    def model_call(input_ids):
        seq_len = input_ids.shape[1]
        output = MagicMock()
        # Uniform logits -- each token gets ~1/VOCAB_SIZE probability
        logits = torch.zeros(1, seq_len, VOCAB_SIZE)
        output.logits = logits
        return output

    model.side_effect = model_call
    return model


def _build_decoder(
    same_tokenizers: bool = True,
    qualifier_rejects: bool = False,
) -> tuple[ThreeModelDecoder, MagicMock]:
    settings = DecoderSettings(
        draft_model_id="test/draft-0.5b",
        qualifier_model_id="test/qualifier-1.5b",
        target_model_id="test/target-7b",
        device="cpu",
        max_draft_tokens=3,
        stage1_threshold=0.05,
        stage2_threshold=0.03,
    )

    registry = ModelRegistry()
    draft_spec = ModelSpec(
        model_id="test/draft-0.5b",
        tier=ModelTier.DRAFT,
        tokenizer_id="shared-tok" if same_tokenizers else "draft-tok",
    )
    qualifier_spec = ModelSpec(
        model_id="test/qualifier-1.5b",
        tier=ModelTier.QUALIFIER,
        tokenizer_id="shared-tok" if same_tokenizers else "qual-tok",
    )
    target_spec = ModelSpec(
        model_id="test/target-7b",
        tier=ModelTier.TARGET,
        tokenizer_id="shared-tok" if same_tokenizers else "target-tok",
    )
    registry.register(draft_spec)
    registry.register(qualifier_spec)
    registry.register(target_spec)

    loader = MagicMock(spec=ModelLoader)
    metrics = MagicMock()
    adaptive = AdaptiveThreshold()

    decoder = ThreeModelDecoder(
        settings=settings,
        registry=registry,
        loader=loader,
        metrics=metrics,
        adaptive=adaptive,
    )

    # Configure loader mock
    draft_tok = _make_mock_tokenizer(name="draft")
    qual_tok = draft_tok if same_tokenizers else _make_mock_tokenizer(name="qual")
    target_tok = draft_tok if same_tokenizers else _make_mock_tokenizer(name="target")

    qual_model = _make_rejecting_model() if qualifier_rejects else _make_mock_model()

    draft_loaded = LoadedModel(
        model=_make_mock_model(),
        tokenizer=draft_tok,
        spec=draft_spec,
    )
    qual_loaded = LoadedModel(
        model=qual_model,
        tokenizer=qual_tok,
        spec=qualifier_spec,
    )
    target_loaded = LoadedModel(
        model=_make_mock_model(),
        tokenizer=target_tok,
        spec=target_spec,
    )

    def load_side_effect(spec, device="cpu"):
        mapping = {
            ModelTier.DRAFT: draft_loaded,
            ModelTier.QUALIFIER: qual_loaded,
            ModelTier.TARGET: target_loaded,
        }
        return mapping[spec.tier]

    loader.load.side_effect = load_side_effect

    return decoder, loader


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestBuildThreeModelDecoder:
    """Construction includes qualifier model."""

    def test_build_three_model_decoder(self):
        decoder, loader = _build_decoder()
        assert decoder.mode == "3model"
        assert not decoder.is_loaded

        decoder.load()

        assert decoder.is_loaded
        assert loader.load.call_count == 3

    def test_load_creates_all_three_models(self):
        decoder, loader = _build_decoder()
        decoder.load()

        assert decoder._draft is not None
        assert decoder._qualifier is not None
        assert decoder._target is not None


class TestQualifyStepFilters:
    """Some tokens are rejected by the qualifier."""

    def test_qualify_step_filters(self):
        decoder, loader = _build_decoder(qualifier_rejects=True)
        decoder.load()

        request = GenerationRequest(prompt="Test filtering", max_new_tokens=5)

        with patch("momo_kibidango.core.three_model.psutil") as mock_psutil:
            mock_psutil.Process.return_value.memory_info.return_value.rss = 2 * 1024**3
            result = decoder.generate(request)

        # Generation should still complete even with qualifier rejections
        assert result.tokens_generated > 0
        assert result.mode == "3model"


class TestFallbackToTargetSample:
    """When qualifier rejects all tokens, fallback to target sample."""

    def test_fallback_to_target_sample(self):
        # Use a very high stage1 threshold so qualifier rejects everything
        decoder, loader = _build_decoder(qualifier_rejects=True)
        # Override stage1 threshold to be very strict
        decoder._settings = DecoderSettings(
            draft_model_id="test/draft-0.5b",
            qualifier_model_id="test/qualifier-1.5b",
            target_model_id="test/target-7b",
            device="cpu",
            max_draft_tokens=3,
            stage1_threshold=0.99,  # nearly impossible to pass
            stage2_threshold=0.03,
        )
        # Disable adaptive so thresholds stay at the configured values
        decoder._adaptive = None
        decoder.load()

        request = GenerationRequest(prompt="Test fallback", max_new_tokens=3)

        with patch("momo_kibidango.core.three_model.psutil") as mock_psutil:
            mock_psutil.Process.return_value.memory_info.return_value.rss = 2 * 1024**3
            result = decoder.generate(request)

        # Should still produce tokens via the fallback path
        assert result.tokens_generated > 0


class TestTokenizerBridgeUsed:
    """Verify bridge is created when tokenizers differ."""

    def test_tokenizer_bridge_used(self):
        decoder, loader = _build_decoder(same_tokenizers=False)
        decoder.load()

        # When tokenizer IDs differ, bridges should be created
        assert decoder._bridge_draft_to_qual is not None
        assert decoder._bridge_qual_to_target is not None
