"""Integration tests for the 2-model speculative decoder pipeline."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
import torch

from momo_kibidango.config.settings import DecoderSettings
from momo_kibidango.core.adaptive import AdaptiveThreshold
from momo_kibidango.core.decoder import GenerationRequest
from momo_kibidango.core.two_model import TwoModelDecoder
from momo_kibidango.models.loader import LoadedModel, ModelLoader
from momo_kibidango.models.registry import ModelRegistry, ModelSpec, ModelTier
from momo_kibidango.monitoring.metrics import MetricsCollector


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

VOCAB_SIZE = 128


def _make_mock_tokenizer(eos_token_id: int = 2) -> MagicMock:
    """Return a mock tokenizer that produces deterministic tensors."""
    tok = MagicMock()
    tok.eos_token_id = eos_token_id
    tok.encode.return_value = torch.tensor([[1, 10, 20]])
    tok.decode.return_value = "hello world"
    return tok


def _make_mock_model_output(batch_size: int = 1, seq_len: int = 5) -> MagicMock:
    """Return a mock model output with logits of the proper shape."""
    output = MagicMock()
    logits = torch.randn(batch_size, seq_len, VOCAB_SIZE)
    # Make token 42 highly probable at every position
    logits[:, :, 42] = 10.0
    output.logits = logits
    return output


def _make_mock_model() -> MagicMock:
    """Return a mock model whose __call__ returns mock output."""
    model = MagicMock()

    def model_call(input_ids):
        seq_len = input_ids.shape[1]
        return _make_mock_model_output(batch_size=1, seq_len=seq_len)

    model.side_effect = model_call
    return model


def _build_decoder(
    with_adaptive: bool = False,
) -> tuple[TwoModelDecoder, MagicMock]:
    """Construct a TwoModelDecoder with mocked loader."""
    settings = DecoderSettings(
        draft_model_id="test/draft-0.5b",
        target_model_id="test/target-7b",
        device="cpu",
        max_draft_tokens=3,
    )

    registry = ModelRegistry()
    registry.register(ModelSpec(model_id="test/draft-0.5b", tier=ModelTier.DRAFT))
    registry.register(ModelSpec(model_id="test/target-7b", tier=ModelTier.TARGET))

    loader = MagicMock(spec=ModelLoader)
    metrics = MagicMock()
    adaptive = AdaptiveThreshold() if with_adaptive else None

    decoder = TwoModelDecoder(
        settings=settings,
        registry=registry,
        loader=loader,
        metrics=metrics,
        adaptive=adaptive,
    )
    return decoder, loader


def _setup_loader_mocks(loader_mock: MagicMock) -> None:
    """Configure the loader mock to return LoadedModels with mock internals."""
    draft_tok = _make_mock_tokenizer()
    target_tok = _make_mock_tokenizer()

    draft_model = _make_mock_model()
    target_model = _make_mock_model()

    draft_loaded = LoadedModel(
        model=draft_model,
        tokenizer=draft_tok,
        spec=ModelSpec(model_id="test/draft-0.5b", tier=ModelTier.DRAFT),
        memory_footprint_gb=0.5,
    )
    target_loaded = LoadedModel(
        model=target_model,
        tokenizer=target_tok,
        spec=ModelSpec(model_id="test/target-7b", tier=ModelTier.TARGET),
        memory_footprint_gb=4.0,
    )

    def load_side_effect(spec, device="cpu"):
        if spec.tier == ModelTier.DRAFT:
            return draft_loaded
        return target_loaded

    loader_mock.load.side_effect = load_side_effect


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestBuildTwoModelDecoder:
    """Verify construction with real settings, registry, and mocked loader."""

    def test_build_two_model_decoder(self):
        decoder, loader = _build_decoder()
        _setup_loader_mocks(loader)

        assert decoder.mode == "2model"
        assert not decoder.is_loaded

        decoder.load()

        assert decoder.is_loaded
        assert loader.load.call_count == 2

    def test_load_called_with_correct_specs(self):
        decoder, loader = _build_decoder()
        _setup_loader_mocks(loader)
        decoder.load()

        calls = loader.load.call_args_list
        assert calls[0][0][0].tier == ModelTier.DRAFT
        assert calls[1][0][0].tier == ModelTier.TARGET


class TestGenerateMockPipeline:
    """Full generate() call with mocked models returning fake logits."""

    def test_generate_mock_pipeline(self):
        decoder, loader = _build_decoder()
        _setup_loader_mocks(loader)
        decoder.load()

        request = GenerationRequest(
            prompt="Hello",
            max_new_tokens=5,
            temperature=0.7,
        )

        with patch("momo_kibidango.core.two_model.psutil") as mock_psutil:
            mock_psutil.Process.return_value.memory_info.return_value.rss = 4 * 1024**3
            result = decoder.generate(request)

        assert result.text == "hello world"
        assert result.mode == "2model"
        assert result.tokens_generated > 0
        assert result.elapsed_seconds > 0

    def test_generate_returns_generation_result(self):
        decoder, loader = _build_decoder()
        _setup_loader_mocks(loader)
        decoder.load()

        request = GenerationRequest(prompt="Test", max_new_tokens=3)

        with patch("momo_kibidango.core.two_model.psutil") as mock_psutil:
            mock_psutil.Process.return_value.memory_info.return_value.rss = 2 * 1024**3
            result = decoder.generate(request)

        assert hasattr(result, "tokens_per_second")
        assert hasattr(result, "acceptance_rate")
        assert hasattr(result, "stage_acceptance_rates")


class TestAdaptiveThresholdIntegration:
    """Verify adaptive thresholds update during generation."""

    def test_adaptive_threshold_integration(self):
        decoder, loader = _build_decoder(with_adaptive=True)
        _setup_loader_mocks(loader)
        decoder.load()

        initial_threshold = decoder._adaptive.stage2_threshold

        request = GenerationRequest(prompt="Hello", max_new_tokens=5)

        with patch("momo_kibidango.core.two_model.psutil") as mock_psutil:
            mock_psutil.Process.return_value.memory_info.return_value.rss = 2 * 1024**3
            decoder.generate(request)

        # The adaptive controller should have been called (update_count > 0)
        assert decoder._adaptive._states["stage2"].update_count > 0


class TestUnloadCleansUp:
    """After unload, is_loaded is False."""

    def test_unload_cleans_up(self):
        decoder, loader = _build_decoder()
        _setup_loader_mocks(loader)
        decoder.load()
        assert decoder.is_loaded

        decoder.unload()

        assert not decoder.is_loaded
        assert decoder._draft is None
        assert decoder._target is None
        loader.unload_all.assert_called_once()
