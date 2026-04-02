"""Unit tests for base decoder interface and data classes."""

import pytest
from dataclasses import FrozenInstanceError
from unittest.mock import MagicMock

from momo_kibidango.core.decoder import (
    BaseDecoder,
    GenerationRequest,
    GenerationResult,
)


class TestGenerationRequest:
    """Test GenerationRequest frozen dataclass."""

    def test_generation_request_frozen(self, sample_request):
        with pytest.raises(FrozenInstanceError):
            sample_request.prompt = "other"

    def test_generation_request_frozen_max_tokens(self, sample_request):
        with pytest.raises(FrozenInstanceError):
            sample_request.max_new_tokens = 100

    def test_generation_request_defaults(self):
        req = GenerationRequest(prompt="test")
        assert req.prompt == "test"
        assert req.max_new_tokens == 256
        assert req.temperature == 0.7
        assert req.top_p == 0.9
        assert req.stop_sequences == []

    def test_generation_request_custom(self):
        req = GenerationRequest(
            prompt="hello",
            max_new_tokens=128,
            temperature=1.0,
            top_p=0.95,
            stop_sequences=["<end>"],
        )
        assert req.prompt == "hello"
        assert req.max_new_tokens == 128
        assert req.temperature == 1.0
        assert req.top_p == 0.95
        assert req.stop_sequences == ["<end>"]

    def test_generation_request_stop_sequences_default_independent(self):
        """Each instance should get its own default list."""
        req1 = GenerationRequest(prompt="a")
        req2 = GenerationRequest(prompt="b")
        assert req1.stop_sequences is not req2.stop_sequences


class TestGenerationResult:
    """Test GenerationResult dataclass."""

    def test_generation_result_fields(self):
        result = GenerationResult(
            text="Hello world",
            tokens_generated=10,
            elapsed_seconds=1.5,
            tokens_per_second=6.67,
            acceptance_rate=0.80,
            stage_acceptance_rates={"stage1": 0.85, "stage2": 0.75},
            peak_memory_gb=4.5,
            mode="2model",
            draft_attempts=12,
            accepted_tokens=8,
        )
        assert result.text == "Hello world"
        assert result.tokens_generated == 10
        assert result.elapsed_seconds == 1.5
        assert result.tokens_per_second == pytest.approx(6.67)
        assert result.acceptance_rate == 0.80
        assert result.stage_acceptance_rates == {"stage1": 0.85, "stage2": 0.75}
        assert result.peak_memory_gb == 4.5
        assert result.mode == "2model"
        assert result.draft_attempts == 12
        assert result.accepted_tokens == 8

    def test_generation_result_mutable(self):
        """GenerationResult is not frozen, so fields can be modified."""
        result = GenerationResult(
            text="a", tokens_generated=1, elapsed_seconds=0.1,
            tokens_per_second=10.0, acceptance_rate=1.0,
            stage_acceptance_rates={}, peak_memory_gb=1.0,
            mode="2model", draft_attempts=1, accepted_tokens=1,
        )
        result.text = "updated"
        assert result.text == "updated"


class TestBaseDecoder:
    """Test BaseDecoder abstract class."""

    def test_base_decoder_abstract(self):
        """Cannot instantiate BaseDecoder directly."""
        with pytest.raises(TypeError):
            BaseDecoder()

    def test_base_decoder_requires_all_abstract(self):
        """Subclass missing abstract methods cannot be instantiated."""

        class IncompleteDecoder(BaseDecoder):
            pass

        with pytest.raises(TypeError):
            IncompleteDecoder()

    def test_stream_default(self):
        """Default stream() yields the full text from generate()."""

        class ConcreteDecoder(BaseDecoder):
            def load(self):
                pass

            def generate(self, request):
                return GenerationResult(
                    text="streamed output",
                    tokens_generated=5,
                    elapsed_seconds=0.5,
                    tokens_per_second=10.0,
                    acceptance_rate=0.8,
                    stage_acceptance_rates={},
                    peak_memory_gb=1.0,
                    mode="2model",
                    draft_attempts=5,
                    accepted_tokens=4,
                )

            def unload(self):
                pass

            @property
            def mode(self):
                return "2model"

            @property
            def is_loaded(self):
                return True

        decoder = ConcreteDecoder()
        request = GenerationRequest(prompt="test")
        chunks = list(decoder.stream(request))
        assert chunks == ["streamed output"]
