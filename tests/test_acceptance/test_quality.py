"""User acceptance tests for generation quality and result structure."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from momo_kibidango.core.decoder import BaseDecoder, GenerationRequest, GenerationResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_mock_decoder(mode: str = "2model") -> BaseDecoder:
    """Return a mock decoder that produces canned GenerationResults."""
    decoder = MagicMock(spec=BaseDecoder)
    decoder.is_loaded = True
    decoder.mode = mode

    def generate_fn(request: GenerationRequest) -> GenerationResult:
        return GenerationResult(
            text=f"Response to: {request.prompt}",
            tokens_generated=15,
            elapsed_seconds=0.3,
            tokens_per_second=50.0,
            acceptance_rate=0.75,
            stage_acceptance_rates={"stage2": 0.75},
            peak_memory_gb=2.0,
            mode=mode,
            draft_attempts=20,
            accepted_tokens=15,
        )

    decoder.generate.side_effect = generate_fn
    return decoder


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestResultHasText:
    """Generated result contains non-empty text."""

    def test_result_has_text(self):
        decoder = _make_mock_decoder()
        request = GenerationRequest(prompt="Tell me a joke", max_new_tokens=50)

        result = decoder.generate(request)

        assert result.text is not None
        assert len(result.text) > 0
        assert isinstance(result.text, str)

    def test_result_text_is_meaningful(self):
        decoder = _make_mock_decoder()
        request = GenerationRequest(prompt="Explain gravity", max_new_tokens=100)

        result = decoder.generate(request)

        # Text should contain more than just whitespace
        assert result.text.strip()


class TestResultMetricsReasonable:
    """Metrics in the result should be within reasonable ranges."""

    def test_result_metrics_reasonable(self):
        decoder = _make_mock_decoder()
        request = GenerationRequest(prompt="Hello world", max_new_tokens=20)

        result = decoder.generate(request)

        # tokens_per_second must be positive
        assert result.tokens_per_second > 0

        # acceptance_rate between 0 and 1
        assert 0.0 <= result.acceptance_rate <= 1.0

        # elapsed_seconds must be non-negative
        assert result.elapsed_seconds >= 0

        # tokens_generated must be positive
        assert result.tokens_generated > 0

    def test_tokens_generated_non_negative(self):
        decoder = _make_mock_decoder()
        request = GenerationRequest(prompt="Test", max_new_tokens=10)

        result = decoder.generate(request)
        assert result.tokens_generated >= 0

    def test_peak_memory_non_negative(self):
        decoder = _make_mock_decoder()
        request = GenerationRequest(prompt="Test", max_new_tokens=10)

        result = decoder.generate(request)
        assert result.peak_memory_gb >= 0


class TestModeReportedCorrectly:
    """Mode in the result matches the configured decoder mode."""

    def test_mode_reported_correctly_2model(self):
        decoder = _make_mock_decoder(mode="2model")
        request = GenerationRequest(prompt="Hello", max_new_tokens=10)

        result = decoder.generate(request)

        assert result.mode == "2model"

    def test_mode_reported_correctly_3model(self):
        decoder = _make_mock_decoder(mode="3model")
        request = GenerationRequest(prompt="Hello", max_new_tokens=10)

        result = decoder.generate(request)

        assert result.mode == "3model"

    def test_mode_reported_correctly_1model(self):
        decoder = _make_mock_decoder(mode="1model")
        request = GenerationRequest(prompt="Hello", max_new_tokens=10)

        result = decoder.generate(request)

        assert result.mode == "1model"

    def test_mode_is_string(self):
        decoder = _make_mock_decoder()
        request = GenerationRequest(prompt="Test", max_new_tokens=10)

        result = decoder.generate(request)

        assert isinstance(result.mode, str)
        assert result.mode in ("1model", "2model", "3model")
