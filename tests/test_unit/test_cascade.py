"""Unit tests for the cascade decoder."""

from unittest.mock import MagicMock, patch

import pytest

from momo_kibidango.core.cascade import CascadeDecoder
from momo_kibidango.core.confidence import ConfidenceResult, ConfidenceScorer
from momo_kibidango.core.decoder import GenerationRequest, GenerationResult
from momo_kibidango.exceptions import CascadeError
from momo_kibidango.models.claude_client import (
    CLAUDE_HAIKU,
    CLAUDE_OPUS,
    CLAUDE_SONNET,
    ClaudeClient,
    CostTracker,
    TokenUsage,
)


def _make_client() -> MagicMock:
    """Create a mock ClaudeClient."""
    client = MagicMock(spec=ClaudeClient)
    client.cost_tracker = CostTracker()
    return client


def _high_confidence() -> ConfidenceResult:
    return ConfidenceResult(score=0.9, reasoning=["good"], component_scores={"length": 0.9})


def _medium_confidence() -> ConfidenceResult:
    return ConfidenceResult(score=0.65, reasoning=["ok"], component_scores={"length": 0.65})


def _low_confidence() -> ConfidenceResult:
    return ConfidenceResult(score=0.3, reasoning=["bad"], component_scores={"length": 0.3})


# ── CascadeDecoder basics ──────────────────────────────────────────


class TestCascadeDecoderBasics:
    def test_mode(self):
        client = _make_client()
        decoder = CascadeDecoder(client=client)
        assert decoder.mode == "cascade"

    def test_not_loaded_initially(self):
        client = _make_client()
        decoder = CascadeDecoder(client=client)
        assert not decoder.is_loaded

    def test_load_unload(self):
        client = _make_client()
        decoder = CascadeDecoder(client=client)
        decoder.load()
        assert decoder.is_loaded
        decoder.unload()
        assert not decoder.is_loaded

    def test_cost_tracker_accessible(self):
        client = _make_client()
        decoder = CascadeDecoder(client=client)
        assert isinstance(decoder.cost_tracker, CostTracker)


# ── Tier escalation logic ──────────────────────────────────────────


class TestTierEscalation:
    def test_high_confidence_stays_at_haiku(self):
        client = _make_client()
        client.complete.return_value = ("Haiku response", TokenUsage(input_tokens=10, output_tokens=20, model=CLAUDE_HAIKU))
        client.cost_tracker = CostTracker()

        scorer = MagicMock(spec=ConfidenceScorer)
        scorer.score.return_value = _high_confidence()

        decoder = CascadeDecoder(client=client, scorer=scorer)
        decoder.load()

        request = GenerationRequest(prompt="Hi", max_new_tokens=100)
        result = decoder.generate(request)

        assert result.text == "Haiku response"
        assert result.stage_acceptance_rates["tier"] == "haiku"
        # Only one API call (Haiku)
        assert client.complete.call_count == 1

    def test_medium_confidence_escalates_to_sonnet(self):
        client = _make_client()
        client.complete.side_effect = [
            ("Haiku draft", TokenUsage(input_tokens=10, output_tokens=20, model=CLAUDE_HAIKU)),
            ("Sonnet refined", TokenUsage(input_tokens=50, output_tokens=40, model=CLAUDE_SONNET)),
        ]
        client.cost_tracker = CostTracker()

        scorer = MagicMock(spec=ConfidenceScorer)
        scorer.score.side_effect = [_medium_confidence(), _high_confidence()]

        decoder = CascadeDecoder(client=client, scorer=scorer)
        decoder.load()

        request = GenerationRequest(prompt="Explain something", max_new_tokens=100)
        result = decoder.generate(request)

        assert result.text == "Sonnet refined"
        assert result.stage_acceptance_rates["tier"] == "sonnet"
        assert client.complete.call_count == 2

    def test_low_confidence_escalates_to_opus(self):
        client = _make_client()
        client.complete.side_effect = [
            ("Haiku bad", TokenUsage(input_tokens=10, output_tokens=5, model=CLAUDE_HAIKU)),
            ("Opus best", TokenUsage(input_tokens=10, output_tokens=100, model=CLAUDE_OPUS)),
        ]
        client.cost_tracker = CostTracker()

        scorer = MagicMock(spec=ConfidenceScorer)
        scorer.score.side_effect = [_low_confidence(), _high_confidence()]

        decoder = CascadeDecoder(client=client, scorer=scorer)
        decoder.load()

        request = GenerationRequest(prompt="Prove P=NP", max_new_tokens=100)
        result = decoder.generate(request)

        assert result.text == "Opus best"
        assert result.stage_acceptance_rates["tier"] == "opus"
        # Haiku + Opus = 2 calls (skips Sonnet)
        assert client.complete.call_count == 2

    def test_custom_thresholds(self):
        client = _make_client()
        client.complete.return_value = ("response", TokenUsage(input_tokens=10, output_tokens=20, model=CLAUDE_HAIKU))
        client.cost_tracker = CostTracker()

        scorer = MagicMock(spec=ConfidenceScorer)
        # Score of 0.85 is below custom high_threshold of 0.95
        scorer.score.side_effect = [
            ConfidenceResult(score=0.85, reasoning=[], component_scores={}),
            _high_confidence(),
        ]

        decoder = CascadeDecoder(
            client=client, scorer=scorer, high_threshold=0.95, low_threshold=0.7
        )
        decoder.load()

        request = GenerationRequest(prompt="test", max_new_tokens=100)
        result = decoder.generate(request)

        # Should escalate to sonnet because 0.85 < 0.95 and >= 0.7
        assert result.stage_acceptance_rates["tier"] == "sonnet"


# ── Cost tracking ──────────────────────────────────────────────────


class TestCascadeCost:
    def test_cost_tracked_in_result(self):
        client = _make_client()
        usage = TokenUsage(input_tokens=100, output_tokens=50, model=CLAUDE_HAIKU)
        client.complete.return_value = ("response", usage)
        # Pre-record the usage so cost_tracker reflects it
        client.cost_tracker = CostTracker()

        def complete_side_effect(*args, **kwargs):
            client.cost_tracker.record(usage)
            return ("response", usage)

        client.complete.side_effect = complete_side_effect

        scorer = MagicMock(spec=ConfidenceScorer)
        scorer.score.return_value = _high_confidence()

        decoder = CascadeDecoder(client=client, scorer=scorer)
        decoder.load()

        request = GenerationRequest(prompt="test", max_new_tokens=100)
        result = decoder.generate(request)

        assert result.stage_acceptance_rates["cost_usd"] >= 0
        assert result.mode == "cascade"

    def test_cost_savings_haiku_vs_opus(self):
        tracker = CostTracker()
        tracker.record(TokenUsage(input_tokens=1000, output_tokens=500, model=CLAUDE_HAIKU))
        assert tracker.savings_usd > 0
        summary = tracker.summary()
        assert summary["savings_usd"] > 0


# ── GenerationResult shape ─────────────────────────────────────────


class TestCascadeResult:
    def test_result_has_correct_fields(self):
        client = _make_client()
        client.complete.return_value = ("output text here", TokenUsage(input_tokens=10, output_tokens=20, model=CLAUDE_HAIKU))
        client.cost_tracker = CostTracker()

        scorer = MagicMock(spec=ConfidenceScorer)
        scorer.score.return_value = _high_confidence()

        decoder = CascadeDecoder(client=client, scorer=scorer)
        decoder.load()

        result = decoder.generate(GenerationRequest(prompt="test"))

        assert isinstance(result, GenerationResult)
        assert isinstance(result.text, str)
        assert result.tokens_generated > 0
        assert result.elapsed_seconds >= 0
        assert result.mode == "cascade"
        assert result.peak_memory_gb == 0.0  # API-based


# ── Error handling ─────────────────────────────────────────────────


class TestCascadeErrors:
    def test_api_error_raises_cascade_error(self):
        client = _make_client()
        client.complete.side_effect = Exception("API failure")
        client.cost_tracker = CostTracker()

        scorer = MagicMock(spec=ConfidenceScorer)

        decoder = CascadeDecoder(client=client, scorer=scorer)
        decoder.load()

        with pytest.raises(CascadeError, match="Cascade generation failed"):
            decoder.generate(GenerationRequest(prompt="test"))

    def test_stream_uses_haiku(self):
        client = _make_client()
        client.stream.return_value = iter(["chunk1", "chunk2"])
        client.cost_tracker = CostTracker()

        decoder = CascadeDecoder(client=client)
        decoder.load()

        chunks = list(decoder.stream(GenerationRequest(prompt="test")))
        assert chunks == ["chunk1", "chunk2"]
        client.stream.assert_called_once()
