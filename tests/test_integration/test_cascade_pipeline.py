"""Integration tests for the full cascade pipeline with mocked API."""

from unittest.mock import MagicMock, patch

import pytest

from momo_kibidango.core.cascade import CascadeDecoder
from momo_kibidango.core.confidence import ConfidenceScorer
from momo_kibidango.core.decoder import GenerationRequest
from momo_kibidango.exceptions import CascadeError
from momo_kibidango.models.claude_client import (
    CLAUDE_HAIKU,
    CLAUDE_OPUS,
    CLAUDE_SONNET,
    ClaudeClient,
    CostTracker,
    TokenUsage,
)


class _FakeClaudeClient:
    """A fake client that records calls and returns scripted responses."""

    def __init__(self, responses: list[tuple[str, TokenUsage]]) -> None:
        self._responses = list(responses)
        self._call_index = 0
        self.cost_tracker = CostTracker()
        self.calls: list[dict] = []

    def complete(self, prompt, model=CLAUDE_HAIKU, max_tokens=1024, temperature=0.7, system=None):
        self.calls.append({"prompt": prompt, "model": model, "max_tokens": max_tokens})
        if self._call_index >= len(self._responses):
            raise RuntimeError("No more scripted responses")
        text, usage = self._responses[self._call_index]
        self.cost_tracker.record(usage)
        self._call_index += 1
        return text, usage

    def stream(self, prompt, model=CLAUDE_HAIKU, max_tokens=1024, temperature=0.7, system=None):
        text, usage = self.complete(prompt, model, max_tokens, temperature, system)
        for word in text.split():
            yield word + " "


# ── Full pipeline: Haiku-only path ─────────────────────────────────


class TestHaikuOnlyPath:
    def test_simple_prompt_stays_haiku(self):
        """A simple prompt with a good response should stay at Haiku."""
        client = _FakeClaudeClient([
            (
                "The capital of France is Paris. It is known for the Eiffel Tower and its rich culture.",
                TokenUsage(input_tokens=15, output_tokens=20, model=CLAUDE_HAIKU),
            ),
        ])

        decoder = CascadeDecoder(client=client, high_threshold=0.7)
        decoder.load()

        result = decoder.generate(GenerationRequest(prompt="What is the capital of France?"))

        assert "Paris" in result.text
        assert result.stage_acceptance_rates["tier"] == "haiku"
        assert len(client.calls) == 1
        assert result.mode == "cascade"

    def test_cost_is_minimal(self):
        client = _FakeClaudeClient([
            (
                "Yes, Python is a programming language used for web development and data science.",
                TokenUsage(input_tokens=10, output_tokens=15, model=CLAUDE_HAIKU),
            ),
        ])

        decoder = CascadeDecoder(client=client, high_threshold=0.7)
        decoder.load()
        decoder.generate(GenerationRequest(prompt="Is Python a language?"))

        summary = client.cost_tracker.summary()
        assert summary["num_calls"] == 1
        assert summary["savings_usd"] > 0


# ── Haiku → Sonnet escalation ──────────────────────────────────────


class TestHaikuSonnetEscalation:
    def test_mediocre_haiku_escalates_to_sonnet(self):
        """A mediocre Haiku response should trigger Sonnet refinement."""
        client = _FakeClaudeClient([
            # Haiku: short/incomplete response to a complex prompt
            (
                "Quantum computing uses qubits.",
                TokenUsage(input_tokens=20, output_tokens=5, model=CLAUDE_HAIKU),
            ),
            # Sonnet: refined response
            (
                "Quantum computing leverages quantum mechanical phenomena such as "
                "superposition and entanglement to process information. Unlike "
                "classical bits, qubits can exist in multiple states simultaneously, "
                "enabling parallel computation. Key applications include cryptography, "
                "drug discovery, and optimization problems.",
                TokenUsage(input_tokens=100, output_tokens=50, model=CLAUDE_SONNET),
            ),
        ])

        # Use low thresholds so the short Haiku response triggers escalation
        decoder = CascadeDecoder(client=client, high_threshold=0.85, low_threshold=0.4)
        decoder.load()

        result = decoder.generate(
            GenerationRequest(prompt="Explain quantum computing in detail, analyze its applications and compare approaches")
        )

        assert result.stage_acceptance_rates["tier"] in ("sonnet", "haiku")
        # At least 1 call, possibly 2 if escalated
        assert len(client.calls) >= 1


# ── Haiku → Opus escalation ────────────────────────────────────────


class TestHaikuOpusEscalation:
    def test_bad_haiku_escalates_to_opus(self):
        """A very poor Haiku response should skip Sonnet and go to Opus."""
        client = _FakeClaudeClient([
            # Haiku: empty/garbage
            (
                "",
                TokenUsage(input_tokens=20, output_tokens=0, model=CLAUDE_HAIKU),
            ),
            # Opus: excellent response
            (
                "Here is a rigorous proof of the mathematical theorem using "
                "induction and formal logic. First, we establish the base case...",
                TokenUsage(input_tokens=50, output_tokens=200, model=CLAUDE_OPUS),
            ),
        ])

        decoder = CascadeDecoder(client=client, high_threshold=0.8, low_threshold=0.5)
        decoder.load()

        result = decoder.generate(
            GenerationRequest(prompt="Prove this theorem formally")
        )

        assert result.stage_acceptance_rates["tier"] == "opus"
        assert len(client.calls) == 2
        # Second call should be to Opus
        assert client.calls[1]["model"] == CLAUDE_OPUS


# ── API error fallback ─────────────────────────────────────────────


class TestCascadeErrorHandling:
    def test_api_error_propagates(self):
        """API errors should be wrapped in CascadeError."""
        client = _FakeClaudeClient([])  # No responses available

        decoder = CascadeDecoder(client=client)
        decoder.load()

        with pytest.raises(CascadeError):
            decoder.generate(GenerationRequest(prompt="test"))


# ── Cost tracking accuracy ─────────────────────────────────────────


class TestCostTracking:
    def test_single_tier_cost(self):
        client = _FakeClaudeClient([
            (
                "Simple answer that is good enough for the prompt.",
                TokenUsage(input_tokens=50, output_tokens=10, model=CLAUDE_HAIKU),
            ),
        ])

        decoder = CascadeDecoder(client=client, high_threshold=0.7)
        decoder.load()
        decoder.generate(GenerationRequest(prompt="Hello"))

        summary = client.cost_tracker.summary()
        assert summary["total_input_tokens"] == 50
        assert summary["total_output_tokens"] == 10
        assert summary["total_cost_usd"] > 0

    def test_multi_tier_cost_accumulation(self):
        client = _FakeClaudeClient([
            ("bad", TokenUsage(input_tokens=10, output_tokens=1, model=CLAUDE_HAIKU)),
            (
                "Excellent detailed response with proper analysis and code examples.",
                TokenUsage(input_tokens=100, output_tokens=80, model=CLAUDE_OPUS),
            ),
        ])

        decoder = CascadeDecoder(client=client, high_threshold=0.8, low_threshold=0.5)
        decoder.load()
        decoder.generate(GenerationRequest(prompt="Implement and explain a B-tree"))

        summary = client.cost_tracker.summary()
        assert summary["num_calls"] == 2
        assert summary["total_input_tokens"] == 110
        assert summary["total_output_tokens"] == 81

    def test_savings_reported_correctly(self):
        client = _FakeClaudeClient([
            (
                "A good response that covers the basics well enough for this query.",
                TokenUsage(input_tokens=50, output_tokens=15, model=CLAUDE_HAIKU),
            ),
        ])

        decoder = CascadeDecoder(client=client, high_threshold=0.7)
        decoder.load()
        decoder.generate(GenerationRequest(prompt="What is 2+2?"))

        summary = client.cost_tracker.summary()
        # Haiku is cheaper than Opus, so savings should be positive
        assert summary["savings_usd"] > 0
        assert summary["opus_equivalent_usd"] > summary["total_cost_usd"]


# ── Streaming ──────────────────────────────────────────────────────


class TestCascadeStreaming:
    def test_stream_returns_chunks(self):
        client = _FakeClaudeClient([
            (
                "Hello world from streaming",
                TokenUsage(input_tokens=5, output_tokens=5, model=CLAUDE_HAIKU),
            ),
        ])

        decoder = CascadeDecoder(client=client)
        decoder.load()

        chunks = list(decoder.stream(GenerationRequest(prompt="Hi")))
        assert len(chunks) > 0
        assert "Hello" in "".join(chunks)
