"""Acceptance tests for cascade quality using realistic mocked responses.

These tests verify that the cascade correctly routes different types of
prompts to appropriate tiers based on response quality.
"""

from unittest.mock import MagicMock

import pytest

from momo_kibidango.core.cascade import CascadeDecoder
from momo_kibidango.core.confidence import ConfidenceScorer
from momo_kibidango.core.decoder import GenerationRequest
from momo_kibidango.models.claude_client import (
    CLAUDE_HAIKU,
    CLAUDE_OPUS,
    CLAUDE_SONNET,
    CostTracker,
    TokenUsage,
)


class _ScriptedClient:
    """Client that returns scripted responses per model tier."""

    def __init__(self, tier_responses: dict[str, str]) -> None:
        self._tier_responses = tier_responses
        self.cost_tracker = CostTracker()
        self.models_called: list[str] = []

    def complete(self, prompt, model=CLAUDE_HAIKU, max_tokens=1024, temperature=0.7, system=None):
        self.models_called.append(model)
        text = self._tier_responses.get(model, "fallback response")
        usage = TokenUsage(
            input_tokens=len(prompt.split()) * 2,
            output_tokens=len(text.split()) * 2,
            model=model,
        )
        self.cost_tracker.record(usage)
        return text, usage

    def stream(self, prompt, model=CLAUDE_HAIKU, max_tokens=1024, temperature=0.7, system=None):
        text, _ = self.complete(prompt, model, max_tokens, temperature, system)
        yield text


# ── Simple prompts → should stay at Haiku ──────────────────────────


class TestSimplePrompts:
    """Simple factual prompts should be handled entirely by Haiku."""

    def test_greeting(self):
        client = _ScriptedClient({
            CLAUDE_HAIKU: "Hello! How can I help you today? I'm happy to assist with any questions.",
        })
        decoder = CascadeDecoder(client=client, high_threshold=0.7)
        decoder.load()

        result = decoder.generate(GenerationRequest(prompt="Hello"))
        assert result.stage_acceptance_rates["tier"] == "haiku"
        assert len(client.models_called) == 1

    def test_simple_factual_question(self):
        client = _ScriptedClient({
            CLAUDE_HAIKU: (
                "Water boils at 100 degrees Celsius (212 degrees Fahrenheit) "
                "at standard atmospheric pressure at sea level."
            ),
        })
        decoder = CascadeDecoder(client=client, high_threshold=0.7)
        decoder.load()

        result = decoder.generate(
            GenerationRequest(prompt="At what temperature does water boil?")
        )
        assert result.stage_acceptance_rates["tier"] == "haiku"

    def test_simple_math(self):
        client = _ScriptedClient({
            CLAUDE_HAIKU: "The answer is 42. When you multiply 6 by 7, you get 42.",
        })
        decoder = CascadeDecoder(client=client, high_threshold=0.7)
        decoder.load()

        result = decoder.generate(GenerationRequest(prompt="What is 6 times 7?"))
        assert result.stage_acceptance_rates["tier"] == "haiku"


# ── Complex reasoning → should escalate ────────────────────────────


class TestComplexReasoning:
    """Complex reasoning prompts should escalate to Sonnet or Opus."""

    def test_complex_analysis_with_poor_draft(self):
        """If Haiku gives a shallow answer to a complex prompt, should escalate."""
        client = _ScriptedClient({
            CLAUDE_HAIKU: "Machine learning is good.",
            CLAUDE_SONNET: (
                "Machine learning encompasses supervised, unsupervised, and "
                "reinforcement learning paradigms. In supervised learning, models "
                "learn from labeled data to make predictions. Unsupervised learning "
                "discovers hidden patterns. Reinforcement learning optimizes decisions "
                "through trial and error with reward signals. Key trade-offs include "
                "bias-variance balance, interpretability versus accuracy, and "
                "computational cost versus model complexity."
            ),
            CLAUDE_OPUS: (
                "A comprehensive analysis of machine learning paradigms reveals "
                "fundamental trade-offs in algorithm design and deployment."
            ),
        })
        decoder = CascadeDecoder(client=client, high_threshold=0.85, low_threshold=0.4)
        decoder.load()

        result = decoder.generate(
            GenerationRequest(
                prompt="Explain and compare machine learning paradigms, analyze trade-offs in detail"
            )
        )

        # Should escalate beyond Haiku due to shallow draft
        assert len(client.models_called) >= 2

    def test_empty_draft_escalates_to_opus(self):
        """An empty draft should immediately escalate to Opus."""
        client = _ScriptedClient({
            CLAUDE_HAIKU: "",
            CLAUDE_OPUS: (
                "Here is a thorough proof by mathematical induction. "
                "Base case: For n=1, the statement holds trivially. "
                "Inductive step: Assume the statement holds for n=k. "
                "We must show it holds for n=k+1..."
            ),
        })
        decoder = CascadeDecoder(client=client, high_threshold=0.8, low_threshold=0.5)
        decoder.load()

        result = decoder.generate(
            GenerationRequest(prompt="Prove by induction that the sum of first n integers equals n(n+1)/2")
        )

        assert result.stage_acceptance_rates["tier"] == "opus"
        assert CLAUDE_OPUS in client.models_called


# ── Code generation → appropriate escalation ───────────────────────


class TestCodeGeneration:
    """Code generation prompts should escalate when the draft lacks actual code."""

    def test_code_request_with_code_response(self):
        client = _ScriptedClient({
            CLAUDE_HAIKU: (
                "Here's a Python function to sort a list:\n\n"
                "def sort_list(items):\n"
                "    return sorted(items)\n\n"
                "This uses Python's built-in sorted() function."
            ),
        })
        decoder = CascadeDecoder(client=client, high_threshold=0.7)
        decoder.load()

        result = decoder.generate(
            GenerationRequest(prompt="Write a function to sort a list")
        )
        # Good code response — should stay at Haiku
        assert result.stage_acceptance_rates["tier"] == "haiku"

    def test_code_request_without_code_response(self):
        client = _ScriptedClient({
            CLAUDE_HAIKU: "You can sort a list by comparing elements and swapping them.",
            CLAUDE_SONNET: (
                "Here's how to sort a list in Python:\n\n"
                "def sort_list(items):\n"
                "    return sorted(items)\n\n"
                "For custom sorting, use the key parameter:\n"
                "def sort_by_key(items, key_func):\n"
                "    return sorted(items, key=key_func)"
            ),
            CLAUDE_OPUS: "def sort_list(items): return sorted(items)",
        })
        decoder = CascadeDecoder(client=client, high_threshold=0.85, low_threshold=0.4)
        decoder.load()

        result = decoder.generate(
            GenerationRequest(prompt="Write a function to sort a list")
        )
        # Should escalate because Haiku response lacks code
        assert len(client.models_called) >= 2


# ── Cost efficiency ────────────────────────────────────────────────


class TestCostEfficiency:
    """Verify that the cascade saves money vs always using Opus."""

    def test_simple_queries_save_money(self):
        """Multiple simple queries should all stay at Haiku, saving vs Opus."""
        prompts = [
            ("What is 2+2?", "The answer is 4."),
            ("Hello", "Hi there! How can I help you today?"),
            ("What color is the sky?", "The sky appears blue during the day due to Rayleigh scattering of sunlight."),
        ]

        client = _ScriptedClient({
            CLAUDE_HAIKU: "placeholder",
        })
        decoder = CascadeDecoder(client=client, high_threshold=0.65)
        decoder.load()

        for prompt_text, expected_response in prompts:
            # Override the response for each prompt
            client._tier_responses[CLAUDE_HAIKU] = expected_response
            decoder.generate(GenerationRequest(prompt=prompt_text))

        summary = client.cost_tracker.summary()
        assert summary["savings_usd"] > 0
        assert summary["total_cost_usd"] < summary["opus_equivalent_usd"]

    def test_mixed_complexity_still_saves(self):
        """Even with some escalations, total cost should be less than all-Opus."""
        client = _ScriptedClient({
            CLAUDE_HAIKU: "Simple and adequate response for the question asked.",
            CLAUDE_SONNET: (
                "A more detailed response that addresses the complexity of the question "
                "with proper analysis and supporting evidence across multiple dimensions."
            ),
            CLAUDE_OPUS: (
                "The most comprehensive response covering all aspects of the query "
                "with rigorous analysis, citations, and multiple perspectives."
            ),
        })
        decoder = CascadeDecoder(client=client, high_threshold=0.7, low_threshold=0.4)
        decoder.load()

        # Mix of simple and complex prompts
        prompts = [
            "What is Python?",
            "Hello",
            "Explain and analyze the trade-offs in distributed systems design, implement examples",
            "What is 1+1?",
            "Compare and contrast functional and object-oriented programming paradigms",
        ]

        for p in prompts:
            decoder.generate(GenerationRequest(prompt=p))

        summary = client.cost_tracker.summary()
        # With haiku being much cheaper, overall savings should be positive
        assert summary["savings_usd"] >= 0
