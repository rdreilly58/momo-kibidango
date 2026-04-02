"""Unit tests for the confidence scoring engine."""

import pytest

from momo_kibidango.core.confidence import (
    CoherenceScore,
    ComplexityMatch,
    ConfidenceResult,
    ConfidenceScorer,
    LengthScore,
    SelfScore,
)


# ── LengthScore ────────────────────────────────────────────────────


class TestLengthScore:
    def setup_method(self):
        self.scorer = LengthScore()

    def test_empty_response(self):
        score, reason = self.scorer.score("Hello world", "")
        assert score == 0.0
        assert "Empty" in reason

    def test_whitespace_only_response(self):
        score, _ = self.scorer.score("Hello world", "   \n  ")
        assert score == 0.0

    def test_very_short_response(self):
        score, reason = self.scorer.score(
            "Explain the theory of relativity in detail", "E=mc2"
        )
        assert score <= 0.5
        assert "short" in reason.lower() or "ratio" in reason.lower()

    def test_reasonable_response(self):
        prompt = "What is Python?"
        response = (
            "Python is a high-level, interpreted programming language known for "
            "its simplicity and readability. It supports multiple paradigms including "
            "object-oriented, functional, and procedural programming. Python is "
            "widely used in web development, data science, and automation."
        )
        score, _ = self.scorer.score(prompt, response)
        assert score >= 0.5

    def test_very_long_response(self):
        prompt = "Hi"
        response = " ".join(["word"] * 200)
        score, reason = self.scorer.score(prompt, response)
        assert score <= 0.8
        assert "ratio" in reason.lower()

    def test_custom_ratios(self):
        scorer = LengthScore(min_ratio=1.0, max_ratio=10.0, ideal_ratio=3.0)
        score, _ = scorer.score("one two three", "a b c d e f g h i")
        assert 0.0 <= score <= 1.0


# ── CoherenceScore ─────────────────────────────────────────────────


class TestCoherenceScore:
    def setup_method(self):
        self.scorer = CoherenceScore()

    def test_empty_response(self):
        score, reason = self.scorer.score("")
        assert score == 0.0
        assert "Empty" in reason

    def test_coherent_response(self):
        response = "Python is a versatile programming language. It is used widely."
        score, reason = self.scorer.score(response)
        assert score >= 0.8
        assert "Coherent" in reason

    def test_repetitive_response(self):
        response = "the cat the cat the cat the cat sat on the mat"
        score, reason = self.scorer.score(response)
        assert score < 0.8
        assert "Repetitive" in reason

    def test_trailing_off(self):
        response = "This is a good point, and there are many others..."
        score, reason = self.scorer.score(response)
        assert score < 1.0
        assert "Trails off" in reason

    def test_trailing_with_etc(self):
        response = "Features include speed, reliability, etc."
        score, _ = self.scorer.score(response)
        assert score < 1.0

    def test_incomplete_sentence(self):
        response = "The answer to this question is that we need to consider the"
        score, reason = self.scorer.score(response)
        assert score < 1.0
        assert "incomplete" in reason.lower()

    def test_properly_terminated(self):
        response = "The answer is 42."
        score, _ = self.scorer.score(response)
        assert score >= 0.9


# ── ComplexityMatch ────────────────────────────────────────────────


class TestComplexityMatch:
    def setup_method(self):
        self.scorer = ComplexityMatch()

    def test_simple_prompt(self):
        score, reason = self.scorer.score("Hello there", "Hi! How can I help?")
        assert score >= 0.9
        assert "Simple" in reason

    def test_code_request_with_code(self):
        prompt = "Write a function to sort a list"
        response = "def sort_list(items):\n    return sorted(items)"
        score, _ = self.scorer.score(prompt, response)
        assert score >= 0.7

    def test_code_request_without_code(self):
        prompt = "Write a function to sort a list"
        response = "You can sort a list by comparing elements."
        score, reason = self.scorer.score(prompt, response)
        assert score <= 0.5
        assert "lacks code" in reason.lower()

    def test_complex_prompt_shallow_response(self):
        prompt = "Explain and compare the design patterns, analyze their trade-offs, and implement examples"
        response = "Design patterns are useful."
        score, reason = self.scorer.score(prompt, response)
        assert score <= 0.5
        assert "shallow" in reason.lower()

    def test_complex_prompt_substantive_response(self):
        prompt = "Explain and compare the design patterns, analyze their trade-offs, and implement examples"
        response = " ".join(["detailed analysis and comparison"] * 20)
        score, _ = self.scorer.score(prompt, response)
        assert score >= 0.7

    def test_moderate_complexity(self):
        prompt = "Explain how Python decorators work"
        response = "Decorators wrap functions to add behaviour."
        score, reason = self.scorer.score(prompt, response)
        assert score >= 0.7
        assert "Moderate" in reason


# ── SelfScore ──────────────────────────────────────────────────────


class TestSelfScore:
    def test_no_client(self):
        scorer = SelfScore(client=None)
        score, reason = scorer.score("prompt", "response")
        assert score == 0.7
        assert "skipped" in reason.lower()

    def test_with_mock_client_high_confidence(self):
        from unittest.mock import MagicMock
        from momo_kibidango.models.claude_client import TokenUsage

        client = MagicMock()
        client.complete.return_value = ("9", TokenUsage())
        scorer = SelfScore(client=client)
        score, reason = scorer.score("prompt", "response")
        assert score == 0.9
        assert "9" in reason

    def test_with_mock_client_low_confidence(self):
        from unittest.mock import MagicMock
        from momo_kibidango.models.claude_client import TokenUsage

        client = MagicMock()
        client.complete.return_value = ("3", TokenUsage())
        scorer = SelfScore(client=client)
        score, _ = scorer.score("prompt", "response")
        assert score == 0.3

    def test_client_failure_returns_default(self):
        from unittest.mock import MagicMock

        client = MagicMock()
        client.complete.side_effect = Exception("API error")
        scorer = SelfScore(client=client)
        score, reason = scorer.score("prompt", "response")
        assert score == 0.7
        assert "unavailable" in reason.lower()


# ── ConfidenceScorer (combined) ────────────────────────────────────


class TestConfidenceScorer:
    def test_default_weights(self):
        scorer = ConfidenceScorer()
        assert "length" in scorer.weights
        assert "coherence" in scorer.weights
        # Self-score disabled by default
        assert scorer.weights["self_score"] == 0.0

    def test_score_returns_confidence_result(self):
        scorer = ConfidenceScorer()
        result = scorer.score("What is Python?", "Python is a programming language.")
        assert isinstance(result, ConfidenceResult)
        assert 0.0 <= result.score <= 1.0
        assert len(result.reasoning) > 0
        assert "length" in result.component_scores

    def test_high_confidence_simple_prompt(self):
        scorer = ConfidenceScorer()
        result = scorer.score(
            "What is 2+2?",
            "The answer is 4. Two plus two equals four."
        )
        assert result.score >= 0.5

    def test_low_confidence_empty_response(self):
        scorer = ConfidenceScorer()
        result = scorer.score("Explain quantum computing in detail", "")
        assert result.score < 0.5

    def test_tier_recommendation_high(self):
        result = ConfidenceResult(score=0.9, reasoning=[], component_scores={})
        assert result.tier_recommendation == "haiku"

    def test_tier_recommendation_medium(self):
        result = ConfidenceResult(score=0.65, reasoning=[], component_scores={})
        assert result.tier_recommendation == "sonnet"

    def test_tier_recommendation_low(self):
        result = ConfidenceResult(score=0.3, reasoning=[], component_scores={})
        assert result.tier_recommendation == "opus"

    def test_custom_weights(self):
        scorer = ConfidenceScorer(weights={"length": 1.0, "coherence": 0.0, "complexity": 0.0, "self_score": 0.0})
        result = scorer.score("Hello", "Hi there friend!")
        assert isinstance(result, ConfidenceResult)

    def test_self_score_enabled(self):
        from unittest.mock import MagicMock
        from momo_kibidango.models.claude_client import TokenUsage

        client = MagicMock()
        client.complete.return_value = ("8", TokenUsage())
        scorer = ConfidenceScorer(enable_self_score=True, client=client)
        assert scorer.weights["self_score"] > 0
        result = scorer.score("Hello", "Hi there!")
        assert "self_score" in result.component_scores

    def test_all_zero_weights_returns_default(self):
        scorer = ConfidenceScorer(
            weights={"length": 0.0, "coherence": 0.0, "complexity": 0.0, "self_score": 0.0}
        )
        result = scorer.score("Hello", "Hi")
        assert result.score == 0.5
