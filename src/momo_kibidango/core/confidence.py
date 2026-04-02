"""Confidence scoring engine for cascade tier escalation.

Evaluates response quality from a draft model to decide whether
to accept the result or escalate to a higher-tier model.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any

from momo_kibidango.exceptions import ConfidenceError

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ConfidenceResult:
    """Result of confidence scoring with score and reasoning."""

    score: float  # 0.0 – 1.0
    reasoning: list[str]
    component_scores: dict[str, float]

    @property
    def tier_recommendation(self) -> str:
        """Suggest which tier should handle this based on score."""
        if self.score >= 0.85:
            return "haiku"
        if self.score >= 0.70:
            return "sonnet"
        return "opus"


# ------------------------------------------------------------------ #
# Individual scoring strategies
# ------------------------------------------------------------------ #


class LengthScore:
    """Score based on response length relative to prompt complexity."""

    def __init__(
        self,
        min_ratio: float = 0.5,
        max_ratio: float = 50.0,
        ideal_ratio: float = 5.0,
    ) -> None:
        self.min_ratio = min_ratio
        self.max_ratio = max_ratio
        self.ideal_ratio = ideal_ratio

    def score(self, prompt: str, response: str) -> tuple[float, str]:
        """Return (score, explanation)."""
        if not response.strip():
            return 0.0, "Empty response"

        prompt_len = max(len(prompt.split()), 1)
        response_len = len(response.split())
        ratio = response_len / prompt_len

        # Short prompts (<15 words) naturally get short answers — don't penalise
        if prompt_len <= 15 and response_len >= 5:
            return 0.95, f"Short prompt, adequate response ({response_len} words)"

        if ratio < self.min_ratio:
            return 0.3, f"Response too short (ratio {ratio:.1f})"
        if ratio > self.max_ratio:
            return 0.5, f"Response unusually long (ratio {ratio:.1f})"

        # Score peaks at ideal_ratio, tapers toward extremes
        distance = abs(ratio - self.ideal_ratio) / self.ideal_ratio
        score = max(0.4, 1.0 - distance * 0.3)
        return min(score, 1.0), f"Length ratio {ratio:.1f}"


class CoherenceScore:
    """Detect repetition, trailing off, and incomplete thoughts."""

    # Patterns that indicate low coherence
    REPETITION_THRESHOLD = 3  # consecutive repeated phrases
    TRAILING_PATTERNS = re.compile(
        r"(?:\.\.\.\s*$|…\s*$|etc\.?\s*$|and so on\.?\s*$)", re.IGNORECASE
    )

    def score(self, response: str) -> tuple[float, str]:
        """Return (score, explanation)."""
        if not response.strip():
            return 0.0, "Empty response"

        reasons: list[str] = []
        penalties = 0.0

        # Check for excessive repetition
        words = response.lower().split()
        if len(words) >= 6:
            # Sliding window bigram repetition
            bigrams = [f"{words[i]} {words[i+1]}" for i in range(len(words) - 1)]
            from collections import Counter

            counts = Counter(bigrams)
            max_repeat = max(counts.values()) if counts else 0
            if max_repeat >= self.REPETITION_THRESHOLD:
                penalties += 0.3
                reasons.append(f"Repetitive (bigram repeated {max_repeat}x)")

        # Check for trailing off
        if self.TRAILING_PATTERNS.search(response):
            penalties += 0.15
            reasons.append("Trails off")

        # Check for incomplete sentences (ends mid-word or no punctuation)
        stripped = response.rstrip()
        if stripped and stripped[-1] not in ".!?\"')]}":
            penalties += 0.1
            reasons.append("May be incomplete")

        score = max(0.0, 1.0 - penalties)
        explanation = "; ".join(reasons) if reasons else "Coherent"
        return score, explanation


class ComplexityMatch:
    """Score whether response complexity matches request complexity."""

    # Keywords that suggest the prompt needs a complex response
    COMPLEX_INDICATORS = re.compile(
        r"\b(explain|analyze|compare|contrast|implement|design|architect|"
        r"debug|optimize|prove|derive|evaluate|synthesize|critique|"
        r"code|function|class|algorithm|theorem|equation)\b",
        re.IGNORECASE,
    )
    CODE_INDICATORS = re.compile(
        r"(?:```|def |class |import |function |const |let |var |public |private )",
    )

    def score(self, prompt: str, response: str) -> tuple[float, str]:
        """Return (score, explanation)."""
        prompt_complexity = len(self.COMPLEX_INDICATORS.findall(prompt))
        is_code_request = bool(self.CODE_INDICATORS.search(prompt))
        response_has_code = bool(self.CODE_INDICATORS.search(response))

        if is_code_request and not response_has_code:
            return 0.4, "Code requested but response lacks code"

        if prompt_complexity >= 3:
            # Complex prompt — check response is substantive
            response_words = len(response.split())
            if response_words < 50:
                return 0.4, "Complex prompt but shallow response"
            return 0.8, "Complex prompt, substantive response"

        if prompt_complexity >= 1:
            return 0.85, "Moderate complexity"

        return 0.95, "Simple prompt"


class SelfScore:
    """Ask the model to self-rate its confidence (optional, costs extra tokens)."""

    SELF_RATE_PROMPT = (
        "Rate your confidence in the following response on a scale of 1-10, "
        "where 10 is completely confident. Reply with ONLY a number.\n\n"
        "Original prompt: {prompt}\n\nYour response: {response}"
    )

    def __init__(self, client: Any = None) -> None:
        self._client = client

    def score(self, prompt: str, response: str) -> tuple[float, str]:
        """Return (score, explanation). Requires a ClaudeClient."""
        if self._client is None:
            return 0.7, "Self-score skipped (no client)"

        rate_prompt = self.SELF_RATE_PROMPT.format(prompt=prompt, response=response)
        try:
            from momo_kibidango.models.claude_client import CLAUDE_HAIKU

            text, _usage = self._client.complete(
                rate_prompt, model=CLAUDE_HAIKU, max_tokens=16, temperature=0.0
            )
            # Extract number from response
            numbers = re.findall(r"\b(\d+(?:\.\d+)?)\b", text.strip())
            if numbers:
                rating = float(numbers[0])
                score = min(max(rating / 10.0, 0.0), 1.0)
                return score, f"Self-rated {rating}/10"
        except Exception as exc:
            logger.warning("Self-scoring failed: %s", exc)

        return 0.7, "Self-score unavailable"


# ------------------------------------------------------------------ #
# Combined confidence scorer
# ------------------------------------------------------------------ #


class ConfidenceScorer:
    """Combines multiple scoring strategies into a single confidence score.

    Weights are configurable. By default, self-scoring is disabled
    (weight=0) to avoid extra API calls.
    """

    DEFAULT_WEIGHTS: dict[str, float] = {
        "length": 0.15,
        "coherence": 0.30,
        "complexity": 0.45,
        "self_score": 0.10,
    }

    def __init__(
        self,
        weights: dict[str, float] | None = None,
        enable_self_score: bool = False,
        client: Any = None,
    ) -> None:
        self.weights = weights or dict(self.DEFAULT_WEIGHTS)
        if not enable_self_score:
            self.weights["self_score"] = 0.0

        self._length = LengthScore()
        self._coherence = CoherenceScore()
        self._complexity = ComplexityMatch()
        self._self_score = SelfScore(client=client)

    def score(self, prompt: str, response: str) -> ConfidenceResult:
        """Score a response and return a ConfidenceResult."""
        try:
            components: dict[str, float] = {}
            reasons: list[str] = []

            len_score, len_reason = self._length.score(prompt, response)
            components["length"] = len_score
            reasons.append(f"Length: {len_reason}")

            coh_score, coh_reason = self._coherence.score(response)
            components["coherence"] = coh_score
            reasons.append(f"Coherence: {coh_reason}")

            cplx_score, cplx_reason = self._complexity.score(prompt, response)
            components["complexity"] = cplx_score
            reasons.append(f"Complexity: {cplx_reason}")

            self_score, self_reason = self._self_score.score(prompt, response)
            components["self_score"] = self_score
            reasons.append(f"Self: {self_reason}")

            # Weighted average (normalise weights for active scorers)
            active_total = sum(
                self.weights.get(k, 0.0) for k in components if self.weights.get(k, 0.0) > 0
            )
            if active_total == 0:
                final_score = 0.5
            else:
                final_score = sum(
                    components[k] * self.weights.get(k, 0.0) / active_total
                    for k in components
                    if self.weights.get(k, 0.0) > 0
                )

            final_score = min(max(final_score, 0.0), 1.0)

            return ConfidenceResult(
                score=final_score,
                reasoning=reasons,
                component_scores=components,
            )

        except Exception as exc:
            raise ConfidenceError(f"Confidence scoring failed: {exc}") from exc
