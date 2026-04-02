"""Claude API cascade decoder.

Implements a 3-tier speculative decoding cascade:
  Tier 1 (Draft):     claude-haiku   — fast, cheap first pass
  Tier 2 (Qualifier): claude-sonnet  — validates/refines uncertain outputs
  Tier 3 (Target):    claude-opus    — final quality for complex/critical tasks

Confidence heuristics decide whether to accept the draft or escalate.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Iterator

from momo_kibidango.core.confidence import ConfidenceResult, ConfidenceScorer
from momo_kibidango.core.decoder import BaseDecoder, GenerationRequest, GenerationResult
from momo_kibidango.exceptions import CascadeError
from momo_kibidango.models.claude_client import (
    CLAUDE_HAIKU,
    CLAUDE_OPUS,
    CLAUDE_SONNET,
    ClaudeClient,
    CostTracker,
)

logger = logging.getLogger(__name__)


class CascadeDecoder(BaseDecoder):
    """Cascade decoder that routes through Claude Haiku → Sonnet → Opus.

    The decoder starts with the cheapest model (Haiku) and only escalates
    to more expensive models when confidence in the draft response is low.
    """

    def __init__(
        self,
        client: ClaudeClient | None = None,
        api_key: str | None = None,
        scorer: ConfidenceScorer | None = None,
        high_threshold: float = 0.8,
        low_threshold: float = 0.5,
        haiku_model: str = CLAUDE_HAIKU,
        sonnet_model: str = CLAUDE_SONNET,
        opus_model: str = CLAUDE_OPUS,
        enable_self_score: bool = False,
    ) -> None:
        if client is not None:
            self._client = client
        else:
            self._client = ClaudeClient(api_key=api_key)

        self._scorer = scorer or ConfidenceScorer(
            enable_self_score=enable_self_score,
            client=self._client if enable_self_score else None,
        )
        self._high_threshold = high_threshold
        self._low_threshold = low_threshold
        self._haiku = haiku_model
        self._sonnet = sonnet_model
        self._opus = opus_model
        self._loaded = False

    # ── BaseDecoder interface ───────────────────────────────────────

    def load(self) -> None:
        """Mark the decoder as ready (no local models to load)."""
        self._loaded = True
        logger.info("CascadeDecoder ready (API-based, no local models)")

    def unload(self) -> None:
        """Release resources."""
        self._loaded = False

    @property
    def mode(self) -> str:
        return "cascade"

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    # ── Generation ──────────────────────────────────────────────────

    def generate(self, request: GenerationRequest) -> GenerationResult:
        """Run cascade generation: Haiku → (Sonnet) → (Opus)."""
        t0 = time.perf_counter()
        cost_before = self._client.cost_tracker.total_cost_usd

        try:
            text, tier_used, confidence = self._cascade(
                prompt=request.prompt,
                max_tokens=request.max_new_tokens,
                temperature=request.temperature,
            )
        except Exception as exc:
            raise CascadeError(f"Cascade generation failed: {exc}") from exc

        elapsed = time.perf_counter() - t0
        cost_after = self._client.cost_tracker.total_cost_usd
        call_cost = cost_after - cost_before

        # Approximate token count from text length
        tokens_generated = max(len(text.split()), 1)
        tps = tokens_generated / elapsed if elapsed > 0 else 0.0

        return GenerationResult(
            text=text,
            tokens_generated=tokens_generated,
            elapsed_seconds=round(elapsed, 3),
            tokens_per_second=round(tps, 1),
            acceptance_rate=confidence.score if confidence else 0.0,
            stage_acceptance_rates={
                "tier": tier_used,
                "confidence": confidence.score if confidence else 0.0,
                "cost_usd": round(call_cost, 6),
            },
            peak_memory_gb=0.0,  # API-based, no local memory
            mode="cascade",
            draft_attempts=1,
            accepted_tokens=tokens_generated,
        )

    def stream(self, request: GenerationRequest) -> Iterator[str]:
        """Stream from the appropriate tier.

        For simplicity, streaming always uses Haiku (lowest latency).
        For high-quality streaming, callers should use generate() instead.
        """
        yield from self._client.stream(
            prompt=request.prompt,
            model=self._haiku,
            max_tokens=request.max_new_tokens,
            temperature=request.temperature,
        )

    # ── Internal cascade logic ──────────────────────────────────────

    def _cascade(
        self,
        prompt: str,
        max_tokens: int = 1024,
        temperature: float = 0.7,
    ) -> tuple[str, str, ConfidenceResult]:
        """Execute the cascade: draft → score → maybe escalate."""

        # Step 1: Draft with Haiku
        logger.info("Cascade: drafting with Haiku")
        haiku_text, _usage = self._client.complete(
            prompt, model=self._haiku, max_tokens=max_tokens, temperature=temperature
        )

        # Step 2: Score the draft
        confidence = self._scorer.score(prompt, haiku_text)
        logger.info(
            "Cascade: Haiku confidence=%.2f (%s)",
            confidence.score,
            confidence.tier_recommendation,
        )

        # Step 3: Decide whether to accept or escalate
        if confidence.score >= self._high_threshold:
            logger.info("Cascade: accepting Haiku result (confidence %.2f)", confidence.score)
            return haiku_text, "haiku", confidence

        if confidence.score >= self._low_threshold:
            # Escalate to Sonnet for refinement
            logger.info("Cascade: escalating to Sonnet (confidence %.2f)", confidence.score)
            refine_prompt = (
                f"The following response was generated for the prompt below, but may need "
                f"improvement. Please provide a better, more complete response.\n\n"
                f"Original prompt: {prompt}\n\n"
                f"Draft response: {haiku_text}\n\n"
                f"Please provide an improved response:"
            )
            sonnet_text, _usage = self._client.complete(
                refine_prompt,
                model=self._sonnet,
                max_tokens=max_tokens,
                temperature=temperature,
            )
            sonnet_confidence = self._scorer.score(prompt, sonnet_text)
            return sonnet_text, "sonnet", sonnet_confidence

        # Low confidence — go straight to Opus
        logger.info("Cascade: escalating to Opus (confidence %.2f)", confidence.score)
        opus_text, _usage = self._client.complete(
            prompt, model=self._opus, max_tokens=max_tokens, temperature=temperature
        )
        opus_confidence = self._scorer.score(prompt, opus_text)
        return opus_text, "opus", opus_confidence

    @property
    def cost_tracker(self) -> CostTracker:
        """Access the underlying cost tracker."""
        return self._client.cost_tracker
