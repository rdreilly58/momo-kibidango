"""Anthropic Claude API client wrapper.

Provides a unified interface to Claude Haiku, Sonnet, and Opus models
with retry logic, rate limiting, streaming support, and cost tracking.
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass, field
from typing import Any, Iterator

from momo_kibidango.exceptions import APIError, ConfigurationError

logger = logging.getLogger(__name__)

# Model IDs for the Claude cascade tiers
CLAUDE_HAIKU = "claude-haiku-4-5-20251001"
CLAUDE_SONNET = "claude-sonnet-4-5-20250514"
CLAUDE_OPUS = "claude-opus-4-20250514"

# Cost per 1M tokens (USD) — input / output
MODEL_COSTS: dict[str, tuple[float, float]] = {
    CLAUDE_HAIKU: (0.80, 4.00),
    CLAUDE_SONNET: (3.00, 15.00),
    CLAUDE_OPUS: (15.00, 75.00),
}


@dataclass
class TokenUsage:
    """Token usage for a single API call."""

    input_tokens: int = 0
    output_tokens: int = 0
    model: str = ""

    @property
    def cost_usd(self) -> float:
        """Estimated cost in USD."""
        rates = MODEL_COSTS.get(self.model, (0.0, 0.0))
        return (self.input_tokens * rates[0] + self.output_tokens * rates[1]) / 1_000_000


@dataclass
class CostTracker:
    """Tracks cumulative costs across cascade calls."""

    calls: list[TokenUsage] = field(default_factory=list)

    @property
    def total_cost_usd(self) -> float:
        return sum(c.cost_usd for c in self.calls)

    @property
    def total_input_tokens(self) -> int:
        return sum(c.input_tokens for c in self.calls)

    @property
    def total_output_tokens(self) -> int:
        return sum(c.output_tokens for c in self.calls)

    @property
    def opus_equivalent_cost(self) -> float:
        """What this would have cost if every call used Opus."""
        opus_rates = MODEL_COSTS[CLAUDE_OPUS]
        return (
            self.total_input_tokens * opus_rates[0]
            + self.total_output_tokens * opus_rates[1]
        ) / 1_000_000

    @property
    def savings_usd(self) -> float:
        return self.opus_equivalent_cost - self.total_cost_usd

    def record(self, usage: TokenUsage) -> None:
        """Record a usage entry."""
        self.calls.append(usage)

    def summary(self) -> dict[str, Any]:
        """Return a summary dict."""
        return {
            "total_cost_usd": round(self.total_cost_usd, 6),
            "opus_equivalent_usd": round(self.opus_equivalent_cost, 6),
            "savings_usd": round(self.savings_usd, 6),
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "num_calls": len(self.calls),
        }

    def reset(self) -> None:
        """Clear all tracked calls."""
        self.calls.clear()


class ClaudeClient:
    """Wrapper around the Anthropic Python SDK.

    Handles authentication, retries, rate limits, streaming,
    and cost tracking for all three Claude tiers.
    """

    def __init__(
        self,
        api_key: str | None = None,
        max_retries: int = 3,
        timeout: float = 120.0,
    ) -> None:
        self._api_key = api_key or os.environ.get("ANTHROPIC_API_KEY", "")
        if not self._api_key:
            raise ConfigurationError(
                "Anthropic API key required. Set ANTHROPIC_API_KEY or pass api_key."
            )
        self._max_retries = max_retries
        self._timeout = timeout
        self._client: Any = None
        self.cost_tracker = CostTracker()

    def _ensure_client(self) -> Any:
        """Lazily initialise the Anthropic client."""
        if self._client is None:
            try:
                import anthropic
            except ImportError as exc:
                raise ConfigurationError(
                    "anthropic package not installed. "
                    "Install with: pip install 'momo-kibidango[cascade]'"
                ) from exc
            self._client = anthropic.Anthropic(
                api_key=self._api_key,
                max_retries=self._max_retries,
                timeout=self._timeout,
            )
        return self._client

    def complete(
        self,
        prompt: str,
        model: str = CLAUDE_HAIKU,
        max_tokens: int = 1024,
        temperature: float = 0.7,
        system: str | None = None,
    ) -> tuple[str, TokenUsage]:
        """Send a single prompt and return (response_text, usage).

        Raises APIError on unrecoverable failures.
        """
        client = self._ensure_client()
        messages = [{"role": "user", "content": prompt}]
        kwargs: dict[str, Any] = {
            "model": model,
            "max_tokens": max_tokens,
            "messages": messages,
            "temperature": temperature,
        }
        if system:
            kwargs["system"] = system

        try:
            response = client.messages.create(**kwargs)
        except Exception as exc:
            logger.error("Anthropic API call failed for model %s: %s", model, exc)
            raise APIError(f"API call failed: {exc}") from exc

        text = response.content[0].text if response.content else ""
        usage = TokenUsage(
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
            model=model,
        )
        self.cost_tracker.record(usage)
        logger.debug(
            "Claude %s: %d in / %d out tokens ($%.6f)",
            model,
            usage.input_tokens,
            usage.output_tokens,
            usage.cost_usd,
        )
        return text, usage

    def stream(
        self,
        prompt: str,
        model: str = CLAUDE_HAIKU,
        max_tokens: int = 1024,
        temperature: float = 0.7,
        system: str | None = None,
    ) -> Iterator[str]:
        """Stream response tokens. Yields text deltas.

        Cost tracking is recorded after the stream completes.
        """
        client = self._ensure_client()
        messages = [{"role": "user", "content": prompt}]
        kwargs: dict[str, Any] = {
            "model": model,
            "max_tokens": max_tokens,
            "messages": messages,
            "temperature": temperature,
        }
        if system:
            kwargs["system"] = system

        try:
            with client.messages.stream(**kwargs) as stream:
                input_tokens = 0
                output_tokens = 0
                for event in stream:
                    if hasattr(event, "type"):
                        if event.type == "content_block_delta":
                            yield event.delta.text
                        elif event.type == "message_delta":
                            output_tokens = getattr(
                                event.usage, "output_tokens", output_tokens
                            )
                        elif event.type == "message_start":
                            input_tokens = getattr(
                                event.message.usage, "input_tokens", 0
                            )
                usage = TokenUsage(
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    model=model,
                )
                self.cost_tracker.record(usage)
        except Exception as exc:
            logger.error("Streaming failed for model %s: %s", model, exc)
            raise APIError(f"Streaming failed: {exc}") from exc
