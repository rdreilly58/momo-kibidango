"""Claude API client wrapper — OpenClaw gateway with direct Anthropic fallback.

Routes requests through OpenClaw's OpenAI-compatible gateway by default.
Falls back to the Anthropic SDK directly if the gateway is unreachable
and an ANTHROPIC_API_KEY is available.
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass, field
from typing import Any, Iterator

from momo_kibidango.exceptions import APIError, ConfigurationError

logger = logging.getLogger(__name__)

# Default gateway URL — OpenRouter (OpenAI-compatible, Claude models available)
DEFAULT_GATEWAY_URL = "https://openrouter.ai/api/v1"

# Model IDs for the Claude cascade tiers (OpenClaw gateway format)
CLAUDE_HAIKU = "anthropic/claude-haiku-4-5"
CLAUDE_SONNET = "anthropic/claude-sonnet-4-5"
CLAUDE_OPUS = "anthropic/claude-opus-4"

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
    """Routes Claude API calls through OpenClaw's gateway (OpenAI-compatible).

    Falls back to direct Anthropic SDK if the gateway is unreachable
    and an ANTHROPIC_API_KEY is available.
    """

    def __init__(
        self,
        api_key: str | None = None,
        max_retries: int = 3,
        timeout: float = 120.0,
        gateway_url: str | None = None,
        fallback_to_direct: bool = True,
    ) -> None:
        self._api_key = api_key or os.environ.get("OPENROUTER_API_KEY", "") or os.environ.get("ANTHROPIC_API_KEY", "")
        self._max_retries = max_retries
        self._timeout = timeout
        self._gateway_url = gateway_url or os.environ.get(
            "OPENCLAW_GATEWAY_URL", DEFAULT_GATEWAY_URL
        )
        self._fallback_to_direct = fallback_to_direct
        self._gateway_client: Any = None
        self._direct_client: Any = None
        self._using_gateway: bool = True
        self.cost_tracker = CostTracker()

    @property
    def backend(self) -> str:
        """Return which backend is currently active."""
        return "gateway" if self._using_gateway else "direct"

    def _ensure_gateway_client(self) -> Any:
        """Lazily initialise the OpenAI-compatible gateway client."""
        if self._gateway_client is None:
            try:
                from openai import OpenAI
            except ImportError as exc:
                raise ConfigurationError(
                    "openai package not installed. "
                    "Install with: pip install 'momo-kibidango[cascade]'"
                ) from exc
            self._gateway_client = OpenAI(
                base_url=self._gateway_url,
                api_key=self._api_key or "not-needed",
                max_retries=self._max_retries,
                timeout=self._timeout,
            )
        return self._gateway_client

    def _ensure_direct_client(self) -> Any:
        """Lazily initialise the direct Anthropic client for fallback."""
        if self._direct_client is None:
            if not self._api_key:
                raise ConfigurationError(
                    "Gateway unreachable and no ANTHROPIC_API_KEY set for fallback."
                )
            try:
                import anthropic
            except ImportError as exc:
                raise ConfigurationError(
                    "anthropic package not installed for direct fallback. "
                    "Install with: pip install anthropic"
                ) from exc
            self._direct_client = anthropic.Anthropic(
                api_key=self._api_key,
                max_retries=self._max_retries,
                timeout=self._timeout,
            )
        return self._direct_client

    def _call_gateway(
        self,
        prompt: str,
        model: str,
        max_tokens: int,
        temperature: float,
        system: str | None,
    ) -> tuple[str, TokenUsage]:
        """Send a completion request via the OpenClaw gateway."""
        client = self._ensure_gateway_client()
        messages: list[dict[str, str]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        response = client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
        )

        text = response.choices[0].message.content if response.choices else ""
        usage = TokenUsage(
            input_tokens=getattr(response.usage, "prompt_tokens", 0),
            output_tokens=getattr(response.usage, "completion_tokens", 0),
            model=model,
        )
        return text or "", usage

    def _call_direct(
        self,
        prompt: str,
        model: str,
        max_tokens: int,
        temperature: float,
        system: str | None,
    ) -> tuple[str, TokenUsage]:
        """Send a completion request directly via Anthropic SDK."""
        client = self._ensure_direct_client()
        messages = [{"role": "user", "content": prompt}]
        kwargs: dict[str, Any] = {
            "model": model,
            "max_tokens": max_tokens,
            "messages": messages,
            "temperature": temperature,
        }
        if system:
            kwargs["system"] = system

        response = client.messages.create(**kwargs)
        text = response.content[0].text if response.content else ""
        usage = TokenUsage(
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
            model=model,
        )
        return text, usage

    def complete(
        self,
        prompt: str,
        model: str = CLAUDE_HAIKU,
        max_tokens: int = 1024,
        temperature: float = 0.7,
        system: str | None = None,
    ) -> tuple[str, TokenUsage]:
        """Send a single prompt and return (response_text, usage).

        Tries the gateway first; falls back to direct Anthropic API
        if the gateway is unreachable and fallback is enabled.
        """
        # Try gateway first
        if self._using_gateway:
            try:
                text, usage = self._call_gateway(
                    prompt, model, max_tokens, temperature, system
                )
                self.cost_tracker.record(usage)
                logger.debug(
                    "Gateway %s: %d in / %d out tokens ($%.6f)",
                    model,
                    usage.input_tokens,
                    usage.output_tokens,
                    usage.cost_usd,
                )
                return text, usage
            except Exception as gw_exc:
                if not self._fallback_to_direct or not self._api_key:
                    logger.error("Gateway call failed for %s: %s", model, gw_exc)
                    raise APIError(f"Gateway call failed: {gw_exc}") from gw_exc
                logger.warning(
                    "Gateway unreachable for %s, falling back to direct API: %s",
                    model,
                    gw_exc,
                )
                self._using_gateway = False

        # Direct Anthropic fallback
        try:
            text, usage = self._call_direct(
                prompt, model, max_tokens, temperature, system
            )
            self.cost_tracker.record(usage)
            logger.debug(
                "Direct %s: %d in / %d out tokens ($%.6f)",
                model,
                usage.input_tokens,
                usage.output_tokens,
                usage.cost_usd,
            )
            return text, usage
        except ConfigurationError:
            raise
        except Exception as exc:
            logger.error("Direct API call failed for %s: %s", model, exc)
            raise APIError(f"API call failed: {exc}") from exc

    def _stream_gateway(
        self,
        prompt: str,
        model: str,
        max_tokens: int,
        temperature: float,
        system: str | None,
    ) -> Iterator[str]:
        """Stream response tokens via the OpenClaw gateway."""
        client = self._ensure_gateway_client()
        messages: list[dict[str, str]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        stream = client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            stream=True,
        )

        input_tokens = 0
        output_tokens = 0
        for chunk in stream:
            if chunk.choices and chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content
            if hasattr(chunk, "usage") and chunk.usage:
                input_tokens = getattr(chunk.usage, "prompt_tokens", input_tokens)
                output_tokens = getattr(
                    chunk.usage, "completion_tokens", output_tokens
                )

        usage = TokenUsage(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            model=model,
        )
        self.cost_tracker.record(usage)

    def _stream_direct(
        self,
        prompt: str,
        model: str,
        max_tokens: int,
        temperature: float,
        system: str | None,
    ) -> Iterator[str]:
        """Stream response tokens directly via Anthropic SDK."""
        client = self._ensure_direct_client()
        messages = [{"role": "user", "content": prompt}]
        kwargs: dict[str, Any] = {
            "model": model,
            "max_tokens": max_tokens,
            "messages": messages,
            "temperature": temperature,
        }
        if system:
            kwargs["system"] = system

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

    def stream(
        self,
        prompt: str,
        model: str = CLAUDE_HAIKU,
        max_tokens: int = 1024,
        temperature: float = 0.7,
        system: str | None = None,
    ) -> Iterator[str]:
        """Stream response tokens. Yields text deltas.

        Tries gateway first; falls back to direct Anthropic if unreachable.
        """
        if self._using_gateway:
            try:
                yield from self._stream_gateway(
                    prompt, model, max_tokens, temperature, system
                )
                return
            except Exception as gw_exc:
                if not self._fallback_to_direct or not self._api_key:
                    logger.error("Gateway stream failed for %s: %s", model, gw_exc)
                    raise APIError(f"Gateway streaming failed: {gw_exc}") from gw_exc
                logger.warning(
                    "Gateway stream unreachable for %s, falling back to direct: %s",
                    model,
                    gw_exc,
                )
                self._using_gateway = False

        try:
            yield from self._stream_direct(
                prompt, model, max_tokens, temperature, system
            )
        except ConfigurationError:
            raise
        except Exception as exc:
            logger.error("Direct streaming failed for %s: %s", model, exc)
            raise APIError(f"Streaming failed: {exc}") from exc
