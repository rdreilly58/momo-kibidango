"""Unit tests for the Claude API client wrapper (gateway + fallback)."""

from unittest.mock import MagicMock, patch, PropertyMock

import pytest

from momo_kibidango.exceptions import APIError, ConfigurationError
from momo_kibidango.models.claude_client import (
    CLAUDE_HAIKU,
    CLAUDE_OPUS,
    CLAUDE_SONNET,
    DEFAULT_GATEWAY_URL,
    MODEL_COSTS,
    ClaudeClient,
    CostTracker,
    TokenUsage,
)


# ── TokenUsage ─────────────────────────────────────────────────────


class TestTokenUsage:
    def test_defaults(self):
        usage = TokenUsage()
        assert usage.input_tokens == 0
        assert usage.output_tokens == 0
        assert usage.cost_usd == 0.0

    def test_haiku_cost(self):
        usage = TokenUsage(input_tokens=1000, output_tokens=500, model=CLAUDE_HAIKU)
        # 1000 * 0.80/1M + 500 * 4.00/1M
        expected = (1000 * 0.80 + 500 * 4.00) / 1_000_000
        assert abs(usage.cost_usd - expected) < 1e-10

    def test_opus_cost(self):
        usage = TokenUsage(input_tokens=1000, output_tokens=500, model=CLAUDE_OPUS)
        expected = (1000 * 15.00 + 500 * 75.00) / 1_000_000
        assert abs(usage.cost_usd - expected) < 1e-10

    def test_unknown_model_zero_cost(self):
        usage = TokenUsage(input_tokens=1000, output_tokens=500, model="unknown")
        assert usage.cost_usd == 0.0


# ── CostTracker ────────────────────────────────────────────────────


class TestCostTracker:
    def test_empty_tracker(self):
        tracker = CostTracker()
        assert tracker.total_cost_usd == 0.0
        assert tracker.total_input_tokens == 0
        assert tracker.total_output_tokens == 0
        assert tracker.savings_usd == 0.0

    def test_record_single(self):
        tracker = CostTracker()
        usage = TokenUsage(input_tokens=100, output_tokens=50, model=CLAUDE_HAIKU)
        tracker.record(usage)
        assert len(tracker.calls) == 1
        assert tracker.total_input_tokens == 100
        assert tracker.total_output_tokens == 50

    def test_record_multiple(self):
        tracker = CostTracker()
        tracker.record(TokenUsage(input_tokens=100, output_tokens=50, model=CLAUDE_HAIKU))
        tracker.record(TokenUsage(input_tokens=200, output_tokens=100, model=CLAUDE_SONNET))
        assert tracker.total_input_tokens == 300
        assert tracker.total_output_tokens == 150
        assert len(tracker.calls) == 2

    def test_savings_positive_for_haiku(self):
        tracker = CostTracker()
        tracker.record(TokenUsage(input_tokens=1000, output_tokens=500, model=CLAUDE_HAIKU))
        assert tracker.savings_usd > 0

    def test_no_savings_for_opus(self):
        tracker = CostTracker()
        tracker.record(TokenUsage(input_tokens=1000, output_tokens=500, model=CLAUDE_OPUS))
        assert abs(tracker.savings_usd) < 1e-10

    def test_summary(self):
        tracker = CostTracker()
        tracker.record(TokenUsage(input_tokens=100, output_tokens=50, model=CLAUDE_HAIKU))
        summary = tracker.summary()
        assert "total_cost_usd" in summary
        assert "opus_equivalent_usd" in summary
        assert "savings_usd" in summary
        assert "num_calls" in summary
        assert summary["num_calls"] == 1

    def test_reset(self):
        tracker = CostTracker()
        tracker.record(TokenUsage(input_tokens=100, output_tokens=50, model=CLAUDE_HAIKU))
        tracker.reset()
        assert len(tracker.calls) == 0
        assert tracker.total_cost_usd == 0.0


# ── ClaudeClient — Gateway mode ───────────────────────────────────


class TestClaudeClientGateway:
    """Tests for the primary gateway path (OpenAI-compatible)."""

    def test_default_gateway_url(self):
        client = ClaudeClient()
        assert client._gateway_url == DEFAULT_GATEWAY_URL

    def test_custom_gateway_url(self):
        client = ClaudeClient(gateway_url="http://custom:9999/v1")
        assert client._gateway_url == "http://custom:9999/v1"

    def test_gateway_url_from_env(self):
        with patch.dict("os.environ", {"OPENCLAW_GATEWAY_URL": "http://env:8888/v1"}):
            client = ClaudeClient()
            assert client._gateway_url == "http://env:8888/v1"

    def test_no_api_key_required_for_gateway(self):
        """Gateway mode should not require an API key."""
        with patch.dict("os.environ", {}, clear=True):
            import os
            env = os.environ.copy()
            env.pop("ANTHROPIC_API_KEY", None)
            env.pop("OPENCLAW_GATEWAY_URL", None)
            with patch.dict("os.environ", env, clear=True):
                client = ClaudeClient()
                assert client.backend == "gateway"

    def test_backend_property_gateway(self):
        client = ClaudeClient()
        assert client.backend == "gateway"

    @patch("momo_kibidango.models.claude_client.ClaudeClient._ensure_gateway_client")
    def test_complete_via_gateway(self, mock_ensure):
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Hello!"
        mock_response.usage.prompt_tokens = 10
        mock_response.usage.completion_tokens = 5
        mock_client.chat.completions.create.return_value = mock_response
        mock_ensure.return_value = mock_client

        client = ClaudeClient()
        text, usage = client.complete("Hi", model=CLAUDE_HAIKU)

        assert text == "Hello!"
        assert usage.input_tokens == 10
        assert usage.output_tokens == 5
        assert usage.model == CLAUDE_HAIKU
        assert len(client.cost_tracker.calls) == 1

    @patch("momo_kibidango.models.claude_client.ClaudeClient._ensure_gateway_client")
    def test_complete_with_system_prompt(self, mock_ensure):
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "response"
        mock_response.usage.prompt_tokens = 10
        mock_response.usage.completion_tokens = 5
        mock_client.chat.completions.create.return_value = mock_response
        mock_ensure.return_value = mock_client

        client = ClaudeClient()
        client.complete("Hi", system="Be helpful")

        call_args = mock_client.chat.completions.create.call_args
        messages = call_args.kwargs.get("messages", call_args[1].get("messages", []))
        system_msgs = [m for m in messages if m["role"] == "system"]
        assert len(system_msgs) == 1
        assert system_msgs[0]["content"] == "Be helpful"

    @patch("momo_kibidango.models.claude_client.ClaudeClient._ensure_gateway_client")
    def test_complete_empty_choices(self, mock_ensure):
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.choices = []
        mock_response.usage.prompt_tokens = 10
        mock_response.usage.completion_tokens = 0
        mock_client.chat.completions.create.return_value = mock_response
        mock_ensure.return_value = mock_client

        client = ClaudeClient()
        text, usage = client.complete("Hi")
        assert text == ""

    @patch("momo_kibidango.models.claude_client.ClaudeClient._ensure_gateway_client")
    def test_gateway_failure_no_fallback_raises(self, mock_ensure):
        """When fallback is disabled, gateway failure should raise APIError."""
        mock_ensure.side_effect = RuntimeError("Connection refused")

        client = ClaudeClient(fallback_to_direct=False)
        with pytest.raises(APIError, match="Gateway call failed"):
            client.complete("Hi")

    @patch("momo_kibidango.models.claude_client.ClaudeClient._ensure_gateway_client")
    def test_stream_via_gateway(self, mock_ensure):
        mock_client = MagicMock()

        # Create mock streaming chunks
        chunk1 = MagicMock()
        chunk1.choices = [MagicMock()]
        chunk1.choices[0].delta.content = "Hello"
        chunk1.usage = None

        chunk2 = MagicMock()
        chunk2.choices = [MagicMock()]
        chunk2.choices[0].delta.content = " world"
        chunk2.usage = None

        mock_client.chat.completions.create.return_value = iter([chunk1, chunk2])
        mock_ensure.return_value = mock_client

        client = ClaudeClient()
        chunks = list(client.stream("Hi", model=CLAUDE_HAIKU))
        assert "Hello" in chunks
        assert " world" in chunks

    def test_ensure_gateway_missing_package(self):
        import sys

        client = ClaudeClient()
        original = sys.modules.get("openai")
        sys.modules["openai"] = None  # type: ignore[assignment]
        try:
            with pytest.raises(ConfigurationError, match="openai"):
                client._ensure_gateway_client()
        finally:
            if original is not None:
                sys.modules["openai"] = original
            else:
                sys.modules.pop("openai", None)


# ── ClaudeClient — Fallback mode ──────────────────────────────────


class TestClaudeClientFallback:
    """Tests for fallback from gateway to direct Anthropic API."""

    @patch("momo_kibidango.models.claude_client.ClaudeClient._ensure_direct_client")
    @patch("momo_kibidango.models.claude_client.ClaudeClient._ensure_gateway_client")
    def test_gateway_failure_falls_back_to_direct(self, mock_gw, mock_direct):
        """When gateway fails and API key is set, should fall back to direct."""
        # Gateway fails
        mock_gw_client = MagicMock()
        mock_gw_client.chat.completions.create.side_effect = ConnectionError("refused")
        mock_gw.return_value = mock_gw_client

        # Direct succeeds
        mock_anthropic = MagicMock()
        mock_response = MagicMock()
        mock_response.content = [MagicMock(text="Direct response")]
        mock_response.usage.input_tokens = 10
        mock_response.usage.output_tokens = 5
        mock_anthropic.messages.create.return_value = mock_response
        mock_direct.return_value = mock_anthropic

        client = ClaudeClient(api_key="test-key", fallback_to_direct=True)
        text, usage = client.complete("Hi")

        assert text == "Direct response"
        assert client.backend == "direct"

    @patch("momo_kibidango.models.claude_client.ClaudeClient._ensure_gateway_client")
    def test_gateway_failure_no_api_key_raises(self, mock_ensure):
        """When gateway fails and no API key, should raise APIError."""
        mock_gw_client = MagicMock()
        mock_gw_client.chat.completions.create.side_effect = ConnectionError("refused")
        mock_ensure.return_value = mock_gw_client

        with patch.dict("os.environ", {}, clear=True):
            import os
            env = os.environ.copy()
            env.pop("ANTHROPIC_API_KEY", None)
            env.pop("OPENCLAW_GATEWAY_URL", None)
            with patch.dict("os.environ", env, clear=True):
                client = ClaudeClient(fallback_to_direct=True)
                with pytest.raises(APIError, match="Gateway call failed"):
                    client.complete("Hi")

    @patch("momo_kibidango.models.claude_client.ClaudeClient._ensure_direct_client")
    @patch("momo_kibidango.models.claude_client.ClaudeClient._ensure_gateway_client")
    def test_stream_fallback_to_direct(self, mock_gw, mock_direct):
        """Streaming should also fall back when gateway fails."""
        # Gateway stream fails
        mock_gw_client = MagicMock()
        mock_gw_client.chat.completions.create.side_effect = ConnectionError("refused")
        mock_gw.return_value = mock_gw_client

        # Direct stream succeeds
        mock_anthropic = MagicMock()
        delta_event = MagicMock()
        delta_event.type = "content_block_delta"
        delta_event.delta.text = "Fallback"

        start_event = MagicMock()
        start_event.type = "message_start"
        start_event.message.usage.input_tokens = 10

        end_event = MagicMock()
        end_event.type = "message_delta"
        end_event.usage.output_tokens = 5

        mock_stream = MagicMock()
        mock_stream.__enter__ = MagicMock(return_value=mock_stream)
        mock_stream.__exit__ = MagicMock(return_value=False)
        mock_stream.__iter__ = MagicMock(
            return_value=iter([start_event, delta_event, end_event])
        )
        mock_anthropic.messages.stream.return_value = mock_stream
        mock_direct.return_value = mock_anthropic

        client = ClaudeClient(api_key="test-key", fallback_to_direct=True)
        chunks = list(client.stream("Hi"))
        assert "Fallback" in chunks

    def test_direct_fallback_no_api_key_raises_config_error(self):
        """Trying direct fallback without API key should raise ConfigurationError."""
        with patch.dict("os.environ", {}, clear=True):
            import os
            env = os.environ.copy()
            env.pop("ANTHROPIC_API_KEY", None)
            with patch.dict("os.environ", env, clear=True):
                client = ClaudeClient()
                client._using_gateway = False
                with pytest.raises(ConfigurationError, match="no ANTHROPIC_API_KEY"):
                    client.complete("Hi")

    def test_ensure_direct_missing_package(self):
        import sys

        client = ClaudeClient(api_key="test-key")
        original = sys.modules.get("anthropic")
        sys.modules["anthropic"] = None  # type: ignore[assignment]
        try:
            with pytest.raises(ConfigurationError, match="anthropic"):
                client._ensure_direct_client()
        finally:
            if original is not None:
                sys.modules["anthropic"] = original
            else:
                sys.modules.pop("anthropic", None)


# ── ClaudeClient — Direct API failure ─────────────────────────────


class TestClaudeClientDirectFailure:
    @patch("momo_kibidango.models.claude_client.ClaudeClient._ensure_direct_client")
    def test_direct_api_failure_raises(self, mock_ensure):
        mock_client = MagicMock()
        mock_client.messages.create.side_effect = RuntimeError("API down")
        mock_ensure.return_value = mock_client

        client = ClaudeClient(api_key="test-key")
        client._using_gateway = False
        with pytest.raises(APIError, match="API call failed"):
            client.complete("Hi")


# ── Model IDs and costs ───────────────────────────────────────────


class TestModelConstants:
    def test_model_ids(self):
        assert "haiku" in CLAUDE_HAIKU
        assert "sonnet" in CLAUDE_SONNET
        assert "opus" in CLAUDE_OPUS

    def test_model_ids_use_gateway_format(self):
        """Model IDs should use the anthropic/ prefix for OpenClaw gateway."""
        assert CLAUDE_HAIKU.startswith("anthropic/")
        assert CLAUDE_SONNET.startswith("anthropic/")
        assert CLAUDE_OPUS.startswith("anthropic/")

    def test_model_costs_defined(self):
        for model in [CLAUDE_HAIKU, CLAUDE_SONNET, CLAUDE_OPUS]:
            assert model in MODEL_COSTS
            in_cost, out_cost = MODEL_COSTS[model]
            assert in_cost > 0
            assert out_cost > 0
        # Haiku should be cheapest
        assert MODEL_COSTS[CLAUDE_HAIKU][0] < MODEL_COSTS[CLAUDE_SONNET][0]
        assert MODEL_COSTS[CLAUDE_SONNET][0] < MODEL_COSTS[CLAUDE_OPUS][0]
