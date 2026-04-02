"""Unit tests for the Claude API client wrapper."""

from unittest.mock import MagicMock, patch

import pytest

from momo_kibidango.exceptions import APIError, ConfigurationError
from momo_kibidango.models.claude_client import (
    CLAUDE_HAIKU,
    CLAUDE_OPUS,
    CLAUDE_SONNET,
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


# ── ClaudeClient ───────────────────────────────────────────────────


class TestClaudeClient:
    def test_no_api_key_raises(self):
        with patch.dict("os.environ", {}, clear=True):
            # Ensure ANTHROPIC_API_KEY is not set
            import os
            env = os.environ.copy()
            env.pop("ANTHROPIC_API_KEY", None)
            with patch.dict("os.environ", env, clear=True):
                with pytest.raises(ConfigurationError, match="API key"):
                    ClaudeClient(api_key="")

    def test_api_key_from_param(self):
        client = ClaudeClient(api_key="test-key-123")
        assert client._api_key == "test-key-123"

    def test_api_key_from_env(self):
        with patch.dict("os.environ", {"ANTHROPIC_API_KEY": "env-key-456"}):
            client = ClaudeClient()
            assert client._api_key == "env-key-456"

    @patch("momo_kibidango.models.claude_client.ClaudeClient._ensure_client")
    def test_complete_success(self, mock_ensure):
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.content = [MagicMock(text="Hello!")]
        mock_response.usage.input_tokens = 10
        mock_response.usage.output_tokens = 5
        mock_client.messages.create.return_value = mock_response
        mock_ensure.return_value = mock_client

        client = ClaudeClient(api_key="test-key")
        text, usage = client.complete("Hi", model=CLAUDE_HAIKU)

        assert text == "Hello!"
        assert usage.input_tokens == 10
        assert usage.output_tokens == 5
        assert usage.model == CLAUDE_HAIKU
        assert len(client.cost_tracker.calls) == 1

    @patch("momo_kibidango.models.claude_client.ClaudeClient._ensure_client")
    def test_complete_with_system_prompt(self, mock_ensure):
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.content = [MagicMock(text="response")]
        mock_response.usage.input_tokens = 10
        mock_response.usage.output_tokens = 5
        mock_client.messages.create.return_value = mock_response
        mock_ensure.return_value = mock_client

        client = ClaudeClient(api_key="test-key")
        client.complete("Hi", system="Be helpful")

        call_kwargs = mock_client.messages.create.call_args
        assert call_kwargs.kwargs.get("system") == "Be helpful" or \
               (call_kwargs[1] if len(call_kwargs) > 1 else {}).get("system") == "Be helpful"

    @patch("momo_kibidango.models.claude_client.ClaudeClient._ensure_client")
    def test_complete_api_failure(self, mock_ensure):
        mock_client = MagicMock()
        mock_client.messages.create.side_effect = RuntimeError("API down")
        mock_ensure.return_value = mock_client

        client = ClaudeClient(api_key="test-key")
        with pytest.raises(APIError, match="API call failed"):
            client.complete("Hi")

    @patch("momo_kibidango.models.claude_client.ClaudeClient._ensure_client")
    def test_complete_empty_content(self, mock_ensure):
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.content = []
        mock_response.usage.input_tokens = 10
        mock_response.usage.output_tokens = 0
        mock_client.messages.create.return_value = mock_response
        mock_ensure.return_value = mock_client

        client = ClaudeClient(api_key="test-key")
        text, usage = client.complete("Hi")
        assert text == ""

    @patch("momo_kibidango.models.claude_client.ClaudeClient._ensure_client")
    def test_stream_success(self, mock_ensure):
        mock_client = MagicMock()

        # Create mock events
        delta_event = MagicMock()
        delta_event.type = "content_block_delta"
        delta_event.delta.text = "Hello"

        start_event = MagicMock()
        start_event.type = "message_start"
        start_event.message.usage.input_tokens = 10

        end_event = MagicMock()
        end_event.type = "message_delta"
        end_event.usage.output_tokens = 5

        # Mock context manager
        mock_stream = MagicMock()
        mock_stream.__enter__ = MagicMock(return_value=mock_stream)
        mock_stream.__exit__ = MagicMock(return_value=False)
        mock_stream.__iter__ = MagicMock(
            return_value=iter([start_event, delta_event, end_event])
        )
        mock_client.messages.stream.return_value = mock_stream

        mock_ensure.return_value = mock_client

        client = ClaudeClient(api_key="test-key")
        chunks = list(client.stream("Hi", model=CLAUDE_HAIKU))
        assert "Hello" in chunks

    @patch("momo_kibidango.models.claude_client.ClaudeClient._ensure_client")
    def test_stream_api_failure(self, mock_ensure):
        mock_client = MagicMock()
        mock_client.messages.stream.side_effect = RuntimeError("Stream error")
        mock_ensure.return_value = mock_client

        client = ClaudeClient(api_key="test-key")
        with pytest.raises(APIError, match="Streaming failed"):
            list(client.stream("Hi"))

    def test_ensure_client_missing_package(self):
        import sys

        client = ClaudeClient(api_key="test-key")
        # Temporarily remove anthropic from modules if present
        original = sys.modules.get("anthropic")
        sys.modules["anthropic"] = None  # type: ignore[assignment]
        try:
            with pytest.raises(ConfigurationError, match="anthropic"):
                client._ensure_client()
        finally:
            if original is not None:
                sys.modules["anthropic"] = original
            else:
                sys.modules.pop("anthropic", None)

    def test_model_ids(self):
        assert "haiku" in CLAUDE_HAIKU
        assert "sonnet" in CLAUDE_SONNET
        assert "opus" in CLAUDE_OPUS

    def test_model_costs_defined(self):
        for model in [CLAUDE_HAIKU, CLAUDE_SONNET, CLAUDE_OPUS]:
            assert model in MODEL_COSTS
            in_cost, out_cost = MODEL_COSTS[model]
            assert in_cost > 0
            assert out_cost > 0
        # Haiku should be cheapest
        assert MODEL_COSTS[CLAUDE_HAIKU][0] < MODEL_COSTS[CLAUDE_SONNET][0]
        assert MODEL_COSTS[CLAUDE_SONNET][0] < MODEL_COSTS[CLAUDE_OPUS][0]
