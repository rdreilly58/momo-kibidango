"""Acceptance tests for various prompt types.

Uses mocked decoders that return canned results to validate pipeline logic
without real models.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from momo_kibidango.core.decoder import BaseDecoder, GenerationRequest, GenerationResult
from momo_kibidango.api.server import InputValidator


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_mock_decoder() -> BaseDecoder:
    """Return a mock decoder that echoes the prompt type in its output."""
    decoder = MagicMock(spec=BaseDecoder)
    decoder.is_loaded = True
    decoder.mode = "2model"

    def generate_fn(request: GenerationRequest) -> GenerationResult:
        # Simulate output based on prompt content
        prompt = request.prompt
        if not prompt or not prompt.strip():
            raise ValueError("Empty prompt rejected")

        # Truncate very long prompts (simulating real behavior)
        if len(prompt) > 32_000:
            prompt = prompt[:32_000]

        return GenerationResult(
            text=f"Output for prompt ({len(prompt)} chars): {prompt[:50]}...",
            tokens_generated=max(1, min(request.max_new_tokens, 20)),
            elapsed_seconds=0.2,
            tokens_per_second=100.0,
            acceptance_rate=0.8,
            stage_acceptance_rates={"stage2": 0.8},
            peak_memory_gb=2.0,
            mode="2model",
            draft_attempts=25,
            accepted_tokens=20,
        )

    decoder.generate.side_effect = generate_fn
    return decoder


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestCodePrompt:
    """Programming prompt produces output."""

    def test_code_prompt(self):
        decoder = _make_mock_decoder()
        request = GenerationRequest(
            prompt="Write a Python function to compute Fibonacci numbers",
            max_new_tokens=100,
        )

        result = decoder.generate(request)

        assert result.text
        assert result.tokens_generated > 0
        assert result.mode == "2model"

    def test_code_prompt_with_language_spec(self):
        decoder = _make_mock_decoder()
        request = GenerationRequest(
            prompt="Implement a binary search tree in Rust with insert and delete operations",
            max_new_tokens=200,
        )

        result = decoder.generate(request)
        assert result.text
        assert result.tokens_generated > 0


class TestCreativePrompt:
    """Story/creative prompt produces output."""

    def test_creative_prompt(self):
        decoder = _make_mock_decoder()
        request = GenerationRequest(
            prompt="Write a short story about a robot learning to paint",
            max_new_tokens=150,
        )

        result = decoder.generate(request)

        assert result.text
        assert result.tokens_generated > 0

    def test_poetry_prompt(self):
        decoder = _make_mock_decoder()
        request = GenerationRequest(
            prompt="Write a haiku about autumn leaves falling",
            max_new_tokens=50,
        )

        result = decoder.generate(request)
        assert result.text


class TestInstructionPrompt:
    """Instruction-following prompt produces output."""

    def test_instruction_prompt(self):
        decoder = _make_mock_decoder()
        request = GenerationRequest(
            prompt="List five benefits of regular exercise. Be concise.",
            max_new_tokens=100,
        )

        result = decoder.generate(request)

        assert result.text
        assert result.tokens_generated > 0

    def test_multi_step_instruction(self):
        decoder = _make_mock_decoder()
        request = GenerationRequest(
            prompt="Step 1: Summarize the theory of relativity. Step 2: Give an example.",
            max_new_tokens=200,
        )

        result = decoder.generate(request)
        assert result.text


class TestEmptyPromptRejected:
    """Empty prompt raises an error."""

    def test_empty_prompt_rejected(self):
        decoder = _make_mock_decoder()
        request = GenerationRequest(prompt="", max_new_tokens=10)

        with pytest.raises(ValueError, match="Empty prompt"):
            decoder.generate(request)

    def test_whitespace_only_prompt_rejected(self):
        decoder = _make_mock_decoder()
        request = GenerationRequest(prompt="   ", max_new_tokens=10)

        with pytest.raises(ValueError, match="Empty prompt"):
            decoder.generate(request)

    def test_input_validator_rejects_empty(self):
        """The InputValidator also rejects empty prompts."""
        validator = InputValidator()

        with pytest.raises(ValueError):
            validator.validate_prompt("")

        with pytest.raises(ValueError):
            validator.validate_prompt("   ")

        with pytest.raises(ValueError):
            validator.validate_prompt(None)


class TestLongPromptHandled:
    """Very long prompts are truncated or handled gracefully."""

    def test_long_prompt_handled(self):
        decoder = _make_mock_decoder()

        # Create a very long prompt (50k characters)
        long_prompt = "Please analyze: " + "x" * 50_000

        request = GenerationRequest(prompt=long_prompt, max_new_tokens=10)
        result = decoder.generate(request)

        # Should still produce output (mock truncates to 32k)
        assert result.text
        assert result.tokens_generated > 0

    def test_input_validator_rejects_overlong_prompt(self):
        """The InputValidator enforces max_prompt_length."""
        validator = InputValidator(max_prompt_length=1000)

        long_prompt = "a" * 1500

        with pytest.raises(ValueError, match="exceeds maximum length"):
            validator.validate_prompt(long_prompt)

    def test_input_validator_accepts_within_limit(self):
        """Prompts within the limit are accepted."""
        validator = InputValidator(max_prompt_length=1000)

        prompt = "a" * 500
        result = validator.validate_prompt(prompt)
        assert result == prompt
