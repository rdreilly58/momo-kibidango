"""REST API server for momo-kibidango speculative decoding.

Provides /health, /ready, /infer, /batch, and /metrics endpoints.
Designed to be instantiated with a BaseDecoder and MetricsCollector.
"""

from __future__ import annotations

import logging
import time
from dataclasses import asdict
from typing import Any

from flask import Flask, jsonify, request

from momo_kibidango.core.decoder import BaseDecoder, GenerationRequest, GenerationResult
from momo_kibidango.monitoring.metrics import MetricsCollector

logger = logging.getLogger(__name__)

OPENCLAW_PORT = 7779


# ------------------------------------------------------------------ #
# Input validation (lightweight, inline)
# ------------------------------------------------------------------ #

class InputValidator:
    """Basic input validation for inference requests."""

    def __init__(self, max_prompt_length: int = 32_000) -> None:
        self.max_prompt_length = max_prompt_length

    def validate_prompt(self, prompt: Any) -> str:
        """Return sanitised prompt or raise ValueError."""
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("'prompt' must be a non-empty string")
        prompt = prompt.strip()
        if len(prompt) > self.max_prompt_length:
            raise ValueError(
                f"Prompt exceeds maximum length ({len(prompt)} > {self.max_prompt_length})"
            )
        return prompt

    def validate_max_tokens(self, value: Any, default: int = 256) -> int:
        if value is None:
            return default
        if not isinstance(value, int) or value < 1:
            raise ValueError("'max_tokens' must be a positive integer")
        return min(value, 4096)

    def validate_temperature(self, value: Any, default: float = 0.7) -> float:
        if value is None:
            return default
        if not isinstance(value, (int, float)) or value < 0.0 or value > 2.0:
            raise ValueError("'temperature' must be between 0.0 and 2.0")
        return float(value)


# ------------------------------------------------------------------ #
# Server
# ------------------------------------------------------------------ #

class InferenceServer:
    """Flask-based REST API wrapping a BaseDecoder."""

    def __init__(
        self,
        decoder: BaseDecoder,
        metrics: MetricsCollector | None = None,
        port: int = OPENCLAW_PORT,
    ) -> None:
        self.decoder = decoder
        self.metrics = metrics or MetricsCollector()
        self.port = port
        self.validator = InputValidator()

        self.app = Flask(__name__)
        self._register_routes()

    # ------------------------------------------------------------ #
    # Routes
    # ------------------------------------------------------------ #

    def _register_routes(self) -> None:

        @self.app.route("/health", methods=["GET"])
        def health():
            return jsonify({"status": "healthy"})

        @self.app.route("/ready", methods=["GET"])
        def ready():
            if not self.decoder.is_loaded:
                return jsonify({"status": "not_ready", "reason": "decoder not loaded"}), 503
            return jsonify({"status": "ready"})

        @self.app.route("/infer", methods=["POST"])
        def infer():
            return self._handle_infer()

        @self.app.route("/batch", methods=["POST"])
        def batch():
            return self._handle_batch()

        @self.app.route("/metrics", methods=["GET"])
        def metrics_endpoint():
            return jsonify(self.metrics.get_summary())

    # ------------------------------------------------------------ #
    # Handlers
    # ------------------------------------------------------------ #

    def _handle_infer(self):
        data = request.get_json(silent=True)
        if not data:
            return jsonify({"error": "Request body must be JSON"}), 400

        try:
            prompt = self.validator.validate_prompt(data.get("prompt"))
            max_tokens = self.validator.validate_max_tokens(data.get("max_tokens"))
            temperature = self.validator.validate_temperature(data.get("temperature"))
        except ValueError as exc:
            return jsonify({"error": str(exc)}), 400

        gen_request = GenerationRequest(
            prompt=prompt,
            max_new_tokens=max_tokens,
            temperature=temperature,
        )

        try:
            t0 = time.perf_counter()
            result: GenerationResult = self.decoder.generate(gen_request)
            duration = time.perf_counter() - t0

            self.metrics.record_inference(
                duration_seconds=duration,
                tokens_generated=result.tokens_generated,
                model_mode=result.mode,
                acceptance_rate=result.acceptance_rate,
                stage_rates=result.stage_acceptance_rates,
            )

            return jsonify({
                "text": result.text,
                "tokens_generated": result.tokens_generated,
                "tokens_per_second": result.tokens_per_second,
                "elapsed_seconds": result.elapsed_seconds,
                "acceptance_rate": result.acceptance_rate,
                "mode": result.mode,
            })

        except Exception as exc:
            logger.exception("Inference failed")
            self.metrics.record_error(type(exc).__name__)
            return jsonify({"error": str(exc)}), 500

    def _handle_batch(self):
        data = request.get_json(silent=True)
        if not data:
            return jsonify({"error": "Request body must be JSON"}), 400

        raw_prompts = data.get("prompts")
        if not isinstance(raw_prompts, list) or not raw_prompts:
            return jsonify({"error": "'prompts' must be a non-empty list"}), 400

        try:
            max_tokens = self.validator.validate_max_tokens(data.get("max_tokens"))
            temperature = self.validator.validate_temperature(data.get("temperature"))
        except ValueError as exc:
            return jsonify({"error": str(exc)}), 400

        results: list[dict[str, Any]] = []
        for raw in raw_prompts:
            try:
                prompt = self.validator.validate_prompt(raw)
            except ValueError as exc:
                results.append({"error": str(exc)})
                continue

            gen_request = GenerationRequest(
                prompt=prompt,
                max_new_tokens=max_tokens,
                temperature=temperature,
            )

            try:
                t0 = time.perf_counter()
                result = self.decoder.generate(gen_request)
                duration = time.perf_counter() - t0

                self.metrics.record_inference(
                    duration_seconds=duration,
                    tokens_generated=result.tokens_generated,
                    model_mode=result.mode,
                    acceptance_rate=result.acceptance_rate,
                    stage_rates=result.stage_acceptance_rates,
                )

                results.append({
                    "text": result.text,
                    "tokens_generated": result.tokens_generated,
                    "tokens_per_second": result.tokens_per_second,
                    "mode": result.mode,
                })
            except Exception as exc:
                logger.exception("Batch item failed")
                self.metrics.record_error(type(exc).__name__)
                results.append({"error": str(exc)})

        return jsonify({"results": results, "batch_size": len(raw_prompts)})

    # ------------------------------------------------------------ #
    # Run
    # ------------------------------------------------------------ #

    def run(self, host: str = "0.0.0.0", debug: bool = False) -> None:
        """Start the Flask development server."""
        self.app.run(host=host, port=self.port, debug=debug, threaded=True)
