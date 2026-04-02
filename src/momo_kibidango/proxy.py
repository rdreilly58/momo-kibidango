"""OpenClaw cascade proxy — intercepts all LLM requests and routes through 3-tier decoding.

Runs as a local OpenAI-compatible proxy on port 7779. OpenClaw points its
provider to this proxy, which then cascades Haiku → Sonnet → Opus based
on confidence scoring.

Usage:
    momo-kibidango serve --mode cascade
    # Then configure OpenClaw to use http://127.0.0.1:7779/v1 as provider
"""

from __future__ import annotations

import json
import logging
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from flask import Flask, Response, jsonify, request

from momo_kibidango.core.cascade import CascadeDecoder
from momo_kibidango.core.confidence import ConfidenceScorer
from momo_kibidango.core.decoder import GenerationRequest
from momo_kibidango.models.claude_client import ClaudeClient, CLAUDE_HAIKU, CLAUDE_SONNET, CLAUDE_OPUS

logger = logging.getLogger(__name__)

# Metrics log file
METRICS_DIR = Path.home() / ".openclaw" / "logs" / "cascade-metrics"
METRICS_DIR.mkdir(parents=True, exist_ok=True)


def _today_log() -> Path:
    return METRICS_DIR / f"{datetime.now().strftime('%Y-%m-%d')}.jsonl"


def _log_request(data: dict[str, Any]) -> None:
    """Append a request record to today's metrics log."""
    data["timestamp"] = datetime.now(timezone.utc).isoformat()
    with open(_today_log(), "a") as f:
        f.write(json.dumps(data) + "\n")


def create_proxy_app(api_key: str | None = None) -> Flask:
    """Create an OpenAI-compatible proxy that cascades through Claude tiers."""

    app = Flask(__name__)

    _api_key = api_key or os.environ.get("ANTHROPIC_API_KEY", "")
    client = ClaudeClient(api_key=_api_key, fallback_to_direct=True)
    scorer = ConfidenceScorer()
    cascade = CascadeDecoder(client=client, scorer=scorer)

    @app.route("/v1/chat/completions", methods=["POST"])
    def chat_completions():
        """OpenAI-compatible chat completions endpoint with cascade routing."""
        start = time.time()
        body = request.get_json(force=True)

        model_requested = body.get("model", "")
        messages = body.get("messages", [])
        max_tokens = body.get("max_tokens", 1024)
        temperature = body.get("temperature", 0.7)

        # Extract prompt from messages
        prompt_parts = []
        system_parts = []
        for msg in messages:
            if msg.get("role") == "system":
                system_parts.append(msg.get("content", ""))
            elif msg.get("role") == "user":
                content = msg.get("content", "")
                if isinstance(content, list):
                    for part in content:
                        if isinstance(part, dict) and part.get("type") == "text":
                            prompt_parts.append(part.get("text", ""))
                else:
                    prompt_parts.append(content)
            elif msg.get("role") == "assistant":
                prompt_parts.append(f"[Previous assistant response: {msg.get('content', '')}]")

        prompt = "\n".join(prompt_parts)
        system = "\n".join(system_parts) if system_parts else None

        try:
            req = GenerationRequest(
                prompt=prompt,
                max_new_tokens=max_tokens,
                temperature=temperature,
            )
            result = cascade.generate(req)
            text = result.text
            elapsed = time.time() - start
            tier = result.stage_acceptance_rates.get("tier", "haiku") if result.stage_acceptance_rates else "haiku"
            confidence = result.stage_acceptance_rates.get("confidence", 0.0) if result.stage_acceptance_rates else 0.0
            cost_usd = result.stage_acceptance_rates.get("cost_usd", 0.0) if result.stage_acceptance_rates else 0.0

            # Log metrics
            _log_request({
                "model_requested": model_requested,
                "tier_used": tier,
                "confidence": round(confidence, 4),
                "input_tokens": result.tokens_generated,
                "output_tokens": result.tokens_generated,
                "cost_usd": round(cost_usd, 8),
                "latency_ms": round(elapsed * 1000, 1),
                "prompt_length": len(prompt),
                "response_length": len(text),
            })

            # Return OpenAI-compatible response
            return jsonify({
                "id": f"cascade-{int(time.time()*1000)}",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": model_requested,
                "choices": [{
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": text,
                    },
                    "finish_reason": "stop",
                }],
                "usage": {
                    "prompt_tokens": result.tokens_generated,
                    "completion_tokens": result.tokens_generated,
                    "total_tokens": result.tokens_generated * 2,
                },
                "cascade_info": {
                    "tier": tier,
                    "confidence": round(confidence, 4),
                    "cost_usd": round(cost_usd, 8),
                },
            })

        except Exception as exc:
            logger.error("Cascade error: %s", exc)
            _log_request({
                "model_requested": model_requested,
                "tier_used": "error",
                "error": str(exc),
                "latency_ms": round((time.time() - start) * 1000, 1),
                "prompt_length": len(prompt),
            })
            return jsonify({"error": {"message": str(exc), "type": "server_error"}}), 500

    @app.route("/v1/models", methods=["GET"])
    def list_models():
        """List available models."""
        return jsonify({
            "object": "list",
            "data": [
                {"id": "anthropic/claude-haiku-4-5", "object": "model"},
                {"id": "anthropic/claude-sonnet-4-5", "object": "model"},
                {"id": "anthropic/claude-opus-4", "object": "model"},
                {"id": "cascade", "object": "model"},
            ]
        })

    @app.route("/health", methods=["GET"])
    def health():
        return jsonify({"status": "ok", "mode": "cascade"})

    @app.route("/v1/metrics", methods=["GET"])
    def metrics():
        """Return today's cascade metrics summary."""
        return jsonify(_get_daily_summary())

    return app


def _get_daily_summary(date_str: str | None = None) -> dict[str, Any]:
    """Parse today's (or a specific day's) metrics log and return summary."""
    if date_str is None:
        date_str = datetime.now().strftime("%Y-%m-%d")
    
    log_file = METRICS_DIR / f"{date_str}.jsonl"
    if not log_file.exists():
        return {"date": date_str, "total_requests": 0, "message": "No data"}

    records = []
    with open(log_file) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    continue

    if not records:
        return {"date": date_str, "total_requests": 0}

    # Aggregate
    total = len(records)
    errors = sum(1 for r in records if r.get("tier_used") == "error")
    successful = [r for r in records if r.get("tier_used") != "error"]
    
    tier_counts = {}
    total_cost = 0.0
    total_input = 0
    total_output = 0
    total_latency = 0.0

    for r in successful:
        tier = r.get("tier_used", "unknown")
        tier_counts[tier] = tier_counts.get(tier, 0) + 1
        total_cost += r.get("cost_usd", 0)
        total_input += r.get("input_tokens", 0)
        total_output += r.get("output_tokens", 0)
        total_latency += r.get("latency_ms", 0)

    # Calculate opus equivalent cost
    opus_cost = (total_input * 15.0 + total_output * 75.0) / 1_000_000

    avg_latency = total_latency / len(successful) if successful else 0
    avg_confidence = (
        sum(r.get("confidence", 0) for r in successful) / len(successful)
        if successful else 0
    )

    return {
        "date": date_str,
        "total_requests": total,
        "successful": len(successful),
        "errors": errors,
        "tier_breakdown": tier_counts,
        "total_cost_usd": round(total_cost, 6),
        "opus_equivalent_usd": round(opus_cost, 6),
        "savings_usd": round(opus_cost - total_cost, 6),
        "savings_pct": round((1 - total_cost / opus_cost) * 100, 1) if opus_cost > 0 else 0,
        "total_input_tokens": total_input,
        "total_output_tokens": total_output,
        "avg_latency_ms": round(avg_latency, 1),
        "avg_confidence": round(avg_confidence, 4),
    }
