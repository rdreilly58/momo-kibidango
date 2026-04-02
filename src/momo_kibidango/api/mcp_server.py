"""MCP (Model Context Protocol) server for momo-kibidango.

Exposes speculative decoding as MCP tools so that LLM agents can
invoke inference and benchmarks programmatically.

Requires the optional ``mcp`` extra::

    pip install momo-kibidango[mcp]
"""

from __future__ import annotations

import asyncio
import json
import logging
import sys
import time
from typing import Any, Optional

# ------------------------------------------------------------------ #
# Optional MCP SDK import
# ------------------------------------------------------------------ #
try:
    from mcp.server import Server
    from mcp.types import Tool, TextContent

    HAS_MCP = True
except ImportError:
    HAS_MCP = False
    print(
        "Warning: MCP SDK not installed. Install with: pip install momo-kibidango[mcp]",
        file=sys.stderr,
    )

from momo_kibidango.core.decoder import BaseDecoder, GenerationRequest, GenerationResult
from momo_kibidango.monitoring.metrics import MetricsCollector

logger = logging.getLogger(__name__)


class MomoKibidangoMCPServer:
    """MCP server exposing speculative-decoding tools."""

    def __init__(
        self,
        decoder: BaseDecoder | None = None,
        metrics: MetricsCollector | None = None,
        log_level: str = "INFO",
    ) -> None:
        if not HAS_MCP:
            raise RuntimeError(
                "MCP SDK not installed. Install with: pip install momo-kibidango[mcp]"
            )

        self.app = Server("momo-kibidango")
        self.decoder = decoder
        self.metrics = metrics or MetricsCollector()
        self._configure_logging(log_level)
        logger.info("Momo-Kibidango MCP Server initialized")

    # ------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------ #

    @staticmethod
    def _configure_logging(level: str) -> None:
        numeric = getattr(logging, level.upper(), logging.INFO)
        logging.basicConfig(
            level=numeric,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        )
        logger.setLevel(numeric)

    def _require_decoder(self) -> BaseDecoder:
        if self.decoder is None:
            raise RuntimeError("No decoder configured — call set_decoder() first")
        if not self.decoder.is_loaded:
            logger.info("Decoder not loaded, calling load()…")
            self.decoder.load()
        return self.decoder

    def set_decoder(self, decoder: BaseDecoder) -> None:
        """Attach (or replace) the decoder used by this server."""
        self.decoder = decoder

    # ------------------------------------------------------------ #
    # MCP handler registration
    # ------------------------------------------------------------ #

    def setup_handlers(self) -> None:
        """Register MCP tool list and call handlers."""

        @self.app.list_tools()
        async def list_tools() -> list[Tool]:
            return [
                Tool(
                    name="run_inference",
                    description=(
                        "Run speculative decoding inference on a prompt. "
                        "Returns generated text, throughput, and acceptance metrics."
                    ),
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "prompt": {
                                "type": "string",
                                "description": "Input text prompt",
                            },
                            "max_tokens": {
                                "type": "integer",
                                "description": "Maximum tokens to generate (default 256)",
                                "default": 256,
                                "minimum": 1,
                                "maximum": 4096,
                            },
                            "temperature": {
                                "type": "number",
                                "description": "Sampling temperature (default 0.7)",
                                "default": 0.7,
                                "minimum": 0.0,
                                "maximum": 2.0,
                            },
                        },
                        "required": ["prompt"],
                    },
                ),
                Tool(
                    name="benchmark_models",
                    description=(
                        "Run a quick benchmark comparing speculative-decoding "
                        "throughput and acceptance rate across several prompts."
                    ),
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "num_prompts": {
                                "type": "integer",
                                "description": "Number of test prompts (default 5)",
                                "default": 5,
                                "minimum": 1,
                                "maximum": 50,
                            },
                            "max_tokens": {
                                "type": "integer",
                                "description": "Tokens per prompt (default 128)",
                                "default": 128,
                                "minimum": 1,
                                "maximum": 2048,
                            },
                        },
                    },
                ),
            ]

        @self.app.call_tool()
        async def call_tool(name: str, arguments: dict) -> list[TextContent]:
            logger.info("Tool called: %s", name)
            try:
                if name == "run_inference":
                    return await self._handle_run_inference(arguments)
                elif name == "benchmark_models":
                    return await self._handle_benchmark(arguments)
                else:
                    raise ValueError(f"Unknown tool: {name}")
            except Exception as exc:
                logger.exception("Error in tool %s", name)
                payload = {"status": "error", "error": str(exc), "tool": name}
                return [TextContent(type="text", text=json.dumps(payload, indent=2))]

    # ------------------------------------------------------------ #
    # Tool implementations
    # ------------------------------------------------------------ #

    async def _handle_run_inference(self, arguments: dict) -> list[TextContent]:
        prompt = arguments.get("prompt", "")
        if not prompt:
            raise ValueError("'prompt' is required")

        max_tokens = int(arguments.get("max_tokens", 256))
        temperature = float(arguments.get("temperature", 0.7))

        decoder = self._require_decoder()

        gen_req = GenerationRequest(
            prompt=prompt,
            max_new_tokens=max_tokens,
            temperature=temperature,
        )

        t0 = time.perf_counter()
        result: GenerationResult = await asyncio.to_thread(decoder.generate, gen_req)
        duration = time.perf_counter() - t0

        self.metrics.record_inference(
            duration_seconds=duration,
            tokens_generated=result.tokens_generated,
            model_mode=result.mode,
            acceptance_rate=result.acceptance_rate,
            stage_rates=result.stage_acceptance_rates,
        )

        response = {
            "status": "success",
            "text": result.text,
            "tokens_generated": result.tokens_generated,
            "tokens_per_second": result.tokens_per_second,
            "acceptance_rate": result.acceptance_rate,
            "mode": result.mode,
            "elapsed_seconds": round(duration, 4),
        }
        return [TextContent(type="text", text=json.dumps(response, indent=2))]

    async def _handle_benchmark(self, arguments: dict) -> list[TextContent]:
        num_prompts = int(arguments.get("num_prompts", 5))
        max_tokens = int(arguments.get("max_tokens", 128))

        decoder = self._require_decoder()

        # Simple built-in benchmark prompts
        sample_prompts = [
            "Explain the concept of speculative decoding in three sentences.",
            "Write a Python function to compute Fibonacci numbers.",
            "What are the key differences between TCP and UDP?",
            "Summarize the benefits of model distillation.",
            "Describe how attention mechanisms work in transformers.",
        ]

        results: list[dict[str, Any]] = []
        total_tokens = 0
        total_time = 0.0

        for i in range(num_prompts):
            prompt = sample_prompts[i % len(sample_prompts)]
            gen_req = GenerationRequest(
                prompt=prompt,
                max_new_tokens=max_tokens,
                temperature=0.7,
            )

            t0 = time.perf_counter()
            result = await asyncio.to_thread(decoder.generate, gen_req)
            elapsed = time.perf_counter() - t0

            total_tokens += result.tokens_generated
            total_time += elapsed

            results.append({
                "prompt_index": i,
                "tokens_generated": result.tokens_generated,
                "tokens_per_second": result.tokens_per_second,
                "acceptance_rate": result.acceptance_rate,
                "elapsed_seconds": round(elapsed, 4),
            })

        response = {
            "status": "success",
            "num_prompts": num_prompts,
            "total_tokens": total_tokens,
            "total_seconds": round(total_time, 4),
            "avg_tokens_per_second": round(total_tokens / total_time, 2) if total_time > 0 else 0,
            "results": results,
        }
        return [TextContent(type="text", text=json.dumps(response, indent=2))]

    # ------------------------------------------------------------ #
    # Run
    # ------------------------------------------------------------ #

    async def run(self) -> None:
        """Start the MCP server in stdio mode."""
        self.setup_handlers()
        logger.info("Starting MCP server (stdio mode)")
        await self.app.run_async()


async def main(
    decoder: BaseDecoder | None = None,
    log_level: str = "INFO",
) -> None:
    """Entry point for running the MCP server."""
    server = MomoKibidangoMCPServer(decoder=decoder, log_level=log_level)
    await server.run()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Momo-Kibidango MCP Server")
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    args = parser.parse_args()
    asyncio.run(main(log_level=args.log_level))
