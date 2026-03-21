#!/usr/bin/env python3
"""
Momo-Kibidango MCP Server

Implements Model Context Protocol (MCP) integration for momo-kibidango speculative decoding.
Provides tools for LLM agents to run inference and benchmarks.

This server can be run in stdio mode (default for direct agent integration)
or HTTP mode (for network-based access).
"""

import asyncio
import json
import logging
import sys
from typing import Any, Optional

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

from .speculative_2model import SpeculativeDecoder, ModelConfig
from .monitoring import PerformanceMonitor
from .production_hardening import ProductionHardener


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


class MomoKibidangoMCPServer:
    """MCP Server for momo-kibidango speculative decoding."""

    def __init__(self, log_level: str = "INFO"):
        """
        Initialize MCP server.

        Args:
            log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
        """
        if not HAS_MCP:
            raise RuntimeError(
                "MCP SDK not installed. Install with: pip install momo-kibidango[mcp]"
            )

        self.app = Server("momo-kibidango")
        self.log_level = log_level
        self._setup_logging()

        # Initialize core components (lazy load on first use)
        self.decoder: Optional[SpeculativeDecoder] = None
        self.monitor: Optional[PerformanceMonitor] = None
        self.hardener: Optional[ProductionHardener] = None

        logger.info("🍑 Momo-Kibidango MCP Server initialized")

    def _setup_logging(self) -> None:
        """Configure logging for the server."""
        level = getattr(logging, self.log_level.upper(), logging.INFO)
        logger.setLevel(level)

    async def _ensure_decoder(self) -> SpeculativeDecoder:
        """Lazy-load speculative decoder on first use."""
        if self.decoder is None:
            logger.info("Loading speculative decoder...")
            try:
                # Use default models or check config
                self.decoder = SpeculativeDecoder()
                logger.info("✓ Speculative decoder loaded")
            except Exception as e:
                error_msg = f"Failed to load decoder: {str(e)}"
                logger.error(error_msg)
                raise RuntimeError(error_msg) from e
        return self.decoder

    async def _ensure_monitor(self) -> PerformanceMonitor:
        """Lazy-load performance monitor on first use."""
        if self.monitor is None:
            logger.info("Initializing performance monitor...")
            try:
                self.monitor = PerformanceMonitor()
                logger.info("✓ Performance monitor initialized")
            except Exception as e:
                error_msg = f"Failed to initialize monitor: {str(e)}"
                logger.error(error_msg)
                raise RuntimeError(error_msg) from e
        return self.monitor

    async def _ensure_hardener(self) -> ProductionHardener:
        """Lazy-load production hardener on first use."""
        if self.hardener is None:
            logger.info("Initializing production hardener...")
            try:
                self.hardener = ProductionHardener()
                logger.info("✓ Production hardener initialized")
            except Exception as e:
                error_msg = f"Failed to initialize hardener: {str(e)}"
                logger.error(error_msg)
                raise RuntimeError(error_msg) from e
        return self.hardener

    def setup_handlers(self) -> None:
        """Register MCP tool handlers."""

        @self.app.list_tools()
        async def list_tools() -> list[Tool]:
            """List available tools."""
            logger.debug("Listing available tools")
            return [
                Tool(
                    name="run_inference",
                    description=(
                        "Run speculative decoding inference on a prompt. "
                        "Compares draft and target models to achieve faster inference "
                        "with speculative decoding."
                    ),
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "prompt": {
                                "type": "string",
                                "description": "Input prompt for inference",
                            },
                            "max_tokens": {
                                "type": "integer",
                                "description": "Maximum tokens to generate (default: 512)",
                                "default": 512,
                                "minimum": 1,
                                "maximum": 4096,
                            },
                            "temperature": {
                                "type": "number",
                                "description": "Sampling temperature for generation (default: 0.7)",
                                "default": 0.7,
                                "minimum": 0.0,
                                "maximum": 2.0,
                            },
                            "draft_model": {
                                "type": "string",
                                "description": (
                                    "Draft model to use (optional, uses config default if not specified)"
                                ),
                            },
                            "target_model": {
                                "type": "string",
                                "description": (
                                    "Target model to use (optional, uses config default if not specified)"
                                ),
                            },
                        },
                        "required": ["prompt"],
                    },
                ),
                Tool(
                    name="benchmark_models",
                    description=(
                        "Run comprehensive benchmark comparing draft and target models. "
                        "Measures speedup, latency, accuracy, and resource usage."
                    ),
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "test_cases": {
                                "type": "integer",
                                "description": "Number of test cases to run (default: 10)",
                                "default": 10,
                                "minimum": 1,
                                "maximum": 100,
                            },
                            "output_format": {
                                "type": "string",
                                "description": "Output format for results (default: json)",
                                "enum": ["json", "csv"],
                                "default": "json",
                            },
                            "save_results": {
                                "type": "boolean",
                                "description": (
                                    "Save benchmark results to file (optional)"
                                ),
                                "default": False,
                            },
                        },
                    },
                ),
            ]

        @self.app.call_tool()
        async def call_tool(name: str, arguments: dict) -> list[TextContent]:
            """Handle tool calls."""
            logger.info(f"Tool called: {name}")
            logger.debug(f"Arguments: {arguments}")

            try:
                if name == "run_inference":
                    return await self._handle_run_inference(arguments)
                elif name == "benchmark_models":
                    return await self._handle_benchmark_models(arguments)
                else:
                    error_msg = f"Unknown tool: {name}"
                    logger.error(error_msg)
                    raise ValueError(error_msg)
            except Exception as e:
                logger.exception(f"Error handling tool {name}")
                error_response = {
                    "status": "error",
                    "error": str(e),
                    "tool": name,
                }
                return [TextContent(type="text", text=json.dumps(error_response, indent=2))]

    async def _handle_run_inference(self, arguments: dict) -> list[TextContent]:
        """Handle run_inference tool call."""
        logger.info("Processing run_inference request")

        # Validate inputs
        prompt = arguments.get("prompt")
        if not prompt or not isinstance(prompt, str):
            raise ValueError("'prompt' is required and must be a string")

        max_tokens = arguments.get("max_tokens", 512)
        if not isinstance(max_tokens, int) or max_tokens < 1:
            raise ValueError("'max_tokens' must be a positive integer")

        temperature = arguments.get("temperature", 0.7)
        if not isinstance(temperature, (int, float)) or temperature < 0 or temperature > 2.0:
            raise ValueError("'temperature' must be between 0.0 and 2.0")

        draft_model = arguments.get("draft_model")
        target_model = arguments.get("target_model")

        logger.info(f"Inference request: prompt length={len(prompt)}, max_tokens={max_tokens}")

        try:
            # Load decoder
            decoder = await self._ensure_decoder()

            # Run inference
            logger.debug("Starting inference...")
            result = await asyncio.to_thread(
                decoder.generate,
                prompt=prompt,
                max_tokens=max_tokens,
                temperature=temperature,
            )

            # Format result
            response = {
                "status": "success",
                "generated_text": result.get("text", ""),
                "tokens_generated": result.get("tokens_generated", 0),
                "tokens_per_second": result.get("tokens_per_second", 0.0),
                "latency_ms": result.get("latency_ms", 0.0),
                "model_config": {
                    "draft_model": draft_model or result.get("draft_model", "unknown"),
                    "target_model": target_model or result.get("target_model", "unknown"),
                    "speculative_decoding_enabled": True,
                },
            }

            logger.info(f"Inference complete: {response['tokens_per_second']:.2f} tok/s")
            return [TextContent(type="text", text=json.dumps(response, indent=2))]

        except Exception as e:
            logger.exception("Inference failed")
            error_response = {
                "status": "error",
                "error": str(e),
                "hint": "Check that models are available and properly configured",
            }
            return [TextContent(type="text", text=json.dumps(error_response, indent=2))]

    async def _handle_benchmark_models(self, arguments: dict) -> list[TextContent]:
        """Handle benchmark_models tool call."""
        logger.info("Processing benchmark_models request")

        # Validate inputs
        test_cases = arguments.get("test_cases", 10)
        if not isinstance(test_cases, int) or test_cases < 1:
            raise ValueError("'test_cases' must be a positive integer")

        output_format = arguments.get("output_format", "json")
        if output_format not in ("json", "csv"):
            raise ValueError("'output_format' must be 'json' or 'csv'")

        save_results = arguments.get("save_results", False)

        logger.info(f"Benchmark request: test_cases={test_cases}, format={output_format}")

        try:
            # Load decoder and monitor
            decoder = await self._ensure_decoder()
            monitor = await self._ensure_monitor()

            # Run benchmark
            logger.debug("Starting benchmark...")
            benchmark_results = await asyncio.to_thread(
                monitor.run_benchmark,
                decoder=decoder,
                num_prompts=test_cases,
            )

            # Format results
            if output_format == "json":
                response = {
                    "status": "success",
                    "test_cases": test_cases,
                    "results": benchmark_results,
                    "summary": self._summarize_benchmark(benchmark_results),
                }
                formatted_output = json.dumps(response, indent=2)
            else:  # csv
                formatted_output = self._format_benchmark_csv(
                    benchmark_results, test_cases
                )

            logger.info(f"Benchmark complete: {test_cases} test cases")

            if save_results:
                logger.info("Saving benchmark results (feature in progress)")

            return [TextContent(type="text", text=formatted_output)]

        except Exception as e:
            logger.exception("Benchmark failed")
            error_response = {
                "status": "error",
                "error": str(e),
                "hint": "Ensure decoder is properly initialized and test data is available",
            }
            return [TextContent(type="text", text=json.dumps(error_response, indent=2))]

    def _summarize_benchmark(self, results: dict) -> dict:
        """Summarize benchmark results."""
        return {
            "avg_speedup": results.get("speedup", 1.0),
            "avg_latency_ms": results.get("latency_ms", 0.0),
            "total_tokens": results.get("total_tokens", 0),
            "test_count": results.get("test_count", 0),
        }

    def _format_benchmark_csv(self, results: dict, test_cases: int) -> str:
        """Format benchmark results as CSV."""
        lines = [
            "metric,value",
            f"test_cases,{test_cases}",
            f"speedup,{results.get('speedup', 1.0):.2f}",
            f"latency_ms,{results.get('latency_ms', 0.0):.2f}",
            f"total_tokens,{results.get('total_tokens', 0)}",
        ]
        return "\n".join(lines)

    async def run(self, host: str = "localhost", port: int = 8000) -> None:
        """
        Run the MCP server.

        Args:
            host: Host to bind to (default: localhost)
            port: Port to bind to (default: 8000)
        """
        logger.info(f"Starting MCP server on {host}:{port}")

        # Set up handlers
        self.setup_handlers()

        # Run in stdio mode (for direct agent integration)
        # This is the default mode for Claude SDK integration
        try:
            await self.app.run_async()
        except KeyboardInterrupt:
            logger.info("Server shutting down...")
        except Exception as e:
            logger.exception("Server error")
            raise


async def main(
    log_level: str = "INFO",
) -> None:
    """
    Main entry point for MCP server.

    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
    """
    try:
        server = MomoKibidangoMCPServer(log_level=log_level)
        await server.run()
    except Exception as e:
        logger.exception("Fatal error")
        sys.exit(1)


if __name__ == "__main__":
    # Parse command-line arguments
    import argparse

    parser = argparse.ArgumentParser(description="Momo-Kibidango MCP Server")
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level (default: INFO)",
    )

    args = parser.parse_args()

    # Run server
    asyncio.run(main(log_level=args.log_level))
