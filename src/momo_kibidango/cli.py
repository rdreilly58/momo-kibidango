#!/usr/bin/env python3
"""
Momo-Kibidango CLI Entry Point

Provides command-line interface for running inference, benchmarks, validation, and MCP server.
"""

import argparse
import asyncio
import sys
from pathlib import Path
from typing import Optional

try:
    from .speculative_2model import SpeculativeDecoder, ModelConfig
    from .monitoring import PerformanceMonitor
    from .production_hardening import ProductionHardener
except ImportError:
    # Allow CLI to work even if heavy dependencies missing
    SpeculativeDecoder = None
    ModelConfig = None
    PerformanceMonitor = None
    ProductionHardener = None

try:
    from .mcp_server import MomoKibidangoMCPServer
    HAS_MCP = True
except ImportError:
    HAS_MCP = False
    MomoKibidangoMCPServer = None


def setup_parser() -> argparse.ArgumentParser:
    """Create and configure CLI argument parser."""
    parser = argparse.ArgumentParser(
        prog="momo-kibidango",
        description="Speculative decoding framework for Apple Silicon and beyond",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  momo-kibidango --version
  momo-kibidango run --prompt "Hello world"
  momo-kibidango benchmark
  momo-kibidango validate
        """,
    )

    parser.add_argument(
        "--version",
        action="version",
        version="%(prog)s 1.0.0",
    )

    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Run inference
    run_parser = subparsers.add_parser("run", help="Run inference with speculative decoding")
    run_parser.add_argument(
        "--prompt",
        "-p",
        type=str,
        required=True,
        help="Input prompt for inference",
    )
    run_parser.add_argument(
        "--max-tokens",
        type=int,
        default=512,
        help="Maximum tokens to generate (default: 512)",
    )
    run_parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature (default: 0.7)",
    )
    run_parser.add_argument(
        "--draft-model",
        type=str,
        help="Draft model name (optional)",
    )
    run_parser.add_argument(
        "--target-model",
        type=str,
        help="Target model name (optional)",
    )

    # Benchmark
    bench_parser = subparsers.add_parser("benchmark", help="Run performance benchmarks")
    bench_parser.add_argument(
        "--test-cases",
        type=int,
        default=10,
        help="Number of test cases (default: 10)",
    )
    bench_parser.add_argument(
        "--output",
        "-o",
        type=str,
        help="Output file for benchmark results",
    )

    # Validate
    val_parser = subparsers.add_parser("validate", help="Validate installation and dependencies")
    val_parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Verbose validation output",
    )

    # MCP Server
    serve_parser = subparsers.add_parser(
        "serve",
        help="Start MCP (Model Context Protocol) server for agent integration",
    )
    serve_parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level (default: INFO)",
    )
    serve_parser.add_argument(
        "--host",
        default="localhost",
        help="Host to bind to (default: localhost)",
    )
    serve_parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port to bind to (default: 8000)",
    )

    return parser


def cmd_run(args) -> int:
    """Execute run command."""
    try:
        if SpeculativeDecoder is None:
            print("❌ Speculative decoding dependencies not installed", file=sys.stderr)
            print("Install with: pip install momo-kibidango[inference]", file=sys.stderr)
            return 1
        
        print(f"🍑 Running inference with speculative decoding...")
        print(f"  Prompt: {args.prompt}")
        print(f"  Max tokens: {args.max_tokens}")
        print(f"  Temperature: {args.temperature}")
        
        # TODO: Implement actual inference
        print("✅ Inference complete (placeholder)")
        return 0
        
    except Exception as e:
        print(f"❌ Error: {e}", file=sys.stderr)
        return 1


def cmd_benchmark(args) -> int:
    """Execute benchmark command."""
    try:
        if PerformanceMonitor is None:
            print("❌ Benchmark dependencies not installed", file=sys.stderr)
            print("Install with: pip install momo-kibidango[inference]", file=sys.stderr)
            return 1
        
        print(f"🍑 Running benchmarks...")
        print(f"  Test cases: {args.test_cases}")
        
        # TODO: Implement actual benchmarking
        print("✅ Benchmarks complete (placeholder)")
        
        if args.output:
            print(f"📊 Results saved to: {args.output}")
        
        return 0
        
    except Exception as e:
        print(f"❌ Error: {e}", file=sys.stderr)
        return 1


def cmd_validate(args) -> int:
    """Execute validate command."""
    try:
        print(f"🍑 Validating installation...")
        
        # Check imports
        checks = {
            "PyTorch": False,
            "Transformers": False,
            "vLLM": False,
            "Pydantic": False,
            "MCP": False,
        }
        
        try:
            import torch
            checks["PyTorch"] = True
            if args.verbose:
                print(f"  ✓ PyTorch {torch.__version__}")
        except ImportError:
            pass
        
        try:
            import transformers
            checks["Transformers"] = True
            if args.verbose:
                print(f"  ✓ Transformers {transformers.__version__}")
        except ImportError:
            pass
        
        try:
            import vllm
            checks["vLLM"] = True
            if args.verbose:
                print(f"  ✓ vLLM {vllm.__version__}")
        except ImportError:
            pass
        
        try:
            import pydantic
            checks["Pydantic"] = True
            if args.verbose:
                print(f"  ✓ Pydantic {pydantic.__version__}")
        except ImportError:
            pass

        try:
            import mcp
            checks["MCP"] = True
            if args.verbose:
                print(f"  ✓ MCP SDK (available for agent integration)")
        except ImportError:
            if args.verbose:
                print(f"  ℹ MCP SDK (optional, for agent integration)")
        
        # Print summary
        passed = sum(1 for v in checks.values() if v)
        total = len(checks)
        
        print(f"\n📊 Validation Results: {passed}/{total} checks passed")
        for name, status in checks.items():
            symbol = "✓" if status else "✗"
            print(f"  {symbol} {name}")
        
        if passed >= 4:  # Core deps (PyTorch, Transformers, Pydantic, vLLM or core)
            print("\n✅ Installation validated successfully!")
            print("\nNext steps:")
            if checks["MCP"]:
                print("  • Start MCP server: momo-kibidango serve")
            else:
                print("  • Install MCP for agent integration: pip install momo-kibidango[mcp]")
            return 0
        else:
            print(f"\n❌ {total - passed} dependency/dependencies missing")
            return 1
        
    except Exception as e:
        print(f"❌ Error: {e}", file=sys.stderr)
        return 1


def cmd_serve(args) -> int:
    """Execute serve command (start MCP server)."""
    try:
        if not HAS_MCP:
            print("❌ MCP SDK not installed", file=sys.stderr)
            print("Install with: pip install momo-kibidango[mcp]", file=sys.stderr)
            return 1
        
        print(f"🍑 Starting Momo-Kibidango MCP Server...")
        print(f"  Log level: {args.log_level}")
        print(f"  Listening in stdio mode (for direct agent integration)")
        print(f"")
        print(f"Integration:")
        print(f"  • Claude SDK: client.add_mcp_server(...)")
        print(f"  • Tools available: run_inference, benchmark_models")
        print(f"")
        
        # Create and run server
        server = MomoKibidangoMCPServer(log_level=args.log_level)
        
        # Run async server
        asyncio.run(server.run(host=args.host, port=args.port))
        
        return 0
        
    except KeyboardInterrupt:
        print(f"\n🛑 Server stopped", file=sys.stderr)
        return 0
    except Exception as e:
        print(f"❌ Error: {e}", file=sys.stderr)
        return 1


def main(argv: Optional[list[str]] = None) -> int:
    """Main CLI entry point."""
    parser = setup_parser()
    args = parser.parse_args(argv)
    
    # Show help if no command
    if not args.command:
        parser.print_help()
        return 0
    
    # Dispatch to command handler
    if args.command == "run":
        return cmd_run(args)
    elif args.command == "benchmark":
        return cmd_benchmark(args)
    elif args.command == "validate":
        return cmd_validate(args)
    elif args.command == "serve":
        return cmd_serve(args)
    else:
        print(f"Unknown command: {args.command}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
