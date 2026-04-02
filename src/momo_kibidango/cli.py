#!/usr/bin/env python3
"""
Momo-Kibidango CLI

Command-line interface for running inference, benchmarks, validation,
and the REST API server.
"""

import argparse
import sys
from typing import Optional


def _load_settings(config_path: Optional[str] = None):
    """Load DecoderSettings from a YAML file or return defaults."""
    from momo_kibidango.config.settings import DecoderSettings

    if config_path:
        return DecoderSettings.from_yaml(config_path)
    return DecoderSettings()


def _build_decoder(
    settings,
    mode: str,
    api_key: str | None = None,
    gateway_url: str | None = None,
):
    """Construct the appropriate decoder based on mode and settings.

    Parameters
    ----------
    settings:
        A ``DecoderSettings`` instance.
    mode:
        One of ``"2model"``, ``"3model"``, ``"auto"``, or ``"cascade"``.
    api_key:
        Optional Anthropic API key for direct fallback.
    gateway_url:
        Optional OpenClaw gateway URL override.

    Returns
    -------
    BaseDecoder
        A decoder instance ready to be loaded.
    """
    if mode == "cascade":
        from momo_kibidango.core.cascade import CascadeDecoder
        from momo_kibidango.models.claude_client import ClaudeClient

        client = ClaudeClient(
            api_key=api_key,
            gateway_url=gateway_url,
        )
        backend = client.backend
        print(f"Backend: {backend} ({client._gateway_url})" if backend == "gateway" else f"Backend: {backend}")
        return CascadeDecoder(client=client)

    from momo_kibidango.models.registry import ModelRegistry, ModelTier
    from momo_kibidango.models.loader import ModelLoader
    from momo_kibidango.monitoring.metrics import MetricsCollector
    from momo_kibidango.core.adaptive import AdaptiveThreshold
    from momo_kibidango.core.two_model import TwoModelDecoder
    from momo_kibidango.core.three_model import ThreeModelDecoder

    registry = ModelRegistry.from_settings(settings)
    loader = ModelLoader(
        device=settings.resolve_device(),
        memory_headroom_gb=settings.memory_headroom_gb,
    )
    metrics = MetricsCollector()

    adaptive = None
    if settings.adaptive_enabled:
        adaptive = AdaptiveThreshold(
            initial_stage1=settings.stage1_threshold,
            initial_stage2=settings.stage2_threshold,
            target_acceptance_rate=settings.adaptive_target_rate,
            ema_alpha=settings.adaptive_ema_alpha,
            warmup_iterations=settings.adaptive_warmup,
        )

    if mode == "3model" or (mode == "auto" and registry.has_tier(ModelTier.QUALIFIER)):
        return ThreeModelDecoder(settings, registry, loader, metrics, adaptive)
    return TwoModelDecoder(settings, registry, loader, metrics, adaptive)


# ---------------------------------------------------------------------------
# Command handlers
# ---------------------------------------------------------------------------


def cmd_run(args) -> int:
    """Run inference with speculative decoding."""
    from momo_kibidango.core.decoder import GenerationRequest

    try:
        settings = _load_settings(args.config)
        if args.device:
            settings = settings.model_copy(update={"device": args.device})

        api_key = getattr(args, "api_key", None)
        gateway_url = getattr(args, "gateway_url", None)
        decoder = _build_decoder(settings, args.mode, api_key=api_key, gateway_url=gateway_url)
        decoder.load()
        try:
            request = GenerationRequest(
                prompt=args.prompt,
                max_new_tokens=args.max_tokens,
                temperature=args.temperature,
            )
            result = decoder.generate(request)

            print(f"\n{result.text}")
            print(f"\n--- Metrics ---")
            print(f"Mode: {result.mode}")
            print(f"Tokens: {result.tokens_generated}")
            print(f"Speed: {result.tokens_per_second:.1f} tok/s")
            print(f"Acceptance: {result.acceptance_rate:.1%}")
            print(f"Memory: {result.peak_memory_gb:.2f} GB")
            print(f"Time: {result.elapsed_seconds:.2f}s")

            # Show cascade cost savings if applicable
            if result.mode == "cascade" and result.stage_acceptance_rates:
                rates = result.stage_acceptance_rates
                print(f"Tier: {rates.get('tier', 'unknown')}")
                print(f"Confidence: {rates.get('confidence', 0):.2f}")
                print(f"Cost: ${rates.get('cost_usd', 0):.6f}")
                from momo_kibidango.core.cascade import CascadeDecoder

                if isinstance(decoder, CascadeDecoder):
                    summary = decoder.cost_tracker.summary()
                    print(f"Savings vs Opus: ${summary['savings_usd']:.6f}")
        finally:
            decoder.unload()
        return 0

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


def cmd_benchmark(args) -> int:
    """Run performance benchmarks across a set of test prompts."""
    from momo_kibidango.core.decoder import GenerationRequest

    test_prompts = [
        "The future of artificial intelligence is",
        "Once upon a time in a land far away",
        "def fibonacci(n):",
        "Explain quantum computing in simple terms:",
        "The best way to learn programming is",
    ][: args.num_prompts]

    try:
        settings = _load_settings(args.config)
        bench_mode = getattr(args, "mode", "auto") or "auto"
        api_key = getattr(args, "api_key", None)
        gateway_url = getattr(args, "gateway_url", None)
        decoder = _build_decoder(settings, bench_mode, api_key=api_key, gateway_url=gateway_url)
        decoder.load()
        try:
            results = []
            for prompt in test_prompts:
                request = GenerationRequest(prompt=prompt, max_new_tokens=64)
                result = decoder.generate(request)
                entry = {
                    "prompt": prompt,
                    "tokens_generated": result.tokens_generated,
                    "tokens_per_second": result.tokens_per_second,
                    "acceptance_rate": result.acceptance_rate,
                    "elapsed_seconds": result.elapsed_seconds,
                    "peak_memory_gb": result.peak_memory_gb,
                    "mode": result.mode,
                }
                results.append(entry)
                print(f"  {prompt[:40]}... => {result.tokens_per_second:.1f} tok/s")

            # Summary
            avg_speed = sum(r["tokens_per_second"] for r in results) / len(results)
            avg_acceptance = sum(r["acceptance_rate"] for r in results) / len(results)
            print(f"\nAverage: {avg_speed:.1f} tok/s, {avg_acceptance:.1%} acceptance")

            if args.output:
                import json

                with open(args.output, "w") as f:
                    json.dump(results, f, indent=2)
                print(f"Results saved to {args.output}")
        finally:
            decoder.unload()
        return 0

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


def cmd_serve(args) -> int:
    """Start the REST API server."""
    try:
        settings = _load_settings(args.config)
        from momo_kibidango.config.settings import ServerSettings

        server_settings = ServerSettings(host=args.host, port=args.port)

        print(f"Starting momo-kibidango server on {args.host}:{args.port}")
        print("Press Ctrl+C to stop.\n")

        # Import and launch the server application
        from momo_kibidango.api.server import create_app

        app = create_app(settings, server_settings)

        import uvicorn

        uvicorn.run(app, host=args.host, port=args.port)
        return 0

    except KeyboardInterrupt:
        print("\nServer stopped.")
        return 0
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


def cmd_validate(args) -> int:
    """Check installation and dependency availability."""
    print("Checking dependencies...\n")
    checks = []

    # -- torch ---------------------------------------------------------------
    try:
        import torch

        checks.append(("torch", True, torch.__version__))
    except ImportError:
        checks.append(("torch", False, "not installed"))

    # -- transformers --------------------------------------------------------
    try:
        import transformers

        checks.append(("transformers", True, transformers.__version__))
    except ImportError:
        checks.append(("transformers", False, "not installed"))

    # -- pydantic ------------------------------------------------------------
    try:
        import pydantic

        checks.append(("pydantic", True, pydantic.__version__))
    except ImportError:
        checks.append(("pydantic", False, "not installed"))

    # -- pyyaml --------------------------------------------------------------
    try:
        import yaml

        checks.append(("pyyaml", True, yaml.__version__))
    except ImportError:
        checks.append(("pyyaml", False, "not installed"))

    # -- Device availability -------------------------------------------------
    cuda_available = False
    mps_available = False
    try:
        import torch as _torch

        cuda_available = _torch.cuda.is_available()
        mps_available = (
            hasattr(_torch.backends, "mps") and _torch.backends.mps.is_available()
        )
    except ImportError:
        pass

    if cuda_available:
        device_count = _torch.cuda.device_count()
        device_name = _torch.cuda.get_device_name(0)
        checks.append(("CUDA", True, f"{device_count} device(s) - {device_name}"))
    else:
        checks.append(("CUDA", False, "not available"))

    if mps_available:
        checks.append(("MPS (Apple Silicon)", True, "available"))
    else:
        checks.append(("MPS (Apple Silicon)", False, "not available"))

    # -- Print results -------------------------------------------------------
    name_width = max(len(name) for name, _, _ in checks)
    passed = 0
    for name, ok, detail in checks:
        status = "OK" if ok else "MISSING"
        print(f"  {name:<{name_width}}  [{status:>7}]  {detail}")
        if ok:
            passed += 1

    print(f"\n{passed}/{len(checks)} checks passed.")

    # Determine recommended device
    if cuda_available:
        rec_device = "cuda"
    elif mps_available:
        rec_device = "mps"
    else:
        rec_device = "cpu"
    print(f"Recommended device: {rec_device}")

    # A non-zero exit only when core deps are missing
    core_ok = all(ok for name, ok, _ in checks if name in ("torch", "transformers", "pydantic"))
    if not core_ok:
        print("\nCore dependencies are missing. Install with: pip install momo-kibidango")
        return 1
    return 0


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="momo-kibidango",
        description="3-tier speculative decoding framework for LLM inference acceleration",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
examples:
  momo-kibidango run -p "Explain gravity"
  momo-kibidango benchmark --num-prompts 3 -o results.json
  momo-kibidango serve --port 7779
  momo-kibidango validate
""",
    )
    parser.add_argument(
        "--version", action="version", version="%(prog)s 2.0.0"
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # -- run -----------------------------------------------------------------
    run_p = subparsers.add_parser("run", help="Run inference with speculative decoding")
    run_p.add_argument("--prompt", "-p", type=str, required=True, help="Input prompt")
    run_p.add_argument(
        "--max-tokens", type=int, default=256, help="Max tokens to generate (default: 256)"
    )
    run_p.add_argument(
        "--temperature", type=float, default=0.7, help="Sampling temperature (default: 0.7)"
    )
    run_p.add_argument(
        "--mode",
        type=str,
        default="auto",
        choices=["2model", "3model", "auto", "cascade"],
        help="Decoding mode (default: auto)",
    )
    run_p.add_argument("--config", "-c", type=str, default=None, help="Path to YAML config file")
    run_p.add_argument(
        "--device", type=str, default=None, help="Override device (auto/cuda/mps/cpu)"
    )
    run_p.add_argument(
        "--api-key", type=str, default=None, help="Anthropic API key for direct fallback (or set ANTHROPIC_API_KEY)"
    )
    run_p.add_argument(
        "--gateway-url", type=str, default=None,
        help="OpenClaw gateway URL (default: http://127.0.0.1:18789/v1)",
    )

    # -- benchmark -----------------------------------------------------------
    bench_p = subparsers.add_parser("benchmark", help="Run performance benchmarks")
    bench_p.add_argument(
        "--num-prompts", type=int, default=5, help="Number of test prompts (default: 5)"
    )
    bench_p.add_argument(
        "--mode",
        type=str,
        default="auto",
        choices=["2model", "3model", "auto", "cascade"],
        help="Decoding mode (default: auto)",
    )
    bench_p.add_argument("--config", "-c", type=str, default=None, help="Path to config file")
    bench_p.add_argument(
        "--output", "-o", type=str, default=None, help="Output file for results (JSON)"
    )
    bench_p.add_argument(
        "--api-key", type=str, default=None, help="Anthropic API key for direct fallback (or set ANTHROPIC_API_KEY)"
    )
    bench_p.add_argument(
        "--gateway-url", type=str, default=None,
        help="OpenClaw gateway URL (default: http://127.0.0.1:18789/v1)",
    )

    # -- serve ---------------------------------------------------------------
    serve_p = subparsers.add_parser("serve", help="Start REST API server")
    serve_p.add_argument("--host", type=str, default="0.0.0.0", help="Server host (default: 0.0.0.0)")
    serve_p.add_argument("--port", type=int, default=7779, help="Server port (default: 7779)")
    serve_p.add_argument("--config", "-c", type=str, default=None, help="Path to config file")

    # -- validate ------------------------------------------------------------
    subparsers.add_parser("validate", help="Check installation and dependencies")

    return parser


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main(argv: Optional[list[str]] = None) -> int:
    """Main CLI entry point."""
    parser = _build_parser()
    args = parser.parse_args(argv)

    if not args.command:
        parser.print_help()
        return 0

    handlers = {
        "run": cmd_run,
        "benchmark": cmd_benchmark,
        "serve": cmd_serve,
        "validate": cmd_validate,
    }

    handler = handlers.get(args.command)
    if handler is None:
        print(f"Unknown command: {args.command}", file=sys.stderr)
        return 1

    return handler(args)


if __name__ == "__main__":
    sys.exit(main())
