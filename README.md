# momo-kibidango

A modular 3-tier speculative decoding framework for accelerating LLM inference.

## Architecture

```
                    ┌─────────────────────────────────────────┐
                    │          User Interfaces                │
                    │   CLI  ·  REST API  ·  MCP Protocol     │
                    └───────────────┬─────────────────────────┘
                                    │
                    ┌───────────────▼─────────────────────────┐
                    │       GenerationRequest / Result         │
                    │          (core/decoder.py)               │
                    └───────────────┬─────────────────────────┘
                                    │
           ┌────────────────────────┼────────────────────────┐
           │                        │                        │
  ┌────────▼─────────┐   ┌─────────▼────────┐   ┌──────────▼─────────┐
  │  TwoModelDecoder  │   │ ThreeModelDecoder │   │  Single Model      │
  │  (draft+target)   │   │ (draft+qual+tgt)  │   │  (fallback only)   │
  └────────┬─────────┘   └─────────┬────────┘   └──────────┬─────────┘
           │                        │                        │
           └────────────────────────┼────────────────────────┘
                                    │
                    ┌───────────────▼─────────────────────────┐
                    │  AdaptiveThreshold  ·  KVCacheManager   │
                    │  TokenizerBridge    ·  MetricsCollector  │
                    └───────────────┬─────────────────────────┘
                                    │
                    ┌───────────────▼─────────────────────────┐
                    │         ModelRegistry + ModelLoader      │
                    │   (memory-aware, dtype fallback chain)   │
                    └───────────────┬─────────────────────────┘
                                    │
           ┌────────────────────────┼────────────────────────┐
           │                        │                        │
  ┌────────▼─────────┐   ┌─────────▼────────┐   ┌──────────▼─────────┐
  │   Tier 1: Draft   │   │  Tier 2: Qualifier│   │  Tier 3: Target    │
  │  Qwen2.5-0.5B     │   │  Phi-2 (2.7B)     │   │  Qwen2.5-7B        │
  │  (ultra-fast)      │   │  (medium filter)   │   │  (final verify)    │
  └───────────────────┘   └──────────────────┘   └────────────────────┘
```

### How speculative decoding works

1. **Draft** — A small, fast model generates K candidate tokens in K sequential steps
2. **Qualify** (3-tier only) — A mid-size model filters candidates, rejecting unlikely tokens
3. **Verify** — The large target model scores all surviving candidates in a *single* forward pass
4. **Accept/Reject** — Tokens where the target probability exceeds the threshold are accepted; the first rejection triggers a resample

Because GPU forward passes process multiple tokens nearly as fast as one, verifying 5 tokens costs roughly the same as generating 1. This yields 1.5-2x+ speedup with identical output quality.

### Adaptive thresholds

Acceptance thresholds auto-tune at runtime using an EMA controller:
- If acceptance rate is too high → threshold increases (saves verification compute)
- If acceptance rate is too low → threshold decreases (avoids excessive fallbacks)

## Package structure

```
src/momo_kibidango/
├── __init__.py              # Lazy-loaded public API
├── exceptions.py            # Custom exception hierarchy
├── utils.py                 # Device detection, logging, memory utils
├── cli.py                   # CLI entry point (argparse)
├── config/
│   ├── settings.py          # Pydantic settings (DecoderSettings, ServerSettings)
│   └── defaults.yaml        # Default configuration
├── core/
│   ├── decoder.py           # BaseDecoder ABC, GenerationRequest/Result
│   ├── two_model.py         # 2-model speculative decoder
│   ├── three_model.py       # 3-tier pyramid decoder
│   ├── adaptive.py          # Adaptive threshold controller
│   └── kv_cache.py          # KV-cache reuse manager
├── models/
│   ├── registry.py          # Model tier registry
│   ├── loader.py            # Memory-aware model loader
│   └── tokenizer_bridge.py  # Cross-tokenizer mapping
├── monitoring/
│   ├── metrics.py           # Lightweight metrics collector
│   └── health.py            # System health checker
└── api/
    ├── server.py            # Flask REST API (port 7779)
    └── mcp_server.py        # MCP protocol server for AI agents
```

## Installation

```bash
pip install momo-kibidango

# With server support
pip install momo-kibidango[server]

# Development
pip install momo-kibidango[dev]
```

### From source

```bash
git clone https://github.com/rdreilly58/momo-kibidango.git
cd momo-kibidango
pip install -e ".[dev,server]"
```

## Quick start

### CLI

```bash
# Run inference (auto-detects 2-model or 3-model based on config)
momo-kibidango run --prompt "The future of AI is" --max-tokens 128

# With a custom config file
momo-kibidango run -p "Hello world" -c config.yaml --device mps

# Run benchmarks
momo-kibidango benchmark --num-prompts 10 --output results.json

# Start REST API server
momo-kibidango serve --port 7779

# Validate installation
momo-kibidango validate
```

### Python API

```python
from momo_kibidango import (
    DecoderSettings, ModelRegistry, ModelLoader,
    TwoModelDecoder, MetricsCollector, AdaptiveThreshold,
    GenerationRequest,
)

# Configure
settings = DecoderSettings(
    draft_model_id="Qwen/Qwen2.5-0.5B-Instruct",
    target_model_id="Qwen/Qwen2.5-7B-Instruct",
    device="auto",
    adaptive_enabled=True,
)

# Build the pipeline
registry = ModelRegistry.from_settings(settings)
loader = ModelLoader(device=settings.resolve_device())
metrics = MetricsCollector()
adaptive = AdaptiveThreshold()

decoder = TwoModelDecoder(settings, registry, loader, metrics, adaptive)
decoder.load()

# Generate
request = GenerationRequest(prompt="Explain quantum computing:", max_new_tokens=256)
result = decoder.generate(request)

print(result.text)
print(f"Speed: {result.tokens_per_second:.1f} tok/s")
print(f"Acceptance rate: {result.acceptance_rate:.1%}")

decoder.unload()
```

### 3-tier mode

```python
settings = DecoderSettings(
    draft_model_id="Qwen/Qwen2.5-0.5B-Instruct",
    qualifier_model_id="microsoft/phi-2",       # enables 3-tier
    target_model_id="Qwen/Qwen2.5-7B-Instruct",
)
# ... same pipeline setup, use ThreeModelDecoder
```

## Configuration

Settings can be provided via:
1. **Constructor arguments** (highest priority)
2. **Environment variables** with `MOMO_` prefix (e.g., `MOMO_TEMPERATURE=0.5`)
3. **YAML file** via `DecoderSettings.from_yaml("config.yaml")`
4. **Defaults** (lowest priority)

### Configuration reference

| Setting | Default | Description |
|---------|---------|-------------|
| `draft_model_id` | `Qwen/Qwen2.5-0.5B-Instruct` | Tier 1 draft model |
| `qualifier_model_id` | `None` | Tier 2 qualifier (None = 2-model mode) |
| `target_model_id` | `Qwen/Qwen2.5-7B-Instruct` | Tier 3 target model |
| `max_draft_tokens` | `5` | Candidates per draft round (1-20) |
| `temperature` | `0.7` | Sampling temperature (0-2) |
| `top_p` | `0.9` | Nucleus sampling threshold (0-1) |
| `stage1_threshold` | `0.10` | Draft→qualifier acceptance threshold |
| `stage2_threshold` | `0.03` | Qualifier→target acceptance threshold |
| `adaptive_enabled` | `True` | Enable adaptive threshold tuning |
| `device` | `auto` | Compute device (auto/cuda/mps/cpu) |
| `memory_headroom_gb` | `2.0` | Reserved memory headroom |

## REST API

Start the server:
```bash
momo-kibidango serve --port 7779
```

### Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/health` | Liveness check |
| GET | `/ready` | Readiness check |
| POST | `/infer` | Run inference |
| POST | `/batch` | Batch inference |
| GET | `/metrics` | Metrics summary |

### Example

```bash
curl -X POST http://localhost:7779/infer \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Hello world", "max_tokens": 64}'
```

## Performance

| Configuration | Speedup | Memory | Acceptance Rate |
|--------------|---------|--------|-----------------|
| Baseline (7B only) | 1.0x | ~7 GB | N/A |
| 2-Model (0.5B + 7B) | ~1.9x | ~10.8 GB | ~70% |
| 3-Model (0.5B + 2.7B + 7B) | ~2.0x | ~11.6 GB | ~80% |

### Benchmark results format

```json
{
  "prompt": "The future of AI is",
  "tokens_generated": 64,
  "tokens_per_second": 45.2,
  "acceptance_rate": 0.72,
  "elapsed_seconds": 1.42,
  "peak_memory_gb": 10.8,
  "mode": "2model"
}
```

## Graceful fallback

The system auto-degrades when resources are insufficient:

```
3-model (full pyramid)
  │ qualifier fails to load / OOM
  ▼
2-model (draft + target)
  │ draft fails to load / OOM
  ▼
1-model (target only)
  │ target fails to load
  ▼
Error with diagnostic message
```

## Testing

```bash
# Run all tests
pytest tests/

# Unit tests only
pytest tests/test_unit/

# Integration tests
pytest tests/test_integration/

# Performance benchmarks (slower)
pytest tests/test_performance/ -m slow

# With coverage
pytest tests/ --cov=src/momo_kibidango --cov-report=term-missing
```

### Test organization

| Directory | Tests | Description |
|-----------|-------|-------------|
| `tests/test_unit/` | 163 | Every module, class, and public method |
| `tests/test_integration/` | 30 | Full pipeline, API, fallback chains |
| `tests/test_performance/` | 16 | Benchmarks, throughput, memory bounds |
| `tests/test_acceptance/` | 18 | Quality validation, prompt variety |

## Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/my-feature`
3. Install dev dependencies: `pip install -e ".[dev]"`
4. Make changes and add tests
5. Run tests: `pytest tests/`
6. Submit a pull request

### Code standards

- Type hints on all functions
- Docstrings on all classes and public methods
- Run `ruff check src/` and `mypy src/` before submitting

## License

MIT License. See [LICENSE](LICENSE) for details.
