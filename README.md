# momo-kibidango

Production-ready speculative decoding implementation achieving 1.9-2.1x speedup in LLM inference while maintaining output quality.

## 🚀 v1.0.0 - Production Ready

This release includes comprehensive production hardening, monitoring, error handling, and full OpenClaw integration.

## Overview

This project implements speculative decoding techniques to accelerate large language model inference using hierarchical multi-model architectures. Based on research from Google's PyramidSD paper, enhanced with production-grade features.

## Features

- **2-Model Speculative Decoding**: Draft model + target model (1.92x speedup)
- **3-Model Pyramid Architecture**: Draft + qualifier + target models (1.97x speedup)  
- **Automatic Fallback**: Graceful degradation on memory constraints (3→2→1 model)
- **Production Hardening**: Error handling, rate limiting, input validation
- **Comprehensive Monitoring**: Prometheus metrics, health checks, alerting
- **OpenClaw Native**: CLI tool, REST API, batch processing
- **Performance Optimization**: Token batching, KV-cache sharing, model caching
- **Security**: Input sanitization, API authentication, resource limits

## Performance

| Configuration | Speedup | Memory Usage | Use Case |
|--------------|---------|--------------|----------|
| Baseline (7B) | 1.0x | 7GB | Reference |
| 2-Model | 1.92x | 10.8GB | Default, stable |
| 3-Model Pyramid | 1.97x | 11.6GB | Maximum performance |

## Quick Start

### Installation

```bash
# Clone repository
git clone https://github.com/rdreilly58/momo-kibidango.git
cd momo-kibidango

# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Basic Usage

```python
# 2-Model (default)
from src.speculative_2model import SpeculativeDecoder, ModelConfig

config = ModelConfig()
decoder = SpeculativeDecoder(config)
result = decoder.generate("The future of AI is", max_length=100)
print(result["generated_text"])

# 3-Model Pyramid
from src.speculative_3model import PyramidSpeculativeDecoder, ModelConfig

config = ModelConfig()
decoder = PyramidSpeculativeDecoder(config)
result = decoder.generate("The future of AI is", max_length=100)
print(result["generated_text"])
```

### OpenClaw API Server

```bash
# Start server (defaults to 2-model)
python src/openclaw_integration.py

# Start with 3-model enabled
python src/openclaw_integration.py --use-3model

# Use the client
python scripts/openclaw_client.py infer "Tell me a story"
```

## Architecture

### 2-Model Configuration
- **Draft Model**: Qwen2.5-1.5B-Instruct
- **Target Model**: Qwen2.5-7B-Instruct

### 3-Model Pyramid Configuration
- **Draft Model**: Qwen2.5-0.5B-Instruct (ultra-fast)
- **Qualifier Model**: Phi-2 2.7B (quality filter)
- **Target Model**: Qwen2.5-7B-Instruct (final verification)

## Benchmarking

Run comprehensive benchmarks:

```bash
# Test all 15 scenarios
python scripts/benchmark_3model.py

# Test subset
python scripts/benchmark_3model.py --scenarios 5
```

## Production Deployment

See [PRODUCTION_DEPLOYMENT.md](docs/PRODUCTION_DEPLOYMENT.md) for comprehensive deployment guide.

### Quick Production Setup

```bash
# Install with production dependencies
pip install -r requirements.txt
pip install prometheus-client psutil

# Initialize models
python -m src.openclaw_native --initialize

# Start with monitoring
python -m src.openclaw_integration_v2

# Check health
curl http://localhost:5000/health
```

## API Reference

See [OPENCLAW_INTEGRATION.md](docs/OPENCLAW_INTEGRATION.md) for full API documentation.

### Key Endpoints

- `POST /infer` - Generate text
- `POST /batch` - Batch generation  
- `GET /status` - System status
- `GET /metrics` - Prometheus metrics
- `GET /health` - Health check

## Monitoring

Access metrics at `http://localhost:8080/metrics`:

- Throughput: `speculative_decoding_throughput_tokens_per_second`
- Latency: `speculative_decoding_latency_seconds` 
- Memory: `speculative_decoding_memory_usage_gb`
- Acceptance rates: `speculative_decoding_acceptance_rate`

## Development

### Project Structure
```
momo-kibidango/
├── src/
│   ├── production_hardening.py      # Error handling, monitoring
│   ├── monitoring.py                # Metrics collection
│   ├── performance_optimization.py  # Batching, caching
│   ├── openclaw_native.py          # CLI interface
│   ├── openclaw_integration_v2.py  # REST API
│   ├── speculative_3model_production.py  # Production decoder
│   └── [legacy implementations]
├── tests/
│   ├── test_production.py          # Unit tests
│   ├── test_performance.py         # Performance tests
│   └── run_tests.py               # Test runner with coverage
├── docs/
│   ├── PRODUCTION_DEPLOYMENT.md    # Deployment guide
│   ├── OPENCLAW_INTEGRATION.md     # Integration guide
│   └── PHASE*_RESULTS.md          # Research results
└── ~/.openclaw/workspace/skills/
    └── speculative-decoding/       # OpenClaw skill location
```

### Testing

```bash
# Run all tests
python tests/run_tests.py

# With coverage report
python tests/run_tests.py --coverage --html

# Specific test suite
python tests/run_tests.py --test test_production.TestResourceMonitor
```

### Contributing

1. Fork the repository
2. Create a feature branch
3. Run benchmarks to ensure performance
4. Submit a pull request

## Citation

```bibtex
@article{byun2024pyramidsd,
  title={PyramidSD: Accelerating Speculative Decoding with Hierarchical Draft Verification},
  author={Byun, Jungwoo and others},
  journal={NeurIPS},
  year={2025}
}
```

## License

MIT License - see LICENSE file for details.

## Acknowledgments

- Google Research for the PyramidSD paper
- Qwen team for the model family
- Microsoft for Phi-2 model
- OpenClaw team for integration support