# momo-kibidango

Fast speculative decoding implementation with 2-model and 3-model pyramid architectures for accelerated LLM inference.

## Overview

This project implements speculative decoding techniques to achieve 1.9-2.1x speedup in large language model inference while maintaining output quality. Based on research from Google's PyramidSD paper.

## Features

- **2-Model Speculative Decoding**: Draft model + target model (1.92x speedup)
- **3-Model Pyramid Architecture**: Draft + qualifier + target models (1.97x speedup)
- **Automatic Fallback**: Graceful degradation on memory constraints
- **OpenClaw Integration**: REST API for easy deployment
- **Comprehensive Benchmarking**: 15 scenarios across different task types

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

## API Reference

### POST /infer
Generate text using speculative decoding.

```json
{
  "prompt": "Your prompt here",
  "max_length": 100
}
```

### GET /config
Get or update configuration.

```json
{
  "use_3model": false,
  "auto_fallback": true,
  "max_memory_gb": 12.0
}
```

### GET /metrics
View inference performance metrics.

## Development

### Project Structure
```
momo-kibidango/
├── src/
│   ├── speculative_2model.py    # 2-model implementation
│   ├── speculative_3model.py    # 3-model pyramid
│   └── openclaw_integration.py  # API server
├── scripts/
│   ├── benchmark_3model.py      # Benchmark suite
│   └── openclaw_client.py       # CLI client
├── docs/
│   ├── PHASE2_RESULTS.md        # 2-model results
│   └── PHASE3_RESULTS.md        # 3-model results
└── results/                      # Benchmark outputs
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