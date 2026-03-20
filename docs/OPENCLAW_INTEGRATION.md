# OpenClaw Integration Guide

This guide explains how to use the Speculative Decoding skill within the OpenClaw ecosystem.

## Table of Contents

1. [Overview](#overview)
2. [Installation](#installation)
3. [Command-Line Usage](#command-line-usage)
4. [API Reference](#api-reference)
5. [Feature Flags](#feature-flags)
6. [Integration Examples](#integration-examples)
7. [Fallback Behavior](#fallback-behavior)
8. [Performance Optimization](#performance-optimization)

---

## Overview

The Speculative Decoding skill provides fast text generation for OpenClaw by using multiple models in a hierarchical architecture:

- **Draft Model (0.5B)**: Ultra-fast token generation
- **Qualifier Model (2.7B)**: Medium-speed quality filtering
- **Target Model (7B)**: Final verification and quality assurance

The system automatically selects the best configuration based on available resources.

## Installation

### Via ClawHub (Recommended)

```bash
# Install the skill
clawhub install speculative-decoding

# Verify installation
openclaw-speculative --status
```

### Manual Installation

```bash
# Clone repository
cd ~/.openclaw/workspace/skills
git clone https://github.com/rdreilly58/momo-kibidango.git speculative-decoding
cd speculative-decoding

# Install dependencies
pip install -r requirements.txt

# Initialize
python -m src.openclaw_native --initialize
```

### Add to OpenClaw Configuration

Add to your OpenClaw session configuration:

```json
{
  "skills": {
    "speculative-decoding": {
      "enabled": true,
      "auto_load": true,
      "config": {
        "default_mode": "2model",
        "enable_monitoring": true
      }
    }
  }
}
```

---

## Command-Line Usage

### Basic Generation

```bash
# Simple generation
openclaw-speculative "What is machine learning?"

# With parameters
openclaw-speculative "Write a story about robots" \
  --max-length 200 \
  --temperature 0.8 \
  --mode 3model

# Streaming output
openclaw-speculative "Explain quantum physics" --stream
```

### Batch Processing

```bash
# Create prompts file
cat > prompts.txt << EOF
What is artificial intelligence?
How do neural networks work?
Explain deep learning
EOF

# Process batch
openclaw-speculative --batch prompts.txt --output results.json

# With specific mode
openclaw-speculative --batch prompts.txt --mode 2model --max-length 100
```

### Configuration Management

```bash
# Show current config
openclaw-speculative --show-config

# Update configuration
openclaw-speculative --config \
  enable_3model=true \
  default_mode=3model \
  max_batch_size=16

# Use custom config file
openclaw-speculative --config-file production.json "Test with custom config"
```

### System Operations

```bash
# Check system status
openclaw-speculative --status

# View metrics
openclaw-speculative --metrics

# Initialize models (usually automatic)
openclaw-speculative --initialize

# Clean shutdown
openclaw-speculative --shutdown
```

---

## API Reference

### REST API Endpoints

Start the API server:

```bash
python -m src.openclaw_integration_v2

# Or with custom port
PORT=8000 python -m src.openclaw_integration_v2
```

#### POST /infer

Generate text from a prompt.

**Request:**
```json
{
  "prompt": "Explain how computers work",
  "max_length": 100,
  "temperature": 0.7,
  "top_p": 0.9,
  "mode": "2model",
  "stream": false
}
```

**Response:**
```json
{
  "output": "Computers work by processing...",
  "tokens_generated": 87,
  "generation_time": 2.34,
  "mode": "2model",
  "acceptance_rates": {
    "stage1": 0.82,
    "combined": 0.76
  },
  "success": true,
  "api_version": "2.0"
}
```

**curl example:**
```bash
curl -X POST http://localhost:5000/infer \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your-api-key" \
  -d '{
    "prompt": "What is the meaning of life?",
    "max_length": 150
  }'
```

#### POST /batch

Process multiple prompts in a batch.

**Request:**
```json
{
  "prompts": [
    "First question?",
    "Second question?",
    "Third question?"
  ],
  "max_length": 50,
  "temperature": 0.7
}
```

**Response:**
```json
{
  "results": [
    {"output": "Answer 1...", "tokens_generated": 45, "success": true},
    {"output": "Answer 2...", "tokens_generated": 48, "success": true},
    {"output": "Answer 3...", "tokens_generated": 42, "success": true}
  ],
  "batch_size": 3,
  "api_version": "2.0"
}
```

#### POST /async

Submit asynchronous inference request.

**Request:**
```json
{
  "prompt": "Long generation task...",
  "max_length": 500,
  "webhook_url": "https://example.com/callback"
}
```

**Response:**
```json
{
  "request_id": "req_1234567890",
  "status": "queued",
  "queue_position": 3
}
```

#### GET /status

Get system status and statistics.

**Response:**
```json
{
  "initialized": true,
  "model_mode": "2model",
  "system_metrics": {
    "memory_gb": 10.5,
    "gpu_memory_gb": 8.2,
    "cpu_percent": 45.3
  },
  "inference_metrics": {
    "total_inferences": 1234,
    "successful_inferences": 1220,
    "failed_inferences": 14,
    "fallback_activations": 3
  },
  "uptime_seconds": 3600
}
```

#### GET /config

Get current configuration.

**Response:**
```json
{
  "default_mode": "2model",
  "enable_3model": true,
  "enable_2model": true,
  "enable_fallback": true,
  "max_batch_size": 8,
  "request_timeout": 300
}
```

#### POST /config

Update configuration dynamically.

**Request:**
```json
{
  "default_mode": "3model",
  "enable_monitoring": true,
  "max_batch_size": 16
}
```

#### GET /metrics

Get Prometheus-formatted metrics.

```bash
curl http://localhost:8080/metrics
```

#### GET /health

Health check endpoint.

**Response:**
```json
{
  "status": "healthy",
  "timestamp": "2024-03-15T10:30:45Z"
}
```

---

## Feature Flags

Control feature rollout and experiments via feature flags.

### Available Flags

| Flag | Default | Description |
|------|---------|-------------|
| `enable_3model` | false | Enable 3-model pyramid mode |
| `enable_2model` | true | Enable 2-model mode |
| `enable_batch_inference` | true | Allow batch processing |
| `enable_streaming` | false | Enable streaming responses |
| `enable_kv_cache_sharing` | true | Share KV-cache between requests |
| `enable_dynamic_thresholds` | false | Adjust acceptance thresholds dynamically |
| `log_performance_metrics` | true | Log detailed performance metrics |
| `enforce_rate_limits` | true | Enforce API rate limits |
| `validate_inputs` | true | Validate and sanitize inputs |

### Managing Feature Flags

**Via API:**
```bash
# Get all flags
curl http://localhost:5000/features

# Set specific flag
curl -X PUT http://localhost:5000/features/enable_3model \
  -H "Content-Type: application/json" \
  -d '{"value": true}'
```

**Via CLI:**
```bash
# Update flags
openclaw-speculative --config \
  enable_3model=true \
  enable_streaming=true
```

**Via Configuration File:**
```json
{
  "feature_flags": {
    "enable_3model": true,
    "enable_streaming": false,
    "enable_dynamic_thresholds": true
  }
}
```

---

## Integration Examples

### Python Integration

```python
from openclaw_native import OpenClawSpeculativeDecoding, OpenClawConfig

# Initialize
config = OpenClawConfig(
    default_mode="2model",
    enable_monitoring=True
)
skill = OpenClawSpeculativeDecoding(config)
skill.initialize()

# Generate text
result = skill.generate(
    "Explain the theory of relativity",
    max_length=200,
    temperature=0.7
)

print(f"Generated: {result['output']}")
print(f"Tokens/sec: {result['tokens_generated'] / result['generation_time']:.1f}")

# Batch generation
prompts = [
    "What is gravity?",
    "How do black holes form?",
    "What is dark matter?"
]

results = skill.batch_generate(prompts, max_length=100)
for i, result in enumerate(results):
    print(f"\nPrompt {i+1}: {result['output'][:50]}...")
```

### OpenClaw Session Integration

```python
# In your OpenClaw session handler
async def handle_generation_request(prompt: str, params: dict):
    # Use speculative decoding for long-form generation
    if params.get("max_length", 0) > 50:
        result = await run_skill(
            "speculative-decoding",
            "generate",
            prompt=prompt,
            **params
        )
        return result["output"]
    else:
        # Use standard model for short responses
        return await standard_generate(prompt, params)
```

### Webhook Integration

```python
from flask import Flask, request
import requests

app = Flask(__name__)

@app.route('/webhook/complete', methods=['POST'])
def generation_complete():
    data = request.json
    request_id = data['request_id']
    output = data['output']
    
    # Process completed generation
    store_result(request_id, output)
    notify_user(request_id, "Generation complete")
    
    return {"status": "acknowledged"}

# Submit async request with webhook
response = requests.post(
    "http://localhost:5000/async",
    json={
        "prompt": "Write a detailed essay about climate change",
        "max_length": 1000,
        "webhook_url": "https://myapp.com/webhook/complete"
    }
)
```

### Streaming Integration

```python
import requests
import json

def stream_generation(prompt: str):
    response = requests.post(
        "http://localhost:5000/infer",
        json={"prompt": prompt, "stream": true},
        stream=True
    )
    
    for line in response.iter_lines():
        if line:
            if line.startswith(b'data: '):
                data = json.loads(line[6:])
                
                if data['event'] == 'token':
                    print(data['content'], end='', flush=True)
                elif data['event'] == 'complete':
                    print(f"\n\nTokens: {data['tokens_generated']}")
                elif data['event'] == 'error':
                    print(f"\nError: {data['error']}")
```

---

## Fallback Behavior

The system implements automatic fallback to ensure reliability:

### Fallback Chain

```
3-Model Pyramid (11.6GB)
    ↓ (on OOM or failure)
2-Model Configuration (10.8GB)
    ↓ (on OOM or failure)
1-Model Baseline (7GB)
    ↓ (on failure)
Error Response
```

### Fallback Triggers

1. **Memory Exhaustion**
   - Detected when memory usage exceeds critical threshold
   - Automatically downgrades to lower memory configuration

2. **Model Loading Failure**
   - If qualifier model fails to load, falls back to 2-model
   - If draft model fails, falls back to 1-model

3. **Inference Timeout**
   - If inference exceeds timeout, falls back to simpler mode
   - Configurable via `request_timeout` parameter

4. **Error Rate Threshold**
   - If error rate exceeds 1%, considers fallback
   - Monitored over 5-minute windows

### Fallback Configuration

```json
{
  "fallback_config": {
    "enable_fallback": true,
    "memory_threshold_gb": 11.5,
    "timeout_seconds": 300,
    "error_rate_threshold": 0.01,
    "fallback_delay_ms": 100
  }
}
```

### Monitoring Fallbacks

```python
# Check fallback statistics
status = skill.get_status()
fallbacks = status["inference_metrics"]["fallback_activations"]
print(f"Fallback activations: {fallbacks}")

# Monitor via Prometheus
# Query: sum(speculative_decoding_fallback_total) by (from_mode, to_mode)
```

---

## Performance Optimization

### Request Batching

Combine multiple requests for better throughput:

```python
# Configure batching
config = OpenClawConfig(
    max_batch_size=16,
    batch_timeout_ms=100,
    optimize_for="throughput"  # or "latency"
)

# Batch requests automatically queued
results = skill.batch_generate(prompts, max_length=100)
```

### Caching Strategies

```json
{
  "cache_config": {
    "enable_model_cache": true,
    "model_cache_size": 3,
    "enable_prompt_cache": true,
    "prompt_cache_size": 100,
    "enable_kv_cache_sharing": true
  }
}
```

### Performance Tuning

```bash
# Find optimal settings for your hardware
python scripts/auto_tune.py --hardware auto --workload mixed

# Results saved to tuning_results.json
{
  "recommended_config": {
    "default_mode": "2model",
    "max_batch_size": 8,
    "temperature": 0.65,
    "acceptance_thresholds": {
      "stage1": 0.12,
      "stage2": 0.04
    }
  }
}
```

### Monitoring Performance

Key metrics to track:

1. **Throughput**: `speculative_decoding_throughput_tokens_per_second`
2. **Latency**: `speculative_decoding_latency_seconds`
3. **Acceptance Rate**: `speculative_decoding_acceptance_rate`
4. **Memory Usage**: `speculative_decoding_memory_usage_gb`

Example Grafana query for throughput:
```promql
rate(speculative_decoding_tokens_generated_total[5m]) 
/ 
rate(speculative_decoding_inference_total[5m])
```

---

## Best Practices

1. **Start Conservative**
   - Begin with 2-model mode in production
   - Enable 3-model after validating performance

2. **Monitor Continuously**
   - Set up alerts for latency and memory
   - Track acceptance rates to detect issues

3. **Tune for Your Workload**
   - Lower temperature for factual content (0.5-0.7)
   - Higher temperature for creative content (0.7-0.9)

4. **Handle Failures Gracefully**
   - Always check `success` field in responses
   - Implement client-side retries with backoff

5. **Optimize Batch Sizes**
   - Larger batches for throughput
   - Smaller batches for latency
   - Use dynamic batching for mixed workloads

---

## Troubleshooting

### Common Integration Issues

**Import Error: No module named 'openclaw_native'**
```bash
# Add to PYTHONPATH
export PYTHONPATH=$PYTHONPATH:~/.openclaw/workspace/skills/speculative-decoding/src

# Or install in development mode
cd ~/.openclaw/workspace/skills/speculative-decoding
pip install -e .
```

**Connection Refused on API calls**
```bash
# Check if service is running
ps aux | grep openclaw_integration

# Check port binding
netstat -tlnp | grep 5000

# Start manually
cd ~/.openclaw/workspace/skills/speculative-decoding
python -m src.openclaw_integration_v2
```

**Slow First Request**
```python
# Pre-warm models on startup
skill = OpenClawSpeculativeDecoding(config)
skill.initialize()

# Run warmup inference
skill.generate("Warmup prompt", max_length=10)
```

---

## Security Considerations

1. **API Authentication**
   - Always use API keys in production
   - Rotate keys regularly
   - Use HTTPS for all API calls

2. **Input Validation**
   - The system validates all inputs by default
   - Additional validation can be added via configuration

3. **Rate Limiting**
   - Default: 60 requests per minute
   - Configurable per API key or IP address

4. **Resource Limits**
   - Set appropriate memory limits
   - Configure request timeouts
   - Monitor for resource exhaustion

---

_For more details, see the [Production Deployment Guide](./PRODUCTION_DEPLOYMENT.md)_