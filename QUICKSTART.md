# Quick Start Guide: momo-kibidango v1.0.0

Get up and running with speculative decoding in 5 minutes.

---

## Table of Contents

1. [Installation](#installation)
2. [First Inference](#first-inference)
3. [Benchmarking](#benchmarking)
4. [MCP/Claude Integration](#mcp-integration)
5. [Troubleshooting](#troubleshooting)

---

## Installation

### Option A: pip (Recommended)

```bash
# Basic install
pip install momo-kibidango

# With MCP support (for Claude integration)
pip install momo-kibidango[mcp]

# Verify
momo-kibidango --version
```

### Option B: Shell Script

```bash
bash install.sh
source ~/.bashrc  # Reload shell
momo-kibidango --version
```

### Option C: Docker

```bash
docker build -t momo-kibidango .
docker run -it momo-kibidango momo-kibidango --version
```

---

## First Inference

### Simple Text Generation

```bash
momo-kibidango infer \
  --prompt "Explain machine learning in one paragraph" \
  --model mistralai/Mistral-7B-Instruct-v0.1 \
  --draft-model mistralai/Mistral-7B-Instruct-v0.1 \
  --max-tokens 256 \
  --temperature 0.7
```

**What happens:**
1. Draft model generates candidate tokens (fast)
2. Verifier model validates them in parallel
3. Accepted tokens output immediately
4. Process repeats until max_tokens reached

**Expected output:**
```
Input: Explain machine learning in one paragraph
Generated: Machine learning is a subset of artificial intelligence 
that enables systems to learn and improve from experience without 
being explicitly programmed...

Statistics:
  Wall time: 14.2s
  Tokens: 56
  Throughput: 3.95 tokens/sec
  Speculative speedup: 2.0x
  Token acceptance rate: 94.9%
```

### Using from Python

```python
from momo_kibidango import SpeculativeDecoder

# Initialize
decoder = SpeculativeDecoder(
    model_name="mistralai/Mistral-7B-Instruct-v0.1",
    draft_model_name="mistralai/Mistral-7B-Instruct-v0.1",
)

# Generate
response = decoder.generate(
    prompt="Write a Python function for binary search",
    max_tokens=256,
    temperature=0.7
)

# Access results
print(f"Generated: {response.text}")
print(f"Speedup: {response.metrics['speedup']:.2f}x")
print(f"Tokens/sec: {response.metrics['throughput']:.2f}")
print(f"Acceptance rate: {response.metrics['acceptance_rate']:.1%}")
```

---

## Benchmarking

### Compare Single vs. Speculative Decoding

```bash
momo-kibidango benchmark \
  --model mistralai/Mistral-7B-Instruct-v0.1 \
  --scenarios basic,long,reasoning \
  --samples 3
```

### Custom Benchmark

```bash
momo-kibidango benchmark \
  --model meta-llama/Llama-2-7b-chat-hf \
  --draft-model mistralai/Mistral-7B-Instruct-v0.1 \
  --prompts custom_prompts.txt \
  --max-tokens 256 \
  --samples 5 \
  --output benchmark_results.json
```

### Benchmark Results Format

```
Running benchmark: basic
  Sample 1/3
    Single model: 7.2s (6.9 tok/s)
    Speculative:  3.6s (13.8 tok/s)
    Speedup: 2.0x ✅

  Sample 2/3
    Single model: 7.1s (7.0 tok/s)
    Speculative:  3.5s (14.0 tok/s)
    Speedup: 2.0x ✅

  Average speedup: 2.0x
  Std deviation: 0.01x

Running benchmark: long
  (Context: 4,096 tokens, Generate: 100 tokens)
  Average speedup: 1.9x
  Memory peak: 11.8GB

Running benchmark: reasoning
  (Chain-of-thought, 256 tokens)
  Average speedup: 2.0x
  Token acceptance: 94.8%

Overall: 1.96x average speedup ✅
```

### Interpreting Results

| Speedup | Interpretation |
|---------|-----------------|
| 2.0x+ | Excellent — draft model is well-calibrated |
| 1.5-2.0x | Good — normal speculative decoding |
| <1.5x | Check — draft model may be poorly chosen or CPU-bound |
| <1.0x | Fail — draft model slower than verification; see troubleshooting |

---

## MCP Integration

### Connect to Claude via MCP

#### Step 1: Install with MCP

```bash
pip install momo-kibidango[mcp]
```

#### Step 2: Start the MCP Server

```bash
mcp-server-momo-kibidango
```

Or in background:
```bash
mcp-server-momo-kibidango > /tmp/momo-kibidango.log 2>&1 &
```

#### Step 3: Configure Claude SDK

**Option A: Direct Server (stdio)**

```python
from anthropic import Anthropic
from mcp.client.stdio import StdioClientSession
from mcp.client.sse import SSEClientSession
import subprocess

# Start momo-kibidango MCP server
server_process = subprocess.Popen(
    ["mcp-server-momo-kibidango"],
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE
)

# Configure Claude client with MCP
client = Anthropic()

response = client.messages.create(
    model="claude-3-5-sonnet-20241022",
    max_tokens=1024,
    tools=[
        {
            "name": "run_inference",
            "description": "Run fast inference with speculative decoding",
            "input_schema": {
                "type": "object",
                "properties": {
                    "prompt": {"type": "string"},
                    "model": {"type": "string"},
                    "max_tokens": {"type": "integer", "default": 256}
                },
                "required": ["prompt"]
            }
        }
    ],
    messages=[{
        "role": "user",
        "content": "Use momo-kibidango to generate creative story ideas"
    }]
)

print(response.content)
```

**Option B: OpenClaw Integration**

```bash
# Register momo-kibidango as an OpenClaw skill
openclaw skill register momo-kibidango

# Use in OpenClaw
openclaw invoke momo-kibidango infer --prompt "Your prompt"
```

#### Step 4: Test the Connection

```bash
# Check server health
curl -s http://localhost:3000/health | jq .

# Call the API directly
curl -X POST http://localhost:3000/infer \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Hello, world!",
    "max_tokens": 50
  }' | jq .
```

### Example: Multi-turn Conversation with Claude

```python
from anthropic import Anthropic

client = Anthropic()

# Tools that Claude can use
tools = [
    {
        "name": "run_inference",
        "description": "Generate text using speculative decoding (2x faster)",
        "input_schema": {
            "type": "object",
            "properties": {
                "prompt": {"type": "string"},
                "max_tokens": {"type": "integer", "default": 256}
            },
            "required": ["prompt"]
        }
    }
]

# Multi-turn conversation
messages = [
    {
        "role": "user",
        "content": "Use momo-kibidango to generate 3 ideas for AI startup names"
    }
]

response = client.messages.create(
    model="claude-3-5-sonnet-20241022",
    max_tokens=2048,
    tools=tools,
    messages=messages
)

print(f"Claude: {response.content}")
```

---

## Troubleshooting

### Q: Installation fails with `externally-managed-environment`

**A:** Use a virtual environment:

```bash
python3 -m venv venv
source venv/bin/activate
pip install momo-kibidango
```

Or use the shell script installer:
```bash
bash install.sh
```

---

### Q: Import error: `No module named 'torch'`

**A:** PyTorch is large; install separately:

```bash
pip install torch
pip install momo-kibidango
```

Or reinstall with all dependencies:
```bash
pip install --upgrade --force-reinstall momo-kibidango
```

---

### Q: "No module named 'transformers'"

**A:** Missing Hugging Face transformers. Install:

```bash
pip install transformers
```

Or reinstall momo-kibidango:
```bash
pip uninstall momo-kibidango && pip install momo-kibidango
```

---

### Q: Speculative decoding slower than single model

**A:** Draft model is poorly calibrated. Try:

1. **Switch draft model:**
   ```bash
   # Use Phi-2 (excellent draft model)
   momo-kibidango infer \
     --model meta-llama/Llama-2-7b \
     --draft-model microsoft/phi-2 \
     --prompt "Your prompt"
   ```

2. **Reduce draft model size:**
   ```bash
   # Use smaller draft (3B instead of 7B)
   momo-kibidango infer \
     --model meta-llama/Llama-2-13b \
     --draft-model mistralai/Mistral-7B-Instruct-v0.1 \
     --prompt "Your prompt"
   ```

3. **Check hardware:**
   - Ensure GPU is being used: `nvidia-smi` (or `moto` for Apple Silicon)
   - Verify PyTorch was compiled with CUDA/Metal support

---

### Q: Out of memory (OOM) when loading both models

**A:** Use quantization to reduce memory:

```bash
momo-kibidango infer \
  --model meta-llama/Llama-2-13b \
  --draft-model mistralai/Mistral-7B \
  --quantize 4bit \
  --prompt "Your prompt"
```

Or use smaller models:
```bash
# Both 3B models = ~6GB
momo-kibidango infer \
  --model microsoft/phi-2 \
  --draft-model microsoft/phi-2 \
  --prompt "Your prompt"
```

---

### Q: MCP Server won't start

**A:** Check installation and logs:

```bash
# Verify MCP is installed
pip list | grep mcp

# If missing, install:
pip install momo-kibidango[mcp]

# Start with verbose logging
mcp-server-momo-kibidango --log-level DEBUG

# Check logs
tail -f /tmp/momo-kibidango.log
```

---

### Q: Slow initial load (2-3 minutes)

**A:** This is normal for first run. Models are downloaded and cached:

```bash
# First run (slow)
momo-kibidango infer --prompt "test" --model mistral-7b
# Downloads: ~30GB of models, compiles PyTorch kernels

# Subsequent runs (cached)
momo-kibidango infer --prompt "test" --model mistral-7b
# Uses cached models: ~30 seconds
```

To pre-download models:
```bash
momo-kibidango download --model mistralai/Mistral-7B-Instruct-v0.1
momo-kibidango download --model mistralai/Mistral-7B-Instruct-v0.1
```

---

### Q: Benchmarks show 1.0x speedup (no improvement)

**A:** Check token acceptance rate:

```bash
momo-kibidango infer --prompt "test" --verbose
```

Look for `token_acceptance_rate`. If <70%, draft model is mismatched.

**Solutions:**
1. Use better-calibrated draft model (Phi-2, Mistral-7B)
2. Reduce draft model size
3. Check if running on CPU (slow; use GPU)

---

### Q: How do I use custom models from Hugging Face?

**A:** Use any HF model ID:

```bash
momo-kibidango infer \
  --model "meta-llama/Llama-2-7b-hf" \
  --draft-model "mistralai/Mistral-7B-Instruct-v0.1" \
  --prompt "Your prompt"
```

Or with authentication:
```bash
huggingface-cli login  # Enter your Hugging Face token
momo-kibidango infer \
  --model "meta-llama/Llama-2-7b-hf" \
  --prompt "Your prompt"
```

---

### Q: How do I contribute?

**A:** See [CONTRIBUTING.md](CONTRIBUTING.md) for:
- Development setup
- Code style and testing
- PR process
- Issue triaging

Quick start for contributors:
```bash
git clone https://github.com/rdreilly58/momo-kibidango.git
cd momo-kibidango
pip install -e ".[dev,mcp]"
pytest  # Run tests
```

---

## Next Steps

1. **Run benchmarks** on your hardware
2. **Check documentation** in [docs/](docs/)
3. **Explore examples** in [examples/](examples/)
4. **Join discussions** at GitHub Issues
5. **Share your results** — we'd love to hear about your speedups!

---

**Questions?** Open an issue: https://github.com/rdreilly58/momo-kibidango/issues

**Ready to make inference 2x faster?**

```bash
momo-kibidango infer --prompt "Let's go!" --model mistral-7b
```
