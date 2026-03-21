# MCP Integration Guide for Momo-Kibidango

This guide explains how to integrate momo-kibidango speculative decoding with LLM agents using the Model Context Protocol (MCP).

## What is MCP?

The **Model Context Protocol (MCP)** is an open standard for AI agents to discover and use tools. It allows Claude and other LLMs to:
- Discover available tools (run_inference, benchmark_models)
- Call tools with validated inputs
- Receive structured results for decision-making

## Quick Start

### 1. Install MCP Dependencies

```bash
# Install momo-kibidango with MCP support
pip install momo-kibidango[mcp]
```

### 2. Start MCP Server

```bash
# Start server in stdio mode (recommended for Claude integration)
momo-kibidango serve

# Or with custom log level
momo-kibidango serve --log-level DEBUG
```

The server starts in **stdio mode** by default, which is ideal for direct integration with Claude SDK.

### 3. Integrate with Claude

```python
from anthropic import Anthropic

client = Anthropic()

# Add momo-kibidango as MCP server
client.add_mcp_server({
    "name": "momo-kibidango",
    "command": "momo-kibidango serve",
})

# Use in conversation
response = client.messages.create(
    model="claude-opus-4-0",
    max_tokens=1024,
    tools=[
        {
            "type": "mcp",
            "mcp_name": "momo-kibidango",
        }
    ],
    messages=[
        {
            "role": "user",
            "content": "Run inference on 'Hello world' and tell me the performance metrics",
        }
    ],
)

print(response.content)
```

## Available Tools

### Tool 1: `run_inference`

Run speculative decoding inference on a prompt.

**Input Schema:**
```json
{
  "prompt": "string (required) - Input prompt for inference",
  "max_tokens": "integer (optional, default: 512) - Maximum tokens to generate (1-4096)",
  "temperature": "number (optional, default: 0.7) - Sampling temperature (0.0-2.0)",
  "draft_model": "string (optional) - Draft model name",
  "target_model": "string (optional) - Target model name"
}
```

**Output Example:**
```json
{
  "status": "success",
  "generated_text": "Hello! How can I help you today?",
  "tokens_generated": 12,
  "tokens_per_second": 45.3,
  "latency_ms": 265.4,
  "model_config": {
    "draft_model": "microsoft/phi-2",
    "target_model": "Qwen/Qwen2-7B",
    "speculative_decoding_enabled": true
  }
}
```

**Example Usage:**
```python
# Via Claude
messages=[
    {
        "role": "user",
        "content": "Generate 100 tokens for 'Write a haiku about AI' and report the tokens/second"
    }
]

# Claude will automatically call:
# {
#   "type": "use_tool",
#   "id": "toolu_...",
#   "name": "run_inference",
#   "input": {
#     "prompt": "Write a haiku about AI",
#     "max_tokens": 100
#   }
# }
```

### Tool 2: `benchmark_models`

Run comprehensive benchmark comparing draft and target models.

**Input Schema:**
```json
{
  "test_cases": "integer (optional, default: 10) - Number of test cases (1-100)",
  "output_format": "string (optional, default: json) - Output format: 'json' or 'csv'",
  "save_results": "boolean (optional, default: false) - Save results to file"
}
```

**Output Example (JSON):**
```json
{
  "status": "success",
  "test_cases": 10,
  "results": {
    "speedup": 2.34,
    "latency_ms": 145.2,
    "accuracy": 0.98,
    "total_tokens": 5240
  },
  "summary": {
    "avg_speedup": 2.34,
    "avg_latency_ms": 145.2,
    "total_tokens": 5240,
    "test_count": 10
  }
}
```

**Output Example (CSV):**
```csv
metric,value
test_cases,10
speedup,2.34
latency_ms,145.2
total_tokens,5240
```

## Architecture

### How MCP Works with Momo-Kibidango

```
┌─────────────────────────────────────────────────────────────┐
│                    Claude AI Agent                           │
│  (uses tools to accomplish tasks)                           │
└────────────────────┬────────────────────────────────────────┘
                     │ JSON-RPC (stdio)
                     ▼
┌─────────────────────────────────────────────────────────────┐
│              MCP Server Process                              │
│  (momo-kibidango serve)                                     │
│                                                              │
│  - list_tools() → returns [run_inference, benchmark_models] │
│  - call_tool(name, args) → executes tool                    │
└────────────────────┬────────────────────────────────────────┘
                     │
        ┌────────────┴────────────┐
        ▼                         ▼
  ┌──────────────┐    ┌────────────────────┐
  │ Decoder      │    │ Performance Monitor │
  │ (speculative │    │ (benchmarking)     │
  │  decoding)   │    │                    │
  └──────────────┘    └────────────────────┘
        │                       │
        ▼                       ▼
   [Models loaded from disk or remote]
```

### Execution Flow

1. **Claude sends request** → "Run inference on 'hello'"
2. **MCP Server receives call** → Validates schema, logs call
3. **Tool handler executes** → `_handle_run_inference()`
4. **Decoder loads** (lazy) → SpeculativeDecoder initialized
5. **Inference runs** → Calls `decoder.generate()`
6. **Results formatted** → Returns JSON with metrics
7. **Response sent back** → Claude receives and uses results

## Production Considerations

### Error Handling

The MCP server handles common errors gracefully:

```python
# Invalid input
{
  "status": "error",
  "error": "'max_tokens' must be a positive integer",
  "tool": "run_inference"
}

# Missing models
{
  "status": "error",
  "error": "Failed to load decoder: Models not found",
  "hint": "Download models with: python -m momo_kibidango.cli validate"
}

# Server issues
{
  "status": "error",
  "error": "Server error details...",
  "hint": "Check logs with: momo-kibidango serve --log-level DEBUG"
}
```

### Logging

Enable debug logging to see detailed MCP communication:

```bash
# Debug mode (verbose logging)
momo-kibidango serve --log-level DEBUG

# Watch logs in another terminal
tail -f ~/.momo-kibidango/logs/mcp_server.log
```

### Performance Notes

- **First request latency**: 2-5 seconds (model loading, LLM initialization)
- **Subsequent requests**: 50-500ms (cached models, inference latency)
- **Max tokens**: Limited to 4096 for safety (tunable)
- **Timeout**: No hard timeout (customize in config if needed)

### Security

- **Schema validation**: All inputs validated against JSON schema
- **Type checking**: Arguments type-checked before use
- **Resource limits**: Max tokens, max test cases bounded
- **Error isolation**: Tool errors don't crash server

## Advanced Configuration

### Custom Model Paths

Set custom model paths via environment variables:

```bash
export MOMO_DRAFT_MODEL="./models/custom-draft"
export MOMO_TARGET_MODEL="./models/custom-target"

momo-kibidango serve
```

### Integration with Other MCP Servers

Combine momo-kibidango with other MCP servers:

```python
client = Anthropic()

# Add multiple MCP servers
client.add_mcp_server({
    "name": "momo-kibidango",
    "command": "momo-kibidango serve",
})

client.add_mcp_server({
    "name": "web-search",
    "command": "web-search serve",
})

# Claude can now use tools from both servers
response = client.messages.create(
    model="claude-opus-4-0",
    tools=[
        {"type": "mcp", "mcp_name": "momo-kibidango"},
        {"type": "mcp", "mcp_name": "web-search"},
    ],
    messages=[...]
)
```

### Programmatic Tool Calls

Use Claude to make decisions based on benchmark results:

```python
response = client.messages.create(
    model="claude-opus-4-0",
    max_tokens=1024,
    tools=[
        {"type": "mcp", "mcp_name": "momo-kibidango"}
    ],
    messages=[
        {
            "role": "user",
            "content": (
                "Compare performance of speculative decoding. "
                "Run 20 benchmark test cases and provide analysis. "
                "Recommend if we should use speculative decoding in production."
            )
        }
    ],
)

# Claude will:
# 1. Call benchmark_models with test_cases=20
# 2. Receive results
# 3. Analyze speedup/latency
# 4. Provide recommendation
```

## Troubleshooting

### "MCP SDK not installed"

```bash
pip install momo-kibidango[mcp]
```

### "Models not found" error

Models are lazy-loaded on first tool call. Ensure they're accessible:

```bash
# Check if models exist
ls ~/.cache/huggingface/hub/

# Or manually download
python -c "from transformers import AutoModel; AutoModel.from_pretrained('Qwen/Qwen2-7B')"
```

### Server not responding

Check stdio connection:

```bash
# Test server manually
echo '{"jsonrpc": "2.0", "method": "tools/list", "id": 1}' | \
  python -m momo_kibidango.mcp_server --log-level DEBUG
```

### Slow inference

- First request is slow due to model loading (expected)
- Check system resources: `top`, `nvidia-smi`
- Verify GPU usage if available
- Consider batch processing for multiple requests

### Claude timeout

If Claude times out waiting for results:

1. Check MCP server logs: `--log-level DEBUG`
2. Verify model loading completes: Monitor during first request
3. Increase Claude timeout (language model dependent)
4. Consider lighter models for faster inference

## API Reference

### MCP Server Methods

**`list_tools()`**
- Returns list of available tools with schemas
- Called automatically when Claude connects
- No arguments

**`call_tool(name: str, arguments: dict)`**
- Executes named tool with provided arguments
- Validates schema before execution
- Returns JSON-formatted result

### Tool Return Codes

All tools return structured responses:

```python
# Success
{
  "status": "success",
  "data": { ... }
}

# Error
{
  "status": "error",
  "error": "Human-readable error message",
  "hint": "Suggested fix or debugging tip"
}
```

## Next Steps

1. **Integrate with Claude**: Use the quick start example above
2. **Customize prompts**: Build agent workflows that leverage inference
3. **Monitor performance**: Use `--log-level DEBUG` to trace calls
4. **Optimize models**: Experiment with different draft/target pairs
5. **Deploy**: Run in production with proper logging and monitoring

## Additional Resources

- [Model Context Protocol Spec](https://modelcontextprotocol.io/)
- [Anthropic SDK Docs](https://docs.anthropic.com/)
- [Momo-Kibidango README](../README.md)
- [Architecture Document](ARCHITECTURE.md)

---

*MCP Integration Guide | Version 1.0 | March 20, 2026*
