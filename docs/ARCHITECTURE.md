# Momo-Kibidango Architecture

High-level system design and component relationships.

## System Overview

```
┌────────────────────────────────────────────────────────────────┐
│                   User Interfaces                               │
├────────────────────────────────────────────────────────────────┤
│                                                                 │
│  CLI Interface          Jupyter Notebook      Claude Agent      │
│  (momo-kibidango)       (Interactive)        (MCP Protocol)    │
│      │                       │                    │             │
└──────┼───────────────────────┼────────────────────┼─────────────┘
       │                       │                    │
       ▼                       ▼                    ▼
┌────────────────────────────────────────────────────────────────┐
│                 Core API & CLI Module                           │
├────────────────────────────────────────────────────────────────┤
│                                                                 │
│  momo_kibidango.cli          momo_kibidango.mcp_server         │
│  - Command routing           - Tool definition                 │
│  - Argument parsing          - Tool execution                  │
│  - Output formatting         - Error handling                  │
│                                                                 │
└──────┬───────────────────────────────────────────────────────┬─┘
       │                                                        │
       ▼                                                        ▼
┌────────────────────────────────┐      ┌────────────────────────┐
│ Inference Engine               │      │ Agent Integration      │
├────────────────────────────────┤      ├────────────────────────┤
│                                │      │                        │
│ SpeculativeDecoder            │      │ MCP Server Process     │
│ - Draft model inference       │      │ - Stdio communication  │
│ - Target model inference      │      │ - Tool dispatching     │
│ - Speculative decoding logic  │      │ - Result formatting    │
│ - Parallel token generation   │      │                        │
│                                │      └────────────────────────┘
└──────┬───────────────────────┬┘
       │                       │
       ▼                       ▼
┌────────────────────────────────────────────────────────────────┐
│              Performance & Monitoring Layer                     │
├────────────────────────────────────────────────────────────────┤
│                                                                 │
│  PerformanceMonitor          ProductionHardener               │
│  - Latency tracking          - Error recovery                 │
│  - Throughput measurement    - Graceful degradation           │
│  - Resource monitoring       - Safe shutdown                  │
│  - Benchmark utilities       - Configuration validation       │
│                                                                 │
└──────────────────────┬─────────────────────────────────────────┘
                       │
                       ▼
┌────────────────────────────────────────────────────────────────┐
│            Model & Runtime Layer                               │
├────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Draft Model (Fast)                Target Model (Accurate)    │
│  ┌─────────────────────┐          ┌──────────────────────┐    │
│  │ Phi-2 (2.7B)        │          │ Qwen2-7B             │    │
│  │ or Custom           │          │ or Custom            │    │
│  └─────────────────────┘          └──────────────────────┘    │
│           │                                │                   │
│           └────────────────┬────────────────┘                  │
│                            │                                   │
│                    Speculative Decoding                        │
│                    (Token-level acceleration)                 │
│                                                                 │
└────────────────────────────────────────────────────────────────┘
```

## Component Details

### 1. CLI Interface (`momo_kibidango/cli.py`)

**Responsibility:** Command-line entry point and argument routing.

**Key Functions:**
- `setup_parser()`: Define CLI arguments and subcommands
- `cmd_run()`: Execute inference command
- `cmd_benchmark()`: Run performance benchmarks
- `cmd_validate()`: Verify installation
- `cmd_serve()`: Start MCP server

**Commands:**
```bash
momo-kibidango run --prompt "..."      # Run inference
momo-kibidango benchmark               # Run benchmarks
momo-kibidango validate                # Validate installation
momo-kibidango serve                   # Start MCP server
```

### 2. Speculative Decoder (`momo_kibidango/speculative_2model.py`)

**Responsibility:** Core inference engine with speculative decoding.

**Key Classes:**
- `ModelConfig`: Configuration for draft/target models
- `SpeculativeDecoder`: Main inference orchestrator

**Key Methods:**
- `generate()`: Run inference with speculative decoding
- `_draft_generate()`: Fast draft model predictions
- `_target_verify()`: Target model verification
- `_accept_tokens()`: Accept/reject mechanism

**Speculative Decoding Flow:**
```
1. Draft model generates K candidate tokens (fast)
2. Target model evaluates each candidate
3. Accept tokens that match target distribution
4. Fallback to target if draft diverges
5. Return completed sequence (usually faster)
```

### 3. Performance Monitor (`momo_kibidango/monitoring.py`)

**Responsibility:** Track performance metrics and benchmarking.

**Key Classes:**
- `PerformanceMonitor`: Metrics collection and analysis

**Key Methods:**
- `run_benchmark()`: Execute benchmark suite
- `measure_latency()`: Measure inference latency
- `measure_throughput()`: Measure tokens/second
- `record_metrics()`: Store performance data

**Metrics Tracked:**
- Latency (ms)
- Throughput (tokens/second)
- Speedup ratio (speculative vs standard)
- Accuracy/acceptance rate
- Resource usage (CPU, memory, GPU if available)

### 4. Production Hardener (`momo_kibidango/production_hardening.py`)

**Responsibility:** Safety, error recovery, and graceful degradation.

**Key Classes:**
- `ProductionHardener`: Error recovery and safety mechanisms

**Key Methods:**
- `validate_config()`: Verify configuration
- `graceful_degrade()`: Fall back to standard inference if needed
- `safe_shutdown()`: Clean resource release
- `handle_error()`: Error recovery strategy

**Safety Features:**
- Schema validation for all inputs
- Graceful fallback to target-only inference
- Timeout handling
- Memory overflow protection
- Clear error messages for debugging

### 5. MCP Server (`momo_kibidango/mcp_server.py`)

**Responsibility:** Model Context Protocol integration for AI agents.

**Key Classes:**
- `MomoKibidangoMCPServer`: MCP server implementation

**Key Methods:**
- `list_tools()`: Report available tools to agents
- `call_tool()`: Execute tool with validated inputs
- `_handle_run_inference()`: Tool handler for inference
- `_handle_benchmark_models()`: Tool handler for benchmarks

**Tools Exposed:**
1. `run_inference`: Speculative decoding on prompt
2. `benchmark_models`: Performance benchmarking

**Communication Protocol:** JSON-RPC over stdio (default for Claude)

## Data Flow

### Inference Request Flow

```
User Request
    │
    ▼
CLI Parser (validate args)
    │
    ▼
SpeculativeDecoder
    ├─ Load draft model
    ├─ Load target model
    └─ Execute decoding:
        ├─ Draft: fast token prediction
        ├─ Target: verify tokens
        ├─ Accept/reject tokens
        └─ Repeat until completion
    │
    ▼
PerformanceMonitor (record metrics)
    │
    ▼
Format Result
    │
    ▼
Return to User (CLI/Agent/Notebook)
```

### Agent Integration Flow

```
Claude Agent
    │ "Run inference on 'hello'"
    ▼
MCP Server (stdio)
    │ Discover tools
    ▼
SpeculativeDecoder
    │ Execute inference
    ▼
Return results (JSON)
    │
    ▼
Claude (uses results for next step)
```

## Module Dependencies

```
momo_kibidango/
├── __init__.py                      (exports main classes)
├── cli.py                           (depends: speculative_2model, mcp_server)
├── speculative_2model.py            (depends: torch, transformers)
├── mcp_server.py                    (depends: mcp SDK, speculative_2model)
├── monitoring.py                    (depends: speculative_2model)
├── performance_optimization.py      (depends: torch, transformers)
├── production_hardening.py          (depends: pydantic)
└── openclaw_native.py              (optional: openclaw integration)

External Dependencies:
├── torch, torchvision, torchaudio  (PyTorch)
├── transformers                    (Hugging Face)
├── vllm                            (optional, for advanced inference)
├── pydantic                        (schema validation)
├── mcp                             (optional, for MCP server)
├── numpy, tqdm                     (utilities)
└── jupyter, ipykernel              (optional, for notebooks)
```

## State Management

### Decoder State

```python
class SpeculativeDecoder:
    def __init__(self):
        self.draft_model = None          # Lazy-loaded
        self.target_model = None         # Lazy-loaded
        self.tokenizer = None            # Shared tokenizer
        self.device = "cuda" if available else "cpu"
        
    # Models stay in memory across calls for performance
    # Cleared only on explicit shutdown or memory pressure
```

### MCP Server State

```python
class MomoKibidangoMCPServer:
    def __init__(self):
        self.decoder = None              # Lazy-loaded on first use
        self.monitor = None              # Lazy-loaded on first use
        self.hardener = None             # Lazy-loaded on first use
        
    # Each component loads only when needed
    # Server is stateless (requests don't affect each other)
```

## Configuration

### Config File Location

```
~/.momo-kibidango/config.yaml
```

### Example Config

```yaml
speculative_decoding:
  enabled: true
  
  models:
    draft:
      name: "microsoft/phi-2"
      local_path: "~/.cache/huggingface/phi-2"
      
    target:
      name: "Qwen/Qwen2-7B"
      local_path: "~/.cache/huggingface/qwen2-7b"
  
  inference:
    max_tokens: 512
    temperature: 0.7
    batch_size: 4

mcp_server:
  log_level: "INFO"
  stdio_mode: true
  
monitoring:
  track_metrics: true
  benchmark_suite: "standard"
```

## Error Handling Strategy

### Graceful Degradation

```
Speculative Decoding Attempt
    │
    ├─ Success? ──→ Return results with metrics
    │
    └─ Failure?
        │
        ├─ Can recover? ──→ Log warning, continue
        │
        └─ Cannot recover?
            │
            ├─ Schema invalid? ──→ Return error with hint
            │
            ├─ Models missing? ──→ Auto-download if enabled
            │                    ├─ Success? ──→ Continue
            │                    └─ Failure? ──→ Clear error message
            │
            └─ Other error? ──→ Log exception, return safe error
```

### Error Response Format

```json
{
  "status": "error",
  "error": "Human-readable error message",
  "hint": "Suggested fix or debugging tip",
  "code": "error_code",
  "context": {
    "model": "...",
    "operation": "...",
    "available_memory": "..."
  }
}
```

## Scaling Considerations

### Single Machine

- Current design: Single SpeculativeDecoder instance
- Models loaded once, shared across requests
- Suitable for: Development, small deployments, local agents

### Multiple Requests

- MCP server handles one request at a time (stdin mode)
- Decoder is thread-safe (PyTorch models have GIL)
- Future: Async queue for handling concurrent requests

### Distributed

- Future enhancement: Model server (vLLM, Ray)
- Remote decoder (HTTP/gRPC)
- Load balancing across multiple inference nodes

## Testing Strategy

### Unit Tests
- Schema validation
- Input/output formatting
- Error handling

### Integration Tests
- End-to-end CLI commands
- MCP tool execution
- Benchmark correctness

### Performance Tests
- Latency baselines
- Throughput measurement
- Speculative decoding speedup

## Deployment

### Local Development
```bash
momo-kibidango serve --log-level DEBUG
```

### Production
```bash
momo-kibidango serve --log-level INFO
# (with monitoring, logging aggregation, etc.)
```

### Agent Integration
```python
client.add_mcp_server({
    "name": "momo-kibidango",
    "command": "momo-kibidango serve",
})
```

## Future Enhancements

1. **Async Queuing**: Handle multiple concurrent requests
2. **Model Serving**: Integration with vLLM/Ray for distributed inference
3. **Web API**: HTTP REST API alongside MCP
4. **Monitoring Dashboard**: Real-time metrics visualization
5. **Custom Models**: Plugin system for draft/target models
6. **Caching**: Prompt/completion caching for repeated requests
7. **Multi-GPU**: Support for multi-GPU inference
8. **Quantization**: INT8/INT4 model support for efficiency

---

*Architecture Document | Version 1.0 | March 20, 2026*
