# Production Deployment Guide

This guide covers deploying the Speculative Decoding system in a production environment with proper monitoring, security, and performance optimization.

## Table of Contents

1. [System Requirements](#system-requirements)
2. [Installation & Setup](#installation--setup)
3. [Configuration Options](#configuration-options)
4. [Monitoring Setup](#monitoring-setup)
5. [Performance Tuning](#performance-tuning)
6. [Security Hardening](#security-hardening)
7. [Troubleshooting Guide](#troubleshooting-guide)
8. [Disaster Recovery](#disaster-recovery)

---

## System Requirements

### Hardware Requirements

**Minimum (1-model fallback only):**
- CPU: 8+ cores (x86_64 or ARM64)
- RAM: 16GB
- Storage: 50GB SSD
- GPU: Optional (CPU inference supported)

**Recommended (2-model deployment):**
- CPU: 16+ cores
- RAM: 32GB
- Storage: 100GB NVMe SSD
- GPU: NVIDIA GPU with 12GB+ VRAM (or Apple Silicon with 16GB+ unified memory)

**Optimal (3-model pyramid):**
- CPU: 24+ cores
- RAM: 64GB
- Storage: 200GB NVMe SSD
- GPU: NVIDIA A10G/A100 with 24GB+ VRAM (or M2/M3 Max with 32GB+ unified memory)

### Software Requirements

- **OS:** Ubuntu 20.04+ / macOS 12+ / RHEL 8+
- **Python:** 3.9 - 3.11 (3.10 recommended)
- **CUDA:** 11.8+ (for NVIDIA GPUs)
- **Docker:** 20.10+ (for containerized deployment)

### Model Memory Requirements

| Configuration | System RAM | GPU VRAM | Total Memory |
|---------------|------------|----------|--------------|
| 1-model | 8GB | 4-6GB | 12-14GB |
| 2-model | 12GB | 8-10GB | 20-22GB |
| 3-model pyramid | 16GB | 10-12GB | 26-28GB |

---

## Installation & Setup

### Step 1: System Preparation

```bash
# Update system packages
sudo apt-get update && sudo apt-get upgrade -y

# Install system dependencies
sudo apt-get install -y \
    build-essential \
    git \
    wget \
    curl \
    libssl-dev \
    libffi-dev \
    python3-dev \
    python3-pip \
    python3-venv

# For NVIDIA GPUs
sudo apt-get install -y nvidia-driver-525 nvidia-cuda-toolkit

# Verify GPU (if applicable)
nvidia-smi
```

### Step 2: Clone Repository

```bash
git clone https://github.com/rdreilly58/momo-kibidango.git
cd momo-kibidango
```

### Step 3: Python Environment Setup

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip wheel setuptools

# Install dependencies
pip install -r requirements.txt

# For production monitoring
pip install prometheus-client psutil
```

### Step 4: Model Download

Models will be automatically downloaded on first use, but for production, pre-download:

```bash
python scripts/download_models.py
```

Or manually:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

# Download models
models = [
    "Qwen/Qwen2.5-0.5B-Instruct",    # Draft
    "microsoft/phi-2",                 # Qualifier
    "Qwen/Qwen2.5-7B-Instruct"       # Target
]

for model_id in models:
    print(f"Downloading {model_id}...")
    AutoModelForCausalLM.from_pretrained(model_id)
    AutoTokenizer.from_pretrained(model_id)
```

### Step 5: Initial Configuration

```bash
# Create configuration directory
mkdir -p ~/.openclaw/workspace/skills/speculative-decoding

# Copy default configuration
cp configs/production.json ~/.openclaw/workspace/skills/speculative-decoding/config.json

# Edit configuration
nano ~/.openclaw/workspace/skills/speculative-decoding/config.json
```

### Step 6: Verify Installation

```bash
# Run system check
python -m src.openclaw_native --status

# Run test inference
python -m src.openclaw_native "Test prompt" --max-length 50
```

---

## Configuration Options

### Main Configuration File

Location: `~/.openclaw/workspace/skills/speculative-decoding/config.json`

```json
{
  "default_mode": "2model",
  "enable_3model": true,
  "enable_2model": true,
  "enable_fallback": true,
  "max_batch_size": 8,
  "request_timeout": 300,
  "enable_monitoring": true,
  "monitoring_port": 8080,
  "metrics_export_interval": 60,
  "rate_limit_per_minute": 60,
  "max_memory_gb": 12.0,
  "warn_memory_gb": 10.0,
  "critical_memory_gb": 11.5,
  "log_level": "INFO",
  "cache_path": "~/.openclaw/workspace/skills/speculative-decoding/cache",
  "log_path": "~/.openclaw/workspace/skills/speculative-decoding/logs"
}
```

### Environment Variables

```bash
# Model selection
export SPECULATIVE_DEFAULT_MODE=2model
export SPECULATIVE_ENABLE_3MODEL=true

# Performance
export SPECULATIVE_BATCH_SIZE=8
export SPECULATIVE_TIMEOUT=300

# Monitoring
export SPECULATIVE_MONITORING_PORT=8080
export SPECULATIVE_METRICS_ENABLED=true

# Paths
export SPECULATIVE_CONFIG=/path/to/config.json
export SPECULATIVE_CACHE_DIR=/path/to/cache
export SPECULATIVE_LOG_DIR=/path/to/logs

# Logging
export LOGLEVEL=INFO
export PYTHONUNBUFFERED=1
```

### Feature Flags

Control feature rollout via `feature_flags.json`:

```json
{
  "enable_3model": false,
  "enable_2model": true,
  "enable_batch_inference": true,
  "enable_streaming": false,
  "enable_kv_cache_sharing": true,
  "enable_dynamic_thresholds": false,
  "log_performance_metrics": true,
  "enforce_rate_limits": true,
  "validate_inputs": true
}
```

---

## Monitoring Setup

### Prometheus Integration

1. **Install Prometheus:**

```bash
# Download Prometheus
wget https://github.com/prometheus/prometheus/releases/download/v2.45.0/prometheus-2.45.0.linux-amd64.tar.gz
tar xvf prometheus-2.45.0.linux-amd64.tar.gz
cd prometheus-2.45.0.linux-amd64/

# Configure Prometheus
cat > prometheus.yml << EOF
global:
  scrape_interval: 15s
  evaluation_interval: 15s

scrape_configs:
  - job_name: 'speculative_decoding'
    static_configs:
      - targets: ['localhost:8080']
EOF

# Start Prometheus
./prometheus --config.file=prometheus.yml
```

2. **Key Metrics to Monitor:**

```yaml
# Prometheus alerts configuration (alerts.yml)
groups:
  - name: speculative_decoding
    rules:
      - alert: HighLatency
        expr: speculative_decoding_latency_seconds{quantile="0.95"} > 5
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "High inference latency detected"
          
      - alert: LowAcceptanceRate
        expr: speculative_decoding_acceptance_rate < 0.6
        for: 10m
        labels:
          severity: warning
        annotations:
          summary: "Low token acceptance rate"
          
      - alert: HighMemoryUsage
        expr: speculative_decoding_memory_usage_gb > 11.5
        for: 2m
        labels:
          severity: critical
        annotations:
          summary: "Critical memory usage"
          
      - alert: HighErrorRate
        expr: rate(speculative_decoding_errors_total[5m]) > 0.01
        for: 5m
        labels:
          severity: critical
        annotations:
          summary: "High error rate detected"
```

### Grafana Dashboard

Import the dashboard from `configs/grafana-dashboard.json` or create with these panels:

1. **Throughput Panel:**
   - Query: `rate(speculative_decoding_tokens_generated_total[5m])`
   - Visualization: Graph

2. **Latency Panel:**
   - Query: `histogram_quantile(0.95, speculative_decoding_latency_seconds_bucket)`
   - Visualization: Graph with P50, P95, P99

3. **Memory Usage Panel:**
   - Query: `speculative_decoding_memory_usage_gb`
   - Visualization: Graph with threshold lines

4. **Acceptance Rate Panel:**
   - Query: `speculative_decoding_acceptance_rate`
   - Visualization: Gauge

### Logging Configuration

```python
# logging_config.py
import logging.config

LOGGING_CONFIG = {
    'version': 1,
    'disable_existing_loggers': False,
    'formatters': {
        'json': {
            'class': 'pythonjsonlogger.jsonlogger.JsonFormatter',
            'format': '%(asctime)s %(name)s %(levelname)s %(message)s'
        }
    },
    'handlers': {
        'console': {
            'class': 'logging.StreamHandler',
            'level': 'INFO',
            'formatter': 'json',
            'stream': 'ext://sys.stdout'
        },
        'file': {
            'class': 'logging.handlers.RotatingFileHandler',
            'level': 'DEBUG',
            'formatter': 'json',
            'filename': 'logs/speculative_decoding.log',
            'maxBytes': 104857600,  # 100MB
            'backupCount': 10
        }
    },
    'loggers': {
        '': {
            'level': 'INFO',
            'handlers': ['console', 'file']
        }
    }
}

logging.config.dictConfig(LOGGING_CONFIG)
```

---

## Performance Tuning

### 1. Model Configuration Optimization

```python
# Optimal settings by hardware
HARDWARE_CONFIGS = {
    "nvidia_a10g": {
        "default_mode": "3model",
        "max_batch_size": 16,
        "use_flash_attention": True,
        "torch_dtype": "float16"
    },
    "apple_m2_max": {
        "default_mode": "2model",
        "max_batch_size": 8,
        "device": "mps",
        "torch_dtype": "float16"
    },
    "cpu_only": {
        "default_mode": "1model",
        "max_batch_size": 2,
        "device": "cpu",
        "torch_dtype": "float32"
    }
}
```

### 2. Batch Size Optimization

```bash
# Find optimal batch size
python scripts/benchmark_batch_sizes.py --min 1 --max 32 --step 2

# Results will show throughput vs latency tradeoff
```

### 3. Memory Optimization

```python
# Enable memory efficient attention
os.environ["PYTORCH_CUDA_MEMORY_FRACTION"] = "0.9"

# Use gradient checkpointing for large models
model.gradient_checkpointing_enable()

# Clear cache periodically
if iteration % 100 == 0:
    torch.cuda.empty_cache()
```

### 4. KV-Cache Tuning

```json
{
  "kv_cache_config": {
    "enable_sharing": true,
    "max_cache_tokens": 2048,
    "cache_dtype": "float16",
    "compression_ratio": 0.5
  }
}
```

### 5. Quantization Settings

```python
# Dynamic quantization selection
from performance_optimization import QuantizationOptimizer

available_memory = get_available_memory_gb()
model_size = "large"  # Based on parameter count

quant_config = QuantizationOptimizer.get_optimal_quantization(
    model_size, available_memory
)
```

---

## Security Hardening

### 1. Input Validation

```python
# Configure strict input validation
INPUT_VALIDATION = {
    "max_prompt_length": 4096,
    "forbidden_patterns": [
        r"(?i)ignore.*previous.*instructions",
        r"(?i)system.*prompt",
        r"<script[^>]*>",
        r"javascript:",
    ],
    "rate_limit_per_ip": 60,
    "rate_limit_window": 60  # seconds
}
```

### 2. API Authentication

```python
# Add API key authentication
from functools import wraps
from flask import request, abort

def require_api_key(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        api_key = request.headers.get('X-API-Key')
        if not api_key or not validate_api_key(api_key):
            abort(401)
        return f(*args, **kwargs)
    return decorated_function

@app.route('/infer', methods=['POST'])
@require_api_key
def infer():
    # ... inference logic
```

### 3. Network Security

```nginx
# Nginx configuration for reverse proxy
server {
    listen 443 ssl http2;
    server_name api.example.com;
    
    ssl_certificate /etc/ssl/certs/cert.pem;
    ssl_certificate_key /etc/ssl/private/key.pem;
    
    # Rate limiting
    limit_req_zone $binary_remote_addr zone=api:10m rate=10r/s;
    limit_req zone=api burst=20 nodelay;
    
    location / {
        proxy_pass http://localhost:5000;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        
        # Timeouts
        proxy_connect_timeout 300s;
        proxy_send_timeout 300s;
        proxy_read_timeout 300s;
    }
}
```

### 4. Container Security

```dockerfile
# Dockerfile with security best practices
FROM python:3.10-slim

# Non-root user
RUN useradd -m -u 1000 speculative
USER speculative

# Copy only necessary files
COPY --chown=speculative:speculative requirements.txt .
RUN pip install --user -r requirements.txt

COPY --chown=speculative:speculative src/ ./src/

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s \
  CMD curl -f http://localhost:5000/health || exit 1

EXPOSE 5000
CMD ["python", "-m", "src.openclaw_integration_v2"]
```

---

## Troubleshooting Guide

### Common Issues and Solutions

#### 1. Out of Memory (OOM) Errors

**Symptoms:**
- `RuntimeError: CUDA out of memory`
- `ResourceExhaustedError`
- System becomes unresponsive

**Solutions:**
```bash
# Check current memory usage
nvidia-smi  # For GPU
free -h     # For system RAM

# Reduce memory usage
export SPECULATIVE_DEFAULT_MODE=2model  # Use 2-model instead of 3
export PYTORCH_CUDA_MEMORY_FRACTION=0.8  # Limit GPU memory usage

# Clear cache
python -c "import torch; torch.cuda.empty_cache()"
```

#### 2. Low Acceptance Rates

**Symptoms:**
- Acceptance rate <60%
- Slow generation despite good hardware

**Solutions:**
```python
# Adjust temperature
config["temperature"] = 0.6  # Lower from 0.7

# Tune acceptance thresholds
config["stage1_threshold"] = 0.15  # Increase from 0.10
config["stage2_threshold"] = 0.05  # Increase from 0.03

# Use different draft model
config["draft_model_id"] = "Qwen/Qwen2.5-1.5B-Instruct"  # Larger draft
```

#### 3. High Latency

**Symptoms:**
- P95 latency >5 seconds
- First inference very slow

**Solutions:**
```bash
# Pre-warm models
python scripts/warmup_models.py

# Enable model caching
export SPECULATIVE_MODEL_CACHE_SIZE=3

# Check for CPU throttling
sudo turbostat --quiet --show Busy%,Bzy_MHz,PkgWatt --interval 5
```

#### 4. API Timeout Errors

**Symptoms:**
- 504 Gateway Timeout
- Connection reset errors

**Solutions:**
```python
# Increase timeouts
config["request_timeout"] = 600  # 10 minutes

# Enable request queuing
config["enable_async_processing"] = True
config["max_queue_size"] = 100

# Use smaller max_length
params["max_length"] = 500  # Instead of 1000+
```

#### 5. Model Loading Failures

**Symptoms:**
- `OSError: Can't load model`
- `HTTPError` during download

**Solutions:**
```bash
# Clear cache and re-download
rm -rf ~/.cache/huggingface/transformers/
python scripts/download_models.py

# Use offline mode
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

# Specify cache directory
export TRANSFORMERS_CACHE=/path/to/models
```

### Debug Commands

```bash
# Check system status
python -m src.openclaw_native --status | jq .

# View recent errors
tail -f ~/.openclaw/workspace/skills/speculative-decoding/logs/error.log | jq .

# Monitor resource usage
htop  # Interactive process viewer
iotop  # I/O monitoring
nvidia-smi dmon  # GPU monitoring

# Test specific model configuration
python -m src.openclaw_native \
  --mode 2model \
  --max-length 10 \
  "Test prompt" \
  --config log_level=DEBUG

# Benchmark performance
python scripts/benchmark_production.py \
  --iterations 100 \
  --mode all \
  --output results.json
```

### Performance Profiling

```python
# Enable profiling
import cProfile
import pstats

profiler = cProfile.Profile()
profiler.enable()

# Run inference
result = decoder.generate(prompt)

profiler.disable()
stats = pstats.Stats(profiler)
stats.sort_stats('cumulative')
stats.print_stats(20)  # Top 20 functions
```

---

## Disaster Recovery

### Backup Strategy

```bash
#!/bin/bash
# backup.sh - Daily backup script

BACKUP_DIR="/backup/speculative-decoding"
DATE=$(date +%Y%m%d)

# Backup configuration
cp -r ~/.openclaw/workspace/skills/speculative-decoding/config* $BACKUP_DIR/config-$DATE/

# Backup logs (last 7 days)
find ~/.openclaw/workspace/skills/speculative-decoding/logs \
  -type f -mtime -7 -exec cp {} $BACKUP_DIR/logs-$DATE/ \;

# Backup metrics database
pg_dump metrics_db > $BACKUP_DIR/metrics-$DATE.sql

# Compress
tar -czf $BACKUP_DIR/backup-$DATE.tar.gz $BACKUP_DIR/*-$DATE

# Cleanup old backups (keep 30 days)
find $BACKUP_DIR -name "backup-*.tar.gz" -mtime +30 -delete
```

### Recovery Procedures

#### 1. Service Recovery

```bash
# Quick restart
systemctl restart speculative-decoding

# Full recovery
systemctl stop speculative-decoding
cd /opt/speculative-decoding

# Restore from backup
tar -xzf /backup/backup-20240315.tar.gz
cp -r backup/config/* ~/.openclaw/workspace/skills/speculative-decoding/

# Clear cache
rm -rf cache/*
rm -rf /tmp/speculative-*

# Restart
systemctl start speculative-decoding

# Verify
curl http://localhost:5000/health
```

#### 2. Model Recovery

```bash
# Re-download models
python scripts/download_models.py --force

# Verify models
python scripts/verify_models.py

# Run test inference
python -m src.openclaw_native "Test recovery" --max-length 10
```

#### 3. Data Recovery

```sql
-- Restore metrics database
psql -U postgres -c "DROP DATABASE IF EXISTS metrics_db;"
psql -U postgres -c "CREATE DATABASE metrics_db;"
psql -U postgres metrics_db < /backup/metrics-20240315.sql

-- Verify
psql -U postgres metrics_db -c "SELECT COUNT(*) FROM inference_metrics;"
```

### Monitoring Recovery

```yaml
# prometheus-recovery.yml
- alert: ServiceDown
  expr: up{job="speculative_decoding"} == 0
  for: 1m
  labels:
    severity: critical
  annotations:
    summary: "Speculative Decoding service is down"
    recovery: "Run: systemctl restart speculative-decoding"

- alert: ServiceRecovered
  expr: up{job="speculative_decoding"} == 1 and 
        increase(up{job="speculative_decoding"}[5m]) > 0
  labels:
    severity: info
  annotations:
    summary: "Service recovered after downtime"
```

---

## Production Checklist

Before deploying to production, ensure:

- [ ] System requirements met (RAM, GPU, storage)
- [ ] All models downloaded and cached
- [ ] Configuration reviewed and optimized
- [ ] Monitoring endpoints accessible
- [ ] Prometheus scraping metrics
- [ ] Alerts configured and tested
- [ ] API authentication enabled
- [ ] Rate limiting configured
- [ ] SSL/TLS certificates installed
- [ ] Backup script scheduled
- [ ] Recovery procedures documented
- [ ] Load testing completed
- [ ] Security scan passed
- [ ] Documentation updated
- [ ] Team trained on operations

---

## Support

For production support:
- GitHub Issues: https://github.com/rdreilly58/momo-kibidango/issues
- Documentation: https://github.com/rdreilly58/momo-kibidango/wiki
- Security: security@example.com

---

_Last updated: March 2024_