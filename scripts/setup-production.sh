#!/bin/bash
# Production Setup Script for Speculative Decoding

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Configuration
PYTHON_VERSION="3.10"
VENV_PATH="venv"
CONFIG_PATH="$HOME/.openclaw/workspace/skills/speculative-decoding"

echo -e "${GREEN}=== Speculative Decoding Production Setup ===${NC}"

# Function to check command availability
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Function to print status
print_status() {
    echo -e "${YELLOW}[$(date +'%Y-%m-%d %H:%M:%S')] $1${NC}"
}

# Function to print error
print_error() {
    echo -e "${RED}[ERROR] $1${NC}"
}

# Function to print success
print_success() {
    echo -e "${GREEN}[SUCCESS] $1${NC}"
}

# Check system requirements
print_status "Checking system requirements..."

# Check Python version
if command_exists python3; then
    PYTHON_VERSION_INSTALLED=$(python3 --version | cut -d' ' -f2 | cut -d'.' -f1,2)
    print_success "Python $PYTHON_VERSION_INSTALLED found"
else
    print_error "Python 3 not found. Please install Python $PYTHON_VERSION+"
    exit 1
fi

# Check available memory
TOTAL_MEM_GB=$(free -g | awk '/^Mem:/{print $2}')
if [ "$TOTAL_MEM_GB" -lt 16 ]; then
    print_error "Insufficient memory: ${TOTAL_MEM_GB}GB found, 16GB+ required"
    exit 1
else
    print_success "Memory check passed: ${TOTAL_MEM_GB}GB available"
fi

# Check GPU (optional)
if command_exists nvidia-smi; then
    GPU_INFO=$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1)
    print_success "GPU detected: $GPU_INFO"
else
    print_status "No NVIDIA GPU detected, will use CPU inference"
fi

# Create virtual environment
print_status "Creating Python virtual environment..."
if [ -d "$VENV_PATH" ]; then
    print_status "Virtual environment already exists, skipping creation"
else
    python3 -m venv "$VENV_PATH"
    print_success "Virtual environment created"
fi

# Activate virtual environment
print_status "Activating virtual environment..."
source "$VENV_PATH/bin/activate"

# Upgrade pip
print_status "Upgrading pip..."
pip install --upgrade pip wheel setuptools

# Install dependencies
print_status "Installing dependencies..."
pip install -r requirements.txt

# Install additional production dependencies
print_status "Installing production dependencies..."
pip install prometheus-client psutil python-json-logger

# Create configuration directories
print_status "Creating configuration directories..."
mkdir -p "$CONFIG_PATH"/{config,cache,logs}

# Download models
print_status "Downloading models (this may take a while)..."
python scripts/download_models.py || {
    print_error "Model download failed"
    exit 1
}

# Create default configuration
print_status "Creating default configuration..."
cat > "$CONFIG_PATH/config.json" <<EOF
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
  "max_memory_gb": $TOTAL_MEM_GB,
  "warn_memory_gb": $(($TOTAL_MEM_GB * 80 / 100)),
  "critical_memory_gb": $(($TOTAL_MEM_GB * 90 / 100)),
  "log_level": "INFO"
}
EOF

# Create systemd service file
print_status "Creating systemd service..."
sudo tee /etc/systemd/system/speculative-decoding.service > /dev/null <<EOF
[Unit]
Description=Speculative Decoding Service
After=network.target

[Service]
Type=simple
User=$USER
WorkingDirectory=$(pwd)
Environment="PATH=$(pwd)/$VENV_PATH/bin:/usr/local/bin:/usr/bin:/bin"
ExecStart=$(pwd)/$VENV_PATH/bin/python -m src.openclaw_integration_v2
Restart=on-failure
RestartSec=10
StandardOutput=journal
StandardError=journal

# Resource limits
LimitNOFILE=65536
LimitNPROC=4096

# Security
NoNewPrivileges=true
PrivateTmp=true

[Install]
WantedBy=multi-user.target
EOF

# Enable service
print_status "Enabling systemd service..."
sudo systemctl daemon-reload
sudo systemctl enable speculative-decoding.service

# Setup log rotation
print_status "Setting up log rotation..."
sudo tee /etc/logrotate.d/speculative-decoding > /dev/null <<EOF
$CONFIG_PATH/logs/*.log {
    daily
    rotate 7
    compress
    delaycompress
    missingok
    notifempty
    create 0644 $USER $USER
}
EOF

# Run initial test
print_status "Running initial test..."
python -m src.openclaw_native "Test installation" --max-length 10 || {
    print_error "Initial test failed"
    exit 1
}

# Setup monitoring
print_status "Setting up monitoring..."
if command_exists prometheus; then
    print_success "Prometheus detected"
    
    # Add scrape config
    cat > prometheus-speculative.yml <<EOF
global:
  scrape_interval: 15s

scrape_configs:
  - job_name: 'speculative_decoding'
    static_configs:
      - targets: ['localhost:8080']
EOF
    print_success "Prometheus configuration created"
else
    print_status "Prometheus not found, skipping monitoring setup"
fi

# Create backup script
print_status "Creating backup script..."
cat > scripts/backup-production.sh <<'EOF'
#!/bin/bash
BACKUP_DIR="/backup/speculative-decoding"
DATE=$(date +%Y%m%d_%H%M%S)
CONFIG_PATH="$HOME/.openclaw/workspace/skills/speculative-decoding"

mkdir -p "$BACKUP_DIR"

# Backup configuration
tar -czf "$BACKUP_DIR/config-$DATE.tar.gz" -C "$CONFIG_PATH" config

# Backup logs (last 7 days)
find "$CONFIG_PATH/logs" -type f -mtime -7 -print0 | \
    tar -czf "$BACKUP_DIR/logs-$DATE.tar.gz" --null -T -

echo "Backup completed: $BACKUP_DIR/*-$DATE.tar.gz"
EOF
chmod +x scripts/backup-production.sh

# Final verification
print_status "Running final verification..."

# Check API endpoint
print_status "Starting service for verification..."
sudo systemctl start speculative-decoding.service
sleep 10

if curl -f http://localhost:5000/health >/dev/null 2>&1; then
    print_success "API health check passed"
else
    print_error "API health check failed"
    sudo systemctl status speculative-decoding.service
fi

# Stop service (will be started by user when ready)
sudo systemctl stop speculative-decoding.service

# Print summary
echo -e "\n${GREEN}=== Setup Complete ===${NC}"
echo -e "Configuration: $CONFIG_PATH/config.json"
echo -e "Logs: $CONFIG_PATH/logs/"
echo -e "Service: systemctl start speculative-decoding"
echo -e "\nTo start the service:"
echo -e "  ${YELLOW}sudo systemctl start speculative-decoding${NC}"
echo -e "\nTo check status:"
echo -e "  ${YELLOW}sudo systemctl status speculative-decoding${NC}"
echo -e "\nTo view logs:"
echo -e "  ${YELLOW}sudo journalctl -u speculative-decoding -f${NC}"
echo -e "\nTo access metrics:"
echo -e "  ${YELLOW}http://localhost:8080/metrics${NC}"

print_success "Production setup completed successfully!"