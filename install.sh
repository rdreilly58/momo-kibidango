#!/bin/bash
set -euo pipefail

# ============================================================================
# Momo-Kibidango Installation Script
# ============================================================================
# Purpose: One-line install for momo-kibidango framework
# Version: 1.0
# Date: March 20, 2026
# ============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_PATH="${HOME}/.momo-kibidango/venv"
CONFIG_DIR="${HOME}/.momo-kibidango/config"
MODELS_DIR="${HOME}/.momo-kibidango/models"
LOG_FILE="${SCRIPT_DIR}/install.log"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# ============================================================================
# Utility Functions
# ============================================================================

log() {
    echo -e "${BLUE}[$(date '+%Y-%m-%d %H:%M:%S')]${NC} $1" | tee -a "$LOG_FILE"
}

success() {
    echo -e "${GREEN}✓ $1${NC}" | tee -a "$LOG_FILE"
}

warning() {
    echo -e "${YELLOW}⚠ $1${NC}" | tee -a "$LOG_FILE"
}

error() {
    echo -e "${RED}❌ $1${NC}" | tee -a "$LOG_FILE"
    exit 1
}

# ============================================================================
# Main Installation
# ============================================================================

main() {
    echo -e "${GREEN}"
    echo "╔════════════════════════════════════════════════════════╗"
    echo "║   🍑 Momo-Kibidango Installation (Phase 1: Script)     ║"
    echo "║   Speculative Decoding Framework for Apple Silicon     ║"
    echo "╚════════════════════════════════════════════════════════╝"
    echo -e "${NC}"
    echo ""
    
    log "Installation started"
    
    # Pre-flight checks
    check_python_version
    check_disk_space
    check_git
    
    # Create environment
    create_venv
    
    # Install dependencies
    install_dependencies
    
    # Create directories
    create_directories
    
    # Create configuration
    create_configuration
    
    # Run validation
    run_validation
    
    # Success message
    print_completion
}

# ============================================================================
# Pre-flight Checks
# ============================================================================

check_python_version() {
    log "Checking Python version..."
    
    if ! command -v python3 &> /dev/null; then
        error "Python 3 is not installed"
    fi
    
    PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
    MAJOR=$(echo "$PYTHON_VERSION" | cut -d. -f1)
    MINOR=$(echo "$PYTHON_VERSION" | cut -d. -f2)
    
    if [ "$MAJOR" -lt 3 ] || ([ "$MAJOR" -eq 3 ] && [ "$MINOR" -lt 10 ]); then
        error "Python 3.10+ required (found $PYTHON_VERSION)"
    fi
    
    success "Python $PYTHON_VERSION"
}

check_disk_space() {
    log "Checking disk space..."
    
    # Check available space in home directory (need ~20GB for models)
    AVAILABLE_SPACE=$(df "$HOME" | tail -1 | awk '{print $4}')
    REQUIRED_SPACE=$((20 * 1024 * 1024))  # 20GB in KB
    
    if [ "$AVAILABLE_SPACE" -lt "$REQUIRED_SPACE" ]; then
        AVAILABLE_GB=$((AVAILABLE_SPACE / 1024 / 1024))
        warning "Low disk space (${AVAILABLE_GB}GB available, 20GB recommended)"
        echo "Install will continue with reduced model set."
    else
        AVAILABLE_GB=$((AVAILABLE_SPACE / 1024 / 1024))
        success "Disk space OK (${AVAILABLE_GB}GB available)"
    fi
}

check_git() {
    log "Checking git..."
    
    if ! command -v git &> /dev/null; then
        warning "git not found (required for some features)"
    else
        GIT_VERSION=$(git --version | awk '{print $3}')
        success "git $GIT_VERSION"
    fi
}

# ============================================================================
# Environment Setup
# ============================================================================

create_venv() {
    log "Creating virtual environment at $VENV_PATH..."
    
    if [ -d "$VENV_PATH" ]; then
        warning "Virtual environment already exists"
        echo "Remove with: rm -rf $VENV_PATH"
    else
        python3 -m venv "$VENV_PATH"
        success "Virtual environment created"
    fi
    
    source "$VENV_PATH/bin/activate"
    success "Virtual environment activated"
}

install_dependencies() {
    log "Installing Python dependencies..."
    
    # Upgrade pip, setuptools, wheel first
    python3 -m pip install --upgrade pip setuptools wheel > /dev/null 2>&1
    
    # Define dependencies
    DEPENDENCIES=(
        "torch>=2.0.0"
        "transformers>=4.30.0"
        "pydantic>=2.0.0"
        "numpy>=1.24.0"
        "tqdm>=4.65.0"
        "pyyaml>=6.0"
    )
    
    # Install each dependency
    for dep in "${DEPENDENCIES[@]}"; do
        echo -n "  Installing $dep... "
        if python3 -m pip install --quiet "$dep" 2>/dev/null; then
            echo -e "${GREEN}✓${NC}"
        else
            error "Failed to install $dep"
        fi
    done
    
    success "All dependencies installed"
}

# ============================================================================
# Directory & Config Setup
# ============================================================================

create_directories() {
    log "Creating directories..."
    
    mkdir -p "$CONFIG_DIR"
    mkdir -p "$MODELS_DIR"
    
    success "Directories created:"
    echo "  Config: $CONFIG_DIR"
    echo "  Models: $MODELS_DIR"
}

create_configuration() {
    log "Creating configuration file..."
    
    CONFIG_FILE="$CONFIG_DIR/config.yaml"
    
    cat > "$CONFIG_FILE" << 'EOF'
# Momo-Kibidango Configuration
# Generated during installation

speculative_decoding:
  enabled: true
  target_model:
    model_name: "Qwen/Qwen2-7B"
    local_path: "${HOME}/.momo-kibidango/models/qwen2-7b"
    cache_dir: "${HOME}/.cache/huggingface/hub"
  
  draft_model:
    model_name: "microsoft/phi-2"
    local_path: "${HOME}/.momo-kibidango/models/phi-2"
    cache_dir: "${HOME}/.cache/huggingface/hub"
  
  # Speculative decoding parameters
  num_speculative_tokens: 5
  temperature: 0.7
  top_k: 50
  top_p: 0.95

inference:
  batch_size: 4
  max_tokens: 512
  temperature: 0.7
  device: "auto"  # auto|cuda|cpu|mps

logging:
  level: "INFO"
  format: "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

performance:
  enable_profiling: false
  benchmark_interval: 100
EOF

    success "Configuration created at $CONFIG_FILE"
}

# ============================================================================
# Validation
# ============================================================================

run_validation() {
    log "Running validation tests..."
    echo ""
    
    python3 << 'PYEOF'
import sys
import importlib

tests = {
    "torch": "PyTorch",
    "transformers": "Transformers",
    "pydantic": "Pydantic",
    "numpy": "NumPy",
    "tqdm": "tqdm",
    "yaml": "PyYAML",
}

all_passed = True
for module, name in tests.items():
    try:
        lib = importlib.import_module(module)
        version = getattr(lib, '__version__', 'unknown')
        print(f"  ✓ {name:20} {version}")
    except ImportError as e:
        print(f"  ✗ {name:20} FAILED: {e}")
        all_passed = False

if not all_passed:
    sys.exit(1)

print("")
print("  System Info:")
try:
    import torch
    device = "CUDA" if torch.cuda.is_available() else "CPU"
    print(f"    • Device: {device}")
    print(f"    • PyTorch version: {torch.__version__}")
except Exception as e:
    print(f"    • Error: {e}")
    sys.exit(1)
PYEOF

    if [ $? -ne 0 ]; then
        error "Validation failed"
    fi
    
    success "All validation tests passed"
}

# ============================================================================
# Completion Message
# ============================================================================

print_completion() {
    echo ""
    echo -e "${GREEN}╔════════════════════════════════════════════════════════╗"
    echo "║          ✅ Installation Complete!                        ║"
    echo "╚════════════════════════════════════════════════════════╝${NC}"
    echo ""
    
    echo "📍 Installation Paths:"
    echo "   Virtual Environment: $VENV_PATH"
    echo "   Configuration:       $CONFIG_DIR"
    echo "   Models Cache:        $MODELS_DIR"
    echo "   Install Log:         $LOG_FILE"
    echo ""
    
    echo "🚀 Next Steps:"
    echo "   1. Activate the virtual environment:"
    echo "      source $VENV_PATH/bin/activate"
    echo ""
    echo "   2. Run validation test:"
    echo "      ${SCRIPT_DIR}/validate-installation.sh"
    echo ""
    echo "   3. Try speculative decoding:"
    echo "      python3 -m momo_kibidango --help"
    echo ""
    
    echo "📚 Documentation:"
    echo "   • README: $SCRIPT_DIR/README.md"
    echo "   • Design: $SCRIPT_DIR/docs/MOMO_KIBIDANGO_INSTALLATION_DESIGN.md"
    echo "   • Troubleshooting: $SCRIPT_DIR/INSTALLATION_TROUBLESHOOTING.md"
    echo ""
    
    echo "🆘 Support:"
    echo "   • Review install log: tail -f $LOG_FILE"
    echo "   • Run validation: ${SCRIPT_DIR}/validate-installation.sh"
    echo "   • Uninstall: ${SCRIPT_DIR}/uninstall.sh"
    echo ""
    
    log "Installation completed successfully"
}

# ============================================================================
# Error Handling
# ============================================================================

trap 'error "Installation failed at line $LINENO"' ERR

# ============================================================================
# Run Main
# ============================================================================

main "$@"
