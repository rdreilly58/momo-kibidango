#!/bin/bash
set -euo pipefail

# ============================================================================
# Momo-Kibidango Update Script
# ============================================================================
# Purpose: Update momo-kibidango to latest version
# Version: 1.0
# Date: March 20, 2026
# ============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_PATH="${HOME}/.momo-kibidango/venv"
CONFIG_DIR="${HOME}/.momo-kibidango/config"
MODELS_DIR="${HOME}/.momo-kibidango/models"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# ============================================================================
# Utility Functions
# ============================================================================

log() {
    echo -e "${BLUE}[$(date '+%H:%M:%S')]${NC} $1"
}

success() {
    echo -e "${GREEN}✓ $1${NC}"
}

warning() {
    echo -e "${YELLOW}⚠ $1${NC}"
}

error() {
    echo -e "${RED}❌ $1${NC}"
    exit 1
}

# ============================================================================
# Main Update
# ============================================================================

main() {
    echo -e "${GREEN}"
    echo "╔════════════════════════════════════════════════════════╗"
    echo "║   🍑 Momo-Kibidango Update                             ║"
    echo "╚════════════════════════════════════════════════════════╝"
    echo -e "${NC}"
    echo ""
    
    # Check installation
    check_installation
    
    # Show current versions
    show_current_versions
    
    # Update repository
    update_repository
    
    # Update dependencies
    update_dependencies
    
    # Validate update
    validate_update
    
    # Print completion
    print_completion
}

# ============================================================================
# Check Installation
# ============================================================================

check_installation() {
    log "Checking existing installation..."
    
    if [ ! -d "$VENV_PATH" ]; then
        error "No existing installation found at $VENV_PATH"
    fi
    success "Installation found"
    
    if [ ! -d "$SCRIPT_DIR/.git" ]; then
        warning "Not a git repository. Using update mode: dependencies only"
        UPDATE_MODE="deps_only"
    else
        UPDATE_MODE="full"
    fi
}

# ============================================================================
# Show Current Versions
# ============================================================================

show_current_versions() {
    log "Current versions:"
    
    source "$VENV_PATH/bin/activate"
    
    python3 << 'PYEOF'
import importlib

modules = {
    "torch": "PyTorch",
    "transformers": "Transformers",
    "pydantic": "Pydantic",
    "numpy": "NumPy",
    "tqdm": "tqdm",
    "yaml": "PyYAML",
}

for module, name in modules.items():
    try:
        mod = importlib.import_module(module)
        version = getattr(mod, '__version__', 'unknown')
        print(f"  {name:20} {version}")
    except:
        print(f"  {name:20} (not installed)")
PYEOF
    
    echo ""
}

# ============================================================================
# Update Repository
# ============================================================================

update_repository() {
    if [ "$UPDATE_MODE" != "full" ]; then
        log "Skipping repository update (not a git repo)"
        return
    fi
    
    log "Updating repository from GitHub..."
    
    CURRENT_BRANCH=$(git -C "$SCRIPT_DIR" rev-parse --abbrev-ref HEAD)
    
    if [ "$CURRENT_BRANCH" != "main" ] && [ "$CURRENT_BRANCH" != "master" ]; then
        warning "On branch '$CURRENT_BRANCH' (not main/master)"
        warning "Skipping git pull to avoid conflicts"
        return
    fi
    
    git -C "$SCRIPT_DIR" fetch origin
    
    if git -C "$SCRIPT_DIR" diff --quiet origin/$([ "$CURRENT_BRANCH" = "main" ] && echo "main" || echo "master")..HEAD; then
        success "Repository is up to date"
    else
        log "Pulling latest changes..."
        git -C "$SCRIPT_DIR" pull origin "$CURRENT_BRANCH"
        success "Repository updated"
    fi
}

# ============================================================================
# Update Dependencies
# ============================================================================

update_dependencies() {
    log "Updating Python dependencies..."
    
    source "$VENV_PATH/bin/activate"
    
    # Define dependencies to update
    DEPENDENCIES=(
        "torch"
        "transformers"
        "pydantic"
        "numpy"
        "tqdm"
        "pyyaml"
    )
    
    # Update pip first
    python3 -m pip install --upgrade pip > /dev/null 2>&1
    success "pip upgraded"
    
    # Update each dependency
    for dep in "${DEPENDENCIES[@]}"; do
        echo -n "  Updating $dep... "
        if python3 -m pip install --upgrade --quiet "$dep" 2>/dev/null; then
            echo -e "${GREEN}✓${NC}"
        else
            warning "Failed to update $dep (continuing)"
        fi
    done
    
    success "Dependencies updated"
}

# ============================================================================
# Validate Update
# ============================================================================

validate_update() {
    log "Validating update..."
    
    source "$VENV_PATH/bin/activate"
    
    python3 << 'PYEOF'
import sys
import importlib

dependencies = [
    "torch",
    "transformers",
    "pydantic",
    "numpy",
    "tqdm",
    "yaml",
]

failed = []
for module in dependencies:
    try:
        mod = importlib.import_module(module)
        version = getattr(mod, '__version__', 'unknown')
        print(f"  ✓ {module:20} {version}")
    except ImportError:
        print(f"  ✗ {module:20} FAILED")
        failed.append(module)

if failed:
    print(f"\n❌ Failed to load: {', '.join(failed)}")
    sys.exit(1)
PYEOF

    if [ $? -ne 0 ]; then
        error "Validation failed"
    fi
    
    success "All dependencies validated"
}

# ============================================================================
# Completion Message
# ============================================================================

print_completion() {
    echo ""
    echo -e "${GREEN}╔════════════════════════════════════════════════════════╗"
    echo "║          ✅ Update Complete!                           ║"
    echo "╚════════════════════════════════════════════════════════╝${NC}"
    echo ""
    
    if [ "$UPDATE_MODE" = "full" ]; then
        echo "Updated:"
        echo "  ✓ Repository"
        echo "  ✓ Dependencies"
    else
        echo "Updated:"
        echo "  ✓ Dependencies (not a git repository)"
    fi
    
    echo ""
    echo "To verify:"
    echo "  source $VENV_PATH/bin/activate"
    echo "  python3 -c 'import torch; print(torch.__version__)'"
    echo ""
    
    echo "Configuration remains at:"
    echo "  $CONFIG_DIR"
    echo ""
}

# ============================================================================
# Error Handling
# ============================================================================

trap 'error "Update failed at line $LINENO"' ERR

# ============================================================================
# Run Main
# ============================================================================

main "$@"
