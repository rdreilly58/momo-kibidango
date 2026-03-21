#!/bin/bash
set -euo pipefail

# ============================================================================
# Momo-Kibidango Uninstall Script
# ============================================================================
# Purpose: Cleanly remove momo-kibidango installation
# Version: 1.0
# Date: March 20, 2026
# ============================================================================

VENV_PATH="${HOME}/.momo-kibidango/venv"
MOMO_DIR="${HOME}/.momo-kibidango"
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

confirm() {
    local prompt="$1"
    local response
    
    read -p "$(echo -e ${YELLOW}${prompt}${NC})" response
    [[ "$response" =~ ^[Yy]$ ]]
}

# ============================================================================
# Main Uninstall
# ============================================================================

main() {
    echo -e "${GREEN}"
    echo "╔════════════════════════════════════════════════════════╗"
    echo "║   🍑 Momo-Kibidango Uninstall                          ║"
    echo "╚════════════════════════════════════════════════════════╝"
    echo -e "${NC}"
    echo ""
    
    # Check if installation exists
    if [ ! -d "$MOMO_DIR" ]; then
        warning "No momo-kibidango installation found at $MOMO_DIR"
        echo "Nothing to uninstall."
        exit 0
    fi
    
    # Show what will be removed
    show_removal_details
    
    # Confirm removal
    if ! confirm "Remove momo-kibidango installation? (y/N) "; then
        echo "Uninstall cancelled."
        exit 0
    fi
    
    # Kill any running processes
    kill_processes
    
    # Remove installation
    remove_installation
    
    # Verify removal
    verify_removal
    
    # Print completion
    print_completion
}

# ============================================================================
# Show Removal Details
# ============================================================================

show_removal_details() {
    log "Installation details:"
    echo ""
    
    if [ -d "$VENV_PATH" ]; then
        VENV_SIZE=$(du -sh "$VENV_PATH" 2>/dev/null | awk '{print $1}')
        echo "  Virtual Environment: $VENV_PATH ($VENV_SIZE)"
    fi
    
    if [ -d "$CONFIG_DIR" ]; then
        echo "  Configuration:       $CONFIG_DIR"
        ls -la "$CONFIG_DIR" 2>/dev/null | sed 's/^/    /'
    fi
    
    if [ -d "$MODELS_DIR" ]; then
        MODELS_SIZE=$(du -sh "$MODELS_DIR" 2>/dev/null | awk '{print $1}')
        echo "  Models Cache:        $MODELS_DIR ($MODELS_SIZE)"
    fi
    
    echo ""
    TOTAL_SIZE=$(du -sh "$MOMO_DIR" 2>/dev/null | awk '{print $1}')
    echo "  Total size to remove: $TOTAL_SIZE"
    echo ""
}

# ============================================================================
# Kill Processes
# ============================================================================

kill_processes() {
    log "Checking for running momo-kibidango processes..."
    
    if pgrep -f "momo.kibidango|python.*momo" > /dev/null 2>&1; then
        warning "Found running momo-kibidango processes"
        
        if confirm "Kill running processes? (y/N) "; then
            pkill -f "momo.kibidango|python.*momo" || true
            success "Processes terminated"
        else
            warning "Skipping process termination"
        fi
    else
        log "No running processes found"
    fi
}

# ============================================================================
# Remove Installation
# ============================================================================

remove_installation() {
    log "Removing installation..."
    
    if [ ! -d "$MOMO_DIR" ]; then
        warning "Installation directory not found"
        return
    fi
    
    # Use trash if available (recoverable), fall back to rm
    if command -v trash &> /dev/null; then
        log "Moving $MOMO_DIR to Trash..."
        trash "$MOMO_DIR"
        success "Installation moved to Trash"
    else
        log "Removing $MOMO_DIR (using rm -rf)..."
        rm -rf "$MOMO_DIR"
        success "Installation removed"
    fi
}

# ============================================================================
# Verify Removal
# ============================================================================

verify_removal() {
    log "Verifying removal..."
    
    if [ -d "$MOMO_DIR" ]; then
        error "Installation directory still exists: $MOMO_DIR"
    fi
    
    success "Installation directory removed"
    
    # Check for leftover processes
    if pgrep -f "momo.kibidango|python.*momo" > /dev/null 2>&1; then
        warning "Leftover momo-kibidango processes still running"
        pkill -9 -f "momo.kibidango|python.*momo" || true
    else
        success "No leftover processes"
    fi
}

# ============================================================================
# Completion Message
# ============================================================================

print_completion() {
    echo ""
    echo -e "${GREEN}╔════════════════════════════════════════════════════════╗"
    echo "║          ✅ Uninstall Complete!                        ║"
    echo "╚════════════════════════════════════════════════════════╝${NC}"
    echo ""
    
    echo "What was removed:"
    echo "  ✓ Virtual environment"
    echo "  ✓ Configuration files"
    echo "  ✓ Model cache"
    echo ""
    
    echo "To reinstall:"
    echo "  ./install.sh"
    echo ""
    
    # Check for cache files outside main directory
    if [ -d "$HOME/.cache/huggingface" ]; then
        echo "Note: Hugging Face model cache may remain at:"
        echo "  $HOME/.cache/huggingface/hub"
        echo ""
        echo "To remove completely:"
        echo "  rm -rf $HOME/.cache/huggingface/hub"
        echo ""
    fi
}

# ============================================================================
# Error Handling
# ============================================================================

trap 'error "Uninstall failed at line $LINENO"' ERR

# ============================================================================
# Run Main
# ============================================================================

main "$@"
