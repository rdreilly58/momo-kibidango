#!/bin/bash

# ============================================================================
# Momo-Kibidango Installation Validator
# ============================================================================
# Purpose: Validate all momo-kibidango components
# Version: 1.0
# Date: March 20, 2026
# ============================================================================

VENV_PATH="${HOME}/.momo-kibidango/venv"
CONFIG_DIR="${HOME}/.momo-kibidango/config"
MODELS_DIR="${HOME}/.momo-kibidango/models"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Tracking
TESTS_PASSED=0
TESTS_FAILED=0

# ============================================================================
# Utility Functions
# ============================================================================

log() {
    echo -e "${BLUE}[$(date '+%H:%M:%S')]${NC} $1"
}

pass() {
    echo -e "${GREEN}  ✓ $1${NC}"
    ((TESTS_PASSED++)) || true
}

fail() {
    echo -e "${RED}  ✗ $1${NC}"
    ((TESTS_FAILED++)) || true
}

# ============================================================================
# Main Validation
# ============================================================================

main() {
    echo -e "${GREEN}"
    echo "╔════════════════════════════════════════════════════════╗"
    echo "║   🍑 Momo-Kibidango Installation Validator              ║"
    echo "╚════════════════════════════════════════════════════════╝"
    echo -e "${NC}"
    echo ""
    
    # Check structure
    check_directory_structure
    check_virtual_environment
    check_configuration
    check_python_environment
    check_dependencies
    
    # Print summary
    print_summary
}

# ============================================================================
# Directory Checks
# ============================================================================

check_directory_structure() {
    log "Checking directory structure..."
    
    if [ ! -d "$VENV_PATH" ]; then
        fail "Virtual environment not found at $VENV_PATH"
    else
        pass "Virtual environment directory exists"
    fi
    
    if [ ! -d "$CONFIG_DIR" ]; then
        fail "Config directory not found at $CONFIG_DIR"
    else
        pass "Config directory exists"
    fi
    
    if [ ! -d "$MODELS_DIR" ]; then
        fail "Models directory not found at $MODELS_DIR"
    else
        pass "Models directory exists"
    fi
}

# ============================================================================
# Virtual Environment Checks
# ============================================================================

check_virtual_environment() {
    log "Checking virtual environment..."
    
    if [ ! -f "$VENV_PATH/bin/activate" ]; then
        fail "Virtual environment activation script not found"
        return
    fi
    pass "Activation script exists"
    
    if [ ! -f "$VENV_PATH/bin/python" ]; then
        fail "Python interpreter not found in venv"
        return
    fi
    pass "Python interpreter available"
    
    # Check Python version in venv
    PYTHON_VERSION=$("$VENV_PATH/bin/python" --version 2>&1 | awk '{print $2}')
    MAJOR=$(echo "$PYTHON_VERSION" | cut -d. -f1)
    MINOR=$(echo "$PYTHON_VERSION" | cut -d. -f2)
    
    if [ "$MAJOR" -lt 3 ] || ([ "$MAJOR" -eq 3 ] && [ "$MINOR" -lt 10 ]); then
        fail "Python 3.10+ required (found $PYTHON_VERSION in venv)"
    else
        pass "Python $PYTHON_VERSION (3.10+ required)"
    fi
}

# ============================================================================
# Configuration Checks
# ============================================================================

check_configuration() {
    log "Checking configuration..."
    
    CONFIG_FILE="$CONFIG_DIR/config.yaml"
    
    if [ ! -f "$CONFIG_FILE" ]; then
        fail "Configuration file not found at $CONFIG_FILE"
        return
    fi
    pass "Configuration file exists"
    
    # Check for required keys
    if grep -q "speculative_decoding:" "$CONFIG_FILE"; then
        pass "Configuration has 'speculative_decoding' section"
    else
        fail "Configuration missing 'speculative_decoding' section"
    fi
    
    if grep -q "inference:" "$CONFIG_FILE"; then
        pass "Configuration has 'inference' section"
    else
        fail "Configuration missing 'inference' section"
    fi
}

# ============================================================================
# Python Environment Checks
# ============================================================================

check_python_environment() {
    log "Checking Python environment..."
    
    # Run Python checks in a subshell with venv activated
    (
        source "$VENV_PATH/bin/activate"
        
        # Check pip version
        if PIP_VERSION=$(python3 -m pip --version 2>/dev/null); then
            echo -e "${GREEN}  ✓ pip: $PIP_VERSION${NC}"
        else
            echo -e "${RED}  ✗ pip not available in virtual environment${NC}"
            exit 1
        fi
    ) && ((TESTS_PASSED++)) || ((TESTS_FAILED++)) || true
}

# ============================================================================
# Dependency Checks
# ============================================================================

check_dependencies() {
    log "Checking required dependencies..."
    
    # Run in subshell with venv
    (
        source "$VENV_PATH/bin/activate"
        
        python3 << 'PYEOF'
import sys
import importlib.util

dependencies = [
    ("torch", "PyTorch"),
    ("transformers", "Transformers"),
    ("pydantic", "Pydantic"),
    ("numpy", "NumPy"),
    ("tqdm", "tqdm"),
    ("yaml", "PyYAML"),
]

failed = []
for module_name, display_name in dependencies:
    spec = importlib.util.find_spec(module_name)
    if spec is None:
        print(f"  ✗ {display_name} not found")
        failed.append(module_name)
    else:
        try:
            mod = importlib.import_module(module_name)
            version = getattr(mod, '__version__', 'unknown')
            print(f"  ✓ {display_name:20} {version}")
        except Exception as e:
            print(f"  ✗ {display_name:20} {e}")
            failed.append(module_name)

if failed:
    print(f"\n❌ Missing: {', '.join(failed)}")
    sys.exit(1)
PYEOF
    ) && ((TESTS_PASSED += 1)) || ((TESTS_FAILED += 1)) || true
}

# ============================================================================
# Summary
# ============================================================================

print_summary() {
    echo ""
    echo -e "${GREEN}╔════════════════════════════════════════════════════════╗"
    printf "║  Validation Results: ${GREEN}%2d passed${NC}  ${RED}%2d failed${GREEN}             ║\n" "$TESTS_PASSED" "$TESTS_FAILED"
    echo -e "${GREEN}╚════════════════════════════════════════════════════════╝${NC}"
    echo ""
    
    if [ "$TESTS_FAILED" -eq 0 ]; then
        echo -e "${GREEN}✅ Installation is valid!${NC}"
        echo ""
        echo "Next steps:"
        echo "  1. Activate environment: source $VENV_PATH/bin/activate"
        echo "  2. Run a test: python3 -c 'import torch; print(torch.__version__)'"
        echo "  3. Check config: cat $CONFIG_DIR/config.yaml"
        echo ""
    else
        echo -e "${RED}❌ Installation has issues. See above for details.${NC}"
        echo ""
        echo "To fix:"
        echo "  1. Review the failed tests above"
        echo "  2. Re-run install.sh: ./install.sh"
        echo "  3. Check install log: tail install.log"
        echo ""
    fi
}

# ============================================================================
# Run Main
# ============================================================================

main "$@"
