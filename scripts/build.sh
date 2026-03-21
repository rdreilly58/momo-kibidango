#!/bin/bash
set -euo pipefail

# ============================================================================
# Momo-Kibidango Build Script
# ============================================================================
# Builds wheel and source distribution for PyPI

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$SCRIPT_DIR"

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${GREEN}🍑 Momo-Kibidango Build${NC}"
echo "=================================="
echo ""

# ============================================================================
# 1. Check dependencies
# ============================================================================

echo -e "${YELLOW}Checking build dependencies...${NC}"

if ! command -v python3 &> /dev/null; then
    echo -e "${RED}❌ Python 3 not found${NC}"
    exit 1
fi

PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}' | cut -d. -f1,2)
if (( $(echo "$PYTHON_VERSION < 3.10" | bc -l) )); then
    echo -e "${RED}❌ Python 3.10+ required (found $PYTHON_VERSION)${NC}"
    exit 1
fi
echo -e "${GREEN}✓ Python $PYTHON_VERSION${NC}"

# Install build tools if needed
echo -e "${YELLOW}Installing build tools...${NC}"
python3 -m pip install --quiet --upgrade pip setuptools wheel build twine

echo -e "${GREEN}✓ Build tools installed${NC}"

# ============================================================================
# 2. Clean previous builds
# ============================================================================

echo -e "${YELLOW}Cleaning previous builds...${NC}"
rm -rf build/ dist/ *.egg-info src/momo_kibidango.egg-info/
echo -e "${GREEN}✓ Clean complete${NC}"

# ============================================================================
# 3. Validate pyproject.toml
# ============================================================================

echo -e "${YELLOW}Validating pyproject.toml...${NC}"

if ! python3 -c "import tomllib" 2>/dev/null; then
    # Python < 3.11 fallback
    python3 -m pip install --quiet tomli
    if ! python3 -c "import tomli" 2>/dev/null; then
        echo -e "${RED}❌ Could not import tomllib or tomli${NC}"
        exit 1
    fi
fi

python3 << 'PYEOF'
import sys
try:
    try:
        import tomllib
    except ImportError:
        import tomli as tomllib
    
    with open('pyproject.toml', 'rb') as f:
        data = tomllib.load(f)
    
    # Check required fields
    assert 'project' in data, "Missing [project] section"
    assert 'name' in data['project'], "Missing project.name"
    assert 'version' in data['project'], "Missing project.version"
    assert 'description' in data['project'], "Missing project.description"
    
    print(f"✓ pyproject.toml valid")
    print(f"  Name: {data['project']['name']}")
    print(f"  Version: {data['project']['version']}")
except Exception as e:
    print(f"❌ pyproject.toml validation failed: {e}", file=sys.stderr)
    sys.exit(1)
PYEOF

echo -e "${GREEN}✓ Validation passed${NC}"

# ============================================================================
# 4. Build wheel and source distribution
# ============================================================================

echo -e "${YELLOW}Building distributions...${NC}"

python3 -m build

if [ ! -d dist ] || [ -z "$(ls -A dist)" ]; then
    echo -e "${RED}❌ Build failed - no distributions created${NC}"
    exit 1
fi

echo -e "${GREEN}✓ Build complete${NC}"

# ============================================================================
# 5. Validate distributions with twine
# ============================================================================

echo -e "${YELLOW}Validating distributions with twine...${NC}"

twine check dist/*

echo -e "${GREEN}✓ Distributions valid${NC}"

# ============================================================================
# 6. Display results
# ============================================================================

echo ""
echo -e "${GREEN}✅ Build complete!${NC}"
echo ""
echo "Distributions created:"
ls -lh dist/

echo ""
echo "Next steps:"
echo "  1. Test locally: pip install dist/momo_kibidango-*.whl"
echo "  2. Upload to TestPyPI: ./scripts/publish-test.sh"
echo "  3. Upload to PyPI: ./scripts/publish-prod.sh"
echo ""
