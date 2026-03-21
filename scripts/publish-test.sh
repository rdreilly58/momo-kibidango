#!/bin/bash
set -euo pipefail

# ============================================================================
# Momo-Kibidango TestPyPI Publishing Script
# ============================================================================
# Uploads to TestPyPI for validation before production release

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$SCRIPT_DIR"

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${GREEN}🍑 Momo-Kibidango TestPyPI Upload${NC}"
echo "========================================"
echo ""

# ============================================================================
# 1. Check for distributions
# ============================================================================

if [ ! -d dist ] || [ -z "$(ls -A dist)" ]; then
    echo -e "${YELLOW}No distributions found. Building...${NC}"
    ./scripts/build.sh
fi

echo -e "${YELLOW}Distributions to upload:${NC}"
ls -lh dist/
echo ""

# ============================================================================
# 2. Check for .pypirc or ask for credentials
# ============================================================================

echo -e "${YELLOW}Checking PyPI credentials...${NC}"

if [ ! -f "$HOME/.pypirc" ]; then
    echo -e "${RED}⚠️  No .pypirc found${NC}"
    echo ""
    echo "To authenticate, you can:"
    echo "  1. Create ~/.pypirc with TestPyPI token (recommended)"
    echo "  2. Use API token as password when prompted"
    echo "  3. Use username '__token__' and paste token as password"
    echo ""
    echo "To get TestPyPI API token:"
    echo "  https://test.pypi.org/manage/account/tokens/"
    echo ""
    read -p "Continue? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# ============================================================================
# 3. Upload to TestPyPI
# ============================================================================

echo -e "${YELLOW}Uploading to TestPyPI...${NC}"

if [ -f "$HOME/.pypirc" ]; then
    # Use credentials from .pypirc
    twine upload --repository testpypi dist/*
else
    # Prompt for credentials
    twine upload --repository testpypi --username __token__ dist/*
fi

echo -e "${GREEN}✓ Upload complete${NC}"

# ============================================================================
# 4. Test installation from TestPyPI
# ============================================================================

echo ""
echo -e "${YELLOW}Testing installation from TestPyPI...${NC}"

# Extract version from the wheel
VERSION=$(ls dist/momo_kibidango-*.whl | sed 's/.*momo_kibidango-\([^-]*\)-.*/\1/')

echo ""
echo "To test installation, run:"
echo ""
echo "  pip install -i https://test.pypi.org/simple/ momo-kibidango==$VERSION"
echo ""
echo "Then verify:"
echo ""
echo "  momo-kibidango --help"
echo "  momo-kibidango validate"
echo ""

echo -e "${GREEN}✅ TestPyPI upload complete!${NC}"
echo ""
echo "Package available at:"
echo "  https://test.pypi.org/project/momo-kibidango/"
echo ""
