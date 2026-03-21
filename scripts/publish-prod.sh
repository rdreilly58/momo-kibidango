#!/bin/bash
set -euo pipefail

# ============================================================================
# Momo-Kibidango Production PyPI Publishing Script
# ============================================================================
# ⚠️  WARNING: This uploads to the PRODUCTION PyPI registry
# Only run after successful TestPyPI validation!

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$SCRIPT_DIR"

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${RED}⚠️  WARNING: PRODUCTION PyPI UPLOAD${NC}"
echo "======================================"
echo ""
echo -e "${YELLOW}This will upload to PRODUCTION PyPI!${NC}"
echo ""
echo "Prerequisites:"
echo "  ✓ Tested on TestPyPI"
echo "  ✓ Tested installation: pip install momo-kibidango==VERSION"
echo "  ✓ Verified CLI works: momo-kibidango --help"
echo "  ✓ Reviewed CHANGELOG.md"
echo ""

# ============================================================================
# 1. Confirm production release
# ============================================================================

read -p "Have you tested this on TestPyPI? (y/n) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo -e "${RED}❌ Aborted${NC}"
    exit 1
fi

read -p "Are you sure you want to upload to PRODUCTION PyPI? (y/n) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo -e "${RED}❌ Aborted${NC}"
    exit 1
fi

# ============================================================================
# 2. Check for distributions
# ============================================================================

if [ ! -d dist ] || [ -z "$(ls -A dist)" ]; then
    echo -e "${YELLOW}No distributions found. Building...${NC}"
    ./scripts/build.sh
fi

echo -e "${YELLOW}Distributions to upload:${NC}"
ls -lh dist/
echo ""

# ============================================================================
# 3. Verify PyPI credentials
# ============================================================================

echo -e "${YELLOW}Checking PyPI credentials...${NC}"

if [ ! -f "$HOME/.pypirc" ]; then
    echo -e "${RED}❌ No .pypirc found${NC}"
    echo ""
    echo "Create ~/.pypirc with PyPI token:"
    echo ""
    echo "[distutils]"
    echo "index-servers ="
    echo "    pypi"
    echo "    testpypi"
    echo ""
    echo "[pypi]"
    echo "repository = https://upload.pypi.org/legacy/"
    echo "username = __token__"
    echo "password = pypi-AgEIcHlwaS5vcmc...(your token)"
    echo ""
    echo "[testpypi]"
    echo "repository = https://test.pypi.org/legacy/"
    echo "username = __token__"
    echo "password = pypi-NG0KfHlwaS5vcmc...(your token)"
    echo ""
    exit 1
fi

echo -e "${GREEN}✓ Credentials configured${NC}"

# ============================================================================
# 4. Upload to production PyPI
# ============================================================================

echo ""
echo -e "${YELLOW}Uploading to production PyPI...${NC}"

twine upload --repository pypi dist/*

echo -e "${GREEN}✓ Upload complete${NC}"

# ============================================================================
# 5. Verify availability
# ============================================================================

VERSION=$(ls dist/momo_kibidango-*.whl | sed 's/.*momo_kibidango-\([^-]*\)-.*/\1/')

echo ""
echo -e "${GREEN}✅ Production upload complete!${NC}"
echo ""
echo "Package now available at:"
echo "  https://pypi.org/project/momo-kibidango/$VERSION/"
echo ""
echo "Users can now install with:"
echo "  pip install momo-kibidango"
echo ""
