# Release Process

This document describes the procedure for releasing new versions of momo-kibidango to PyPI.

## Pre-Release Checklist

Before starting a release:

- [ ] All tests passing locally and in CI/CD
- [ ] Code review completed
- [ ] No known critical issues or TODOs
- [ ] Documentation updated
- [ ] CHANGELOG.md updated with new changes
- [ ] Merged to `main` branch
- [ ] Commits are clean and well-documented

## Version Numbering

Follow [Semantic Versioning](https://semver.org/):

- **MAJOR.MINOR.PATCH** (e.g., 1.0.0)
- **MAJOR**: Breaking API changes
- **MINOR**: New features (backward compatible)
- **PATCH**: Bug fixes

Examples:
- 1.0.0 → 1.0.1 (patch: bug fix)
- 1.0.0 → 1.1.0 (minor: new feature)
- 1.0.0 → 2.0.0 (major: breaking change)

## Release Steps

### 1. Update Version

Update version in `pyproject.toml`:

```toml
[project]
version = "1.1.0"  # Update this
```

### 2. Update CHANGELOG.md

Add new section at top:

```markdown
## [1.1.0] - 2026-03-27

### Added
- New feature X
- New feature Y

### Fixed
- Bug fix A
- Bug fix B

### Changed
- Behavior change C

### Removed
- Deprecated API D
```

### 3. Create Git Tag

```bash
git checkout main
git pull origin main
git tag -a v1.1.0 -m "Release version 1.1.0"
git push origin v1.1.0
```

### 4. Build Distributions

```bash
# Clean previous builds
rm -rf dist build *.egg-info

# Install build tools
pip install --upgrade build twine

# Build wheel and source distribution
python -m build

# Verify distributions
twine check dist/*
```

### 5. Test on TestPyPI (Required)

```bash
# Upload to TestPyPI
twine upload --repository testpypi dist/*

# Create temporary venv for testing
python3 -m venv test_venv
source test_venv/bin/activate

# Install from TestPyPI
pip install -i https://test.pypi.org/simple/ momo-kibidango==1.1.0

# Verify CLI works
momo-kibidango --version
momo-kibidango --help
momo-kibidango validate

# Test optional dependencies
pip install -i https://test.pypi.org/simple/ 'momo-kibidango[dev]'==1.1.0

# Clean up
deactivate
rm -rf test_venv
```

### 6. Publish to Production PyPI

```bash
# Upload to production PyPI
twine upload dist/*

# Verify it's available (wait ~1 minute for index to update)
pip index versions momo-kibidango
```

### 7. Create GitHub Release

1. Go to https://github.com/rdreilly58/momo-kibidango/releases
2. Click "Draft a new release"
3. Select tag: v1.1.0
4. Title: "Release v1.1.0"
5. Description: Copy from CHANGELOG.md
6. Attach wheel and sdist files from dist/
7. Mark as "Pre-release" if applicable
8. Publish release

## Automated Publishing (Future)

GitHub Actions workflow to auto-publish on tag (`.github/workflows/publish.yml`):

```yaml
name: Publish to PyPI

on:
  push:
    tags:
      - 'v*'

jobs:
  publish:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.10'
      
      - name: Install dependencies
        run: |
          python -m pip install --upgrade pip build twine
      
      - name: Build distributions
        run: python -m build
      
      - name: Publish to PyPI
        env:
          TWINE_USERNAME: __token__
          TWINE_PASSWORD: ${{ secrets.PYPI_API_TOKEN }}
        run: twine upload dist/*
```

## Post-Release

### 1. Verify Package
```bash
# Check PyPI listing
curl https://pypi.org/pypi/momo-kibidango/json | jq '.info.version'

# Install and test
pip install --upgrade momo-kibidango
momo-kibidango --version
```

### 2. Announce Release
- Update GitHub releases page
- Post to relevant communities:
  - Reddit: r/MachineLearning, r/Python
  - Twitter/X
  - Dev communities

### 3. Prepare for Next Release
- Create milestone for next version
- Update version to next SNAPSHOT (e.g., 1.1.0 → 1.1.1-dev)
- Plan next features

## Troubleshooting

### Issue: "File already exists" on PyPI

PyPI doesn't allow re-uploading same version. Solutions:
1. Bump to next patch version (1.0.1 → 1.0.2)
2. Delete version from TestPyPI if testing
3. Contact PyPI support for removal (very rare)

### Issue: TestPyPI succeeds but production fails

Common causes:
- Missing credentials in .pypirc
- Network/firewall issues
- Dependencies not available on PyPI

Solutions:
```bash
# Check .pypirc format
cat ~/.pypirc

# Test upload with verbose output
twine upload --verbose dist/*

# Check token validity
curl -u __token__:your_token https://upload.pypi.org/legacy/
```

### Issue: Installation fails from PyPI

Possible causes:
- Missing dependency (check pyproject.toml)
- Platform-specific issue
- Version constraints too strict

Solutions:
```bash
# Install verbose to see dependency resolution
pip install -v momo-kibidango

# Check what got installed
pip show momo-kibidango
pip show -f momo-kibidango | grep Location
```

## PyPI Configuration (.pypirc)

Create `~/.pypirc` with API tokens:

```ini
[distutils]
index-servers =
    pypi
    testpypi

[pypi]
repository = https://upload.pypi.org/legacy/
username = __token__
password = pypi-AgEIcHlwaS5vcmc...

[testpypi]
repository = https://test.pypi.org/legacy/
username = __token__
password = pypi-NG0KfHlwaS5vcmc...
```

**To get API tokens:**
1. https://pypi.org/account/tokens/ (production)
2. https://test.pypi.org/account/tokens/ (testing)
3. Create token with "Entire repository" scope
4. Copy and paste into .pypirc

## Rollback Procedure

If a release has critical bugs:

1. **Yank the version** on PyPI:
   ```bash
   twine upload --skip-existing --skip-metadata dist/*
   # Then mark as "yanked" in PyPI web interface
   ```

2. **Hotfix and re-release**:
   ```bash
   # Fix the bug
   git commit -am "fix: critical bug in 1.1.0"
   
   # Version it as patch
   # 1.1.0 → 1.1.1
   
   # Follow normal release process
   ```

3. **Notify users**:
   - Update GitHub release with warning
   - Post in discussions
   - Include upgrade instructions

## Version History Format

Keep CHANGELOG.md consistent:

```markdown
## [1.1.0] - 2026-03-27

### Added
- Brief feature description

### Fixed
- Brief bug fix description

### Changed
- Brief change description

### Removed
- Removed feature/API

### Security
- Security issue fixes

### Deprecated
- Deprecated APIs/features

### Notes
- Important notes for upgraders
```

## Long-Term Support (LTS)

For critical versions (1.0.0, 2.0.0, etc.):
- Maintain for 12 months minimum
- Backport critical security fixes
- Announce EOL date in advance
- Tag maintenance branch: `1.0-lts`

## Questions?

Reach out:
- Open GitHub issue
- Email: robert.reilly@reillydesignstudio.com
- Check existing release issues/PRs

---

Last updated: March 20, 2026
