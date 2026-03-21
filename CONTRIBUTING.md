# Contributing to Momo-Kibidango

Thank you for your interest in contributing! This document provides guidelines and instructions for participating in the project.

## Code of Conduct

- Be respectful and constructive
- Assume good intentions
- Focus on the work, not the person
- Report issues privately if sensitive

## Getting Started

### Prerequisites
- Python 3.10 or higher
- Git
- ~20GB disk space (for model downloads)
- Basic familiarity with PyTorch and transformers

### Setup Development Environment

```bash
# Clone the repository
git clone https://github.com/rdreilly58/momo-kibidango.git
cd momo-kibidango

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install in editable mode with dev dependencies
pip install -e ".[dev]"

# Run tests
pytest

# Check code quality
black src/ tests/
ruff check src/ tests/
mypy src/
```

## Development Workflow

### 1. Create a Feature Branch
```bash
git checkout -b feature/your-feature-name
```

**Branch naming conventions:**
- `feature/` - New features
- `bugfix/` - Bug fixes
- `docs/` - Documentation
- `perf/` - Performance improvements
- `test/` - Test coverage

### 2. Make Changes
- Write clean, well-documented code
- Add type hints (Python 3.10+ style)
- Include docstrings for public APIs
- Write tests for new functionality

### 3. Code Quality Checks

```bash
# Format code
black src/ tests/

# Check for style issues
ruff check --fix src/ tests/

# Type checking
mypy src/

# Run tests with coverage
pytest --cov=src/momo_kibidango --cov-report=html

# Check distributions
python -m build
twine check dist/*
```

### 4. Commit Messages

Follow [Conventional Commits](https://www.conventionalcommits.org/):

```
type(scope): description

body (optional)

footer (optional)
```

**Types:**
- `feat` - New feature
- `fix` - Bug fix
- `docs` - Documentation
- `style` - Code style (formatting, missing semicolons)
- `refactor` - Code refactoring
- `perf` - Performance improvements
- `test` - Test coverage
- `ci` - CI/CD changes
- `chore` - Maintenance

**Examples:**
```
feat(cli): add progress bar to inference command
fix(decoder): handle edge case in token sampling
docs: update installation guide for Windows
perf(monitor): optimize metrics collection
test(speculative): increase coverage to 95%
```

### 5. Submit Pull Request

1. Push your branch: `git push origin feature/your-feature-name`
2. Open PR on GitHub
3. Fill in the PR template
4. Link related issues: `Closes #123`
5. Request reviewers (maintainers)

**PR Guidelines:**
- One feature per PR (focused changes)
- Include tests for new code
- Update documentation
- Add entry to CHANGELOG.md
- Ensure CI passes

## Testing

### Writing Tests

```python
# tests/test_speculative_decoder.py
import pytest
from momo_kibidango import SpeculativeDecoder, ModelConfig


def test_decoder_initialization():
    """Test decoder initialization with valid config."""
    config = ModelConfig(
        target_model="microsoft/phi-2",
        draft_model="microsoft/phi-2",
    )
    decoder = SpeculativeDecoder(config)
    assert decoder is not None


@pytest.mark.parametrize("max_tokens", [1, 10, 100, 512])
def test_decoder_max_tokens(max_tokens):
    """Test decoder respects max_tokens parameter."""
    # Implementation
    pass


@pytest.fixture
def decoder():
    """Fixture providing initialized decoder."""
    config = ModelConfig(...)
    return SpeculativeDecoder(config)
```

### Running Tests

```bash
# Run all tests
pytest

# Run specific test file
pytest tests/test_speculative_decoder.py

# Run with verbose output
pytest -v

# Run with coverage report
pytest --cov=src/momo_kibidango --cov-report=html

# Run specific test
pytest tests/test_speculative_decoder.py::test_decoder_initialization

# Run tests matching pattern
pytest -k "test_decoder"
```

## Documentation

### Writing Documentation

- Use Markdown for .md files
- Include code examples for technical docs
- Keep sentences concise
- Update README.md for user-facing changes
- Update docstrings for API changes

### Documentation Structure
```
docs/
├── INSTALLATION_DESIGN.md     # Architecture & design decisions
├── PRODUCTION_DEPLOYMENT.md   # Deployment guide
└── API.md                      # API reference (future)
```

### Docstring Format

```python
def run_inference(
    prompt: str,
    max_tokens: int = 512,
    temperature: float = 0.7,
) -> str:
    """
    Run inference with speculative decoding.
    
    Executes inference using draft and target models to accelerate
    token generation through speculative decoding.
    
    Args:
        prompt: Input text to continue from
        max_tokens: Maximum tokens to generate (default: 512)
        temperature: Sampling temperature, 0-2 (default: 0.7)
        
    Returns:
        Generated text continuation
        
    Raises:
        ValueError: If temperature not in valid range
        RuntimeError: If model initialization fails
        
    Example:
        >>> from momo_kibidango import SpeculativeDecoder
        >>> decoder = SpeculativeDecoder(config)
        >>> result = decoder.run_inference("Hello")
        >>> print(result)
    """
    pass
```

## Release Process

See [RELEASE_PROCESS.md](RELEASE_PROCESS.md) for version bumping, testing, and publishing procedures.

## Reporting Issues

### Security Issues
**Do not open public issues for security vulnerabilities.**

Email: robert.reilly@reillydesignstudio.com with:
- Description of vulnerability
- Steps to reproduce
- Potential impact
- Suggested fix (optional)

### Bug Reports

Include:
- Clear description
- Steps to reproduce
- Expected vs actual behavior
- Python version and OS
- Relevant error messages/logs
- Minimal reproducible example

**Template:**
```
### Description
Describe the bug clearly.

### Steps to Reproduce
1. Do this
2. Then this
3. See error

### Expected Behavior
What should happen

### Actual Behavior
What actually happens

### Environment
- Python: 3.11.0
- OS: macOS 13.0
- momo-kibidango: 1.0.0

### Error Message
```
error traceback here
```
```

## Feature Requests

Include:
- Clear use case
- Why it's needed
- How you'd use it
- Potential implementation approach (optional)

## Project Structure

```
momo-kibidango/
├── src/momo_kibidango/          # Main package
│   ├── __init__.py              # Package exports
│   ├── cli.py                   # CLI interface
│   ├── speculative_2model.py    # 2-model decoder
│   ├── monitoring.py            # Performance monitoring
│   └── production_hardening.py  # Safeguards
├── tests/                        # Test suite
├── docs/                         # Documentation
├── scripts/                      # Utility scripts
├── pyproject.toml               # Project configuration
└── README.md                     # Overview
```

## Community & Support

- **GitHub Issues**: Bug reports and feature requests
- **GitHub Discussions**: General questions and ideas
- **Email**: robert.reilly@reillydesignstudio.com

## License

By contributing, you agree to license your work under the Apache-2.0 license. See [LICENSE](LICENSE) for details.

## Recognition

Contributors will be recognized in:
- CHANGELOG.md releases
- GitHub contributors page
- Project documentation (with permission)

---

Thank you for contributing to momo-kibidango! 🍑
