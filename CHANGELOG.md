# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] - 2026-03-20

### Added
- Initial public release of momo-kibidango
- Speculative decoding framework with 2-model and 3-model implementations
- CLI interface for inference, benchmarking, and validation
- PyPI package distribution with modern Python packaging (PEP 621)
- Optional dependencies for advanced features (MCP, Jupyter, vLLM)
- Performance monitoring and production hardening utilities
- OpenClaw integration support
- Comprehensive documentation and examples

### Features
- **Speculative Decoding**: Draft model + target model inference acceleration
- **Performance Monitoring**: Real-time performance metrics and logging
- **Production Hardening**: Error handling, validation, and safeguards
- **CLI Tool**: `momo-kibidango` command-line utility
- **Flexible Installation**: pip, git, editable, and script-based options

### Installation Methods
- `pip install momo-kibidango` (PyPI)
- `pip install git+https://github.com/rdreilly58/momo-kibidango.git` (GitHub)
- `pip install -e .` (Development/editable)
- One-line script: `curl -fsSL https://github.com/.../install.sh | bash`

### Documentation
- README with quick-start guide
- INSTALLATION_DESIGN.md with architecture overview
- API reference and code examples
- Troubleshooting guide

### Technical Details
- **Python**: 3.10+ support
- **Dependencies**: PyTorch 2.0+, Transformers 4.30+, Pydantic 2.0+
- **Optional**: vLLM 0.3+ (inference), MCP (agent integration), Jupyter
- **License**: Apache-2.0

---

## Planned Releases

### [1.1.0] - TBD
- [ ] MCP server implementation for agent integration
- [ ] Advanced caching strategies
- [ ] ONNX export support
- [ ] Quantization improvements

### [1.2.0] - TBD
- [ ] GPU optimization for other accelerators
- [ ] Improved error recovery
- [ ] Extended model support

### [2.0.0] - TBD
- [ ] Multi-machine distributed inference
- [ ] Custom model fine-tuning support
- [ ] Advanced profiling and tracing

---

## Notes

- **Week 1 (March 13-20, 2026)**: Script-based installation and deployment
- **Week 2 (March 20-27, 2026)**: PyPI package distribution (current)
- **Week 3 (March 27-Apr 3, 2026)**: MCP server implementation
- **Week 4+ (Apr 3+, 2026)**: Documentation, testing, production release
