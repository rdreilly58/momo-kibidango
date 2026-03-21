"""
Momo-Kibidango: Speculative Decoding Framework for Apple Silicon and Beyond

A high-performance inference framework implementing speculative decoding
to accelerate LLM inference using draft models for faster token generation.
"""

__version__ = "1.0.0"
__author__ = "Robert Reilly"
__email__ = "robert.reilly@reillydesignstudio.com"
__license__ = "Apache-2.0"

try:
    from .speculative_2model import SpeculativeDecoder, ModelConfig
    from .monitoring import PerformanceMonitor
    from .production_hardening import ProductionHardener
    
    __all__ = [
        "SpeculativeDecoder",
        "ModelConfig",
        "PerformanceMonitor",
        "ProductionHardener",
    ]
except ImportError:
    # Graceful degradation if optional dependencies missing
    __all__ = []
