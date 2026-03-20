#!/usr/bin/env python3
"""
Quick test of 3-model pyramid implementation
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from speculative_3model import PyramidSpeculativeDecoder, ModelConfig, test_3model_loading

print("Running quick 3-model test...")
print("=" * 80)

# First test model loading
if test_3model_loading():
    print("\n✅ Model loading test passed!")
else:
    print("\n❌ Model loading test failed!")
    sys.exit(1)

print("\nTest completed successfully!")
print("Ready to run full benchmarks with: python scripts/benchmark_3model.py")