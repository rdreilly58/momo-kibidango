#!/usr/bin/env python3
"""Quick test to verify model loading and memory usage"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.speculative_2model_minimal import MinimalSpeculativeDecoder

def test_loading():
    print("="*80)
    print("Testing Model Loading - Phase 2 Optimization")
    print("="*80)
    
    print("\nExpected models:")
    print("- Draft: Qwen/Qwen2.5-1.5B-Instruct")
    print("- Target: Qwen/Qwen2.5-7B-Instruct")
    
    try:
        decoder = MinimalSpeculativeDecoder()
        print("\n✅ Models loaded successfully!")
        
        # Test basic generation
        test_prompt = "The key to successful speculative decoding is"
        print(f"\nTest prompt: '{test_prompt}'")
        
        result = decoder.simple_generate(test_prompt, max_tokens=20)
        print(f"\nGenerated text: {result['generated_text']}")
        print(f"Memory usage: {result['memory_gb']:.2f} GB")
        print(f"Throughput: {result['throughput']:.2f} tokens/sec")
        
        return True
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_loading()
    exit(0 if success else 1)