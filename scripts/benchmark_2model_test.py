#!/usr/bin/env python3
"""
Test benchmark with smaller models to verify the approach
Uses Qwen2.5-0.5B → Qwen2.5-1.5B for quick testing
"""

import os
import sys
import json
import time
import torch
import psutil
from datetime import datetime

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.speculative_2model_minimal import MinimalSpeculativeDecoder

def quick_test():
    """Quick test with smaller model pair"""
    print("="*80)
    print("QUICK TEST - Verifying Speculative Decoding Implementation")
    print("Using 0.5B → 1.5B for faster testing (will scale to 1.5B → 7B)")
    print("="*80)
    
    # Override with smaller models for testing
    decoder = MinimalSpeculativeDecoder(
        draft_model_path="Qwen/Qwen2.5-0.5B-Instruct",
        target_model_path="Qwen/Qwen2.5-1.5B-Instruct"
    )
    
    test_prompts = [
        "The capital of France is",
        "Machine learning algorithms can",
        "To solve climate change, we need to"
    ]
    
    results = []
    
    for prompt in test_prompts:
        print(f"\nTesting prompt: '{prompt}'")
        
        # Baseline
        print("  Running baseline...")
        baseline = decoder.simple_generate(prompt, max_tokens=30)
        print(f"    Baseline throughput: {baseline['throughput']:.2f} tok/s")
        
        # Speculative
        print("  Running speculative...")
        spec = decoder.speculative_generate(prompt, max_tokens=30)
        print(f"    Speculative throughput: {spec['throughput']:.2f} tok/s")
        print(f"    Acceptance rate: {spec['acceptance_rate']:.2%}")
        print(f"    Speedup: {spec['throughput'] / baseline['throughput']:.2f}x")
        
        results.append({
            "prompt": prompt,
            "baseline_throughput": baseline['throughput'],
            "speculative_throughput": spec['throughput'],
            "acceptance_rate": spec['acceptance_rate'],
            "speedup": spec['throughput'] / baseline['throughput']
        })
    
    # Summary
    avg_speedup = sum(r['speedup'] for r in results) / len(results)
    avg_acceptance = sum(r['acceptance_rate'] for r in results) / len(results)
    
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"Average speedup: {avg_speedup:.2f}x")
    print(f"Average acceptance rate: {avg_acceptance:.2%}")
    print(f"Memory usage: {decoder.get_memory_usage():.2f} GB")
    
    if avg_speedup > 1.2:
        print("\n✅ Implementation working! Ready to scale to 1.5B → 7B")
    else:
        print("\n⚠️  Low speedup detected. May need tuning for 1.5B → 7B")
        
    # Save test results
    with open("results/quick_test_results.json", "w") as f:
        json.dump({
            "test_config": "0.5B → 1.5B (verification test)",
            "results": results,
            "summary": {
                "avg_speedup": avg_speedup,
                "avg_acceptance_rate": avg_acceptance,
                "memory_gb": decoder.get_memory_usage()
            }
        }, f, indent=2)

if __name__ == "__main__":
    os.makedirs("results", exist_ok=True)
    quick_test()