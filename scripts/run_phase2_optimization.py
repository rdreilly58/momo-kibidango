#!/usr/bin/env python3
"""
Phase 2 Optimization Runner
Tests multiple model configurations to find optimal pairing
"""

import os
import sys
import json
import time
from datetime import datetime

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def run_optimization_tests():
    """Run optimization tests with different model pairings"""
    
    print("="*80)
    print("PHASE 2 OPTIMIZATION - Finding Optimal Model Pairing")
    print("="*80)
    
    configurations = [
        {
            "name": "Previous (Failed)",
            "draft": "Qwen/Qwen2.5-0.5B-Instruct",
            "target": "Qwen/Qwen2.5-1.5B-Instruct",
            "expected": "0.8x (too small draft)"
        },
        {
            "name": "Optimized (Phase 2)",
            "draft": "Qwen/Qwen2.5-1.5B-Instruct", 
            "target": "Qwen/Qwen2.5-3B-Instruct",  # Using 3B instead of 7B for faster testing
            "expected": "1.5-2.0x (better match)"
        }
    ]
    
    results = {
        "optimization_date": datetime.now().isoformat(),
        "configurations": []
    }
    
    for config in configurations:
        print(f"\nTesting configuration: {config['name']}")
        print(f"  Draft: {config['draft']}")
        print(f"  Target: {config['target']}")
        print(f"  Expected: {config['expected']}")
        
        # Import here to reload with new config
        from src.speculative_2model_minimal import MinimalSpeculativeDecoder
        
        try:
            # Initialize decoder with config
            decoder = MinimalSpeculativeDecoder(
                draft_model_path=config['draft'],
                target_model_path=config['target']
            )
            
            # Test on representative prompt
            test_prompt = "The key to successful speculative decoding is"
            
            # Baseline
            print("\n  Running baseline...")
            baseline = decoder.simple_generate(test_prompt, max_tokens=50)
            baseline_throughput = baseline['throughput']
            
            # Speculative
            print("  Running speculative...")
            spec = decoder.speculative_generate(test_prompt, max_tokens=50)
            spec_throughput = spec['throughput']
            acceptance_rate = spec['acceptance_rate']
            
            speedup = spec_throughput / baseline_throughput
            
            print(f"\n  Results:")
            print(f"    Baseline: {baseline_throughput:.2f} tok/s")
            print(f"    Speculative: {spec_throughput:.2f} tok/s")
            print(f"    Acceptance rate: {acceptance_rate:.2%}")
            print(f"    Speedup: {speedup:.2f}x")
            print(f"    Memory: {decoder.get_memory_usage():.2f} GB")
            
            config_result = {
                "config": config,
                "baseline_throughput": baseline_throughput,
                "speculative_throughput": spec_throughput,
                "acceptance_rate": acceptance_rate,
                "speedup": speedup,
                "memory_gb": decoder.get_memory_usage(),
                "status": "success"
            }
            
            # Clean up models to free memory
            del decoder
            import gc
            gc.collect()
            
        except Exception as e:
            print(f"\n  ❌ Failed: {str(e)}")
            config_result = {
                "config": config,
                "status": "failed",
                "error": str(e)
            }
        
        results["configurations"].append(config_result)
    
    # Find best configuration
    successful = [c for c in results["configurations"] if c["status"] == "success"]
    if successful:
        best = max(successful, key=lambda x: x.get("speedup", 0))
        results["best_configuration"] = best["config"]["name"]
        results["best_speedup"] = best.get("speedup", 0)
        
        print("\n" + "="*80)
        print("OPTIMIZATION RESULTS")
        print("="*80)
        print(f"Best configuration: {results['best_configuration']}")
        print(f"Best speedup: {results['best_speedup']:.2f}x")
        
        if results['best_speedup'] >= 1.5:
            print("\n✅ Target speedup achieved! Ready for full benchmarks.")
            results["recommendation"] = "PROCEED with optimized configuration"
        else:
            print("\n⚠️  Target speedup not achieved. Further optimization needed.")
            results["recommendation"] = "NEED larger target model (7B) or different approach"
    
    # Save results
    os.makedirs("results", exist_ok=True)
    with open("results/phase2_optimization_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to: results/phase2_optimization_results.json")
    
    return results

if __name__ == "__main__":
    results = run_optimization_tests()