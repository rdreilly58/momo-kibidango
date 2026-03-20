#!/usr/bin/env python3
"""
Comprehensive Benchmark Suite for 2-Model Speculative Decoding
Phase 2 Testing for momo-kibidango project

Tests 10 diverse scenarios to validate performance and quality.
"""

import os
import sys
import json
import time
import psutil
import torch
from typing import List, Dict, Any
from datetime import datetime

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.speculative_2model_minimal import MinimalSpeculativeDecoder


class Benchmark2Model:
    """Benchmark suite for 2-model speculative decoding"""
    
    def __init__(self):
        self.test_scenarios = [
            # Core 5 scenarios
            {
                "name": "math_reasoning",
                "prompt": "Let's solve this step by step: If a train travels 120 miles in 2 hours, and then increases its speed by 25%, how far will it travel in the next 3 hours?",
                "category": "logic-heavy",
                "max_tokens": 150
            },
            {
                "name": "creative_writing",
                "prompt": "Write a vivid description of a sunset over a futuristic city where buildings are made of crystallized light:",
                "category": "generation-quality",
                "max_tokens": 200
            },
            {
                "name": "code_generation",
                "prompt": "Write a Python function that implements binary search on a sorted list and returns the index of the target element:",
                "category": "precision",
                "max_tokens": 150
            },
            {
                "name": "analysis",
                "prompt": "Analyze the economic implications of universal basic income on labor markets, considering both potential benefits and drawbacks:",
                "category": "reasoning-depth",
                "max_tokens": 200
            },
            {
                "name": "simple_qa",
                "prompt": "What is the capital of France? Explain why it became the capital.",
                "category": "straightforward",
                "max_tokens": 50
            },
            # Additional 5 diverse scenarios
            {
                "name": "technical_explanation",
                "prompt": "Explain how a transformer neural network architecture works, focusing on the attention mechanism:",
                "category": "technical",
                "max_tokens": 150
            },
            {
                "name": "dialogue",
                "prompt": "Customer: My package hasn't arrived and it's been 2 weeks.\nSupport Agent:",
                "category": "conversational",
                "max_tokens": 100
            },
            {
                "name": "summarization",
                "prompt": "Summarize the following in 2-3 sentences: Machine learning has revolutionized numerous fields including healthcare, finance, and transportation. In healthcare, ML algorithms can detect diseases earlier than traditional methods. Financial institutions use ML for fraud detection and algorithmic trading. Self-driving cars rely on ML for perception and decision-making. However, challenges remain including bias in algorithms, data privacy concerns, and the need for interpretability.",
                "category": "compression",
                "max_tokens": 80
            },
            {
                "name": "translation",
                "prompt": "Translate to Spanish: 'The early bird catches the worm, but the second mouse gets the cheese.'",
                "category": "linguistic",
                "max_tokens": 50
            },
            {
                "name": "instruction_following",
                "prompt": "List exactly 5 benefits of regular exercise. Format each as a bullet point starting with an action verb.",
                "category": "structured-output",
                "max_tokens": 100
            }
        ]
        
        self.results = {
            "metadata": {
                "timestamp": datetime.now().isoformat(),
                "platform": sys.platform,
                "python_version": sys.version,
                "torch_version": torch.__version__,
                "device": self._get_device_info()
            },
            "scenarios": [],
            "summary": {}
        }
    
    def _get_device_info(self):
        """Get device information"""
        if torch.cuda.is_available():
            return {
                "type": "cuda",
                "name": torch.cuda.get_device_name(),
                "memory_gb": torch.cuda.get_device_properties(0).total_memory / 1e9
            }
        elif torch.backends.mps.is_available():
            return {
                "type": "mps",
                "name": "Apple Silicon",
                "memory_gb": psutil.virtual_memory().total / 1e9
            }
        else:
            return {
                "type": "cpu",
                "name": "CPU",
                "memory_gb": psutil.virtual_memory().total / 1e9
            }
    
    def run_single_scenario(self, decoder: MinimalSpeculativeDecoder, scenario: Dict) -> Dict:
        """Run a single test scenario"""
        print(f"\n{'='*80}")
        print(f"Running: {scenario['name']} ({scenario['category']})")
        print(f"{'='*80}")
        
        results = {
            "scenario": scenario,
            "baseline": None,
            "speculative": None,
            "speedup": None,
            "quality_check": None
        }
        
        try:
            # Run baseline (target model only)
            print("\n1. Running baseline...")
            baseline = decoder.simple_generate(scenario["prompt"], scenario["max_tokens"])
            results["baseline"] = baseline
            
            # Run speculative decoding
            print("\n2. Running speculative decoding...")
            speculative = decoder.speculative_generate(scenario["prompt"], scenario["max_tokens"])
            results["speculative"] = speculative
            
            # Calculate speedup
            speedup = speculative["throughput"] / baseline["throughput"]
            results["speedup"] = speedup
            
            # Simple quality check (length similarity)
            len_baseline = len(baseline["generated_text"].split())
            len_spec = len(speculative["generated_text"].split())
            quality_score = 1.0 - abs(len_baseline - len_spec) / max(len_baseline, 1)
            
            results["quality_check"] = {
                "length_similarity": quality_score,
                "baseline_length": len_baseline,
                "speculative_length": len_spec
            }
            
            print(f"\n📊 Results:")
            print(f"   Baseline throughput: {baseline['throughput']:.2f} tok/s")
            print(f"   Speculative throughput: {speculative['throughput']:.2f} tok/s")
            print(f"   Speedup: {speedup:.2f}x")
            print(f"   Acceptance rate: {speculative.get('acceptance_rate', 0):.2%}")
            print(f"   Quality score: {quality_score:.2f}")
            
        except Exception as e:
            print(f"\n❌ Error in scenario {scenario['name']}: {e}")
            results["error"] = str(e)
            
        return results
    
    def run_all_scenarios(self):
        """Run all benchmark scenarios"""
        print("="*80)
        print("2-MODEL SPECULATIVE DECODING BENCHMARK")
        print("="*80)
        
        # Initialize decoder once
        print("\nInitializing models...")
        start_init = time.time()
        decoder = MinimalSpeculativeDecoder()
        init_time = time.time() - start_init
        print(f"Models initialized in {init_time:.2f}s")
        
        # Run all scenarios
        all_speedups = []
        all_acceptance_rates = []
        all_memory = []
        
        for i, scenario in enumerate(self.test_scenarios):
            print(f"\n[{i+1}/{len(self.test_scenarios)}] {scenario['name']}")
            
            result = self.run_single_scenario(decoder, scenario)
            self.results["scenarios"].append(result)
            
            if result.get("speedup"):
                all_speedups.append(result["speedup"])
                all_acceptance_rates.append(result["speculative"].get("acceptance_rate", 0))
                all_memory.append(result["speculative"]["memory_gb"])
        
        # Calculate summary statistics
        if all_speedups:
            self.results["summary"] = {
                "initialization_time": init_time,
                "average_speedup": sum(all_speedups) / len(all_speedups),
                "min_speedup": min(all_speedups),
                "max_speedup": max(all_speedups),
                "average_acceptance_rate": sum(all_acceptance_rates) / len(all_acceptance_rates),
                "peak_memory_gb": max(all_memory),
                "average_memory_gb": sum(all_memory) / len(all_memory),
                "successful_scenarios": len(all_speedups),
                "total_scenarios": len(self.test_scenarios)
            }
            
            # Check success criteria
            avg_speedup = self.results["summary"]["average_speedup"]
            peak_memory = self.results["summary"]["peak_memory_gb"]
            
            self.results["summary"]["phase2_criteria"] = {
                "target_speedup": "1.8-2.2x",
                "achieved_speedup": f"{avg_speedup:.2f}x",
                "speedup_met": 1.8 <= avg_speedup <= 2.2,
                "target_memory": "<12GB",
                "achieved_memory": f"{peak_memory:.2f}GB",
                "memory_met": peak_memory < 12,
                "overall_success": (1.8 <= avg_speedup <= 2.2) and (peak_memory < 12)
            }
        
        return self.results
    
    def save_results(self, filename: str = "phase2_benchmark.json"):
        """Save benchmark results to file"""
        os.makedirs("results", exist_ok=True)
        filepath = os.path.join("results", filename)
        
        with open(filepath, "w") as f:
            json.dump(self.results, f, indent=2)
        
        print(f"\n✅ Results saved to: {filepath}")
        
    def print_summary(self):
        """Print a summary of the benchmark results"""
        if not self.results.get("summary"):
            print("\n❌ No summary available - benchmarks may have failed")
            return
            
        summary = self.results["summary"]
        criteria = summary.get("phase2_criteria", {})
        
        print("\n" + "="*80)
        print("BENCHMARK SUMMARY")
        print("="*80)
        
        print(f"\nPerformance Metrics:")
        print(f"  • Average Speedup: {summary['average_speedup']:.2f}x")
        print(f"  • Speedup Range: {summary['min_speedup']:.2f}x - {summary['max_speedup']:.2f}x")
        print(f"  • Average Acceptance Rate: {summary['average_acceptance_rate']:.2%}")
        
        print(f"\nMemory Usage:")
        print(f"  • Peak Memory: {summary['peak_memory_gb']:.2f} GB")
        print(f"  • Average Memory: {summary['average_memory_gb']:.2f} GB")
        
        print(f"\nPhase 2 Success Criteria:")
        print(f"  • Speedup Target: {criteria.get('target_speedup', 'N/A')} → {criteria.get('achieved_speedup', 'N/A')} {'✅' if criteria.get('speedup_met') else '❌'}")
        print(f"  • Memory Target: {criteria.get('target_memory', 'N/A')} → {criteria.get('achieved_memory', 'N/A')} {'✅' if criteria.get('memory_met') else '❌'}")
        
        print(f"\n{'✅ PHASE 2 CRITERIA MET!' if criteria.get('overall_success') else '❌ PHASE 2 CRITERIA NOT MET'}")


def main():
    """Main entry point"""
    benchmark = Benchmark2Model()
    
    # Run all benchmarks
    results = benchmark.run_all_scenarios()
    
    # Save results
    benchmark.save_results()
    
    # Print summary
    benchmark.print_summary()
    
    # Return exit code based on success
    success = results.get("summary", {}).get("phase2_criteria", {}).get("overall_success", False)
    return 0 if success else 1


if __name__ == "__main__":
    exit(main())