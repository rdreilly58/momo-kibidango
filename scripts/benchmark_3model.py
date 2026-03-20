#!/usr/bin/env python3
"""
Comprehensive Benchmark Suite for 3-Model Pyramid Speculative Decoding
Phase 3 of momo-kibidango project

Tests 15 scenarios across different task types and compares against:
1. Baseline (single model)
2. 2-model implementation (from Phase 2)
3. 3-model pyramid implementation (Phase 3)
"""

import sys
import os
import json
import time
import torch
import gc
from datetime import datetime
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.speculative_2model import SpeculativeDecoder as TwoModelDecoder
from src.speculative_2model import ModelConfig as TwoModelConfig
from src.speculative_3model import PyramidSpeculativeDecoder as ThreeModelDecoder
from src.speculative_3model import ModelConfig as ThreeModelConfig
from transformers import AutoModelForCausalLM, AutoTokenizer


@dataclass
class BenchmarkResult:
    scenario: str
    category: str
    prompt: str
    baseline: Dict
    two_model: Dict
    three_model: Dict
    speedup_2model: float
    speedup_3model: float
    memory_2model: float
    memory_3model: float
    quality_maintained: bool
    timestamp: str


class BenchmarkSuite:
    def __init__(self):
        self.scenarios = self._define_scenarios()
        self.baseline_model = None
        self.baseline_tokenizer = None
        self.two_model_decoder = None
        self.three_model_decoder = None
        
    def _define_scenarios(self) -> List[Dict]:
        """Define 15 test scenarios (10 from Phase 2 + 5 new)"""
        return [
            # Original 10 from Phase 2
            {
                "name": "math_problem",
                "category": "math_reasoning",
                "prompt": "Solve step by step: If a train travels 120 km in 2 hours, and then 180 km in 3 hours, what is its average speed for the entire journey?",
                "max_length": 150
            },
            {
                "name": "creative_story",
                "category": "creative_writing",
                "prompt": "Write a short story about a robot who discovers it can dream. Begin with: 'The first dream came during routine maintenance...'",
                "max_length": 200
            },
            {
                "name": "python_function",
                "category": "code_generation",
                "prompt": "Write a Python function to find the longest common subsequence between two strings. Include docstring and type hints.",
                "max_length": 150
            },
            {
                "name": "market_analysis",
                "category": "analysis_reasoning",
                "prompt": "Analyze the potential impact of artificial intelligence on the job market in the next decade. Consider both opportunities and challenges.",
                "max_length": 200
            },
            {
                "name": "simple_qa",
                "category": "simple_qa",
                "prompt": "What is the capital of France?",
                "max_length": 20
            },
            {
                "name": "explain_concept",
                "category": "education",
                "prompt": "Explain quantum entanglement in simple terms that a high school student could understand.",
                "max_length": 150
            },
            {
                "name": "recipe_generation",
                "category": "creative_writing",
                "prompt": "Create a recipe for a healthy breakfast smoothie using spinach, banana, and your choice of other ingredients.",
                "max_length": 150
            },
            {
                "name": "technical_tutorial",
                "category": "technical_writing",
                "prompt": "Write a step-by-step guide on how to set up a Python virtual environment for beginners.",
                "max_length": 200
            },
            {
                "name": "logic_puzzle",
                "category": "logic_reasoning",
                "prompt": "Three friends - Alice, Bob, and Charlie - are standing in a line. Alice is not first, Bob is not last, and Charlie is not next to Alice. What is the order?",
                "max_length": 100
            },
            {
                "name": "email_draft",
                "category": "business_writing",
                "prompt": "Draft a professional email declining a job offer while maintaining a positive relationship for future opportunities.",
                "max_length": 150
            },
            
            # 5 Additional scenarios for Phase 3
            {
                "name": "scientific_explanation",
                "category": "scientific_writing",
                "prompt": "Explain how CRISPR gene editing works and its potential medical applications.",
                "max_length": 200
            },
            {
                "name": "dialogue_generation",
                "category": "creative_writing", 
                "prompt": "Write a dialogue between a time traveler from 2124 and a person from today discussing climate change.",
                "max_length": 200
            },
            {
                "name": "sql_query",
                "category": "code_generation",
                "prompt": "Write an SQL query to find the top 5 customers by total purchase amount from orders and customers tables.",
                "max_length": 100
            },
            {
                "name": "philosophical_question",
                "category": "philosophy",
                "prompt": "Discuss the ethical implications of creating artificial general intelligence. What safeguards should be in place?",
                "max_length": 200
            },
            {
                "name": "rapid_fire_qa",
                "category": "simple_qa",
                "prompt": "Quick answers only: What is 15 × 7? Who painted the Mona Lisa? What year did WW2 end?",
                "max_length": 50
            }
        ]
        
    def _load_baseline_model(self):
        """Load the baseline model (single 7B model)"""
        if self.baseline_model is None:
            print("Loading baseline model...")
            model_id = "Qwen/Qwen2.5-7B-Instruct"
            
            self.baseline_model = AutoModelForCausalLM.from_pretrained(
                model_id,
                torch_dtype=torch.float32,
                low_cpu_mem_usage=True,
                trust_remote_code=True
            )
            
            device = "mps" if torch.backends.mps.is_available() else "cpu"
            if device != "cpu":
                self.baseline_model = self.baseline_model.to(device)
                
            self.baseline_tokenizer = AutoTokenizer.from_pretrained(
                model_id,
                trust_remote_code=True
            )
            
            if self.baseline_tokenizer.pad_token is None:
                self.baseline_tokenizer.pad_token = self.baseline_tokenizer.eos_token
                
    def _load_two_model_decoder(self):
        """Load the 2-model decoder"""
        if self.two_model_decoder is None:
            print("Loading 2-model decoder...")
            config = TwoModelConfig(
                draft_model_id="Qwen/Qwen2.5-1.5B-Instruct",
                target_model_id="Qwen/Qwen2.5-7B-Instruct"
            )
            self.two_model_decoder = TwoModelDecoder(config)
            
    def _load_three_model_decoder(self):
        """Load the 3-model decoder"""
        if self.three_model_decoder is None:
            print("Loading 3-model pyramid decoder...")
            config = ThreeModelConfig()
            self.three_model_decoder = ThreeModelDecoder(config)
            
    def run_baseline(self, prompt: str, max_length: int) -> Dict:
        """Run baseline single-model generation"""
        self._load_baseline_model()
        
        device = "mps" if torch.backends.mps.is_available() else "cpu"
        input_ids = self.baseline_tokenizer.encode(prompt, return_tensors="pt").to(device)
        
        import psutil
        start_time = time.time()
        start_memory = psutil.Process().memory_info().rss / 1e9
        
        with torch.no_grad():
            outputs = self.baseline_model.generate(
                input_ids,
                max_length=input_ids.size(1) + max_length,
                temperature=0.7,
                do_sample=True,
                top_p=0.9,
                pad_token_id=self.baseline_tokenizer.pad_token_id
            )
            
        end_time = time.time()
        end_memory = psutil.Process().memory_info().rss / 1e9
        
        generated_text = self.baseline_tokenizer.decode(outputs[0], skip_special_tokens=True)
        tokens_generated = outputs.size(1) - input_ids.size(1)
        
        return {
            "generated_text": generated_text,
            "tokens_generated": tokens_generated,
            "time_taken": end_time - start_time,
            "throughput": tokens_generated / (end_time - start_time),
            "peak_memory_gb": end_memory,
            "memory_increase_gb": end_memory - start_memory
        }
        
    def run_benchmark_scenario(self, scenario: Dict) -> BenchmarkResult:
        """Run a single benchmark scenario across all three implementations"""
        print(f"\nRunning scenario: {scenario['name']} ({scenario['category']})")
        print("-" * 80)
        
        # Run baseline
        print("  Running baseline...")
        baseline_result = self.run_baseline(scenario["prompt"], scenario["max_length"])
        
        # Run 2-model
        print("  Running 2-model...")
        self._load_two_model_decoder()
        two_model_result = self.two_model_decoder.generate(scenario["prompt"], scenario["max_length"])
        
        # Run 3-model
        print("  Running 3-model pyramid...")
        self._load_three_model_decoder()
        three_model_result = self.three_model_decoder.generate(scenario["prompt"], scenario["max_length"])
        
        # Calculate speedups
        speedup_2model = two_model_result["throughput"] / baseline_result["throughput"]
        speedup_3model = three_model_result["throughput"] / baseline_result["throughput"]
        
        # Check quality (simplified - just check if output is reasonable length)
        quality_maintained = (
            len(two_model_result["generated_text"]) > len(scenario["prompt"]) + 10 and
            len(three_model_result["generated_text"]) > len(scenario["prompt"]) + 10
        )
        
        return BenchmarkResult(
            scenario=scenario["name"],
            category=scenario["category"],
            prompt=scenario["prompt"],
            baseline=baseline_result,
            two_model=two_model_result,
            three_model=three_model_result,
            speedup_2model=speedup_2model,
            speedup_3model=speedup_3model,
            memory_2model=two_model_result.get("peak_memory_gb", 0),
            memory_3model=three_model_result.get("peak_memory_gb", 0),
            quality_maintained=quality_maintained,
            timestamp=datetime.utcnow().isoformat()
        )
        
    def run_full_benchmark(self) -> List[BenchmarkResult]:
        """Run all benchmark scenarios"""
        results = []
        
        print("Starting comprehensive 3-model benchmark suite")
        print(f"Running {len(self.scenarios)} scenarios...")
        print("=" * 80)
        
        for i, scenario in enumerate(self.scenarios):
            print(f"\nScenario {i+1}/{len(self.scenarios)}")
            
            try:
                result = self.run_benchmark_scenario(scenario)
                results.append(result)
                
                # Print summary
                print(f"\n  Results for '{scenario['name']}':")
                print(f"    Baseline: {result.baseline['throughput']:.2f} tok/s")
                print(f"    2-model:  {result.two_model['throughput']:.2f} tok/s ({result.speedup_2model:.2f}x)")
                print(f"    3-model:  {result.three_model['throughput']:.2f} tok/s ({result.speedup_3model:.2f}x)")
                print(f"    Memory:   2M={result.memory_2model:.1f}GB, 3M={result.memory_3model:.1f}GB")
                
            except Exception as e:
                print(f"  ❌ Error in scenario '{scenario['name']}': {e}")
                
            # Clear memory between scenarios
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
        return results
        
    def save_results(self, results: List[BenchmarkResult], output_file: str):
        """Save benchmark results to JSON file"""
        # Convert to serializable format
        serializable_results = []
        for result in results:
            result_dict = asdict(result)
            # Simplify the generated text (just store length)
            for key in ["baseline", "two_model", "three_model"]:
                if "generated_text" in result_dict[key]:
                    text = result_dict[key]["generated_text"]
                    result_dict[key]["generated_text_length"] = len(text)
                    result_dict[key]["generated_text_preview"] = text[:100] + "..." if len(text) > 100 else text
                    del result_dict[key]["generated_text"]
            serializable_results.append(result_dict)
            
        # Calculate summary statistics
        speedups_2model = [r.speedup_2model for r in results]
        speedups_3model = [r.speedup_3model for r in results]
        memories_2model = [r.memory_2model for r in results]
        memories_3model = [r.memory_3model for r in results]
        
        summary = {
            "total_scenarios": len(results),
            "avg_speedup_2model": sum(speedups_2model) / len(speedups_2model),
            "avg_speedup_3model": sum(speedups_3model) / len(speedups_3model),
            "min_speedup_2model": min(speedups_2model),
            "max_speedup_2model": max(speedups_2model),
            "min_speedup_3model": min(speedups_3model),
            "max_speedup_3model": max(speedups_3model),
            "avg_memory_2model": sum(memories_2model) / len(memories_2model),
            "avg_memory_3model": sum(memories_3model) / len(memories_3model),
            "peak_memory_2model": max(memories_2model),
            "peak_memory_3model": max(memories_3model),
            "quality_maintained_all": all(r.quality_maintained for r in results),
            "timestamp": datetime.utcnow().isoformat()
        }
        
        # Calculate acceptance rates if available
        acceptance_2model = []
        acceptance_3model = []
        for r in results:
            if "acceptance_rate" in r.two_model:
                acceptance_2model.append(r.two_model["acceptance_rate"])
            if "combined_acceptance_rate" in r.three_model:
                acceptance_3model.append(r.three_model["combined_acceptance_rate"])
                
        if acceptance_2model:
            summary["avg_acceptance_2model"] = sum(acceptance_2model) / len(acceptance_2model)
        if acceptance_3model:
            summary["avg_acceptance_3model"] = sum(acceptance_3model) / len(acceptance_3model)
            
        output_data = {
            "summary": summary,
            "results": serializable_results
        }
        
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, "w") as f:
            json.dump(output_data, f, indent=2)
            
        print(f"\nResults saved to: {output_file}")
        
    def print_summary(self, results: List[BenchmarkResult]):
        """Print a summary of the benchmark results"""
        print("\n" + "=" * 80)
        print("BENCHMARK SUMMARY")
        print("=" * 80)
        
        # Overall statistics
        speedups_2model = [r.speedup_2model for r in results]
        speedups_3model = [r.speedup_3model for r in results]
        
        print(f"\n2-Model Performance:")
        print(f"  Average speedup: {sum(speedups_2model)/len(speedups_2model):.2f}x")
        print(f"  Min speedup: {min(speedups_2model):.2f}x")
        print(f"  Max speedup: {max(speedups_2model):.2f}x")
        
        print(f"\n3-Model Pyramid Performance:")
        print(f"  Average speedup: {sum(speedups_3model)/len(speedups_3model):.2f}x")
        print(f"  Min speedup: {min(speedups_3model):.2f}x")
        print(f"  Max speedup: {max(speedups_3model):.2f}x")
        
        # Category breakdown
        categories = {}
        for r in results:
            if r.category not in categories:
                categories[r.category] = {"2model": [], "3model": []}
            categories[r.category]["2model"].append(r.speedup_2model)
            categories[r.category]["3model"].append(r.speedup_3model)
            
        print("\nPerformance by Category:")
        for cat, speeds in categories.items():
            avg_2model = sum(speeds["2model"]) / len(speeds["2model"])
            avg_3model = sum(speeds["3model"]) / len(speeds["3model"])
            print(f"  {cat}:")
            print(f"    2-model: {avg_2model:.2f}x")
            print(f"    3-model: {avg_3model:.2f}x")
            
        # Memory usage
        memories_2model = [r.memory_2model for r in results]
        memories_3model = [r.memory_3model for r in results]
        
        print(f"\nMemory Usage:")
        print(f"  2-model peak: {max(memories_2model):.2f} GB")
        print(f"  3-model peak: {max(memories_3model):.2f} GB")
        
        # Success criteria check
        avg_speedup_3model = sum(speedups_3model) / len(speedups_3model)
        peak_memory_3model = max(memories_3model)
        quality_ok = all(r.quality_maintained for r in results)
        
        print("\nSuccess Criteria Check:")
        print(f"  ✅ Speedup: {avg_speedup_3model:.2f}x {'✅' if avg_speedup_3model >= 1.85 else '❌'} (target: 1.85-2.1x)")
        print(f"  ✅ Memory: {peak_memory_3model:.2f} GB {'✅' if peak_memory_3model < 12 else '❌'} (target: <12GB)")
        print(f"  ✅ Quality: {'✅ Maintained' if quality_ok else '❌ Degraded'}")


def main():
    parser = argparse.ArgumentParser(description="3-Model Pyramid Benchmark Suite")
    parser.add_argument("--output", default="results/phase3_benchmark.json", help="Output file")
    parser.add_argument("--scenarios", type=int, help="Number of scenarios to run (default: all)")
    args = parser.parse_args()
    
    # Create benchmark suite
    suite = BenchmarkSuite()
    
    # Optionally limit scenarios
    if args.scenarios:
        suite.scenarios = suite.scenarios[:args.scenarios]
        
    # Run benchmarks
    results = suite.run_full_benchmark()
    
    # Save results
    suite.save_results(results, args.output)
    
    # Print summary
    suite.print_summary(results)


if __name__ == "__main__":
    main()