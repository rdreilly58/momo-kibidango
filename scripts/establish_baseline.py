#!/usr/bin/env python3
"""
Establish single-model baseline performance
This will give us the reference throughput to compare against
"""

import torch
import time
import json
import os
from transformers import AutoModelForCausalLM, AutoTokenizer

def get_baseline_throughput():
    """Measure baseline throughput with Qwen2.5-7B model"""
    
    # Load model
    model_id = "Qwen/Qwen2.5-1.5B-Instruct"  # Using 1.5B for faster testing
    print(f"Loading baseline model: {model_id}")
    
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16 if device != "cpu" else torch.float32,
        low_cpu_mem_usage=True,
        trust_remote_code=True
    )
    
    if device != "cpu":
        model = model.to(device)
    
    print(f"Model loaded on {device}")
    
    # Test prompts
    test_prompts = [
        "The future of artificial intelligence is",
        "Once upon a time in a distant galaxy",
        "def fibonacci(n):",
        "The economic implications of climate change include"
    ]
    
    results = []
    
    for prompt in test_prompts:
        print(f"\nTesting: '{prompt}'")
        
        # Tokenize
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        
        # Warm up
        with torch.no_grad():
            _ = model.generate(**inputs, max_new_tokens=10, do_sample=False)
        
        # Actual benchmark
        start_time = time.time()
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=100,
                do_sample=True,
                temperature=0.7,
                top_p=0.9
            )
        end_time = time.time()
        
        # Calculate metrics
        tokens_generated = outputs.shape[1] - inputs['input_ids'].shape[1]
        time_taken = end_time - start_time
        throughput = tokens_generated / time_taken
        
        result = {
            "prompt": prompt,
            "tokens_generated": tokens_generated,
            "time_taken": time_taken,
            "throughput": throughput
        }
        results.append(result)
        
        print(f"  Throughput: {throughput:.2f} tokens/sec")
    
    # Calculate average
    avg_throughput = sum(r["throughput"] for r in results) / len(results)
    
    print(f"\n{'='*60}")
    print(f"BASELINE PERFORMANCE ESTABLISHED")
    print(f"Average throughput: {avg_throughput:.2f} tokens/sec")
    print(f"{'='*60}")
    
    # Save results
    baseline_data = {
        "model": model_id,
        "device": device,
        "results": results,
        "average_throughput": avg_throughput
    }
    
    os.makedirs("results", exist_ok=True)
    with open("results/baseline_performance.json", "w") as f:
        json.dump(baseline_data, f, indent=2)
    
    return avg_throughput


if __name__ == "__main__":
    get_baseline_throughput()