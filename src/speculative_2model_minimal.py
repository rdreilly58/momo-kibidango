#!/usr/bin/env python3
"""
Minimal Speculative Decoding with 2 Models using Qwen2 Family
Phase 2 Implementation for momo-kibidango project

This is a simplified version that focuses on getting the models loaded and basic inference working.
"""

import os
import torch
import time
import json
import psutil
from typing import List, Dict, Optional
from transformers import AutoModelForCausalLM, AutoTokenizer

# Disable warning about unauthenticated requests
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"


class MinimalSpeculativeDecoder:
    """Simplified two-model speculative decoding"""
    
    def __init__(self, draft_model_path: Optional[str] = None, target_model_path: Optional[str] = None):
        # Default to Qwen2.5 models - OPTIMIZED PAIRING FOR PHASE 2
        self.draft_model_id = draft_model_path or "Qwen/Qwen2.5-1.5B-Instruct"  # UPDATED: 1.5B draft for better match
        self.target_model_id = target_model_path or "Qwen/Qwen2.5-7B-Instruct"  # FIXED: Use 7B target as intended
        
        # Note: Expected memory usage with fp16:
        # - Draft (1.5B): ~3GB
        # - Target (7B): ~14GB  
        # - Total: ~17GB (may exceed 12GB target without quantization)
        
        # Detect device
        if torch.cuda.is_available():
            self.device = "cuda"
        elif torch.backends.mps.is_available():
            self.device = "mps" 
        else:
            self.device = "cpu"
        
        print(f"Using device: {self.device}")
        
        # Load models
        self.load_models()
        
    def load_models(self):
        """Load the draft and target models"""
        print(f"\nLoading draft model: {self.draft_model_id}")
        start_time = time.time()
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.draft_model_id,
            trust_remote_code=True
        )
        
        # Load draft model
        self.draft_model = AutoModelForCausalLM.from_pretrained(
            self.draft_model_id,
            torch_dtype=torch.float16 if self.device != "cpu" else torch.float32,
            low_cpu_mem_usage=True,
            trust_remote_code=True
        )
        
        # Move to device if needed
        if self.device != "cpu":
            self.draft_model = self.draft_model.to(self.device)
            
        print(f"Draft model loaded in {time.time() - start_time:.2f}s")
        
        # Load target model
        print(f"\nLoading target model: {self.target_model_id}")
        start_time = time.time()
        
        self.target_model = AutoModelForCausalLM.from_pretrained(
            self.target_model_id,
            torch_dtype=torch.float16 if self.device != "cpu" else torch.float32,
            low_cpu_mem_usage=True,
            trust_remote_code=True
        )
        
        if self.device != "cpu":
            self.target_model = self.target_model.to(self.device)
            
        print(f"Target model loaded in {time.time() - start_time:.2f}s")
        print(f"\nMemory usage: {self.get_memory_usage():.2f} GB")
        
    def get_memory_usage(self) -> float:
        """Get current memory usage in GB"""
        if self.device == "cuda":
            return torch.cuda.memory_allocated() / 1e9
        else:
            return psutil.Process().memory_info().rss / 1e9
            
    def simple_generate(self, prompt: str, max_tokens: int = 50) -> Dict:
        """Simple generation without speculative decoding for testing"""
        print(f"\nGenerating with target model only...")
        start_time = time.time()
        
        # Tokenize
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        # Generate
        with torch.no_grad():
            outputs = self.target_model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                do_sample=True,
                temperature=0.7,
                top_p=0.9
            )
        
        # Decode
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        end_time = time.time()
        tokens_generated = outputs.shape[1] - inputs['input_ids'].shape[1]
        
        return {
            "generated_text": generated_text,
            "tokens_generated": tokens_generated,
            "time_taken": end_time - start_time,
            "throughput": tokens_generated / (end_time - start_time),
            "memory_gb": self.get_memory_usage()
        }
        
    def speculative_generate(self, prompt: str, max_tokens: int = 50, draft_len: int = 4) -> Dict:
        """Basic speculative decoding implementation"""
        print(f"\nGenerating with speculative decoding...")
        start_time = time.time()
        
        # Tokenize prompt
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        input_ids = inputs['input_ids']
        
        generated_ids = input_ids.clone()
        total_accepted = 0
        total_drafted = 0
        
        while generated_ids.shape[1] - input_ids.shape[1] < max_tokens:
            # Draft phase - generate multiple tokens with small model
            with torch.no_grad():
                draft_outputs = self.draft_model.generate(
                    generated_ids,
                    max_new_tokens=draft_len,
                    do_sample=True,
                    temperature=0.7,
                    return_dict_in_generate=True,
                    output_scores=True
                )
            
            draft_ids = draft_outputs.sequences
            draft_new_tokens = draft_ids[:, generated_ids.shape[1]:]
            total_drafted += draft_new_tokens.shape[1]
            
            # Verification phase - check with target model
            accepted = 0
            with torch.no_grad():
                # Get target model logits for the draft sequence
                target_outputs = self.target_model(draft_ids)
                target_logits = target_outputs.logits
                
                # Simple acceptance criterion: accept if draft token is in top-k of target
                for i in range(draft_new_tokens.shape[1]):
                    pos = generated_ids.shape[1] + i - 1
                    if pos >= 0 and pos < target_logits.shape[1]:
                        # Get top-k tokens from target model
                        top_k = torch.topk(target_logits[:, pos, :], k=10).indices
                        
                        if i < draft_new_tokens.shape[1] and draft_new_tokens[:, i] in top_k:
                            accepted += 1
                        else:
                            break
            
            # Accept verified tokens
            if accepted > 0:
                generated_ids = torch.cat([generated_ids, draft_new_tokens[:, :accepted]], dim=1)
                total_accepted += accepted
            else:
                # If no tokens accepted, generate one with target model
                with torch.no_grad():
                    target_out = self.target_model.generate(
                        generated_ids,
                        max_new_tokens=1,
                        do_sample=True,
                        temperature=0.7
                    )
                generated_ids = target_out
            
            # Check for EOS
            if self.tokenizer.eos_token_id in generated_ids[0, -draft_len:]:
                break
        
        # Decode final output
        generated_text = self.tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        
        end_time = time.time()
        total_time = end_time - start_time
        tokens_generated = generated_ids.shape[1] - input_ids.shape[1]
        
        return {
            "generated_text": generated_text,
            "tokens_generated": tokens_generated,
            "time_taken": total_time,
            "throughput": tokens_generated / total_time,
            "memory_gb": self.get_memory_usage(),
            "acceptance_rate": total_accepted / total_drafted if total_drafted > 0 else 0,
            "total_accepted": total_accepted,
            "total_drafted": total_drafted
        }


def test_loading():
    """Test that models load successfully"""
    print("=" * 80)
    print("TESTING MODEL LOADING")
    print("=" * 80)
    
    try:
        decoder = MinimalSpeculativeDecoder()
        print("\n✅ Models loaded successfully!")
        
        # Test tokenizer
        test_text = "Hello, world!"
        tokens = decoder.tokenizer.encode(test_text)
        decoded = decoder.tokenizer.decode(tokens)
        print(f"✅ Tokenizer test: '{test_text}' -> {tokens} -> '{decoded}'")
        
        return True
    except Exception as e:
        print(f"\n❌ Error loading models: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_benchmark(prompt: str = "The future of AI is", max_tokens: int = 50):
    """Run a simple benchmark comparing regular vs speculative generation"""
    print("\n" + "=" * 80)
    print("RUNNING BENCHMARK")
    print("=" * 80)
    
    # Initialize decoder
    decoder = MinimalSpeculativeDecoder()
    
    # Run regular generation
    print("\n1. BASELINE (Target Model Only)")
    baseline_results = decoder.simple_generate(prompt, max_tokens)
    print(f"Generated: {baseline_results['generated_text']}")
    print(f"Throughput: {baseline_results['throughput']:.2f} tokens/sec")
    print(f"Memory: {baseline_results['memory_gb']:.2f} GB")
    
    # Run speculative generation
    print("\n2. SPECULATIVE DECODING")
    spec_results = decoder.speculative_generate(prompt, max_tokens)
    print(f"Generated: {spec_results['generated_text']}")
    print(f"Throughput: {spec_results['throughput']:.2f} tokens/sec")
    print(f"Memory: {spec_results['memory_gb']:.2f} GB")
    print(f"Acceptance rate: {spec_results['acceptance_rate']:.2%}")
    
    # Calculate speedup
    speedup = spec_results['throughput'] / baseline_results['throughput']
    print(f"\n📊 SPEEDUP: {speedup:.2f}x")
    
    # Save results
    results = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "prompt": prompt,
        "baseline": baseline_results,
        "speculative": spec_results,
        "speedup": speedup
    }
    
    os.makedirs("results", exist_ok=True)
    with open("results/minimal_test.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✅ Results saved to results/minimal_test.json")
    return results


if __name__ == "__main__":
    import sys
    
    if "--test" in sys.argv:
        success = test_loading()
        sys.exit(0 if success else 1)
    elif "--benchmark" in sys.argv:
        run_benchmark()
    else:
        # Default: run both test and benchmark
        if test_loading():
            run_benchmark()