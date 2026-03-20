#!/usr/bin/env python3
"""
4-bit Quantized Speculative Decoding with 2 Models using Qwen2 Family
Phase 2 Implementation with Memory Optimization

Draft Model: Qwen2-1.5B (fp16)
Target Model: Qwen2-7B (4-bit quantized)
"""

import os
import torch
import time
import json
import psutil
from typing import List, Dict, Optional
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# Disable warning about unauthenticated requests
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"


class SpeculativeDecoder4bit:
    """Two-model speculative decoding with 4-bit target model"""
    
    def __init__(self, draft_model_path: Optional[str] = None, target_model_path: Optional[str] = None):
        # Optimized model pairing for Phase 2
        self.draft_model_id = draft_model_path or "Qwen/Qwen2.5-1.5B-Instruct"
        self.target_model_id = target_model_path or "Qwen/Qwen2.5-7B-Instruct"
        
        # Detect device
        if torch.cuda.is_available():
            self.device = "cuda"
        elif torch.backends.mps.is_available():
            self.device = "mps" 
        else:
            self.device = "cpu"
        
        print(f"Using device: {self.device}")
        print(f"Phase 2 Optimization: 1.5B draft → 7B-4bit target")
        
        # Load models
        self.load_models()
        
    def load_models(self):
        """Load the draft and target models with optimized memory usage"""
        print(f"\nLoading draft model: {self.draft_model_id}")
        start_time = time.time()
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.draft_model_id,
            trust_remote_code=True
        )
        
        # Load draft model in fp16
        self.draft_model = AutoModelForCausalLM.from_pretrained(
            self.draft_model_id,
            torch_dtype=torch.float16 if self.device != "cpu" else torch.float32,
            low_cpu_mem_usage=True,
            trust_remote_code=True
        )
        
        if self.device != "cpu":
            self.draft_model = self.draft_model.to(self.device)
            
        print(f"Draft model loaded in {time.time() - start_time:.2f}s")
        print(f"Memory usage after draft: {self.get_memory_usage():.2f} GB")
        
        # Load target model with 4-bit quantization
        print(f"\nLoading target model with 4-bit quantization: {self.target_model_id}")
        start_time = time.time()
        
        # 4-bit configuration
        if self.device == "cuda":
            # CUDA supports bitsandbytes 4-bit
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16
            )
            
            self.target_model = AutoModelForCausalLM.from_pretrained(
                self.target_model_id,
                quantization_config=bnb_config,
                low_cpu_mem_usage=True,
                trust_remote_code=True
            )
        else:
            # For MPS/CPU, use fp16 but warn about memory
            print("⚠️  4-bit quantization not available on MPS/CPU, using fp16")
            print("   Memory usage may exceed 12GB target!")
            
            self.target_model = AutoModelForCausalLM.from_pretrained(
                self.target_model_id,
                torch_dtype=torch.float16 if self.device != "cpu" else torch.float32,
                low_cpu_mem_usage=True,
                trust_remote_code=True
            )
            
            if self.device != "cpu":
                self.target_model = self.target_model.to(self.device)
            
        print(f"Target model loaded in {time.time() - start_time:.2f}s")
        print(f"Total memory usage: {self.get_memory_usage():.2f} GB")
        
    def get_memory_usage(self) -> float:
        """Get current memory usage in GB"""
        if self.device == "cuda":
            return torch.cuda.memory_allocated() / 1e9
        else:
            return psutil.Process().memory_info().rss / 1e9
            
    def generate_draft_tokens(self, input_ids: torch.Tensor, n_tokens: int = 5) -> List[int]:
        """Generate draft tokens using the smaller model"""
        draft_tokens = []
        current_ids = input_ids.clone()
        
        with torch.no_grad():
            for _ in range(n_tokens):
                outputs = self.draft_model(current_ids)
                logits = outputs.logits[:, -1, :]
                
                # Simple greedy decoding for draft
                next_token = torch.argmax(logits, dim=-1)
                draft_tokens.append(next_token.item())
                
                # Update input for next iteration
                current_ids = torch.cat([current_ids, next_token.unsqueeze(0)], dim=1)
                
        return draft_tokens
    
    def verify_and_accept(self, input_ids: torch.Tensor, draft_tokens: List[int]) -> tuple:
        """Verify draft tokens with target model and accept/reject"""
        accepted_tokens = []
        
        with torch.no_grad():
            # Prepare input with all draft tokens
            draft_ids = torch.tensor(draft_tokens, device=self.device).unsqueeze(0)
            extended_ids = torch.cat([input_ids, draft_ids], dim=1)
            
            # Get target model logits
            outputs = self.target_model(extended_ids)
            
            # Check each draft token
            for i, draft_token in enumerate(draft_tokens):
                position = input_ids.size(1) + i
                target_logits = outputs.logits[:, position - 1, :]
                
                # Get target's distribution
                target_probs = torch.softmax(target_logits, dim=-1)
                
                # Accept if draft token has reasonable probability
                # Tuned threshold for better acceptance
                if target_probs[0, draft_token] > 0.03:  # Lower threshold for better acceptance
                    accepted_tokens.append(draft_token)
                else:
                    # Reject and sample from target
                    new_token = torch.multinomial(target_probs, num_samples=1).squeeze().item()
                    accepted_tokens.append(new_token)
                    break
                    
        return accepted_tokens, len([t for t in accepted_tokens if t in draft_tokens])
    
    def speculative_generate(self, prompt: str, max_tokens: int = 100) -> Dict:
        """Generate text using speculative decoding"""
        # Tokenize prompt
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        input_ids = inputs.input_ids
        
        # Track metrics
        start_time = time.time()
        generated_tokens = []
        total_draft = 0
        total_accepted = 0
        memory_samples = []
        
        while len(generated_tokens) < max_tokens:
            # Sample memory periodically
            if len(generated_tokens) % 10 == 0:
                memory_samples.append(self.get_memory_usage())
            
            # Generate draft tokens
            draft_tokens = self.generate_draft_tokens(input_ids, n_tokens=5)
            total_draft += len(draft_tokens)
            
            # Verify with target model
            accepted, n_accepted_from_draft = self.verify_and_accept(input_ids, draft_tokens)
            total_accepted += n_accepted_from_draft
            
            # Update generated tokens and input_ids
            for token in accepted:
                generated_tokens.append(token)
                if token == self.tokenizer.eos_token_id or len(generated_tokens) >= max_tokens:
                    break
                    
            # Update input_ids for next iteration
            if accepted:
                new_tokens = torch.tensor(accepted, device=self.device).unsqueeze(0)
                input_ids = torch.cat([input_ids, new_tokens], dim=1)
                
            # Check for EOS
            if generated_tokens and generated_tokens[-1] == self.tokenizer.eos_token_id:
                break
                
        # Calculate metrics
        end_time = time.time()
        total_time = end_time - start_time
        
        # Decode result
        full_ids = torch.cat([
            inputs.input_ids,
            torch.tensor(generated_tokens, device=self.device).unsqueeze(0)
        ], dim=1)
        generated_text = self.tokenizer.decode(full_ids[0], skip_special_tokens=True)
        
        return {
            "generated_text": generated_text,
            "tokens_generated": len(generated_tokens),
            "throughput": len(generated_tokens) / total_time,
            "acceptance_rate": total_accepted / total_draft if total_draft > 0 else 0,
            "total_time": total_time,
            "memory_gb": max(memory_samples) if memory_samples else self.get_memory_usage(),
            "avg_memory_gb": sum(memory_samples) / len(memory_samples) if memory_samples else self.get_memory_usage()
        }
    
    def simple_generate(self, prompt: str, max_tokens: int = 100) -> Dict:
        """Baseline generation with target model only"""
        # Tokenize prompt
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        # Generate
        start_time = time.time()
        
        with torch.no_grad():
            outputs = self.target_model.generate(
                inputs.input_ids,
                max_new_tokens=max_tokens,
                do_sample=False,  # Greedy for fair comparison
                pad_token_id=self.tokenizer.pad_token_id
            )
        
        end_time = time.time()
        total_time = end_time - start_time
        
        # Calculate metrics
        generated_ids = outputs[0][inputs.input_ids.size(1):]
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        return {
            "generated_text": generated_text,
            "tokens_generated": len(generated_ids),
            "throughput": len(generated_ids) / total_time,
            "total_time": total_time,
            "memory_gb": self.get_memory_usage()
        }


def main():
    """Test the 4-bit implementation"""
    print("="*80)
    print("Phase 2 Optimization Test - 4-bit Target Model")
    print("="*80)
    
    decoder = SpeculativeDecoder4bit()
    
    # Test prompt
    test_prompt = "The key to achieving high speedup in speculative decoding is"
    
    print(f"\nTest prompt: '{test_prompt}'")
    print("\nRunning baseline generation...")
    baseline = decoder.simple_generate(test_prompt, max_tokens=50)
    
    print(f"\nBaseline throughput: {baseline['throughput']:.2f} tokens/sec")
    print(f"Baseline memory: {baseline['memory_gb']:.2f} GB")
    
    print("\nRunning speculative generation...")
    speculative = decoder.speculative_generate(test_prompt, max_tokens=50)
    
    print(f"\nSpeculative throughput: {speculative['throughput']:.2f} tokens/sec")
    print(f"Acceptance rate: {speculative['acceptance_rate']:.2%}")
    print(f"Speedup: {speculative['throughput'] / baseline['throughput']:.2f}x")
    print(f"Peak memory: {speculative['memory_gb']:.2f} GB")
    
    print("\nGenerated text:")
    print(speculative['generated_text'])


if __name__ == "__main__":
    main()