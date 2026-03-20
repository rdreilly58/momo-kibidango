#!/usr/bin/env python3
"""
Optimized 2-Model Speculative Decoding Implementation
Targets 1.8-2.2x speedup with better acceptance rates
"""

import torch
import time
import json
import os
import numpy as np
from typing import List, Tuple, Dict, Optional
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch.nn.functional as F

os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"


class OptimizedSpeculativeDecoder:
    """Optimized two-model speculative decoding"""
    
    def __init__(self, 
                 draft_model_id: str = "Qwen/Qwen2.5-0.5B-Instruct",
                 target_model_id: str = "Qwen/Qwen2.5-1.5B-Instruct",
                 device: Optional[str] = None):
        
        # Detect device
        if device:
            self.device = device
        elif torch.cuda.is_available():
            self.device = "cuda"
        elif torch.backends.mps.is_available():
            self.device = "mps"
        else:
            self.device = "cpu"
        
        print(f"Using device: {self.device}")
        
        # Load models
        self._load_models(draft_model_id, target_model_id)
        
        # Optimization parameters
        self.max_draft_len = 6  # Increased for better throughput
        self.temperature = 0.8  # Slightly higher for better diversity
        self.top_p = 0.95
        self.acceptance_threshold = 0.1  # Lower threshold for higher acceptance
        
    def _load_models(self, draft_id: str, target_id: str):
        """Load models with optimizations"""
        print(f"\nLoading draft model: {draft_id}")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(draft_id, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load draft model with optimizations
        self.draft_model = AutoModelForCausalLM.from_pretrained(
            draft_id,
            torch_dtype=torch.float16 if self.device != "cpu" else torch.float32,
            low_cpu_mem_usage=True,
            trust_remote_code=True
        )
        
        if self.device != "cpu":
            self.draft_model = self.draft_model.to(self.device)
        self.draft_model.eval()
        
        print(f"\nLoading target model: {target_id}")
        
        # Load target model
        self.target_model = AutoModelForCausalLM.from_pretrained(
            target_id,
            torch_dtype=torch.float16 if self.device != "cpu" else torch.float32,
            low_cpu_mem_usage=True,
            trust_remote_code=True
        )
        
        if self.device != "cpu":
            self.target_model = self.target_model.to(self.device)
        self.target_model.eval()
        
        print("Models loaded successfully!")
        
    @torch.no_grad()
    def _sample_token(self, logits: torch.Tensor, temperature: float = 1.0, top_p: float = 0.9) -> Tuple[int, torch.Tensor]:
        """Sample a token from logits with temperature and top-p"""
        # Apply temperature
        logits = logits / temperature
        
        # Convert to probabilities
        probs = F.softmax(logits, dim=-1)
        
        # Apply top-p (nucleus) sampling
        sorted_probs, sorted_indices = torch.sort(probs, descending=True)
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
        
        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probs > top_p
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0
        
        indices_to_remove = sorted_indices_to_remove.scatter(dim=-1, index=sorted_indices, src=sorted_indices_to_remove)
        probs[indices_to_remove] = 0.0
        
        # Renormalize
        probs = probs / probs.sum(dim=-1, keepdim=True)
        
        # Sample
        next_token = torch.multinomial(probs, num_samples=1).squeeze()
        
        return next_token.item(), probs
    
    @torch.no_grad()
    def generate_speculative(self, prompt: str, max_new_tokens: int = 100) -> Dict:
        """Optimized speculative decoding generation"""
        # Tokenize prompt
        inputs = self.tokenizer(prompt, return_tensors="pt", padding=True).to(self.device)
        input_ids = inputs.input_ids
        attention_mask = inputs.attention_mask
        
        # Metrics
        start_time = time.time()
        generated_ids = input_ids.clone()
        generated_mask = attention_mask.clone()
        
        n_drafted = 0
        n_accepted = 0
        n_iterations = 0
        
        # Generation loop
        while generated_ids.shape[1] - input_ids.shape[1] < max_new_tokens:
            n_iterations += 1
            
            # Draft phase - generate multiple tokens with draft model
            draft_ids = generated_ids.clone()
            draft_mask = generated_mask.clone()
            draft_tokens = []
            
            for _ in range(self.max_draft_len):
                outputs = self.draft_model(
                    input_ids=draft_ids,
                    attention_mask=draft_mask
                )
                logits = outputs.logits[:, -1, :]
                
                # Sample next token
                next_token, _ = self._sample_token(logits, self.temperature, self.top_p)
                draft_tokens.append(next_token)
                
                # Update draft sequence
                draft_ids = torch.cat([draft_ids, torch.tensor([[next_token]], device=self.device)], dim=1)
                draft_mask = torch.cat([draft_mask, torch.ones((1, 1), device=self.device)], dim=1)
                
                # Stop if EOS
                if next_token == self.tokenizer.eos_token_id:
                    break
            
            n_drafted += len(draft_tokens)
            
            # Verification phase - check draft tokens with target model
            # Run target model on the full draft sequence
            target_outputs = self.target_model(
                input_ids=draft_ids,
                attention_mask=draft_mask
            )
            
            # Verify each draft token
            n_accepted_this_round = 0
            for i, draft_token in enumerate(draft_tokens):
                # Get target model's distribution at this position
                target_logits = target_outputs.logits[:, generated_ids.shape[1] + i - 1, :]
                _, target_probs = self._sample_token(target_logits, self.temperature, self.top_p)
                
                # Accept if draft token has sufficient probability
                draft_prob = target_probs[0, draft_token].item()
                
                if draft_prob >= self.acceptance_threshold:
                    # Accept the token
                    generated_ids = torch.cat([
                        generated_ids, 
                        torch.tensor([[draft_token]], device=self.device)
                    ], dim=1)
                    generated_mask = torch.cat([
                        generated_mask,
                        torch.ones((1, 1), device=self.device)
                    ], dim=1)
                    n_accepted_this_round += 1
                    n_accepted += 1
                else:
                    # Reject and sample from target distribution
                    new_token, _ = self._sample_token(target_logits, self.temperature, self.top_p)
                    generated_ids = torch.cat([
                        generated_ids,
                        torch.tensor([[new_token]], device=self.device)
                    ], dim=1)
                    generated_mask = torch.cat([
                        generated_mask,
                        torch.ones((1, 1), device=self.device)
                    ], dim=1)
                    n_accepted += 1
                    break  # Stop verifying after first rejection
            
            # Check for EOS
            if generated_ids[0, -1].item() == self.tokenizer.eos_token_id:
                break
        
        # Calculate metrics
        end_time = time.time()
        generation_time = end_time - start_time
        tokens_generated = generated_ids.shape[1] - input_ids.shape[1]
        
        # Decode text
        generated_text = self.tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        
        return {
            "generated_text": generated_text,
            "tokens_generated": tokens_generated,
            "time_taken": generation_time,
            "throughput": tokens_generated / generation_time if generation_time > 0 else 0,
            "acceptance_rate": n_accepted / n_drafted if n_drafted > 0 else 0,
            "n_iterations": n_iterations,
            "n_drafted": n_drafted,
            "n_accepted": n_accepted
        }
    
    @torch.no_grad()
    def generate_baseline(self, prompt: str, max_new_tokens: int = 100) -> Dict:
        """Baseline generation with target model only"""
        # Tokenize
        inputs = self.tokenizer(prompt, return_tensors="pt", padding=True).to(self.device)
        
        start_time = time.time()
        
        # Generate
        outputs = self.target_model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=self.temperature,
            top_p=self.top_p,
            pad_token_id=self.tokenizer.pad_token_id
        )
        
        end_time = time.time()
        
        # Calculate metrics
        generation_time = end_time - start_time
        tokens_generated = outputs.shape[1] - inputs.input_ids.shape[1]
        
        # Decode
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        return {
            "generated_text": generated_text,
            "tokens_generated": tokens_generated,
            "time_taken": generation_time,
            "throughput": tokens_generated / generation_time if generation_time > 0 else 0
        }


def test_speedup():
    """Quick test to verify speedup"""
    print("="*80)
    print("OPTIMIZED SPECULATIVE DECODING TEST")
    print("="*80)
    
    decoder = OptimizedSpeculativeDecoder()
    
    test_prompt = "The key advantages of renewable energy include"
    max_tokens = 100
    
    # Baseline
    print("\n1. BASELINE (Target only)")
    baseline = decoder.generate_baseline(test_prompt, max_tokens)
    print(f"Throughput: {baseline['throughput']:.2f} tok/s")
    print(f"Generated: {baseline['generated_text'][:100]}...")
    
    # Speculative
    print("\n2. SPECULATIVE DECODING")
    speculative = decoder.generate_speculative(test_prompt, max_tokens)
    print(f"Throughput: {speculative['throughput']:.2f} tok/s")
    print(f"Acceptance rate: {speculative['acceptance_rate']:.2%}")
    print(f"Generated: {speculative['generated_text'][:100]}...")
    
    # Speedup
    speedup = speculative['throughput'] / baseline['throughput']
    print(f"\n📊 SPEEDUP: {speedup:.2f}x")
    
    # Save results
    os.makedirs("results", exist_ok=True)
    with open("results/optimized_test.json", "w") as f:
        json.dump({
            "baseline": baseline,
            "speculative": speculative,
            "speedup": speedup
        }, f, indent=2)
    
    return speedup


if __name__ == "__main__":
    speedup = test_speedup()
    print(f"\nTarget speedup: 1.8-2.2x")
    print(f"Achieved: {speedup:.2f}x {'✅' if 1.8 <= speedup <= 2.2 else '❌'}")