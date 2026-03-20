#!/usr/bin/env python3
"""
Speculative Decoding with 2 Models using Qwen2 Family
Phase 2 Implementation for momo-kibidango project

Draft Model: Qwen2-0.5B or Qwen2-1.5B
Target Model: Qwen2-7B-4bit (cached)
"""

import os
import torch
import time
import json
import psutil
import argparse
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass
from transformers import AutoModelForCausalLM, AutoTokenizer

# Configuration
@dataclass
class ModelConfig:
    draft_model_id: str = "Qwen/Qwen2.5-0.5B-Instruct"  # Start with smallest available
    target_model_id: str = "Qwen/Qwen2.5-7B-Instruct"  # Will load 4-bit version
    max_draft_tokens: int = 5
    temperature: float = 0.7
    top_p: float = 0.9
    device: str = "mps" if torch.backends.mps.is_available() else "cpu"  # Use Apple Silicon GPU


class SpeculativeDecoder:
    """Two-model speculative decoding implementation"""
    
    def __init__(self, config: ModelConfig):
        self.config = config
        self.device = config.device
        
        print("Loading models...")
        
        # Load draft model
        print(f"Loading draft model: {config.draft_model_id}")
        self.draft_model = AutoModelForCausalLM.from_pretrained(
            config.draft_model_id,
            torch_dtype=torch.float32,  # Use float32 for better compatibility
            low_cpu_mem_usage=True,
            trust_remote_code=True
        )
        if self.device != "cpu":
            self.draft_model = self.draft_model.to(self.device)
        
        # Load target model (we'll quantize later if needed)
        print(f"Loading target model: {config.target_model_id}")
        self.target_model = AutoModelForCausalLM.from_pretrained(
            config.target_model_id,
            torch_dtype=torch.float32,
            low_cpu_mem_usage=True,
            trust_remote_code=True
        )
        if self.device != "cpu":
            self.target_model = self.target_model.to(self.device)
        
        # Load tokenizer (same for both models in Qwen2 family)
        print("Loading tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            config.draft_model_id,
            trust_remote_code=True
        )
        
        # Set padding token if not set
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        print("Models loaded successfully!")
        
    def draft_tokens(self, input_ids: torch.Tensor, n_tokens: int) -> Tuple[List[int], torch.Tensor]:
        """Generate draft tokens using the smaller model"""
        draft_tokens = []
        draft_probs = []
        
        with torch.no_grad():
            current_ids = input_ids.clone()
            
            for _ in range(n_tokens):
                outputs = self.draft_model(current_ids)
                logits = outputs.logits[:, -1, :]
                
                # Apply temperature and top-p sampling
                probs = torch.softmax(logits / self.config.temperature, dim=-1)
                
                # Sample token
                next_token = torch.multinomial(probs, num_samples=1).squeeze()
                draft_tokens.append(next_token.item())
                draft_probs.append(probs)
                
                # Append token for next iteration
                current_ids = torch.cat([current_ids, next_token.unsqueeze(0).unsqueeze(0)], dim=1)
        
        return draft_tokens, torch.stack(draft_probs)
    
    def verify_tokens(self, input_ids: torch.Tensor, draft_tokens: List[int]) -> Tuple[List[int], int]:
        """Verify draft tokens using the larger model"""
        accepted_tokens = []
        
        with torch.no_grad():
            # Prepare input with draft tokens
            draft_ids = torch.tensor(draft_tokens, device=self.device).unsqueeze(0)
            extended_ids = torch.cat([input_ids, draft_ids], dim=1)
            
            # Get target model predictions
            outputs = self.target_model(extended_ids)
            
            # Check each draft token
            for i, draft_token in enumerate(draft_tokens):
                target_logits = outputs.logits[:, input_ids.size(1) + i - 1, :]
                target_probs = torch.softmax(target_logits / self.config.temperature, dim=-1)
                
                # Accept if draft token has reasonable probability
                if target_probs[0, draft_token] > 0.05:  # Acceptance threshold
                    accepted_tokens.append(draft_token)
                else:
                    # Reject and sample from target distribution
                    new_token = torch.multinomial(target_probs, num_samples=1).squeeze().item()
                    accepted_tokens.append(new_token)
                    break
        
        return accepted_tokens, len(accepted_tokens)
    
    def generate(self, prompt: str, max_length: int = 100) -> Dict:
        """Generate text using speculative decoding"""
        # Tokenize prompt
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)
        
        # Metrics tracking
        metrics = {
            "prompt": prompt,
            "start_time": time.time(),
            "tokens_generated": 0,
            "draft_attempts": 0,
            "accepted_tokens": 0,
            "memory_usage": []
        }
        
        generated_tokens = []
        
        while len(generated_tokens) < max_length:
            # Record memory usage
            if self.device == "cuda":
                memory_used = torch.cuda.memory_allocated() / 1e9  # GB
            elif self.device == "mps":
                # For Apple Silicon, use process memory
                memory_used = psutil.Process().memory_info().rss / 1e9  # GB
            else:
                memory_used = psutil.Process().memory_info().rss / 1e9  # GB
            metrics["memory_usage"].append(memory_used)
            
            # Draft phase
            draft_tokens, _ = self.draft_tokens(input_ids, self.config.max_draft_tokens)
            metrics["draft_attempts"] += len(draft_tokens)
            
            # Verification phase
            accepted, n_accepted = self.verify_tokens(input_ids, draft_tokens)
            metrics["accepted_tokens"] += n_accepted
            
            # Update generated tokens
            for token in accepted[:n_accepted]:
                generated_tokens.append(token)
                if token == self.tokenizer.eos_token_id or len(generated_tokens) >= max_length:
                    break
            
            # Update input_ids for next iteration
            if accepted:
                new_tokens = torch.tensor(accepted[:n_accepted], device=self.device).unsqueeze(0)
                input_ids = torch.cat([input_ids, new_tokens], dim=1)
            
            # Check for EOS
            if generated_tokens and generated_tokens[-1] == self.tokenizer.eos_token_id:
                break
        
        # Calculate final metrics
        metrics["end_time"] = time.time()
        metrics["total_time"] = metrics["end_time"] - metrics["start_time"]
        metrics["tokens_generated"] = len(generated_tokens)
        metrics["throughput"] = metrics["tokens_generated"] / metrics["total_time"]
        metrics["acceptance_rate"] = metrics["accepted_tokens"] / metrics["draft_attempts"] if metrics["draft_attempts"] > 0 else 0
        metrics["peak_memory_gb"] = max(metrics["memory_usage"])
        metrics["avg_memory_gb"] = sum(metrics["memory_usage"]) / len(metrics["memory_usage"])
        
        # Decode generated text
        full_ids = torch.cat([
            input_ids[:, :len(self.tokenizer.encode(prompt))],
            torch.tensor(generated_tokens, device=self.device).unsqueeze(0)
        ], dim=1)
        generated_text = self.tokenizer.decode(full_ids[0], skip_special_tokens=True)
        
        metrics["generated_text"] = generated_text
        
        return metrics


def test_model_loading():
    """Test that models load successfully"""
    print("Testing model loading...")
    config = ModelConfig()
    
    try:
        decoder = SpeculativeDecoder(config)
        print("✅ Models loaded successfully!")
        
        # Test tokenizer compatibility
        test_text = "Hello, world!"
        tokens = decoder.tokenizer.encode(test_text)
        decoded = decoder.tokenizer.decode(tokens)
        print(f"✅ Tokenizer test: '{test_text}' -> {tokens} -> '{decoded}'")
        
        return True
    except Exception as e:
        print(f"❌ Model loading failed: {e}")
        return False


def main():
    """Main entry point for testing"""
    parser = argparse.ArgumentParser(description="2-Model Speculative Decoding")
    parser.add_argument("--test", action="store_true", help="Run model loading test")
    parser.add_argument("--prompt", type=str, default="The future of AI is", help="Prompt to generate from")
    parser.add_argument("--max-length", type=int, default=50, help="Maximum tokens to generate")
    args = parser.parse_args()
    
    if args.test:
        success = test_model_loading()
        exit(0 if success else 1)
    
    # Run generation
    config = ModelConfig()
    decoder = SpeculativeDecoder(config)
    
    print(f"\nGenerating from prompt: '{args.prompt}'")
    print("-" * 80)
    
    results = decoder.generate(args.prompt, max_length=args.max_length)
    
    print(f"\nGenerated text:\n{results['generated_text']}")
    print(f"\nMetrics:")
    print(f"  Throughput: {results['throughput']:.2f} tokens/sec")
    print(f"  Acceptance rate: {results['acceptance_rate']:.2%}")
    print(f"  Peak memory: {results['peak_memory_gb']:.2f} GB")
    print(f"  Total time: {results['total_time']:.2f} seconds")


if __name__ == "__main__":
    main()