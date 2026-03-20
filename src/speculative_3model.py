#!/usr/bin/env python3
"""
3-Model Pyramid Speculative Decoding Implementation
Phase 3 of momo-kibidango project

Architecture:
- Draft Model: Qwen2-0.5B (ultra-fast generation)
- Qualifier Model: Phi-2 2.7B (medium-quality filter)
- Target Model: Qwen2-7B-4bit (final verification)
"""

import os
import torch
import time
import json
import psutil
import argparse
import gc
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# Configuration
@dataclass
class ModelConfig:
    # Model IDs
    draft_model_id: str = "Qwen/Qwen2.5-0.5B-Instruct"
    qualifier_model_id: str = "microsoft/phi-2"  
    target_model_id: str = "Qwen/Qwen2.5-7B-Instruct"
    
    # Speculative parameters
    max_draft_tokens: int = 6  # Slightly more than Phase 2
    temperature: float = 0.7
    top_p: float = 0.9
    
    # Acceptance thresholds
    stage1_threshold: float = 0.10  # Draft → Qualifier (lenient)
    stage2_threshold: float = 0.03  # Qualifier → Target (strict)
    
    # Device configuration
    device: str = "mps" if torch.backends.mps.is_available() else "cpu"
    use_4bit: bool = True  # Quantize target model


class PyramidSpeculativeDecoder:
    """3-Model Pyramid Speculative Decoding implementation"""
    
    def __init__(self, config: ModelConfig):
        self.config = config
        self.device = config.device
        
        print("Loading 3-model pyramid architecture...")
        print(f"Device: {self.device}")
        
        # Load models in order of size for memory efficiency
        self._load_draft_model()
        self._check_memory("After draft model")
        
        self._load_qualifier_model()
        self._check_memory("After qualifier model")
        
        self._load_target_model()
        self._check_memory("After target model")
        
        # Load tokenizers
        self._load_tokenizers()
        
        print("✅ All models loaded successfully!")
        
    def _check_memory(self, stage: str):
        """Monitor memory usage at each stage"""
        if self.device == "cuda":
            allocated = torch.cuda.memory_allocated() / 1e9
            reserved = torch.cuda.memory_reserved() / 1e9
            print(f"{stage}: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved")
        else:
            # For MPS/CPU, use process memory
            process = psutil.Process()
            memory_gb = process.memory_info().rss / 1e9
            print(f"{stage}: {memory_gb:.2f}GB process memory")
            
    def _load_draft_model(self):
        """Load the fast draft model"""
        print(f"Loading draft model: {self.config.draft_model_id}")
        self.draft_model = AutoModelForCausalLM.from_pretrained(
            self.config.draft_model_id,
            torch_dtype=torch.float16 if self.device != "cpu" else torch.float32,
            low_cpu_mem_usage=True,
            trust_remote_code=True
        )
        if self.device != "cpu":
            self.draft_model = self.draft_model.to(self.device)
        self.draft_model.eval()
        
    def _load_qualifier_model(self):
        """Load the medium-quality qualifier model"""
        print(f"Loading qualifier model: {self.config.qualifier_model_id}")
        self.qualifier_model = AutoModelForCausalLM.from_pretrained(
            self.config.qualifier_model_id,
            torch_dtype=torch.float16 if self.device != "cpu" else torch.float32,
            low_cpu_mem_usage=True,
            trust_remote_code=True
        )
        if self.device != "cpu":
            self.qualifier_model = self.qualifier_model.to(self.device)
        self.qualifier_model.eval()
        
    def _load_target_model(self):
        """Load the target model with optional 4-bit quantization"""
        print(f"Loading target model: {self.config.target_model_id}")
        
        if self.config.use_4bit and self.device == "cuda":
            # 4-bit quantization for CUDA devices
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16
            )
            self.target_model = AutoModelForCausalLM.from_pretrained(
                self.config.target_model_id,
                quantization_config=bnb_config,
                low_cpu_mem_usage=True,
                trust_remote_code=True
            )
        else:
            # Full precision for MPS/CPU
            self.target_model = AutoModelForCausalLM.from_pretrained(
                self.config.target_model_id,
                torch_dtype=torch.float16 if self.device != "cpu" else torch.float32,
                low_cpu_mem_usage=True,
                trust_remote_code=True
            )
            if self.device != "cpu":
                self.target_model = self.target_model.to(self.device)
        
        self.target_model.eval()
        
    def _load_tokenizers(self):
        """Load tokenizers for each model"""
        print("Loading tokenizers...")
        
        # Draft and target use same tokenizer (Qwen family)
        self.draft_tokenizer = AutoTokenizer.from_pretrained(
            self.config.draft_model_id,
            trust_remote_code=True
        )
        self.target_tokenizer = self.draft_tokenizer  # Same for Qwen family
        
        # Phi-2 has its own tokenizer
        self.qualifier_tokenizer = AutoTokenizer.from_pretrained(
            self.config.qualifier_model_id,
            trust_remote_code=True
        )
        
        # Set padding tokens
        for tokenizer in [self.draft_tokenizer, self.qualifier_tokenizer]:
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
                
    def _map_tokens(self, tokens: List[int], from_tokenizer, to_tokenizer) -> List[int]:
        """Map tokens between different tokenizers"""
        if from_tokenizer == to_tokenizer:
            return tokens
            
        # Decode to text and re-encode
        text = from_tokenizer.decode(tokens, skip_special_tokens=True)
        mapped_tokens = to_tokenizer.encode(text, add_special_tokens=False)
        return mapped_tokens
        
    def draft_tokens(self, input_ids: torch.Tensor, n_tokens: int) -> Tuple[List[int], List[float]]:
        """Stage 1: Generate draft tokens using the smallest model"""
        draft_tokens = []
        draft_probs = []
        
        with torch.no_grad():
            current_ids = input_ids.clone()
            
            for _ in range(n_tokens):
                outputs = self.draft_model(current_ids)
                logits = outputs.logits[:, -1, :]
                
                # Apply temperature and softmax
                probs = torch.softmax(logits / self.config.temperature, dim=-1)
                
                # Sample token
                next_token = torch.multinomial(probs, num_samples=1).squeeze()
                token_id = next_token.item()
                token_prob = probs[0, token_id].item()
                
                draft_tokens.append(token_id)
                draft_probs.append(token_prob)
                
                # Append for next iteration
                current_ids = torch.cat([current_ids, next_token.unsqueeze(0).unsqueeze(0)], dim=1)
        
        return draft_tokens, draft_probs
        
    def qualify_tokens(self, input_ids: torch.Tensor, draft_tokens: List[int]) -> Tuple[List[int], List[bool]]:
        """Stage 2: Verify draft tokens with qualifier model"""
        # Map tokens from Qwen to Phi-2 tokenizer
        input_text = self.draft_tokenizer.decode(input_ids[0], skip_special_tokens=True)
        qualifier_input_ids = self.qualifier_tokenizer.encode(input_text, return_tensors="pt").to(self.device)
        
        qualified_tokens = []
        accept_decisions = []
        
        with torch.no_grad():
            current_ids = qualifier_input_ids
            
            for draft_token in draft_tokens:
                # Get qualifier model's prediction
                outputs = self.qualifier_model(current_ids)
                logits = outputs.logits[:, -1, :]
                probs = torch.softmax(logits / self.config.temperature, dim=-1)
                
                # Map draft token to qualifier tokenizer space
                draft_text = self.draft_tokenizer.decode([draft_token], skip_special_tokens=True)
                if draft_text:  # Skip empty tokens
                    qualifier_tokens = self.qualifier_tokenizer.encode(draft_text, add_special_tokens=False)
                    if qualifier_tokens:
                        qualifier_token = qualifier_tokens[0]
                        token_prob = probs[0, qualifier_token].item() if qualifier_token < probs.size(1) else 0.0
                        
                        # Stage 1 acceptance decision
                        if token_prob > self.config.stage1_threshold:
                            qualified_tokens.append(draft_token)
                            accept_decisions.append(True)
                            # Update context for next prediction
                            current_ids = torch.cat([current_ids, torch.tensor([[qualifier_token]], device=self.device)], dim=1)
                        else:
                            accept_decisions.append(False)
                            break
                else:
                    accept_decisions.append(False)
                    break
                    
        return qualified_tokens, accept_decisions
        
    def verify_tokens(self, input_ids: torch.Tensor, qualified_tokens: List[int]) -> Tuple[List[int], int]:
        """Stage 3: Final verification with target model"""
        accepted_tokens = []
        
        with torch.no_grad():
            # Prepare input with qualified tokens
            if qualified_tokens:
                qualified_ids = torch.tensor(qualified_tokens, device=self.device).unsqueeze(0)
                extended_ids = torch.cat([input_ids, qualified_ids], dim=1)
            else:
                extended_ids = input_ids
            
            # Get target model predictions
            outputs = self.target_model(extended_ids)
            
            # Check each qualified token
            for i, token in enumerate(qualified_tokens):
                target_logits = outputs.logits[:, input_ids.size(1) + i - 1, :]
                target_probs = torch.softmax(target_logits / self.config.temperature, dim=-1)
                
                # Stage 2 acceptance decision
                if target_probs[0, token] > self.config.stage2_threshold:
                    accepted_tokens.append(token)
                else:
                    # Reject and sample from target
                    new_token = torch.multinomial(target_probs, num_samples=1).squeeze().item()
                    accepted_tokens.append(new_token)
                    break
                    
        return accepted_tokens, len(accepted_tokens)
        
    def generate(self, prompt: str, max_length: int = 100) -> Dict:
        """Generate text using 3-model pyramid speculative decoding"""
        # Tokenize prompt
        input_ids = self.draft_tokenizer.encode(prompt, return_tensors="pt").to(self.device)
        prompt_length = input_ids.size(1)
        
        # Initialize metrics
        metrics = {
            "prompt": prompt,
            "start_time": time.time(),
            "tokens_generated": 0,
            "draft_attempts": 0,
            "stage1_accepted": 0,
            "stage2_accepted": 0,
            "memory_usage": []
        }
        
        generated_tokens = []
        
        while len(generated_tokens) < max_length:
            # Monitor memory
            self._check_memory(f"Iteration {len(generated_tokens)//self.config.max_draft_tokens}")
            if self.device == "mps":
                memory_gb = psutil.Process().memory_info().rss / 1e9
                metrics["memory_usage"].append(memory_gb)
            
            # Stage 1: Draft generation
            draft_tokens, draft_probs = self.draft_tokens(input_ids, self.config.max_draft_tokens)
            metrics["draft_attempts"] += len(draft_tokens)
            
            # Stage 2: Qualifier verification
            qualified_tokens, stage1_accepts = self.qualify_tokens(input_ids, draft_tokens)
            metrics["stage1_accepted"] += sum(stage1_accepts)
            
            # Stage 3: Target verification
            if qualified_tokens:
                accepted_tokens, n_accepted = self.verify_tokens(input_ids, qualified_tokens)
                metrics["stage2_accepted"] += n_accepted
            else:
                # If no tokens passed Stage 2, fall back to target model generation
                with torch.no_grad():
                    outputs = self.target_model(input_ids)
                    logits = outputs.logits[:, -1, :]
                    probs = torch.softmax(logits / self.config.temperature, dim=-1)
                    new_token = torch.multinomial(probs, num_samples=1).squeeze().item()
                    accepted_tokens = [new_token]
                    n_accepted = 1
            
            # Update generated tokens
            for token in accepted_tokens[:n_accepted]:
                generated_tokens.append(token)
                if token == self.draft_tokenizer.eos_token_id or len(generated_tokens) >= max_length:
                    break
                    
            # Update input_ids
            if accepted_tokens:
                new_tokens = torch.tensor(accepted_tokens[:n_accepted], device=self.device).unsqueeze(0)
                input_ids = torch.cat([input_ids, new_tokens], dim=1)
                
            # Check for EOS
            if generated_tokens and generated_tokens[-1] == self.draft_tokenizer.eos_token_id:
                break
                
        # Calculate metrics
        metrics["end_time"] = time.time()
        metrics["total_time"] = metrics["end_time"] - metrics["start_time"]
        metrics["tokens_generated"] = len(generated_tokens)
        metrics["throughput"] = metrics["tokens_generated"] / metrics["total_time"]
        
        # Acceptance rates
        metrics["stage1_acceptance_rate"] = (
            metrics["stage1_accepted"] / metrics["draft_attempts"] 
            if metrics["draft_attempts"] > 0 else 0
        )
        metrics["stage2_acceptance_rate"] = (
            metrics["stage2_accepted"] / metrics["stage1_accepted"]
            if metrics["stage1_accepted"] > 0 else 0
        )
        metrics["combined_acceptance_rate"] = (
            metrics["stage2_accepted"] / metrics["draft_attempts"]
            if metrics["draft_attempts"] > 0 else 0
        )
        
        # Memory metrics
        if metrics["memory_usage"]:
            metrics["peak_memory_gb"] = max(metrics["memory_usage"])
            metrics["avg_memory_gb"] = sum(metrics["memory_usage"]) / len(metrics["memory_usage"])
        else:
            metrics["peak_memory_gb"] = 0
            metrics["avg_memory_gb"] = 0
            
        # Decode generated text
        full_ids = torch.cat([
            input_ids[:, :prompt_length],
            torch.tensor(generated_tokens, device=self.device).unsqueeze(0)
        ], dim=1)
        generated_text = self.draft_tokenizer.decode(full_ids[0], skip_special_tokens=True)
        
        metrics["generated_text"] = generated_text
        
        return metrics
        
    def __del__(self):
        """Clean up models on deletion"""
        if hasattr(self, 'draft_model'):
            del self.draft_model
        if hasattr(self, 'qualifier_model'):
            del self.qualifier_model
        if hasattr(self, 'target_model'):
            del self.target_model
        gc.collect()
        if self.device == "cuda":
            torch.cuda.empty_cache()


def fallback_2model_generate(prompt: str, max_length: int = 100) -> Dict:
    """Fallback to proven 2-model implementation"""
    print("⚠️ Falling back to 2-model implementation...")
    from speculative_2model import SpeculativeDecoder, ModelConfig as Config2Model
    
    config = Config2Model(
        draft_model_id="Qwen/Qwen2.5-1.5B-Instruct",
        target_model_id="Qwen/Qwen2.5-7B-Instruct"
    )
    decoder = SpeculativeDecoder(config)
    return decoder.generate(prompt, max_length)


def fallback_1model_generate(prompt: str, max_length: int = 100) -> Dict:
    """Ultimate fallback to single model"""
    print("⚠️ Falling back to single model...")
    from transformers import pipeline
    
    generator = pipeline('text-generation', model='Qwen/Qwen2.5-7B-Instruct', device='mps')
    result = generator(prompt, max_length=max_length, return_full_text=True)
    
    return {
        "generated_text": result[0]['generated_text'],
        "throughput": 0,  # Not measured
        "memory_usage": 0
    }


def test_3model_loading():
    """Test that all 3 models load successfully"""
    print("Testing 3-model loading...")
    config = ModelConfig()
    
    try:
        decoder = PyramidSpeculativeDecoder(config)
        print("✅ All models loaded successfully!")
        
        # Test basic generation
        test_prompt = "Hello, world!"
        print(f"\nTesting generation with: '{test_prompt}'")
        
        # Quick test with short generation
        config.max_draft_tokens = 3
        result = decoder.generate(test_prompt, max_length=10)
        
        print(f"Generated: {result['generated_text']}")
        print(f"Throughput: {result['throughput']:.2f} tokens/sec")
        print(f"Memory: {result['peak_memory_gb']:.2f} GB")
        
        return True
    except Exception as e:
        print(f"❌ 3-model loading failed: {e}")
        return False


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="3-Model Pyramid Speculative Decoding")
    parser.add_argument("--test", action="store_true", help="Test model loading")
    parser.add_argument("--prompt", type=str, default="The future of AI is", help="Prompt to generate from")
    parser.add_argument("--max-length", type=int, default=50, help="Maximum tokens to generate")
    parser.add_argument("--fallback", action="store_true", help="Test fallback mechanism")
    args = parser.parse_args()
    
    if args.test:
        success = test_3model_loading()
        exit(0 if success else 1)
        
    if args.fallback:
        # Test fallback chain
        print("Testing fallback mechanism...")
        try:
            # Force memory error for testing
            raise MemoryError("Simulated OOM")
        except MemoryError:
            try:
                result = fallback_2model_generate(args.prompt, args.max_length)
            except:
                result = fallback_1model_generate(args.prompt, args.max_length)
        
        print(f"Fallback result: {result['generated_text']}")
        exit(0)
    
    # Normal generation
    config = ModelConfig()
    
    try:
        decoder = PyramidSpeculativeDecoder(config)
        
        print(f"\nGenerating from prompt: '{args.prompt}'")
        print("-" * 80)
        
        results = decoder.generate(args.prompt, max_length=args.max_length)
        
        print(f"\nGenerated text:\n{results['generated_text']}")
        print(f"\nMetrics:")
        print(f"  Throughput: {results['throughput']:.2f} tokens/sec")
        print(f"  Stage 1 acceptance: {results['stage1_acceptance_rate']:.2%}")
        print(f"  Stage 2 acceptance: {results['stage2_acceptance_rate']:.2%}")
        print(f"  Combined acceptance: {results['combined_acceptance_rate']:.2%}")
        print(f"  Peak memory: {results['peak_memory_gb']:.2f} GB")
        print(f"  Total time: {results['total_time']:.2f} seconds")
        
    except (RuntimeError, torch.cuda.OutOfMemoryError, MemoryError) as e:
        print(f"Memory error: {e}")
        # Try 2-model fallback
        try:
            results = fallback_2model_generate(args.prompt, args.max_length)
            print("Successfully used 2-model fallback")
        except:
            results = fallback_1model_generate(args.prompt, args.max_length)
            print("Successfully used 1-model fallback")


if __name__ == "__main__":
    main()