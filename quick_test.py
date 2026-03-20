#!/usr/bin/env python3
"""Quick test for model pairing without full download"""

import torch
from transformers import AutoTokenizer, AutoConfig

def test_model_compatibility():
    """Test if the model pairing makes sense"""
    print("Testing model compatibility...")
    
    draft_model = "Qwen/Qwen2.5-1.5B-Instruct"
    target_model = "Qwen/Qwen2.5-7B-Instruct"
    
    # Check model configs
    print(f"\nDraft model: {draft_model}")
    draft_config = AutoConfig.from_pretrained(draft_model)
    print(f"  - Hidden size: {draft_config.hidden_size}")
    print(f"  - Num layers: {draft_config.num_hidden_layers}")
    print(f"  - Vocab size: {draft_config.vocab_size}")
    
    print(f"\nTarget model: {target_model}")
    target_config = AutoConfig.from_pretrained(target_model)
    print(f"  - Hidden size: {target_config.hidden_size}")
    print(f"  - Num layers: {target_config.num_hidden_layers}")  
    print(f"  - Vocab size: {target_config.vocab_size}")
    
    # Check tokenizer compatibility
    print("\nChecking tokenizer compatibility...")
    draft_tokenizer = AutoTokenizer.from_pretrained(draft_model, trust_remote_code=True)
    target_tokenizer = AutoTokenizer.from_pretrained(target_model, trust_remote_code=True)
    
    test_text = "Hello, world!"
    draft_tokens = draft_tokenizer.encode(test_text)
    target_tokens = target_tokenizer.encode(test_text)
    
    if draft_tokens == target_tokens:
        print("✅ Tokenizers are compatible!")
    else:
        print("❌ Tokenizers are NOT compatible!")
        print(f"   Draft tokens: {draft_tokens}")
        print(f"   Target tokens: {target_tokens}")
        
    # Estimate memory usage
    print("\nEstimated memory usage (fp16):")
    draft_params = draft_config.num_hidden_layers * draft_config.hidden_size * draft_config.hidden_size * 4 / 1e9
    target_params = target_config.num_hidden_layers * target_config.hidden_size * target_config.hidden_size * 4 / 1e9
    
    print(f"  - Draft (1.5B): ~{draft_params * 2:.1f} GB")
    print(f"  - Target (7B): ~{target_params * 2:.1f} GB")
    print(f"  - Total: ~{(draft_params + target_params) * 2:.1f} GB")
    
    print("\n⚠️  Note: Actual 7B model may use 14-16GB in fp16")
    print("   Consider using model sharding or CPU offloading if memory limited")

if __name__ == "__main__":
    test_model_compatibility()