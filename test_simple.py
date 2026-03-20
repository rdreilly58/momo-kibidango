#!/usr/bin/env python3
"""Simple test to verify models can be loaded"""

import torch
print(f"PyTorch version: {torch.__version__}")
print(f"MPS available: {torch.backends.mps.is_available()}")
print(f"Device: {'mps' if torch.backends.mps.is_available() else 'cpu'}")

# Try loading a tiny model first
from transformers import AutoTokenizer, AutoModelForCausalLM

print("\nTrying to load Qwen2.5-0.5B-Instruct...")
try:
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct", trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2.5-0.5B-Instruct",
        torch_dtype=torch.float32,
        low_cpu_mem_usage=True,
        trust_remote_code=True
    )
    print("✅ Model loaded successfully!")
    
    # Test tokenization
    text = "Hello world"
    tokens = tokenizer.encode(text)
    decoded = tokenizer.decode(tokens)
    print(f"Tokenization test: '{text}' -> {tokens} -> '{decoded}'")
    
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()