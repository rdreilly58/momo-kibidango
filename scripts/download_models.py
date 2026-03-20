#!/usr/bin/env python3
"""
Download all required models for speculative decoding
Ensures models are cached before production deployment
"""

import os
import sys
import time
from pathlib import Path
from typing import List, Dict, Any

from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer,
    BitsAndBytesConfig
)
import torch


def print_status(message: str):
    """Print status message with timestamp"""
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {message}")


def get_model_info() -> List[Dict[str, Any]]:
    """Get list of models to download"""
    return [
        {
            "id": "Qwen/Qwen2.5-0.5B-Instruct",
            "name": "Draft Model (0.5B)",
            "size": "1.2GB",
            "quantize": False
        },
        {
            "id": "Qwen/Qwen2.5-1.5B-Instruct", 
            "name": "Alternative Draft Model (1.5B)",
            "size": "3.5GB",
            "quantize": False
        },
        {
            "id": "microsoft/phi-2",
            "name": "Qualifier Model (2.7B)",
            "size": "5.5GB", 
            "quantize": False
        },
        {
            "id": "Qwen/Qwen2.5-7B-Instruct",
            "name": "Target Model (7B)",
            "size": "14GB (4-bit: 4GB)",
            "quantize": True
        }
    ]


def check_disk_space(required_gb: int = 30) -> bool:
    """Check if enough disk space is available"""
    import shutil
    stat = shutil.disk_usage(Path.home())
    available_gb = stat.free / (1024**3)
    
    print_status(f"Available disk space: {available_gb:.1f} GB")
    
    if available_gb < required_gb:
        print(f"ERROR: Insufficient disk space. Need {required_gb}GB, have {available_gb:.1f}GB")
        return False
    
    return True


def download_model(model_info: Dict[str, Any], force: bool = False) -> bool:
    """Download a single model"""
    model_id = model_info["id"]
    name = model_info["name"]
    
    print_status(f"Downloading {name}: {model_id}")
    
    try:
        # Check if already cached
        cache_dir = Path.home() / ".cache" / "huggingface"
        model_files = list(cache_dir.glob(f"*{model_id.replace('/', '--')}*"))
        
        if model_files and not force:
            print_status(f"  Model already cached, skipping download")
            return True
        
        # Download tokenizer first (smaller)
        print_status(f"  Downloading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        print_status(f"  ✓ Tokenizer downloaded")
        
        # Download model
        print_status(f"  Downloading model weights ({model_info['size']})...")
        
        if model_info.get("quantize", False):
            # For models that will be quantized
            print_status(f"  Downloading with 4-bit quantization config...")
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16
            )
            
            # Only download, don't load into memory
            model = AutoModelForCausalLM.from_pretrained(
                model_id,
                quantization_config=quantization_config,
                device_map="auto",
                torch_dtype=torch.float16,
                low_cpu_mem_usage=True
            )
            
            # Immediately delete to free memory
            del model
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
        else:
            # For smaller models
            model = AutoModelForCausalLM.from_pretrained(
                model_id,
                torch_dtype=torch.float16,
                low_cpu_mem_usage=True
            )
            
            # Delete to free memory
            del model
            
        print_status(f"  ✓ Model downloaded successfully")
        return True
        
    except Exception as e:
        print(f"ERROR: Failed to download {model_id}: {e}")
        return False


def verify_models(models: List[Dict[str, Any]]) -> bool:
    """Verify all models are accessible"""
    print_status("\nVerifying model accessibility...")
    
    all_good = True
    for model_info in models:
        model_id = model_info["id"]
        try:
            # Just try to load tokenizer (quick check)
            tokenizer = AutoTokenizer.from_pretrained(model_id, local_files_only=True)
            print_status(f"  ✓ {model_id} - OK")
        except Exception:
            print_status(f"  ✗ {model_id} - NOT FOUND")
            all_good = False
            
    return all_good


def main():
    """Main download process"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Download models for speculative decoding")
    parser.add_argument("--force", action="store_true", help="Force re-download even if cached")
    parser.add_argument("--verify-only", action="store_true", help="Only verify models, don't download")
    args = parser.parse_args()
    
    print("=" * 60)
    print("Speculative Decoding Model Downloader")
    print("=" * 60)
    
    models = get_model_info()
    
    if args.verify_only:
        if verify_models(models):
            print_status("\nAll models verified successfully!")
            return 0
        else:
            print_status("\nSome models are missing!")
            return 1
    
    # Check disk space
    if not check_disk_space(30):
        return 1
    
    # Download models
    print_status("\nStarting model downloads...")
    print_status("This may take 10-30 minutes depending on connection speed\n")
    
    failed = []
    for i, model_info in enumerate(models, 1):
        print(f"\n[{i}/{len(models)}] " + "="*40)
        
        if not download_model(model_info, force=args.force):
            failed.append(model_info["id"])
            
        # Small delay between downloads
        if i < len(models):
            time.sleep(2)
    
    # Summary
    print("\n" + "=" * 60)
    if not failed:
        print_status("✓ All models downloaded successfully!")
        
        # Verify accessibility
        if verify_models(models):
            print_status("✓ All models verified and ready for use!")
            return 0
        else:
            print_status("✗ Some models failed verification")
            return 1
    else:
        print_status(f"✗ Failed to download {len(failed)} model(s):")
        for model_id in failed:
            print(f"  - {model_id}")
        return 1


if __name__ == "__main__":
    sys.exit(main())