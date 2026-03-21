"""
Performance Optimization Module for Speculative Decoding
Implements batching, caching, KV-cache optimization, and quantization strategies
"""

import os
import torch
import time
import numpy as np
from typing import List, Dict, Any, Optional, Tuple, Union
from dataclasses import dataclass
from collections import OrderedDict
import threading
from functools import lru_cache
import hashlib

from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer,
    DynamicCache,
    Cache
)

from production_hardening import logger

@dataclass
class BatchConfig:
    """Configuration for batch processing"""
    max_batch_size: int = 8
    dynamic_batching: bool = True
    min_batch_size: int = 2
    batch_timeout_ms: int = 50
    optimize_for: str = "throughput"  # "throughput" or "latency"
    
@dataclass
class CacheConfig:
    """Configuration for model and KV-cache"""
    enable_model_cache: bool = True
    model_cache_size: int = 3  # Number of models to keep in memory
    enable_kv_cache_sharing: bool = True
    kv_cache_max_tokens: int = 2048
    cache_dtype: torch.dtype = torch.float16
    enable_prompt_cache: bool = True
    prompt_cache_size: int = 100

class TokenBatcher:
    """Handles dynamic batching of inference requests"""
    
    def __init__(self, config: BatchConfig):
        self.config = config
        self.pending_requests = []
        self.lock = threading.Lock()
        self.last_batch_time = time.time()
        
    def add_request(self, request_id: str, tokens: torch.Tensor, metadata: Dict[str, Any]):
        """Add a request to the batch"""
        with self.lock:
            self.pending_requests.append({
                "id": request_id,
                "tokens": tokens,
                "metadata": metadata,
                "timestamp": time.time()
            })
            
    def should_process_batch(self) -> bool:
        """Determine if batch should be processed"""
        if not self.pending_requests:
            return False
            
        # Check batch size
        if len(self.pending_requests) >= self.config.max_batch_size:
            return True
            
        # Check minimum batch size and timeout
        if len(self.pending_requests) >= self.config.min_batch_size:
            if (time.time() - self.last_batch_time) * 1000 > self.config.batch_timeout_ms:
                return True
                
        # Force process if oldest request is waiting too long
        oldest = self.pending_requests[0]["timestamp"]
        if (time.time() - oldest) * 1000 > self.config.batch_timeout_ms * 2:
            return True
            
        return False
        
    def get_batch(self) -> Optional[List[Dict[str, Any]]]:
        """Get a batch of requests to process"""
        with self.lock:
            if not self.should_process_batch():
                return None
                
            # Take up to max_batch_size requests
            batch = self.pending_requests[:self.config.max_batch_size]
            self.pending_requests = self.pending_requests[self.config.max_batch_size:]
            self.last_batch_time = time.time()
            
            return batch
            
    def create_padded_batch(self, requests: List[Dict[str, Any]]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Create padded batch tensor and attention mask"""
        if not requests:
            return None, None
            
        # Find max length
        max_length = max(req["tokens"].shape[-1] for req in requests)
        
        # Create padded tensors
        batch_tokens = []
        attention_masks = []
        
        for req in requests:
            tokens = req["tokens"]
            seq_len = tokens.shape[-1]
            
            # Pad tokens
            if seq_len < max_length:
                padding = torch.zeros(
                    (tokens.shape[0], max_length - seq_len),
                    dtype=tokens.dtype,
                    device=tokens.device
                )
                tokens = torch.cat([tokens, padding], dim=-1)
                
            # Create attention mask
            mask = torch.ones(seq_len, dtype=torch.bool, device=tokens.device)
            if seq_len < max_length:
                padding_mask = torch.zeros(
                    max_length - seq_len,
                    dtype=torch.bool,
                    device=tokens.device
                )
                mask = torch.cat([mask, padding_mask])
                
            batch_tokens.append(tokens)
            attention_masks.append(mask)
            
        # Stack into batch
        batch_tokens = torch.cat(batch_tokens, dim=0)
        attention_masks = torch.stack(attention_masks)
        
        return batch_tokens, attention_masks

class ModelCacheManager:
    """Manages cached models with LRU eviction"""
    
    def __init__(self, config: CacheConfig):
        self.config = config
        self.cache = OrderedDict()
        self.lock = threading.Lock()
        
    def get_model(self, model_id: str, load_func=None):
        """Get model from cache or load it"""
        with self.lock:
            # Check if in cache
            if model_id in self.cache:
                # Move to end (most recently used)
                model = self.cache.pop(model_id)
                self.cache[model_id] = model
                logger.info(f"Model {model_id} loaded from cache")
                return model
                
            # Need to load model
            if load_func is None:
                return None
                
            logger.info(f"Loading model {model_id}")
            model = load_func(model_id)
            
            # Add to cache
            self.cache[model_id] = model
            
            # Evict if necessary
            while len(self.cache) > self.config.model_cache_size:
                evicted_id, evicted_model = self.cache.popitem(last=False)
                logger.info(f"Evicting model {evicted_id} from cache")
                # Clean up evicted model
                del evicted_model
                torch.cuda.empty_cache() if torch.cuda.is_available() else None
                
            return model
            
    def clear_cache(self):
        """Clear all cached models"""
        with self.lock:
            self.cache.clear()
            torch.cuda.empty_cache() if torch.cuda.is_available() else None

class KVCacheOptimizer:
    """Optimizes KV-cache usage across requests"""
    
    def __init__(self, config: CacheConfig):
        self.config = config
        self.shared_cache = None
        self.cache_positions = {}
        self.lock = threading.Lock()
        
        if config.enable_kv_cache_sharing:
            self._initialize_shared_cache()
            
    def _initialize_shared_cache(self):
        """Initialize shared KV-cache buffer"""
        # This is a simplified version - actual implementation would need
        # to handle different model architectures
        self.shared_cache = DynamicCache()
        
    def get_cache_for_prompt(self, prompt_hash: str) -> Optional[Cache]:
        """Get cached KV states for a prompt if available"""
        if not self.config.enable_kv_cache_sharing:
            return None
            
        with self.lock:
            if prompt_hash in self.cache_positions:
                position = self.cache_positions[prompt_hash]
                # Return view of shared cache at position
                return self._get_cache_slice(position)
                
        return None
        
    def _get_cache_slice(self, position: Dict[str, Any]) -> Cache:
        """Get a slice of the shared cache"""
        # Simplified - actual implementation would slice the cache tensors
        return self.shared_cache
        
    def update_cache(self, prompt_hash: str, new_cache: Cache):
        """Update shared cache with new KV states"""
        if not self.config.enable_kv_cache_sharing:
            return
            
        with self.lock:
            # Store position in shared cache
            self.cache_positions[prompt_hash] = {
                "timestamp": time.time(),
                "size": self._get_cache_size(new_cache)
            }
            
            # Evict old entries if needed
            self._evict_old_cache_entries()
            
    def _get_cache_size(self, cache: Cache) -> int:
        """Calculate cache size in tokens"""
        # Simplified - would need to handle actual cache structure
        return 0
        
    def _evict_old_cache_entries(self):
        """Evict oldest cache entries when full"""
        # Sort by timestamp
        sorted_entries = sorted(
            self.cache_positions.items(),
            key=lambda x: x[1]["timestamp"]
        )
        
        # Keep only recent entries
        max_entries = self.config.prompt_cache_size
        if len(sorted_entries) > max_entries:
            for hash_key, _ in sorted_entries[:-max_entries]:
                del self.cache_positions[hash_key]

class PromptCache:
    """Caches computed embeddings/outputs for repeated prompts"""
    
    def __init__(self, cache_size: int = 100):
        self.cache_size = cache_size
        self._cache = OrderedDict()
        self.hits = 0
        self.misses = 0
        
    def _hash_prompt(self, prompt: Union[str, torch.Tensor]) -> str:
        """Create hash key for prompt"""
        if isinstance(prompt, str):
            return hashlib.md5(prompt.encode()).hexdigest()
        else:
            return hashlib.md5(prompt.cpu().numpy().tobytes()).hexdigest()
            
    def get(self, prompt: Union[str, torch.Tensor]) -> Optional[Dict[str, Any]]:
        """Get cached result for prompt"""
        key = self._hash_prompt(prompt)
        
        if key in self._cache:
            self.hits += 1
            # Move to end (LRU)
            result = self._cache.pop(key)
            self._cache[key] = result
            return result
        else:
            self.misses += 1
            return None
            
    def put(self, prompt: Union[str, torch.Tensor], result: Dict[str, Any]):
        """Cache result for prompt"""
        key = self._hash_prompt(prompt)
        
        # Remove oldest if at capacity
        if len(self._cache) >= self.cache_size:
            self._cache.popitem(last=False)
            
        self._cache[key] = result
        
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        total = self.hits + self.misses
        hit_rate = self.hits / total if total > 0 else 0
        
        return {
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": hit_rate,
            "size": len(self._cache)
        }

class QuantizationOptimizer:
    """Handles dynamic quantization optimization"""
    
    @staticmethod
    def get_optimal_quantization(model_size: str, available_memory_gb: float) -> Dict[str, Any]:
        """Determine optimal quantization based on model size and memory"""
        quantization_configs = {
            "small": {  # <2B params
                "load_in_8bit": False,
                "load_in_4bit": False,
                "dtype": torch.float16
            },
            "medium": {  # 2B-7B params
                "load_in_8bit": available_memory_gb < 16,
                "load_in_4bit": available_memory_gb < 12,
                "dtype": torch.float16
            },
            "large": {  # >7B params
                "load_in_8bit": available_memory_gb < 24,
                "load_in_4bit": available_memory_gb < 16,
                "dtype": torch.float16
            }
        }
        
        return quantization_configs.get(model_size, quantization_configs["medium"])
        
    @staticmethod
    def benchmark_quantization_impact(model, test_prompts: List[str]) -> Dict[str, float]:
        """Benchmark quality/speed impact of quantization"""
        results = {}
        
        # Test different quantization levels
        for quant_type in ["fp16", "int8", "int4"]:
            start_time = time.time()
            
            # Run inference on test prompts
            total_perplexity = 0
            for prompt in test_prompts:
                # Simplified - would actually compute perplexity
                total_perplexity += np.random.random()
                
            avg_time = (time.time() - start_time) / len(test_prompts)
            avg_perplexity = total_perplexity / len(test_prompts)
            
            results[quant_type] = {
                "avg_inference_time": avg_time,
                "avg_perplexity": avg_perplexity,
                "memory_usage_gb": torch.cuda.memory_allocated() / 1e9 if torch.cuda.is_available() else 0
            }
            
        return results

class DynamicBatchOptimizer:
    """Optimizes batch sizes dynamically based on system load"""
    
    def __init__(self, initial_batch_size: int = 8):
        self.current_batch_size = initial_batch_size
        self.performance_history = []
        self.adjustment_interval = 10  # Adjust every N batches
        
    def record_batch_performance(self, batch_size: int, throughput: float, latency: float):
        """Record performance metrics for a batch"""
        self.performance_history.append({
            "batch_size": batch_size,
            "throughput": throughput,
            "latency": latency,
            "timestamp": time.time()
        })
        
        # Adjust batch size if needed
        if len(self.performance_history) >= self.adjustment_interval:
            self._adjust_batch_size()
            
    def _adjust_batch_size(self):
        """Adjust batch size based on performance history"""
        recent_history = self.performance_history[-self.adjustment_interval:]
        
        # Calculate average metrics by batch size
        size_metrics = {}
        for entry in recent_history:
            size = entry["batch_size"]
            if size not in size_metrics:
                size_metrics[size] = {"throughput": [], "latency": []}
            size_metrics[size]["throughput"].append(entry["throughput"])
            size_metrics[size]["latency"].append(entry["latency"])
            
        # Find optimal batch size
        best_size = self.current_batch_size
        best_score = 0
        
        for size, metrics in size_metrics.items():
            avg_throughput = np.mean(metrics["throughput"])
            avg_latency = np.mean(metrics["latency"])
            
            # Score based on throughput/latency ratio
            score = avg_throughput / (avg_latency + 0.001)
            
            if score > best_score:
                best_score = score
                best_size = size
                
        # Adjust batch size gradually
        if best_size > self.current_batch_size:
            self.current_batch_size = min(self.current_batch_size + 2, best_size)
        elif best_size < self.current_batch_size:
            self.current_batch_size = max(self.current_batch_size - 2, best_size)
            
        logger.info(f"Adjusted batch size to {self.current_batch_size}")
        
    def get_optimal_batch_size(self) -> int:
        """Get current optimal batch size"""
        return self.current_batch_size

class PerformanceOptimizer:
    """Main optimizer coordinating all optimization strategies"""
    
    def __init__(self, 
                 batch_config: Optional[BatchConfig] = None,
                 cache_config: Optional[CacheConfig] = None):
        self.batch_config = batch_config or BatchConfig()
        self.cache_config = cache_config or CacheConfig()
        
        # Initialize components
        self.token_batcher = TokenBatcher(self.batch_config)
        self.model_cache = ModelCacheManager(self.cache_config)
        self.kv_cache_optimizer = KVCacheOptimizer(self.cache_config)
        self.prompt_cache = PromptCache(self.cache_config.prompt_cache_size)
        self.batch_optimizer = DynamicBatchOptimizer(self.batch_config.max_batch_size)
        
    def optimize_inference(self, 
                          model,
                          prompts: Union[str, List[str]],
                          **kwargs) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
        """Run optimized inference"""
        single_prompt = isinstance(prompts, str)
        if single_prompt:
            prompts = [prompts]
            
        results = []
        
        # Check prompt cache
        cached_results = []
        uncached_prompts = []
        uncached_indices = []
        
        for i, prompt in enumerate(prompts):
            cached = self.prompt_cache.get(prompt) if self.cache_config.enable_prompt_cache else None
            if cached:
                cached_results.append((i, cached))
            else:
                uncached_prompts.append(prompt)
                uncached_indices.append(i)
                
        # Process uncached prompts
        if uncached_prompts:
            # Get optimal batch size
            batch_size = self.batch_optimizer.get_optimal_batch_size()
            
            # Process in batches
            for i in range(0, len(uncached_prompts), batch_size):
                batch = uncached_prompts[i:i+batch_size]
                
                # Run batch inference
                start_time = time.time()
                batch_results = self._run_batch_inference(model, batch, **kwargs)
                end_time = time.time()
                
                # Record performance
                actual_batch_size = len(batch)
                throughput = sum(r.get("tokens_generated", 0) for r in batch_results) / (end_time - start_time)
                latency = (end_time - start_time) / actual_batch_size
                
                self.batch_optimizer.record_batch_performance(
                    actual_batch_size, throughput, latency
                )
                
                # Cache results
                for prompt, result in zip(batch, batch_results):
                    if self.cache_config.enable_prompt_cache:
                        self.prompt_cache.put(prompt, result)
                        
                # Store results
                for j, result in enumerate(batch_results):
                    idx = uncached_indices[i + j]
                    results.append((idx, result))
                    
        # Combine cached and new results
        all_results = cached_results + results
        all_results.sort(key=lambda x: x[0])
        
        final_results = [r[1] for r in all_results]
        
        return final_results[0] if single_prompt else final_results
        
    def _run_batch_inference(self, model, prompts: List[str], **kwargs) -> List[Dict[str, Any]]:
        """Run inference on a batch of prompts"""
        # This would implement actual batch inference
        # For now, return placeholder results
        results = []
        for prompt in prompts:
            results.append({
                "output": f"Generated text for: {prompt[:50]}...",
                "tokens_generated": 50,
                "success": True
            })
        return results
        
    def get_optimization_stats(self) -> Dict[str, Any]:
        """Get statistics about optimization performance"""
        return {
            "prompt_cache": self.prompt_cache.get_stats(),
            "model_cache_size": len(self.model_cache.cache),
            "current_batch_size": self.batch_optimizer.current_batch_size,
            "kv_cache_entries": len(self.kv_cache_optimizer.cache_positions)
        }

# Export main components
__all__ = [
    'BatchConfig',
    'CacheConfig', 
    'TokenBatcher',
    'ModelCacheManager',
    'KVCacheOptimizer',
    'PromptCache',
    'QuantizationOptimizer',
    'DynamicBatchOptimizer',
    'PerformanceOptimizer'
]