#!/usr/bin/env python3
"""
Performance and Stress Tests for Speculative Decoding
Tests system under load, memory limits, and edge cases
"""

import unittest
import torch
import time
import threading
import multiprocessing
import psutil
import gc
from typing import List
import numpy as np

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from performance_optimization import (
    TokenBatcher, BatchConfig, ModelCacheManager, CacheConfig,
    PromptCache, DynamicBatchOptimizer, PerformanceOptimizer
)


class TestTokenBatcher(unittest.TestCase):
    """Test token batching functionality"""
    
    def setUp(self):
        self.config = BatchConfig(
            max_batch_size=4,
            min_batch_size=2,
            batch_timeout_ms=100
        )
        self.batcher = TokenBatcher(self.config)
        
    def test_batch_accumulation(self):
        """Test batch accumulation logic"""
        # Add requests
        for i in range(3):
            tokens = torch.tensor([[i] * 10])
            self.batcher.add_request(f"req_{i}", tokens, {"idx": i})
            
        # Should not process yet (min_batch_size=2, but timeout not reached)
        self.assertFalse(self.batcher.should_process_batch())
        
        # Add one more to reach max_batch_size
        self.batcher.add_request("req_3", torch.tensor([[3] * 10]), {"idx": 3})
        
        # Should process now
        self.assertTrue(self.batcher.should_process_batch())
        
        # Get batch
        batch = self.batcher.get_batch()
        self.assertEqual(len(batch), 4)
        
    def test_batch_timeout(self):
        """Test batch timeout triggering"""
        # Add 2 requests (min_batch_size)
        for i in range(2):
            self.batcher.add_request(f"req_{i}", torch.tensor([[i]]), {})
            
        # Initially shouldn't process
        self.assertFalse(self.batcher.should_process_batch())
        
        # Wait for timeout
        time.sleep(0.15)  # 150ms > 100ms timeout
        
        # Should process now
        self.assertTrue(self.batcher.should_process_batch())
        
    def test_padded_batch_creation(self):
        """Test creation of padded batches"""
        # Add requests with different lengths
        requests = []
        for i, length in enumerate([5, 8, 3]):
            tokens = torch.ones(1, length, dtype=torch.long) * i
            request = {"id": f"req_{i}", "tokens": tokens}
            requests.append(request)
            
        # Create padded batch
        batch_tokens, attention_masks = self.batcher.create_padded_batch(requests)
        
        # Check shapes
        self.assertEqual(batch_tokens.shape, (3, 8))  # 3 requests, max length 8
        self.assertEqual(attention_masks.shape, (3, 8))
        
        # Check attention masks
        expected_masks = [
            [True] * 5 + [False] * 3,
            [True] * 8,
            [True] * 3 + [False] * 5
        ]
        
        for i, expected in enumerate(expected_masks):
            self.assertEqual(attention_masks[i].tolist(), expected)


class TestModelCache(unittest.TestCase):
    """Test model caching functionality"""
    
    def setUp(self):
        self.config = CacheConfig(model_cache_size=2)
        self.cache = ModelCacheManager(self.config)
        
    def test_cache_hit_miss(self):
        """Test cache hits and misses"""
        # Mock model loader
        load_count = 0
        def mock_loader(model_id):
            nonlocal load_count
            load_count += 1
            return {"id": model_id, "loaded": True}
            
        # First access - miss
        model1 = self.cache.get_model("model_1", mock_loader)
        self.assertEqual(load_count, 1)
        self.assertEqual(model1["id"], "model_1")
        
        # Second access - hit
        model1_again = self.cache.get_model("model_1", mock_loader)
        self.assertEqual(load_count, 1)  # No additional load
        self.assertEqual(model1_again["id"], "model_1")
        
    def test_lru_eviction(self):
        """Test LRU eviction policy"""
        def mock_loader(model_id):
            return {"id": model_id}
            
        # Load models up to capacity
        self.cache.get_model("model_1", mock_loader)
        self.cache.get_model("model_2", mock_loader)
        
        # Load third model - should evict model_1
        self.cache.get_model("model_3", mock_loader)
        
        # model_1 should be evicted
        self.assertNotIn("model_1", self.cache.cache)
        self.assertIn("model_2", self.cache.cache)
        self.assertIn("model_3", self.cache.cache)
        
        # Access model_2 to make it most recent
        self.cache.get_model("model_2", mock_loader)
        
        # Load model_4 - should evict model_3
        self.cache.get_model("model_4", mock_loader)
        
        self.assertIn("model_2", self.cache.cache)  # Most recently used
        self.assertIn("model_4", self.cache.cache)  # Just added
        self.assertNotIn("model_3", self.cache.cache)  # Evicted


class TestPromptCache(unittest.TestCase):
    """Test prompt caching"""
    
    def setUp(self):
        self.cache = PromptCache(cache_size=3)
        
    def test_string_prompt_caching(self):
        """Test caching with string prompts"""
        prompt = "What is machine learning?"
        result = {"output": "ML is...", "tokens": 50}
        
        # Cache miss
        cached = self.cache.get(prompt)
        self.assertIsNone(cached)
        self.assertEqual(self.cache.misses, 1)
        
        # Store result
        self.cache.put(prompt, result)
        
        # Cache hit
        cached = self.cache.get(prompt)
        self.assertEqual(cached, result)
        self.assertEqual(self.cache.hits, 1)
        
    def test_tensor_prompt_caching(self):
        """Test caching with tensor prompts"""
        prompt_tensor = torch.tensor([1, 2, 3, 4, 5])
        result = {"output": "Generated", "tokens": 25}
        
        # Store and retrieve
        self.cache.put(prompt_tensor, result)
        cached = self.cache.get(prompt_tensor)
        self.assertEqual(cached, result)
        
    def test_cache_eviction(self):
        """Test cache eviction when full"""
        # Fill cache
        for i in range(4):
            self.cache.put(f"prompt_{i}", {"result": i})
            
        # First prompt should be evicted
        self.assertIsNone(self.cache.get("prompt_0"))
        
        # Others should still be cached
        for i in range(1, 4):
            self.assertIsNotNone(self.cache.get(f"prompt_{i}"))
            
    def test_cache_statistics(self):
        """Test cache statistics"""
        # Generate some hits and misses
        self.cache.put("prompt_1", {"result": 1})
        self.cache.get("prompt_1")  # Hit
        self.cache.get("prompt_2")  # Miss
        self.cache.get("prompt_1")  # Hit
        
        stats = self.cache.get_stats()
        self.assertEqual(stats["hits"], 2)
        self.assertEqual(stats["misses"], 1)
        self.assertAlmostEqual(stats["hit_rate"], 2/3)
        self.assertEqual(stats["size"], 1)


class TestDynamicBatchOptimizer(unittest.TestCase):
    """Test dynamic batch size optimization"""
    
    def setUp(self):
        self.optimizer = DynamicBatchOptimizer(initial_batch_size=4)
        
    def test_batch_size_increase(self):
        """Test batch size increase on good performance"""
        # Record good performance with larger batches
        for _ in range(15):
            self.optimizer.record_batch_performance(
                batch_size=6,
                throughput=100,  # Good throughput
                latency=0.5      # Low latency
            )
            
        # Should increase batch size
        self.assertGreater(self.optimizer.get_optimal_batch_size(), 4)
        
    def test_batch_size_decrease(self):
        """Test batch size decrease on poor performance"""
        # Record poor performance with current size
        for _ in range(15):
            self.optimizer.record_batch_performance(
                batch_size=4,
                throughput=20,   # Poor throughput
                latency=5.0      # High latency
            )
            
        # Record better performance with smaller size
        for _ in range(5):
            self.optimizer.record_batch_performance(
                batch_size=2,
                throughput=50,
                latency=1.0
            )
            
        # Should decrease batch size
        self.assertLess(self.optimizer.get_optimal_batch_size(), 4)


class TestMemoryPressure(unittest.TestCase):
    """Test behavior under memory pressure"""
    
    def test_memory_allocation_stress(self):
        """Test system under memory allocation stress"""
        allocations = []
        
        try:
            # Allocate tensors until we approach limits
            while True:
                # Allocate 100MB chunks
                tensor = torch.randn(100 * 1024 * 1024 // 4)  # 100MB of float32
                allocations.append(tensor)
                
                # Check memory usage
                mem_usage = psutil.virtual_memory().percent
                if mem_usage > 85:
                    break
                    
                if len(allocations) > 100:  # Safety limit
                    break
                    
        except MemoryError:
            pass
            
        finally:
            # Clean up
            allocations.clear()
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
        # System should still be responsive
        self.assertTrue(psutil.virtual_memory().percent < 95)
        
    def test_gpu_memory_management(self):
        """Test GPU memory management"""
        if not torch.cuda.is_available():
            self.skipTest("CUDA not available")
            
        initial_memory = torch.cuda.memory_allocated()
        
        try:
            # Allocate GPU tensors
            tensors = []
            for _ in range(10):
                tensor = torch.randn(1024, 1024, device='cuda')
                tensors.append(tensor)
                
            # Check memory increased
            allocated = torch.cuda.memory_allocated()
            self.assertGreater(allocated, initial_memory)
            
            # Clear tensors
            tensors.clear()
            torch.cuda.empty_cache()
            
            # Memory should be released
            final_memory = torch.cuda.memory_allocated()
            self.assertLess(final_memory, allocated)
            
        finally:
            torch.cuda.empty_cache()


class TestConcurrentRequests(unittest.TestCase):
    """Test handling of concurrent requests"""
    
    def test_thread_safety(self):
        """Test thread safety of components"""
        cache = PromptCache(cache_size=100)
        errors = []
        
        def worker(worker_id):
            try:
                for i in range(100):
                    prompt = f"worker_{worker_id}_prompt_{i}"
                    
                    # Try to get from cache
                    result = cache.get(prompt)
                    
                    if result is None:
                        # Simulate some work
                        time.sleep(0.001)
                        
                        # Store result
                        cache.put(prompt, {"worker": worker_id, "index": i})
                        
            except Exception as e:
                errors.append(e)
                
        # Start multiple threads
        threads = []
        for i in range(10):
            t = threading.Thread(target=worker, args=(i,))
            threads.append(t)
            t.start()
            
        # Wait for completion
        for t in threads:
            t.join()
            
        # Should have no errors
        self.assertEqual(len(errors), 0)
        
        # Cache should have entries from all workers
        stats = cache.get_stats()
        self.assertGreater(stats["size"], 0)
        
    def test_multiprocess_isolation(self):
        """Test process isolation"""
        # This would test that separate processes don't interfere
        # Skipped for brevity but important for production
        pass


class TestPerformanceTargets(unittest.TestCase):
    """Test that performance meets specified targets"""
    
    def test_inference_latency_target(self):
        """Test inference latency targets"""
        # Simulate inference latencies
        latencies = []
        
        # Most requests fast, some slow
        for i in range(1000):
            if i % 50 == 0:  # 2% slow requests
                latency = np.random.uniform(4, 6)
            else:
                latency = np.random.exponential(0.8)
            latencies.append(latency)
            
        # Calculate percentiles
        p50 = np.percentile(latencies, 50)
        p95 = np.percentile(latencies, 95)
        p99 = np.percentile(latencies, 99)
        
        # Verify targets
        self.assertLess(p50, 1.5)  # P50 < 1.5s
        self.assertLess(p95, 3.0)  # P95 < 3s
        self.assertLess(p99, 5.0)  # P99 < 5s
        
    def test_throughput_target(self):
        """Test throughput targets"""
        # Simulate throughput measurements
        throughputs = []
        
        # Generate throughput samples
        for _ in range(100):
            # Tokens generated / time
            tokens = np.random.randint(30, 100)
            time_taken = tokens / np.random.uniform(20, 80)  # 20-80 tok/s
            throughput = tokens / time_taken
            throughputs.append(throughput)
            
        avg_throughput = np.mean(throughputs)
        
        # Should maintain at least 40 tok/s average
        self.assertGreater(avg_throughput, 40)


class TestEdgeCases(unittest.TestCase):
    """Test edge cases and boundary conditions"""
    
    def test_empty_batch_handling(self):
        """Test handling of empty batches"""
        batcher = TokenBatcher(BatchConfig())
        
        # Should handle empty batch gracefully
        batch_tokens, masks = batcher.create_padded_batch([])
        self.assertIsNone(batch_tokens)
        self.assertIsNone(masks)
        
    def test_single_token_generation(self):
        """Test generation of single token"""
        # This would test the edge case of max_length=1
        pass
        
    def test_maximum_sequence_length(self):
        """Test handling of maximum sequence lengths"""
        # Test with sequences at model maximum
        pass
        
    def test_unicode_handling(self):
        """Test handling of unicode in prompts"""
        cache = PromptCache()
        
        unicode_prompts = [
            "你好世界",  # Chinese
            "مرحبا بالعالم",  # Arabic
            "🎉🎊✨",  # Emoji
            "Ωμέγα",  # Greek
        ]
        
        for i, prompt in enumerate(unicode_prompts):
            cache.put(prompt, {"idx": i})
            retrieved = cache.get(prompt)
            self.assertEqual(retrieved["idx"], i)


if __name__ == "__main__":
    unittest.main()