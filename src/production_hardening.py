"""
Production Hardening Module for Speculative Decoding
Implements error handling, resource management, logging, and security features.
"""

import os
import json
import time
import psutil
import logging
import threading
from typing import Optional, Dict, Any, List, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
from functools import wraps
import torch
import gc
import re
from contextlib import contextmanager

# Configure structured logging
logging.basicConfig(
    level=logging.INFO,
    format='{"time": "%(asctime)s", "level": "%(levelname)s", "module": "%(module)s", "message": "%(message)s"}',
    datefmt='%Y-%m-%dT%H:%M:%S'
)
logger = logging.getLogger(__name__)

# Security patterns for input validation
PROMPT_INJECTION_PATTERNS = [
    r'(?i)ignore.*previous.*instructions',
    r'(?i)disregard.*above',
    r'(?i)new.*instructions.*:',
    r'(?i)system.*prompt.*:',
    r'(?i)admin.*mode',
    r'<script[^>]*>',
    r'javascript:',
    r'data:text/html',
]

@dataclass
class ResourceLimits:
    """Resource usage limits and thresholds"""
    max_memory_gb: float = 12.0
    warn_memory_gb: float = 10.0
    critical_memory_gb: float = 11.5
    max_prompt_length: int = 4096
    max_output_length: int = 4096
    request_timeout_seconds: int = 300
    rate_limit_per_minute: int = 60
    max_batch_size: int = 16

@dataclass
class SystemMetrics:
    """Current system resource metrics"""
    timestamp: str
    memory_gb: float
    memory_percent: float
    gpu_memory_gb: float
    gpu_memory_percent: float
    cpu_percent: float
    active_threads: int
    inference_count: int
    error_count: int

class ResourceMonitor:
    """Monitors system resources and enforces limits"""
    
    def __init__(self, limits: ResourceLimits):
        self.limits = limits
        self.start_time = time.time()
        self.inference_count = 0
        self.error_count = 0
        self._lock = threading.Lock()
        
    def get_memory_usage(self) -> Tuple[float, float]:
        """Get current memory usage in GB and percentage"""
        mem = psutil.virtual_memory()
        gb = (mem.total - mem.available) / (1024**3)
        return gb, mem.percent
        
    def get_gpu_memory_usage(self) -> Tuple[float, float]:
        """Get GPU memory usage if available"""
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / (1024**3)
            reserved = torch.cuda.memory_reserved() / (1024**3)
            total = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            percent = (reserved / total) * 100
            return reserved, percent
        return 0.0, 0.0
        
    def get_metrics(self) -> SystemMetrics:
        """Get current system metrics"""
        mem_gb, mem_percent = self.get_memory_usage()
        gpu_gb, gpu_percent = self.get_gpu_memory_usage()
        
        return SystemMetrics(
            timestamp=datetime.utcnow().isoformat(),
            memory_gb=round(mem_gb, 2),
            memory_percent=round(mem_percent, 1),
            gpu_memory_gb=round(gpu_gb, 2),
            gpu_memory_percent=round(gpu_percent, 1),
            cpu_percent=round(psutil.cpu_percent(interval=0.1), 1),
            active_threads=threading.active_count(),
            inference_count=self.inference_count,
            error_count=self.error_count
        )
        
    def check_memory_critical(self) -> bool:
        """Check if memory usage is critical"""
        mem_gb, _ = self.get_memory_usage()
        gpu_gb, _ = self.get_gpu_memory_usage()
        total_gb = mem_gb + gpu_gb
        
        if total_gb >= self.limits.critical_memory_gb:
            logger.error(f"CRITICAL: Memory usage {total_gb:.1f}GB exceeds limit {self.limits.critical_memory_gb}GB")
            return True
        elif total_gb >= self.limits.warn_memory_gb:
            logger.warning(f"Memory usage {total_gb:.1f}GB approaching limit")
        return False
        
    def increment_inference(self):
        """Increment inference counter"""
        with self._lock:
            self.inference_count += 1
            
    def increment_error(self):
        """Increment error counter"""
        with self._lock:
            self.error_count += 1

class InputValidator:
    """Validates and sanitizes input prompts"""
    
    def __init__(self, max_length: int = 4096):
        self.max_length = max_length
        self.injection_patterns = [re.compile(p) for p in PROMPT_INJECTION_PATTERNS]
        
    def validate_prompt(self, prompt: str) -> Tuple[bool, Optional[str]]:
        """Validate prompt for safety and constraints"""
        if not prompt or not isinstance(prompt, str):
            return False, "Invalid prompt type"
            
        if len(prompt) > self.max_length:
            return False, f"Prompt exceeds maximum length of {self.max_length}"
            
        # Check for potential injection attacks
        for pattern in self.injection_patterns:
            if pattern.search(prompt):
                logger.warning(f"Potential prompt injection detected: {prompt[:50]}...")
                return False, "Prompt contains suspicious patterns"
                
        return True, None
        
    def sanitize_prompt(self, prompt: str) -> str:
        """Sanitize prompt by removing potentially harmful content"""
        # Remove any HTML/script tags
        prompt = re.sub(r'<[^>]+>', '', prompt)
        # Remove excessive whitespace
        prompt = ' '.join(prompt.split())
        # Truncate if needed
        if len(prompt) > self.max_length:
            prompt = prompt[:self.max_length]
        return prompt

class RateLimiter:
    """Simple rate limiter for API requests"""
    
    def __init__(self, max_requests_per_minute: int = 60):
        self.max_requests = max_requests_per_minute
        self.requests = []
        self._lock = threading.Lock()
        
    def check_rate_limit(self) -> Tuple[bool, Optional[str]]:
        """Check if request is within rate limit"""
        now = time.time()
        with self._lock:
            # Remove requests older than 1 minute
            self.requests = [t for t in self.requests if now - t < 60]
            
            if len(self.requests) >= self.max_requests:
                return False, f"Rate limit exceeded: {self.max_requests} requests per minute"
                
            self.requests.append(now)
            return True, None

class ConnectionPool:
    """Connection pool for OpenClaw clients"""
    
    def __init__(self, max_connections: int = 10):
        self.max_connections = max_connections
        self.connections = []
        self.in_use = set()
        self._lock = threading.Lock()
        
    @contextmanager
    def get_connection(self):
        """Get a connection from the pool"""
        conn = None
        try:
            with self._lock:
                if self.connections:
                    conn = self.connections.pop()
                elif len(self.in_use) < self.max_connections:
                    conn = self._create_connection()
                else:
                    raise RuntimeError("Connection pool exhausted")
                self.in_use.add(conn)
            yield conn
        finally:
            if conn:
                with self._lock:
                    self.in_use.discard(conn)
                    self.connections.append(conn)
                    
    def _create_connection(self):
        """Create a new connection (placeholder)"""
        return {"id": len(self.in_use), "created": time.time()}

class ModelCleanup:
    """Handles proper model cleanup to prevent memory leaks"""
    
    @staticmethod
    def cleanup_model(model):
        """Clean up a model and free memory"""
        if model is not None:
            try:
                # Move to CPU first to free GPU memory
                if hasattr(model, 'cpu'):
                    model.cpu()
                    
                # Delete the model
                del model
                
                # Force garbage collection
                gc.collect()
                
                # Clear GPU cache if available
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    
                logger.info("Model cleanup completed successfully")
            except Exception as e:
                logger.error(f"Error during model cleanup: {e}")

def with_timeout(timeout_seconds: int):
    """Decorator to add timeout to functions"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            result = [None]
            exception = [None]
            
            def target():
                try:
                    result[0] = func(*args, **kwargs)
                except Exception as e:
                    exception[0] = e
                    
            thread = threading.Thread(target=target)
            thread.daemon = True
            thread.start()
            thread.join(timeout_seconds)
            
            if thread.is_alive():
                logger.error(f"Function {func.__name__} timed out after {timeout_seconds}s")
                raise TimeoutError(f"Operation timed out after {timeout_seconds} seconds")
                
            if exception[0]:
                raise exception[0]
                
            return result[0]
        return wrapper
    return decorator

def with_graceful_degradation(fallback_modes: List[str]):
    """Decorator to implement graceful degradation"""
    def decorator(func):
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            current_mode = kwargs.get('mode', '3model')
            
            for mode in [current_mode] + fallback_modes:
                try:
                    kwargs['mode'] = mode
                    logger.info(f"Attempting inference with mode: {mode}")
                    return func(self, *args, **kwargs)
                except MemoryError:
                    logger.warning(f"Memory error with mode {mode}, degrading...")
                    ModelCleanup.cleanup_model(getattr(self, f'{mode}_model', None))
                except Exception as e:
                    logger.error(f"Error with mode {mode}: {e}")
                    
            raise RuntimeError("All inference modes failed")
        return wrapper
    return decorator

class FeatureFlags:
    """Manage feature flags for gradual rollout"""
    
    def __init__(self, config_file: Optional[str] = None):
        self.flags = {
            "enable_3model": False,
            "enable_2model": True,
            "enable_batch_inference": False,
            "enable_streaming": False,
            "enable_kv_cache_sharing": False,
            "enable_dynamic_thresholds": False,
            "log_performance_metrics": True,
            "enforce_rate_limits": True,
            "validate_inputs": True,
        }
        
        if config_file and os.path.exists(config_file):
            self.load_from_file(config_file)
            
    def load_from_file(self, config_file: str):
        """Load feature flags from JSON file"""
        try:
            with open(config_file, 'r') as f:
                loaded_flags = json.load(f)
                self.flags.update(loaded_flags)
                logger.info(f"Loaded feature flags from {config_file}")
        except Exception as e:
            logger.error(f"Failed to load feature flags: {e}")
            
    def get(self, flag_name: str, default: bool = False) -> bool:
        """Get feature flag value"""
        return self.flags.get(flag_name, default)
        
    def set(self, flag_name: str, value: bool):
        """Set feature flag value"""
        self.flags[flag_name] = value
        logger.info(f"Feature flag '{flag_name}' set to {value}")

class PerformanceLogger:
    """Logs performance metrics in structured format"""
    
    def __init__(self, log_file: str = "logs/performance_metrics.jsonl"):
        self.log_file = log_file
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        
    def log_inference(self, metrics: Dict[str, Any]):
        """Log inference metrics"""
        entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "type": "inference",
            **metrics
        }
        
        try:
            with open(self.log_file, 'a') as f:
                json.dump(entry, f)
                f.write('\n')
        except Exception as e:
            logger.error(f"Failed to log performance metrics: {e}")

# Error types for better error handling
class SpeculativeDecodingError(Exception):
    """Base error for speculative decoding"""
    pass

class TokenizerMismatchError(SpeculativeDecodingError):
    """Error when tokenizers don't match"""
    pass

class ModelLoadError(SpeculativeDecodingError):
    """Error loading model"""
    pass

class InferenceError(SpeculativeDecodingError):
    """Error during inference"""
    pass

class ResourceExhaustedError(SpeculativeDecodingError):
    """Error when resources are exhausted"""
    pass

def safe_model_load(model_path: str, device: str = "cpu") -> Any:
    """Safely load a model with error handling"""
    try:
        logger.info(f"Loading model from {model_path}")
        # Model loading logic would go here
        # This is a placeholder
        return {"path": model_path, "device": device}
    except MemoryError:
        logger.error(f"Out of memory loading model from {model_path}")
        raise ResourceExhaustedError("Insufficient memory to load model")
    except Exception as e:
        logger.error(f"Failed to load model from {model_path}: {e}")
        raise ModelLoadError(f"Failed to load model: {e}")

def handle_tokenizer_mismatch(tokens: List[int], from_tokenizer, to_tokenizer) -> List[int]:
    """Handle tokenizer mismatches safely"""
    try:
        # Decode with source tokenizer
        text = from_tokenizer.decode(tokens, skip_special_tokens=True)
        # Re-encode with target tokenizer
        new_tokens = to_tokenizer.encode(text, add_special_tokens=False)
        return new_tokens
    except Exception as e:
        logger.error(f"Tokenizer mismatch error: {e}")
        raise TokenizerMismatchError(f"Failed to map tokens: {e}")

# Export main components
__all__ = [
    'ResourceLimits',
    'ResourceMonitor',
    'InputValidator',
    'RateLimiter',
    'ConnectionPool',
    'ModelCleanup',
    'FeatureFlags',
    'PerformanceLogger',
    'with_timeout',
    'with_graceful_degradation',
    'safe_model_load',
    'handle_tokenizer_mismatch',
    'SpeculativeDecodingError',
    'TokenizerMismatchError',
    'ModelLoadError',
    'InferenceError',
    'ResourceExhaustedError'
]