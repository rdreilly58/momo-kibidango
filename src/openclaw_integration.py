#!/usr/bin/env python3
"""
OpenClaw Integration for Speculative Decoding
Supports 1-model, 2-model, and 3-model pyramid architectures

API endpoints:
- POST /infer - Generate text using best available model configuration
- GET /config - Get current configuration
- POST /config - Update configuration
"""

import os
import json
import torch
import gc
from flask import Flask, request, jsonify
from typing import Dict, Optional
from dataclasses import asdict
from datetime import datetime

# Import our implementations
from speculative_2model import SpeculativeDecoder as TwoModelDecoder
from speculative_2model import ModelConfig as TwoModelConfig
from speculative_3model import PyramidSpeculativeDecoder as ThreeModelDecoder
from speculative_3model import ModelConfig as ThreeModelConfig

app = Flask(__name__)

# Global configuration
class IntegrationConfig:
    def __init__(self):
        self.use_3model = False  # Default to proven 2-model
        self.auto_fallback = True  # Automatically fall back on OOM
        self.max_memory_gb = 12.0  # Maximum memory before fallback
        self.log_metrics = True
        self.metrics_file = "logs/inference_metrics.jsonl"
        
        # Model-specific configs
        self.two_model_config = TwoModelConfig(
            draft_model_id="Qwen/Qwen2.5-1.5B-Instruct",
            target_model_id="Qwen/Qwen2.5-7B-Instruct"
        )
        self.three_model_config = ThreeModelConfig()
        
        # Cached decoders
        self._two_model_decoder = None
        self._three_model_decoder = None
        
config = IntegrationConfig()


def log_metrics(metrics: Dict):
    """Log metrics to file for analysis"""
    if not config.log_metrics:
        return
        
    # Ensure logs directory exists
    os.makedirs("logs", exist_ok=True)
    
    # Add timestamp
    metrics["timestamp"] = datetime.utcnow().isoformat()
    
    # Append to JSONL file
    with open(config.metrics_file, "a") as f:
        f.write(json.dumps(metrics) + "\n")


def get_memory_usage() -> float:
    """Get current memory usage in GB"""
    import psutil
    process = psutil.Process()
    return process.memory_info().rss / 1e9


def cleanup_memory():
    """Force garbage collection and clear caches"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    elif hasattr(torch.mps, 'empty_cache'):
        # For Apple Silicon (if available in your PyTorch version)
        try:
            torch.mps.empty_cache()
        except:
            pass


def get_two_model_decoder() -> TwoModelDecoder:
    """Get or create 2-model decoder (lazy loading)"""
    if config._two_model_decoder is None:
        print("Initializing 2-model decoder...")
        config._two_model_decoder = TwoModelDecoder(config.two_model_config)
    return config._two_model_decoder


def get_three_model_decoder() -> Optional[ThreeModelDecoder]:
    """Get or create 3-model decoder with memory checking"""
    if config._three_model_decoder is None:
        # Check memory before loading
        current_memory = get_memory_usage()
        if current_memory > config.max_memory_gb - 4.0:  # Need 4GB headroom
            print(f"⚠️ Memory too high ({current_memory:.1f}GB) for 3-model")
            return None
            
        try:
            print("Initializing 3-model pyramid decoder...")
            config._three_model_decoder = ThreeModelDecoder(config.three_model_config)
        except (RuntimeError, MemoryError) as e:
            print(f"Failed to load 3-model: {e}")
            cleanup_memory()
            return None
            
    return config._three_model_decoder


@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "memory_gb": get_memory_usage(),
        "use_3model": config.use_3model,
        "models_loaded": {
            "2model": config._two_model_decoder is not None,
            "3model": config._three_model_decoder is not None
        }
    })


@app.route('/config', methods=['GET', 'POST'])
def handle_config():
    """Get or update configuration"""
    if request.method == 'GET':
        return jsonify({
            "use_3model": config.use_3model,
            "auto_fallback": config.auto_fallback,
            "max_memory_gb": config.max_memory_gb,
            "log_metrics": config.log_metrics,
            "two_model_config": asdict(config.two_model_config),
            "three_model_config": asdict(config.three_model_config)
        })
    
    # POST - update configuration
    data = request.get_json()
    
    if "use_3model" in data:
        config.use_3model = bool(data["use_3model"])
        
    if "auto_fallback" in data:
        config.auto_fallback = bool(data["auto_fallback"])
        
    if "max_memory_gb" in data:
        config.max_memory_gb = float(data["max_memory_gb"])
        
    if "log_metrics" in data:
        config.log_metrics = bool(data["log_metrics"])
    
    # Clear cached decoders if switching modes
    if "use_3model" in data:
        if data["use_3model"] and config._two_model_decoder:
            print("Clearing 2-model decoder to free memory")
            del config._two_model_decoder
            config._two_model_decoder = None
            cleanup_memory()
        elif not data["use_3model"] and config._three_model_decoder:
            print("Clearing 3-model decoder to free memory")
            del config._three_model_decoder  
            config._three_model_decoder = None
            cleanup_memory()
    
    return jsonify({"status": "updated"})


@app.route('/infer', methods=['POST'])
def infer():
    """Main inference endpoint"""
    data = request.get_json()
    
    # Validate input
    if not data or "prompt" not in data:
        return jsonify({"error": "Missing 'prompt' in request"}), 400
        
    prompt = data["prompt"]
    max_length = data.get("max_length", 100)
    
    # Track inference mode
    inference_mode = "3model" if config.use_3model else "2model"
    
    try:
        # Try 3-model if enabled
        if config.use_3model:
            decoder = get_three_model_decoder()
            if decoder:
                print(f"Using 3-model pyramid for inference")
                metrics = decoder.generate(prompt, max_length)
                metrics["mode"] = "3model"
                log_metrics(metrics)
                return jsonify({
                    "text": metrics["generated_text"],
                    "metrics": {
                        "throughput": metrics["throughput"],
                        "mode": "3model",
                        "memory_gb": metrics["peak_memory_gb"],
                        "stage1_acceptance": metrics.get("stage1_acceptance_rate", 0),
                        "stage2_acceptance": metrics.get("stage2_acceptance_rate", 0),
                        "combined_acceptance": metrics.get("combined_acceptance_rate", 0)
                    }
                })
            elif config.auto_fallback:
                print("3-model unavailable, falling back to 2-model")
                inference_mode = "2model_fallback"
            else:
                return jsonify({"error": "3-model unavailable and auto_fallback disabled"}), 503
                
        # Use 2-model (default or fallback)
        decoder = get_two_model_decoder()
        print(f"Using 2-model for inference")
        metrics = decoder.generate(prompt, max_length)
        metrics["mode"] = inference_mode
        log_metrics(metrics)
        
        return jsonify({
            "text": metrics["generated_text"],
            "metrics": {
                "throughput": metrics["throughput"],
                "mode": inference_mode,
                "memory_gb": metrics["peak_memory_gb"],
                "acceptance_rate": metrics.get("acceptance_rate", 0)
            }
        })
        
    except (RuntimeError, MemoryError, torch.cuda.OutOfMemoryError) as e:
        if config.auto_fallback and inference_mode == "3model":
            # Try 2-model fallback
            print(f"3-model OOM, falling back to 2-model: {e}")
            cleanup_memory()
            
            try:
                decoder = get_two_model_decoder()
                metrics = decoder.generate(prompt, max_length)
                metrics["mode"] = "2model_oom_fallback"
                log_metrics(metrics)
                
                return jsonify({
                    "text": metrics["generated_text"],
                    "metrics": {
                        "throughput": metrics["throughput"],
                        "mode": "2model_oom_fallback",
                        "memory_gb": metrics["peak_memory_gb"],
                        "acceptance_rate": metrics.get("acceptance_rate", 0),
                        "fallback_reason": "3model_oom"
                    }
                })
            except Exception as e2:
                return jsonify({"error": f"All models failed: {e2}"}), 503
        else:
            return jsonify({"error": f"Inference failed: {e}"}), 503
            
    except Exception as e:
        return jsonify({"error": f"Unexpected error: {e}"}), 500


@app.route('/metrics', methods=['GET'])
def get_metrics():
    """Get recent inference metrics"""
    if not os.path.exists(config.metrics_file):
        return jsonify({"metrics": []})
        
    # Read last 100 metrics
    metrics = []
    with open(config.metrics_file, "r") as f:
        lines = f.readlines()
        for line in lines[-100:]:
            try:
                metrics.append(json.loads(line))
            except:
                pass
                
    # Calculate summary statistics
    if metrics:
        throughputs = [m["throughput"] for m in metrics if "throughput" in m]
        memories = [m.get("peak_memory_gb", 0) for m in metrics]
        modes = {}
        for m in metrics:
            mode = m.get("mode", "unknown")
            modes[mode] = modes.get(mode, 0) + 1
            
        summary = {
            "total_inferences": len(metrics),
            "avg_throughput": sum(throughputs) / len(throughputs) if throughputs else 0,
            "max_throughput": max(throughputs) if throughputs else 0,
            "avg_memory_gb": sum(memories) / len(memories) if memories else 0,
            "mode_distribution": modes
        }
    else:
        summary = {}
        
    return jsonify({
        "summary": summary,
        "recent_metrics": metrics[-10:]  # Last 10 inferences
    })


@app.route('/models/reload', methods=['POST'])
def reload_models():
    """Force reload models (useful after configuration change)"""
    # Clear existing models
    if config._two_model_decoder:
        del config._two_model_decoder
        config._two_model_decoder = None
        
    if config._three_model_decoder:
        del config._three_model_decoder
        config._three_model_decoder = None
        
    cleanup_memory()
    
    # Pre-load based on configuration
    if config.use_3model:
        decoder = get_three_model_decoder()
        if not decoder:
            return jsonify({"status": "failed", "error": "Could not load 3-model"}), 503
    else:
        decoder = get_two_model_decoder()
        
    return jsonify({
        "status": "reloaded",
        "mode": "3model" if config.use_3model else "2model",
        "memory_gb": get_memory_usage()
    })


def run_server(host='0.0.0.0', port=5000):
    """Run the Flask server"""
    print(f"Starting OpenClaw Speculative Decoding server on {host}:{port}")
    print(f"Default mode: {'3-model pyramid' if config.use_3model else '2-model'}")
    print(f"Auto-fallback: {config.auto_fallback}")
    
    # Pre-load default model
    if config.use_3model:
        get_three_model_decoder()
    else:
        get_two_model_decoder()
        
    app.run(host=host, port=port, debug=False)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="OpenClaw Speculative Decoding Server")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=5000, help="Port to listen on")
    parser.add_argument("--use-3model", action="store_true", help="Start with 3-model enabled")
    args = parser.parse_args()
    
    if args.use_3model:
        config.use_3model = True
        
    run_server(args.host, args.port)