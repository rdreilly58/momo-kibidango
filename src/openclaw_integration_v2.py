#!/usr/bin/env python3
"""
Enhanced OpenClaw REST API Integration
With production hardening, monitoring, and feature flags
"""

import os
import json
import time
import asyncio
import queue
import threading
from flask import Flask, request, jsonify, Response
from flask_cors import CORS
from typing import Dict, Optional, Any, List
from dataclasses import asdict
from datetime import datetime

# Import production components
from openclaw_native import OpenClawSpeculativeDecoding, OpenClawConfig
from monitoring import MetricsCollector
from production_hardening import FeatureFlags, RateLimiter, logger

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for web frontends

# Global instances
skill: Optional[OpenClawSpeculativeDecoding] = None
request_queue: Optional[queue.Queue] = None
worker_thread: Optional[threading.Thread] = None


class RequestProcessor(threading.Thread):
    """Background thread for processing queued requests"""
    
    def __init__(self, skill: OpenClawSpeculativeDecoding, request_queue: queue.Queue):
        super().__init__(daemon=True)
        self.skill = skill
        self.queue = request_queue
        self.running = True
        
    def run(self):
        """Process requests from queue"""
        while self.running:
            try:
                # Get request from queue with timeout
                request_data = self.queue.get(timeout=1.0)
                
                if request_data is None:  # Shutdown signal
                    break
                    
                # Process request
                request_id = request_data["id"]
                prompt = request_data["prompt"]
                params = request_data["params"]
                
                # Generate response
                result = self.skill.generate(prompt, **params)
                
                # Store result for retrieval
                request_data["result"] = result
                request_data["status"] = "complete"
                
                self.queue.task_done()
                
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Error processing request: {e}")
                if "request_data" in locals():
                    request_data["result"] = {"error": str(e), "success": False}
                    request_data["status"] = "error"


@app.before_first_request
def initialize():
    """Initialize the system on first request"""
    global skill, request_queue, worker_thread
    
    if skill is None:
        logger.info("Initializing OpenClaw Speculative Decoding API")
        
        # Load configuration
        config = OpenClawConfig()
        config_path = os.environ.get("SPECULATIVE_CONFIG", config.config_path)
        if os.path.exists(os.path.expanduser(config_path)):
            config = OpenClawConfig.from_file(config_path)
            
        # Create skill instance
        skill = OpenClawSpeculativeDecoding(config)
        
        # Initialize models
        skill.initialize()
        
        # Create request queue if batch processing enabled
        if config.max_batch_size > 1:
            request_queue = queue.Queue(maxsize=100)
            worker_thread = RequestProcessor(skill, request_queue)
            worker_thread.start()
            
        logger.info("API initialization complete")


@app.route("/health", methods=["GET"])
def health():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "version": "1.0.0"
    })


@app.route("/ready", methods=["GET"])
def ready():
    """Readiness check endpoint"""
    if skill is None:
        return jsonify({
            "status": "not_ready",
            "reason": "not initialized"
        }), 503
        
    status = skill.get_status()
    
    if not status.get("initialized"):
        return jsonify({
            "status": "not_ready",
            "reason": "models not loaded"
        }), 503
        
    return jsonify({
        "status": "ready",
        "timestamp": datetime.utcnow().isoformat()
    })


@app.route("/infer", methods=["POST"])
def infer():
    """Main inference endpoint"""
    try:
        # Get request data
        data = request.json
        if not data or "prompt" not in data:
            return jsonify({"error": "Missing 'prompt' field"}), 400
            
        prompt = data["prompt"]
        
        # Extract parameters
        params = {
            "max_length": data.get("max_length", 100),
            "temperature": data.get("temperature", 0.7),
            "top_p": data.get("top_p", 0.9),
            "mode": data.get("mode")  # None = auto-select
        }
        
        # Check if streaming requested
        if data.get("stream", False) and skill.config.enable_streaming:
            return stream_inference(prompt, params)
            
        # Perform inference
        result = skill.generate(prompt, **params)
        
        # Add metadata
        result["api_version"] = "2.0"
        result["request_id"] = request.headers.get("X-Request-ID", "")
        
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Inference error: {e}")
        return jsonify({
            "error": str(e),
            "success": False
        }), 500


@app.route("/batch", methods=["POST"])
def batch_infer():
    """Batch inference endpoint"""
    try:
        data = request.json
        if not data or "prompts" not in data:
            return jsonify({"error": "Missing 'prompts' field"}), 400
            
        prompts = data["prompts"]
        if not isinstance(prompts, list):
            return jsonify({"error": "'prompts' must be a list"}), 400
            
        # Extract common parameters
        params = {
            "max_length": data.get("max_length", 100),
            "temperature": data.get("temperature", 0.7),
            "top_p": data.get("top_p", 0.9),
            "mode": data.get("mode")
        }
        
        # Process batch
        results = skill.batch_generate(prompts, **params)
        
        return jsonify({
            "results": results,
            "batch_size": len(prompts),
            "api_version": "2.0"
        })
        
    except Exception as e:
        logger.error(f"Batch inference error: {e}")
        return jsonify({
            "error": str(e),
            "success": False
        }), 500


@app.route("/async", methods=["POST"])
def async_infer():
    """Asynchronous inference endpoint"""
    if not request_queue:
        return jsonify({
            "error": "Async processing not enabled",
            "success": False
        }), 501
        
    try:
        data = request.json
        if not data or "prompt" not in data:
            return jsonify({"error": "Missing 'prompt' field"}), 400
            
        # Create request object
        request_id = f"req_{int(time.time() * 1000)}"
        request_data = {
            "id": request_id,
            "prompt": data["prompt"],
            "params": {
                "max_length": data.get("max_length", 100),
                "temperature": data.get("temperature", 0.7),
                "top_p": data.get("top_p", 0.9),
                "mode": data.get("mode")
            },
            "status": "queued",
            "result": None
        }
        
        # Add to queue
        try:
            request_queue.put_nowait(request_data)
        except queue.Full:
            return jsonify({
                "error": "Request queue full",
                "success": False
            }), 503
            
        return jsonify({
            "request_id": request_id,
            "status": "queued",
            "queue_position": request_queue.qsize()
        }), 202
        
    except Exception as e:
        logger.error(f"Async inference error: {e}")
        return jsonify({
            "error": str(e),
            "success": False
        }), 500


@app.route("/async/<request_id>", methods=["GET"])
def get_async_result(request_id):
    """Get result of async request"""
    # In production, this would check a persistent store
    # For now, return not implemented
    return jsonify({
        "error": "Result retrieval not implemented",
        "success": False
    }), 501


def stream_inference(prompt: str, params: Dict[str, Any]):
    """Stream inference results using Server-Sent Events"""
    def generate():
        # TODO: Implement true streaming generation
        # For now, simulate streaming with chunks
        
        yield f"data: {json.dumps({'event': 'start', 'request_id': time.time()})}\n\n"
        
        # Generate full response
        result = skill.generate(prompt, **params)
        
        if result.get("success"):
            # Simulate streaming by sending tokens in chunks
            output = result.get("output", "")
            words = output.split()
            
            for i in range(0, len(words), 5):
                chunk = " ".join(words[i:i+5])
                yield f"data: {json.dumps({'event': 'token', 'content': chunk})}\n\n"
                time.sleep(0.1)  # Simulate generation delay
                
            yield f"data: {json.dumps({'event': 'complete', 'tokens_generated': result.get('tokens_generated', 0)})}\n\n"
        else:
            yield f"data: {json.dumps({'event': 'error', 'error': result.get('error', 'Unknown error')})}\n\n"
            
    return Response(generate(), mimetype="text/event-stream")


@app.route("/config", methods=["GET"])
def get_config():
    """Get current configuration"""
    if skill is None:
        return jsonify({"error": "Not initialized"}), 503
        
    return jsonify(skill.config.to_dict())


@app.route("/config", methods=["POST", "PATCH"])
def update_config():
    """Update configuration"""
    if skill is None:
        return jsonify({"error": "Not initialized"}), 503
        
    try:
        updates = request.json
        if not updates:
            return jsonify({"error": "No updates provided"}), 400
            
        # Apply updates
        skill.update_config(updates)
        
        return jsonify({
            "success": True,
            "updated": list(updates.keys()),
            "config": skill.config.to_dict()
        })
        
    except Exception as e:
        logger.error(f"Config update error: {e}")
        return jsonify({
            "error": str(e),
            "success": False
        }), 500


@app.route("/metrics", methods=["GET"])
def metrics():
    """Get metrics in JSON format"""
    if skill is None or not skill.metrics_collector:
        return jsonify({"error": "Monitoring not enabled"}), 501
        
    metrics_summary = skill.metrics_collector.get_metrics_summary()
    return jsonify(metrics_summary)


@app.route("/status", methods=["GET"])
def status():
    """Get comprehensive system status"""
    if skill is None:
        return jsonify({
            "initialized": False,
            "error": "System not initialized"
        })
        
    return jsonify(skill.get_status())


@app.route("/features", methods=["GET"])
def get_features():
    """Get feature flags"""
    if skill is None:
        return jsonify({"error": "Not initialized"}), 503
        
    return jsonify(skill.feature_flags.flags)


@app.route("/features/<flag_name>", methods=["PUT"])
def set_feature(flag_name):
    """Set a feature flag"""
    if skill is None:
        return jsonify({"error": "Not initialized"}), 503
        
    try:
        data = request.json
        if "value" not in data:
            return jsonify({"error": "Missing 'value' field"}), 400
            
        skill.feature_flags.set(flag_name, bool(data["value"]))
        
        return jsonify({
            "success": True,
            "flag": flag_name,
            "value": skill.feature_flags.get(flag_name)
        })
        
    except Exception as e:
        logger.error(f"Feature flag error: {e}")
        return jsonify({
            "error": str(e),
            "success": False
        }), 500


@app.route("/shutdown", methods=["POST"])
def shutdown():
    """Graceful shutdown endpoint"""
    if skill is None:
        return jsonify({"error": "Not initialized"}), 503
        
    try:
        # Stop worker thread if running
        if worker_thread and worker_thread.is_alive():
            request_queue.put(None)  # Shutdown signal
            worker_thread.join(timeout=5)
            
        # Shutdown skill
        skill.shutdown()
        
        return jsonify({
            "success": True,
            "message": "Shutdown complete"
        })
        
    except Exception as e:
        logger.error(f"Shutdown error: {e}")
        return jsonify({
            "error": str(e),
            "success": False
        }), 500


def create_app():
    """Factory function to create Flask app"""
    # Initialize on app creation
    initialize()
    return app


if __name__ == "__main__":
    # Run development server
    port = int(os.environ.get("PORT", 5000))
    debug = os.environ.get("DEBUG", "false").lower() == "true"
    
    app.run(host="0.0.0.0", port=port, debug=debug)