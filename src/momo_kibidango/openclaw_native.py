#!/usr/bin/env python3
"""
Native OpenClaw Integration for Speculative Decoding
Provides command-line interface and skill-compatible functionality
"""

import os
import sys
import json
import argparse
import yaml
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, asdict
import time

# Import production components
from speculative_3model_production import ProductionPyramidDecoder, ModelConfig, create_production_decoder
from monitoring import MetricsCollector, start_monitoring
from production_hardening import FeatureFlags, logger

@dataclass
class OpenClawConfig:
    """Configuration for OpenClaw integration"""
    # Model selection
    default_mode: str = "2model"  # Start conservative
    enable_3model: bool = True
    enable_2model: bool = True
    enable_fallback: bool = True
    
    # Performance settings
    max_batch_size: int = 8
    request_timeout: int = 300
    enable_streaming: bool = False
    
    # Monitoring
    enable_monitoring: bool = True
    monitoring_port: int = 8080
    metrics_export_interval: int = 60
    
    # Paths
    config_path: str = "~/.openclaw/workspace/skills/speculative-decoding/config.json"
    cache_path: str = "~/.openclaw/workspace/skills/speculative-decoding/cache"
    log_path: str = "~/.openclaw/workspace/skills/speculative-decoding/logs"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return asdict(self)
        
    @classmethod
    def from_file(cls, path: str) -> 'OpenClawConfig':
        """Load from JSON file"""
        path = os.path.expanduser(path)
        if os.path.exists(path):
            with open(path, 'r') as f:
                data = json.load(f)
                return cls(**data)
        return cls()
        
    def save(self, path: Optional[str] = None):
        """Save to JSON file"""
        path = os.path.expanduser(path or self.config_path)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)


class OpenClawSpeculativeDecoding:
    """OpenClaw skill implementation for speculative decoding"""
    
    def __init__(self, config: Optional[OpenClawConfig] = None):
        self.config = config or OpenClawConfig()
        self.decoder = None
        self.metrics_collector = None
        self.monitoring_server = None
        self.background_monitor = None
        
        # Initialize paths
        self._setup_paths()
        
        # Load feature flags
        self.feature_flags = self._load_feature_flags()
        
    def _setup_paths(self):
        """Create necessary directories"""
        for path_attr in ['cache_path', 'log_path']:
            path = os.path.expanduser(getattr(self.config, path_attr))
            os.makedirs(path, exist_ok=True)
            
    def _load_feature_flags(self) -> FeatureFlags:
        """Load feature flags from config"""
        flags = FeatureFlags()
        flags.set("enable_3model", self.config.enable_3model)
        flags.set("enable_2model", self.config.enable_2model)
        flags.set("enable_batch_inference", self.config.max_batch_size > 1)
        flags.set("enable_streaming", self.config.enable_streaming)
        flags.set("log_performance_metrics", self.config.enable_monitoring)
        return flags
        
    def initialize(self):
        """Initialize the decoder and monitoring"""
        logger.info("Initializing OpenClaw Speculative Decoding")
        
        try:
            # Create model configuration
            model_config = ModelConfig()
            model_config.enable_monitoring = self.config.enable_monitoring
            model_config.enable_fallback = self.config.enable_fallback
            model_config.timeout_seconds = self.config.request_timeout
            
            # Create decoder
            self.decoder = ProductionPyramidDecoder(model_config, self.feature_flags)
            
            # Load models
            logger.info("Loading models...")
            self.decoder.load_models()
            
            # Start monitoring if enabled
            if self.config.enable_monitoring:
                logger.info(f"Starting monitoring on port {self.config.monitoring_port}")
                self.metrics_collector, self.monitoring_server, self.background_monitor = \
                    start_monitoring(self.config.monitoring_port)
                    
            logger.info("Initialization complete")
            
        except Exception as e:
            logger.error(f"Failed to initialize: {e}")
            raise
            
    def generate(self, prompt: str, **kwargs) -> Dict[str, Any]:
        """Generate text using speculative decoding"""
        if not self.decoder:
            return {"error": "Decoder not initialized", "success": False}
            
        # Set default parameters
        params = {
            "max_length": kwargs.get("max_length", 100),
            "temperature": kwargs.get("temperature", 0.7),
            "top_p": kwargs.get("top_p", 0.9),
            "mode": kwargs.get("mode", self.config.default_mode)
        }
        
        # Perform generation
        start_time = time.time()
        result = self.decoder.generate(prompt, **params)
        
        # Add timing information
        result["generation_time"] = time.time() - start_time
        
        # Record metrics if available
        if self.metrics_collector and result.get("success"):
            self.metrics_collector.record_inference(
                duration_seconds=result["generation_time"],
                tokens_generated=result.get("tokens_generated", 0),
                model_mode=result.get("mode", "unknown"),
                success=True,
                acceptance_metrics=result.get("acceptance_metrics")
            )
            
        return result
        
    def batch_generate(self, prompts: List[str], **kwargs) -> List[Dict[str, Any]]:
        """Batch generation for multiple prompts"""
        if not self.config.max_batch_size > 1:
            # Fall back to sequential processing
            return [self.generate(prompt, **kwargs) for prompt in prompts]
            
        # TODO: Implement true batch processing
        # For now, process sequentially
        results = []
        for prompt in prompts[:self.config.max_batch_size]:
            results.append(self.generate(prompt, **kwargs))
            
        return results
        
    def get_status(self) -> Dict[str, Any]:
        """Get current system status"""
        status = {
            "initialized": self.decoder is not None,
            "config": self.config.to_dict(),
            "feature_flags": self.feature_flags.flags if self.feature_flags else {}
        }
        
        if self.decoder:
            status.update(self.decoder.get_status())
            
        if self.metrics_collector:
            status["metrics"] = self.metrics_collector.get_metrics_summary()
            
        return status
        
    def update_config(self, updates: Dict[str, Any]):
        """Update configuration dynamically"""
        for key, value in updates.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)
                logger.info(f"Updated config: {key} = {value}")
                
        # Update feature flags
        if "enable_3model" in updates:
            self.feature_flags.set("enable_3model", updates["enable_3model"])
        if "enable_2model" in updates:
            self.feature_flags.set("enable_2model", updates["enable_2model"])
            
        # Save updated config
        self.config.save()
        
    def shutdown(self):
        """Clean shutdown"""
        logger.info("Shutting down OpenClaw Speculative Decoding")
        
        if self.decoder:
            self.decoder.shutdown()
            
        if self.background_monitor:
            self.background_monitor.stop()
            
        logger.info("Shutdown complete")


def create_cli_parser() -> argparse.ArgumentParser:
    """Create command-line argument parser"""
    parser = argparse.ArgumentParser(
        description="OpenClaw Speculative Decoding - Fast text generation with quality preservation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate text with default settings
  openclaw-speculative "What is machine learning?"
  
  # Use 3-model pyramid for maximum speed
  openclaw-speculative --mode 3model "Explain quantum computing"
  
  # Batch generation from file
  openclaw-speculative --batch prompts.txt --output results.json
  
  # Check system status
  openclaw-speculative --status
  
  # Update configuration
  openclaw-speculative --config enable_3model=true default_mode=3model
        """
    )
    
    # Main arguments
    parser.add_argument("prompt", nargs="?", help="Text prompt for generation")
    parser.add_argument("--mode", choices=["1model", "2model", "3model"], 
                       help="Generation mode (default: from config)")
    parser.add_argument("--max-length", type=int, default=100,
                       help="Maximum length of generated text")
    parser.add_argument("--temperature", type=float, default=0.7,
                       help="Sampling temperature (0.0-1.0)")
    parser.add_argument("--top-p", type=float, default=0.9,
                       help="Nucleus sampling threshold")
    
    # Batch processing
    parser.add_argument("--batch", help="File containing prompts (one per line)")
    parser.add_argument("--output", help="Output file for results (JSON format)")
    
    # System operations
    parser.add_argument("--initialize", action="store_true",
                       help="Initialize models and monitoring")
    parser.add_argument("--status", action="store_true",
                       help="Show system status")
    parser.add_argument("--shutdown", action="store_true",
                       help="Clean shutdown of system")
    
    # Configuration
    parser.add_argument("--config", nargs="+",
                       help="Update configuration (key=value pairs)")
    parser.add_argument("--config-file", 
                       help="Path to configuration file")
    parser.add_argument("--show-config", action="store_true",
                       help="Show current configuration")
    
    # Monitoring
    parser.add_argument("--metrics", action="store_true",
                       help="Show current metrics")
    parser.add_argument("--monitoring-port", type=int,
                       help="Port for monitoring server")
    
    return parser


def main():
    """Main CLI entry point"""
    parser = create_cli_parser()
    args = parser.parse_args()
    
    # Load configuration
    config = OpenClawConfig()
    if args.config_file:
        config = OpenClawConfig.from_file(args.config_file)
    if args.monitoring_port:
        config.monitoring_port = args.monitoring_port
        
    # Create skill instance
    skill = OpenClawSpeculativeDecoding(config)
    
    # Handle configuration updates
    if args.config:
        updates = {}
        for item in args.config:
            if "=" in item:
                key, value = item.split("=", 1)
                # Try to parse value as JSON
                try:
                    value = json.loads(value)
                except:
                    # Keep as string if not valid JSON
                    pass
                updates[key] = value
        skill.update_config(updates)
        print("Configuration updated successfully")
        
    # Show configuration
    if args.show_config:
        print(json.dumps(skill.config.to_dict(), indent=2))
        return
        
    # Initialize if needed
    if args.initialize or (args.prompt and not skill.decoder):
        print("Initializing speculative decoding system...")
        skill.initialize()
        print("Initialization complete")
        
    # Show status
    if args.status:
        status = skill.get_status()
        print(json.dumps(status, indent=2))
        return
        
    # Show metrics
    if args.metrics:
        if skill.metrics_collector:
            metrics = skill.metrics_collector.get_metrics_summary()
            print(json.dumps(metrics, indent=2))
        else:
            print("Monitoring not enabled")
        return
        
    # Shutdown
    if args.shutdown:
        skill.shutdown()
        print("Shutdown complete")
        return
        
    # Batch processing
    if args.batch:
        if not os.path.exists(args.batch):
            print(f"Error: Batch file '{args.batch}' not found")
            sys.exit(1)
            
        with open(args.batch, 'r') as f:
            prompts = [line.strip() for line in f if line.strip()]
            
        print(f"Processing {len(prompts)} prompts...")
        results = skill.batch_generate(prompts, mode=args.mode, 
                                      max_length=args.max_length,
                                      temperature=args.temperature,
                                      top_p=args.top_p)
                                      
        # Save results
        if args.output:
            with open(args.output, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"Results saved to {args.output}")
        else:
            for i, result in enumerate(results):
                print(f"\n--- Prompt {i+1} ---")
                print(f"Output: {result.get('output', 'Error: ' + result.get('error', 'Unknown'))}")
                
    # Single prompt generation
    elif args.prompt:
        result = skill.generate(
            args.prompt,
            mode=args.mode,
            max_length=args.max_length,
            temperature=args.temperature,
            top_p=args.top_p
        )
        
        if result.get("success"):
            print(f"\nGenerated text:\n{result['output']}")
            print(f"\nTokens: {result.get('tokens_generated', 'Unknown')}")
            print(f"Time: {result.get('generation_time', 0):.2f}s")
            print(f"Mode: {result.get('mode', 'Unknown')}")
        else:
            print(f"Error: {result.get('error', 'Unknown error')}")
            
    else:
        parser.print_help()


if __name__ == "__main__":
    main()