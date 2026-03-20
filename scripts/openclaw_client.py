#!/usr/bin/env python3
"""
OpenClaw Client - CLI tool to interact with the speculative decoding server
"""

import requests
import json
import argparse
import sys
from typing import Dict, Optional


class OpenClawClient:
    def __init__(self, base_url: str = "http://localhost:5000"):
        self.base_url = base_url
        
    def health(self) -> Dict:
        """Check server health"""
        response = requests.get(f"{self.base_url}/health")
        return response.json()
        
    def get_config(self) -> Dict:
        """Get current configuration"""
        response = requests.get(f"{self.base_url}/config")
        return response.json()
        
    def update_config(self, **kwargs) -> Dict:
        """Update configuration"""
        response = requests.post(f"{self.base_url}/config", json=kwargs)
        return response.json()
        
    def infer(self, prompt: str, max_length: int = 100) -> Dict:
        """Run inference"""
        response = requests.post(f"{self.base_url}/infer", json={
            "prompt": prompt,
            "max_length": max_length
        })
        return response.json()
        
    def get_metrics(self) -> Dict:
        """Get inference metrics"""
        response = requests.get(f"{self.base_url}/metrics")
        return response.json()
        
    def reload_models(self) -> Dict:
        """Force reload models"""
        response = requests.post(f"{self.base_url}/models/reload")
        return response.json()


def main():
    parser = argparse.ArgumentParser(description="OpenClaw Speculative Decoding Client")
    parser.add_argument("--server", default="http://localhost:5000", help="Server URL")
    
    subparsers = parser.add_subparsers(dest="command", help="Commands")
    
    # Health command
    subparsers.add_parser("health", help="Check server health")
    
    # Config commands
    config_parser = subparsers.add_parser("config", help="Get/set configuration")
    config_parser.add_argument("--get", action="store_true", help="Get current config")
    config_parser.add_argument("--use-3model", type=bool, help="Enable/disable 3-model")
    config_parser.add_argument("--auto-fallback", type=bool, help="Enable/disable auto fallback")
    config_parser.add_argument("--max-memory", type=float, help="Max memory in GB")
    
    # Infer command
    infer_parser = subparsers.add_parser("infer", help="Run inference")
    infer_parser.add_argument("prompt", help="Text prompt")
    infer_parser.add_argument("--max-length", type=int, default=100, help="Max tokens to generate")
    
    # Metrics command
    subparsers.add_parser("metrics", help="Get inference metrics")
    
    # Reload command
    subparsers.add_parser("reload", help="Reload models")
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        sys.exit(1)
        
    client = OpenClawClient(args.server)
    
    try:
        if args.command == "health":
            result = client.health()
            print(json.dumps(result, indent=2))
            
        elif args.command == "config":
            if args.get or not any([args.use_3model is not None, 
                                   args.auto_fallback is not None,
                                   args.max_memory is not None]):
                result = client.get_config()
                print(json.dumps(result, indent=2))
            else:
                kwargs = {}
                if args.use_3model is not None:
                    kwargs["use_3model"] = args.use_3model
                if args.auto_fallback is not None:
                    kwargs["auto_fallback"] = args.auto_fallback
                if args.max_memory is not None:
                    kwargs["max_memory_gb"] = args.max_memory
                    
                result = client.update_config(**kwargs)
                print(f"Configuration updated: {result}")
                
        elif args.command == "infer":
            print(f"Generating from prompt: '{args.prompt}'")
            print("-" * 80)
            
            result = client.infer(args.prompt, args.max_length)
            
            if "error" in result:
                print(f"Error: {result['error']}")
                sys.exit(1)
                
            print(f"\nGenerated text:\n{result['text']}")
            print(f"\nMetrics:")
            metrics = result["metrics"]
            print(f"  Mode: {metrics['mode']}")
            print(f"  Throughput: {metrics['throughput']:.2f} tokens/sec")
            print(f"  Memory: {metrics['memory_gb']:.2f} GB")
            
            if "stage1_acceptance" in metrics:
                print(f"  Stage 1 acceptance: {metrics['stage1_acceptance']:.2%}")
                print(f"  Stage 2 acceptance: {metrics['stage2_acceptance']:.2%}")
                print(f"  Combined acceptance: {metrics['combined_acceptance']:.2%}")
            elif "acceptance_rate" in metrics:
                print(f"  Acceptance rate: {metrics['acceptance_rate']:.2%}")
                
        elif args.command == "metrics":
            result = client.metrics()
            
            if "summary" in result and result["summary"]:
                print("Summary Statistics:")
                summary = result["summary"]
                print(f"  Total inferences: {summary.get('total_inferences', 0)}")
                print(f"  Avg throughput: {summary.get('avg_throughput', 0):.2f} tokens/sec")
                print(f"  Max throughput: {summary.get('max_throughput', 0):.2f} tokens/sec")
                print(f"  Avg memory: {summary.get('avg_memory_gb', 0):.2f} GB")
                
                if "mode_distribution" in summary:
                    print("\n  Mode distribution:")
                    for mode, count in summary["mode_distribution"].items():
                        print(f"    {mode}: {count}")
                        
            if "recent_metrics" in result and result["recent_metrics"]:
                print("\nRecent inferences:")
                for m in result["recent_metrics"][-5:]:
                    print(f"  - {m.get('timestamp', 'N/A')}: "
                          f"{m.get('throughput', 0):.1f} tok/s, "
                          f"{m.get('mode', 'unknown')}")
                          
        elif args.command == "reload":
            print("Reloading models...")
            result = client.reload_models()
            print(json.dumps(result, indent=2))
            
    except requests.exceptions.ConnectionError:
        print(f"Error: Could not connect to server at {args.server}")
        print("Is the server running? Start with: python src/openclaw_integration.py")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()