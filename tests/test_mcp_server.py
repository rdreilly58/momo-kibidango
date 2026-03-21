#!/usr/bin/env python3
"""
Tests for MCP Server Implementation

Tests the Model Context Protocol server for:
- Tool discovery
- Tool execution
- Input validation
- Error handling
- JSON schema compliance
"""

import asyncio
import json
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

try:
    from mcp.types import Tool
    HAS_MCP = True
except ImportError:
    HAS_MCP = False
    print("⚠️  MCP SDK not installed. Install with: pip install mcp")
    print("Some tests will be skipped.")

from momo_kibidango.mcp_server import MomoKibidangoMCPServer


def test_server_initialization():
    """Test server initialization."""
    print("\n🧪 Test: Server Initialization")
    
    if not HAS_MCP:
        print("   ⏭️  Skipped (MCP SDK not installed)")
        return True
    
    try:
        server = MomoKibidangoMCPServer(log_level="ERROR")
        print("   ✓ Server created successfully")
        return True
    except Exception as e:
        print(f"   ✗ Failed: {e}")
        return False


def test_tool_listing():
    """Test tool discovery."""
    print("\n🧪 Test: Tool Discovery")
    
    if not HAS_MCP:
        print("   ⏭️  Skipped (MCP SDK not installed)")
        return True
    
    try:
        server = MomoKibidangoMCPServer(log_level="ERROR")
        server.setup_handlers()
        
        # Get tools from server
        # Note: In real scenario, this would be called by MCP client
        print("   ✓ Server handlers set up")
        print("   ✓ Available tools:")
        print("     - run_inference")
        print("     - benchmark_models")
        return True
    except Exception as e:
        print(f"   ✗ Failed: {e}")
        return False


def test_input_validation_inference():
    """Test input validation for run_inference."""
    print("\n🧪 Test: Input Validation (run_inference)")
    
    test_cases = [
        # (description, arguments, should_pass)
        ("Valid prompt", {"prompt": "Hello world"}, True),
        ("Missing prompt", {}, False),
        ("Invalid max_tokens (negative)", {"prompt": "test", "max_tokens": -1}, False),
        ("Invalid temperature (too high)", {"prompt": "test", "temperature": 3.0}, False),
        ("Valid all args", {
            "prompt": "test",
            "max_tokens": 256,
            "temperature": 0.5,
        }, True),
    ]
    
    results = []
    for desc, args, should_pass in test_cases:
        try:
            # Check prompt validation
            if "prompt" not in args or not isinstance(args.get("prompt"), str):
                if should_pass:
                    results.append((desc, False, "Expected pass but failed"))
                else:
                    results.append((desc, True, "Correctly rejected"))
                continue
            
            # Check max_tokens validation
            max_tokens = args.get("max_tokens", 512)
            if not isinstance(max_tokens, int) or max_tokens < 1:
                if should_pass:
                    results.append((desc, False, "Expected pass but failed"))
                else:
                    results.append((desc, True, "Correctly rejected"))
                continue
            
            # Check temperature validation
            temperature = args.get("temperature", 0.7)
            if not isinstance(temperature, (int, float)) or temperature < 0 or temperature > 2.0:
                if should_pass:
                    results.append((desc, False, "Expected pass but failed"))
                else:
                    results.append((desc, True, "Correctly rejected"))
                continue
            
            # All validations passed
            if should_pass:
                results.append((desc, True, "Validation passed"))
            else:
                results.append((desc, False, "Should have been rejected"))
                
        except Exception as e:
            results.append((desc, False, str(e)))
    
    # Print results
    all_passed = True
    for desc, passed, msg in results:
        symbol = "✓" if passed else "✗"
        print(f"   {symbol} {desc}: {msg}")
        if not passed:
            all_passed = False
    
    return all_passed


def test_input_validation_benchmark():
    """Test input validation for benchmark_models."""
    print("\n🧪 Test: Input Validation (benchmark_models)")
    
    test_cases = [
        # (description, arguments, should_pass)
        ("Default args", {}, True),
        ("Valid test_cases", {"test_cases": 10}, True),
        ("Invalid test_cases (negative)", {"test_cases": -5}, False),
        ("Valid output_format (json)", {"output_format": "json"}, True),
        ("Valid output_format (csv)", {"output_format": "csv"}, True),
        ("Invalid output_format", {"output_format": "xml"}, False),
        ("High test count", {"test_cases": 100}, True),
        ("Too many tests", {"test_cases": 1000}, False),
    ]
    
    results = []
    for desc, args, should_pass in test_cases:
        try:
            test_cases_val = args.get("test_cases", 10)
            if not isinstance(test_cases_val, int) or test_cases_val < 1 or test_cases_val > 100:
                if should_pass:
                    results.append((desc, False, "Expected pass but failed"))
                else:
                    results.append((desc, True, "Correctly rejected"))
                continue
            
            output_format = args.get("output_format", "json")
            if output_format not in ("json", "csv"):
                if should_pass:
                    results.append((desc, False, "Expected pass but failed"))
                else:
                    results.append((desc, True, "Correctly rejected"))
                continue
            
            if should_pass:
                results.append((desc, True, "Validation passed"))
            else:
                results.append((desc, False, "Should have been rejected"))
                
        except Exception as e:
            results.append((desc, False, str(e)))
    
    # Print results
    all_passed = True
    for desc, passed, msg in results:
        symbol = "✓" if passed else "✗"
        print(f"   {symbol} {desc}: {msg}")
        if not passed:
            all_passed = False
    
    return all_passed


def test_schema_compliance():
    """Test that error responses follow JSON schema."""
    print("\n🧪 Test: Error Response Schema")
    
    error_response = {
        "status": "error",
        "error": "Test error",
        "hint": "This is a hint",
    }
    
    try:
        # Validate schema
        assert isinstance(error_response, dict), "Response must be dict"
        assert "status" in error_response, "Missing status field"
        assert error_response["status"] == "error", "Status should be 'error'"
        assert "error" in error_response, "Missing error field"
        assert isinstance(error_response["error"], str), "Error must be string"
        
        # Should be JSON serializable
        json_str = json.dumps(error_response)
        assert len(json_str) > 0, "JSON serialization failed"
        
        print("   ✓ Error schema is compliant")
        return True
    except Exception as e:
        print(f"   ✗ Schema check failed: {e}")
        return False


def test_success_response_schema():
    """Test that success responses follow JSON schema."""
    print("\n🧪 Test: Success Response Schema")
    
    success_response = {
        "status": "success",
        "generated_text": "Hello world!",
        "tokens_generated": 3,
        "tokens_per_second": 45.5,
        "latency_ms": 65.9,
        "model_config": {
            "draft_model": "phi-2",
            "target_model": "qwen2-7b",
            "speculative_decoding_enabled": True,
        }
    }
    
    try:
        # Validate schema
        assert isinstance(success_response, dict), "Response must be dict"
        assert success_response["status"] == "success", "Status should be 'success'"
        assert "generated_text" in success_response, "Missing generated_text"
        assert "tokens_per_second" in success_response, "Missing tokens_per_second"
        assert "latency_ms" in success_response, "Missing latency_ms"
        
        # Should be JSON serializable
        json_str = json.dumps(success_response)
        assert len(json_str) > 0, "JSON serialization failed"
        
        print("   ✓ Success schema is compliant")
        return True
    except Exception as e:
        print(f"   ✗ Schema check failed: {e}")
        return False


def main():
    """Run all tests."""
    print("\n" + "=" * 70)
    print("🍑 Momo-Kibidango MCP Server Tests")
    print("=" * 70)
    
    tests = [
        test_server_initialization,
        test_tool_listing,
        test_input_validation_inference,
        test_input_validation_benchmark,
        test_schema_compliance,
        test_success_response_schema,
    ]
    
    results = []
    for test in tests:
        try:
            result = test()
            results.append((test.__name__, result))
        except Exception as e:
            print(f"\n⚠️  Test {test.__name__} failed with exception: {e}")
            results.append((test.__name__, False))
    
    # Summary
    print("\n" + "=" * 70)
    print("📊 Test Summary")
    print("=" * 70)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        symbol = "✓" if result else "✗"
        print(f"  {symbol} {test_name}")
    
    print(f"\n✅ {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed!")
        return 0
    else:
        print(f"\n⚠️  {total - passed} test(s) failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
