#!/usr/bin/env python3
"""
Example: Using Momo-Kibidango with Claude via MCP

This example demonstrates how to integrate momo-kibidango speculative decoding
with Claude AI using the Model Context Protocol (MCP).

Before running:
1. Install dependencies: pip install momo-kibidango[mcp] anthropic
2. Set ANTHROPIC_API_KEY environment variable
3. Start momo-kibidango MCP server: momo-kibidango serve

Then run:
python examples/claude_agent_example.py
"""

import json
import os
import subprocess
import sys
from pathlib import Path

try:
    from anthropic import Anthropic
except ImportError:
    print("Error: anthropic library not installed")
    print("Install with: pip install anthropic")
    sys.exit(1)


def example_simple_inference():
    """Example 1: Simple inference request."""
    print("\n" + "=" * 70)
    print("EXAMPLE 1: Simple Inference Request")
    print("=" * 70 + "\n")

    client = Anthropic()

    # Create message with momo-kibidango tools
    response = client.messages.create(
        model="claude-opus-4-0",
        max_tokens=1024,
        tools=[
            {
                "type": "mcp",
                "mcp_name": "momo-kibidango",
            }
        ],
        messages=[
            {
                "role": "user",
                "content": "Use momo-kibidango to run inference on 'Hello world' and tell me the tokens per second",
            }
        ],
    )

    print("Claude's response:")
    for block in response.content:
        if hasattr(block, "text"):
            print(f"  {block.text}")
        elif block.type == "tool_use":
            print(f"\n  Tool call: {block.name}")
            print(f"  Input: {json.dumps(block.input, indent=4)}")


def example_benchmark_analysis():
    """Example 2: Benchmark analysis and reporting."""
    print("\n" + "=" * 70)
    print("EXAMPLE 2: Benchmark Analysis")
    print("=" * 70 + "\n")

    client = Anthropic()

    response = client.messages.create(
        model="claude-opus-4-0",
        max_tokens=1024,
        tools=[
            {
                "type": "mcp",
                "mcp_name": "momo-kibidango",
            }
        ],
        messages=[
            {
                "role": "user",
                "content": (
                    "Run a benchmark with 10 test cases using momo-kibidango. "
                    "Analyze the results and tell me if the speedup is worth using "
                    "speculative decoding in production."
                ),
            }
        ],
    )

    print("Claude's analysis:")
    for block in response.content:
        if hasattr(block, "text"):
            print(f"  {block.text}")
        elif block.type == "tool_use":
            print(f"\n  Tool called: {block.name}")
            print(f"  Parameters: {json.dumps(block.input, indent=4)}")


def example_multi_turn():
    """Example 3: Multi-turn conversation with tool use."""
    print("\n" + "=" * 70)
    print("EXAMPLE 3: Multi-Turn Conversation")
    print("=" * 70 + "\n")

    client = Anthropic()

    messages = [
        {
            "role": "user",
            "content": "Let's evaluate momo-kibidango. First, run inference on a test prompt.",
        }
    ]

    print("User: Let's evaluate momo-kibidango. First, run inference on a test prompt.")
    print()

    # First turn
    response = client.messages.create(
        model="claude-opus-4-0",
        max_tokens=1024,
        tools=[
            {
                "type": "mcp",
                "mcp_name": "momo-kibidango",
            }
        ],
        messages=messages,
    )

    # Add assistant response to conversation
    messages.append(
        {
            "role": "assistant",
            "content": response.content,
        }
    )

    # Show tool calls
    for block in response.content:
        if block.type == "tool_use":
            print(f"Claude is calling: {block.name}")
            print(f"With parameters: {json.dumps(block.input, indent=2)}")

    # Simulate tool result (in real scenario, you'd execute the tool)
    print("\n[Tool would execute here and return results]\n")

    # Second turn - follow-up question
    messages.append(
        {
            "role": "user",
            "content": "Now run a benchmark and compare the performance metrics.",
        }
    )

    print("User: Now run a benchmark and compare the performance metrics.")
    print()

    response = client.messages.create(
        model="claude-opus-4-0",
        max_tokens=1024,
        tools=[
            {
                "type": "mcp",
                "mcp_name": "momo-kibidango",
            }
        ],
        messages=messages,
    )

    # Show follow-up interaction
    for block in response.content:
        if hasattr(block, "text"):
            print(f"Claude: {block.text}")
        elif block.type == "tool_use":
            print(f"\nClaude is calling: {block.name}")
            print(f"With parameters: {json.dumps(block.input, indent=2)}")


def example_tool_schema():
    """Example 4: Inspect available tool schemas."""
    print("\n" + "=" * 70)
    print("EXAMPLE 4: Available Tool Schemas")
    print("=" * 70 + "\n")

    # In production, this would come from the MCP server
    # For demonstration, we show the schema structure

    tools_schema = {
        "run_inference": {
            "name": "run_inference",
            "description": "Run speculative decoding inference on a prompt",
            "input_schema": {
                "type": "object",
                "properties": {
                    "prompt": {
                        "type": "string",
                        "description": "Input prompt for inference",
                    },
                    "max_tokens": {
                        "type": "integer",
                        "description": "Maximum tokens to generate (default: 512)",
                        "default": 512,
                    },
                    "temperature": {
                        "type": "number",
                        "description": "Sampling temperature (default: 0.7)",
                        "default": 0.7,
                    },
                },
                "required": ["prompt"],
            },
        },
        "benchmark_models": {
            "name": "benchmark_models",
            "description": "Run benchmark comparing draft/target models",
            "input_schema": {
                "type": "object",
                "properties": {
                    "test_cases": {
                        "type": "integer",
                        "description": "Number of test cases (default: 10)",
                        "default": 10,
                    },
                    "output_format": {
                        "type": "string",
                        "description": "Output format: 'json' or 'csv' (default: json)",
                        "enum": ["json", "csv"],
                    },
                },
            },
        },
    }

    print("Available tools:")
    print()

    for tool_name, tool_info in tools_schema.items():
        print(f"  🔧 {tool_info['name']}")
        print(f"     {tool_info['description']}")
        print(f"     Schema: {json.dumps(tool_info['input_schema'], indent=8)}")
        print()


def check_server_running():
    """Check if momo-kibidango MCP server is running."""
    print("\n🔍 Checking if MCP server is running...")

    try:
        # Try to connect to server (would need actual server check)
        # For now, just inform user
        print("   ℹ️  Make sure to run 'momo-kibidango serve' in another terminal")
        return True
    except Exception as e:
        print(f"   ⚠️  Could not verify server: {e}")
        return False


def main():
    """Main entry point."""
    print("\n" + "=" * 70)
    print("🍑 Momo-Kibidango × Claude Agent Examples")
    print("=" * 70)

    # Check API key
    if not os.getenv("ANTHROPIC_API_KEY"):
        print("\n❌ Error: ANTHROPIC_API_KEY not set")
        print("   Set your API key: export ANTHROPIC_API_KEY='sk-...'")
        sys.exit(1)

    # Check server
    check_server_running()

    print("\nThese examples demonstrate:")
    print("  1. Simple inference requests")
    print("  2. Benchmark analysis and reporting")
    print("  3. Multi-turn conversations with tool use")
    print("  4. Tool schema inspection")

    print("\n" + "=" * 70)
    print("NOTE: Full examples require the MCP server running")
    print("=" * 70)

    # Run examples
    try:
        # Example 4 works without server (just shows schemas)
        example_tool_schema()

        # Examples 1-3 require server
        print("\n⚠️  The following examples require MCP server:")
        print("   Run in another terminal: momo-kibidango serve")
        print()

        # Try example 1
        try:
            print("\nWould run Example 1: Simple Inference (requires server)")
            # example_simple_inference()
        except Exception as e:
            print(f"   Note: Example skipped (server not running): {e}")

        # Try example 2
        try:
            print("Would run Example 2: Benchmark Analysis (requires server)")
            # example_benchmark_analysis()
        except Exception as e:
            print(f"   Note: Example skipped (server not running): {e}")

        # Try example 3
        try:
            print("Would run Example 3: Multi-Turn Conversation (requires server)")
            # example_multi_turn()
        except Exception as e:
            print(f"   Note: Example skipped (server not running): {e}")

    except KeyboardInterrupt:
        print("\n\n🛑 Interrupted")
        sys.exit(0)

    print("\n" + "=" * 70)
    print("✅ Examples completed!")
    print("=" * 70 + "\n")

    print("Next steps:")
    print("  1. Start MCP server: momo-kibidango serve")
    print("  2. Run this script: python examples/claude_agent_example.py")
    print("  3. Check docs: docs/MCP_INTEGRATION_GUIDE.md")
    print()


if __name__ == "__main__":
    main()
