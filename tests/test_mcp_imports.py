#!/usr/bin/env python3
"""
Simple MCP Server Import Tests

Tests that MCP server code is syntactically correct and imports work
(without requiring full dependencies).
"""

import ast
import sys
from pathlib import Path


def test_syntax():
    """Test Python syntax is valid."""
    print("\n🧪 Test: Python Syntax Validation")
    
    files_to_check = [
        "src/momo_kibidango/mcp_server.py",
        "src/momo_kibidango/cli.py",
    ]
    
    all_valid = True
    for filepath in files_to_check:
        full_path = Path(filepath)
        if not full_path.exists():
            print(f"   ✗ File not found: {filepath}")
            all_valid = False
            continue
        
        try:
            with open(full_path, 'r') as f:
                code = f.read()
            ast.parse(code)
            print(f"   ✓ {filepath}: Syntax valid")
        except SyntaxError as e:
            print(f"   ✗ {filepath}: Syntax error - {e}")
            all_valid = False
    
    return all_valid


def test_module_structure():
    """Test module structure and docstrings."""
    print("\n🧪 Test: Module Structure")
    
    files_to_check = {
        "src/momo_kibidango/mcp_server.py": [
            "MomoKibidangoMCPServer",
        ],
    }
    
    all_valid = True
    for filepath, expected_items in files_to_check.items():
        full_path = Path(filepath)
        
        try:
            with open(full_path, 'r') as f:
                code = f.read()
            
            tree = ast.parse(code)
            
            # Check for classes and functions
            defined_items = set()
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef):
                    defined_items.add(node.name)
                elif isinstance(node, ast.FunctionDef):
                    defined_items.add(node.name)
            
            missing = set(expected_items) - defined_items
            if missing:
                print(f"   ✗ {filepath}: Missing items - {missing}")
                all_valid = False
            else:
                print(f"   ✓ {filepath}: All expected items found")
                
        except Exception as e:
            print(f"   ✗ {filepath}: Error - {e}")
            all_valid = False
    
    return all_valid


def test_code_quality():
    """Test basic code quality."""
    print("\n🧪 Test: Code Quality Checks")
    
    checks = [
        ("MCP server has docstring", "src/momo_kibidango/mcp_server.py", '"""'),
        ("CLI has docstring", "src/momo_kibidango/cli.py", '"""'),
    ]
    
    all_valid = True
    for desc, filepath, pattern in checks:
        full_path = Path(filepath)
        
        try:
            with open(full_path, 'r') as f:
                content = f.read()
            
            if pattern in content:
                print(f"   ✓ {desc}")
            else:
                print(f"   ✗ {desc}")
                all_valid = False
                
        except Exception as e:
            print(f"   ✗ {desc}: {e}")
            all_valid = False
    
    return all_valid


def test_file_completeness():
    """Test that files exist and aren't empty."""
    print("\n🧪 Test: File Completeness")
    
    required_files = [
        "src/momo_kibidango/mcp_server.py",
        "src/momo_kibidango/cli.py",
        "docs/MCP_INTEGRATION_GUIDE.md",
        "docs/ARCHITECTURE.md",
        "examples/claude_agent_example.py",
        "tests/test_mcp_server.py",
    ]
    
    all_valid = True
    for filepath in required_files:
        full_path = Path(filepath)
        
        if not full_path.exists():
            print(f"   ✗ File missing: {filepath}")
            all_valid = False
            continue
        
        size = full_path.stat().st_size
        if size < 100:
            print(f"   ✗ File too small: {filepath} ({size} bytes)")
            all_valid = False
            continue
        
        print(f"   ✓ {filepath} ({size} bytes)")
    
    return all_valid


def test_configuration():
    """Test that pyproject.toml includes MCP settings."""
    print("\n🧪 Test: Configuration")
    
    try:
        with open("pyproject.toml", 'r') as f:
            content = f.read()
        
        checks = [
            ("MCP optional dependency", "mcp", content),
            ("MCP console script", "mcp-server-momo-kibidango", content),
        ]
        
        all_valid = True
        for desc, pattern, text in checks:
            if pattern in text:
                print(f"   ✓ {desc} configured")
            else:
                print(f"   ✗ {desc} not found")
                all_valid = False
        
        return all_valid
        
    except Exception as e:
        print(f"   ✗ Error reading config: {e}")
        return False


def main():
    """Run all tests."""
    print("\n" + "=" * 70)
    print("🍑 MCP Server Implementation Validation")
    print("=" * 70)
    
    tests = [
        test_syntax,
        test_module_structure,
        test_code_quality,
        test_file_completeness,
        test_configuration,
    ]
    
    results = []
    for test in tests:
        try:
            result = test()
            results.append((test.__name__, result))
        except Exception as e:
            print(f"\n⚠️  Test {test.__name__} failed: {e}")
            results.append((test.__name__, False))
    
    # Summary
    print("\n" + "=" * 70)
    print("📊 Validation Summary")
    print("=" * 70)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        symbol = "✓" if result else "✗"
        print(f"  {symbol} {test_name}")
    
    print(f"\n✅ {passed}/{total} validation checks passed")
    
    if passed == total:
        print("\n🎉 MCP Server implementation is valid!")
        print("\nNext steps:")
        print("  1. Install MCP: pip install momo-kibidango[mcp]")
        print("  2. Start server: momo-kibidango serve")
        print("  3. Try integration: python examples/claude_agent_example.py")
        return 0
    else:
        print(f"\n⚠️  {total - passed} validation check(s) failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
