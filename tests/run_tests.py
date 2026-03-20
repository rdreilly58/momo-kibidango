#!/usr/bin/env python3
"""
Test Runner with Coverage Reporting
Runs all tests and generates coverage report
"""

import sys
import os
import unittest
import argparse
from pathlib import Path

# Try to import coverage
try:
    import coverage
    COVERAGE_AVAILABLE = True
except ImportError:
    COVERAGE_AVAILABLE = False
    print("Warning: coverage module not available. Install with: pip install coverage")


def run_tests_with_coverage(test_dir, source_dir, html_report=False):
    """Run tests with coverage measurement"""
    if not COVERAGE_AVAILABLE:
        print("Coverage not available, running tests without coverage")
        return run_tests_without_coverage(test_dir)
        
    # Create coverage instance
    cov = coverage.Coverage(source=[source_dir])
    
    # Start coverage
    cov.start()
    
    # Discover and run tests
    loader = unittest.TestLoader()
    suite = loader.discover(test_dir, pattern='test_*.py')
    
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Stop coverage
    cov.stop()
    cov.save()
    
    # Generate report
    print("\n" + "="*70)
    print("COVERAGE REPORT")
    print("="*70)
    
    # Console report
    cov.report()
    
    # HTML report if requested
    if html_report:
        html_dir = os.path.join(os.path.dirname(test_dir), 'coverage_html')
        cov.html_report(directory=html_dir)
        print(f"\nHTML coverage report generated in: {html_dir}")
        
    # Get coverage percentage
    total_coverage = cov.report(show_missing=False)
    
    print(f"\nTotal coverage: {total_coverage:.1f}%")
    
    # Check if coverage meets target
    target_coverage = 80.0
    if total_coverage >= target_coverage:
        print(f"✅ Coverage target met ({target_coverage}%)")
    else:
        print(f"❌ Coverage below target ({target_coverage}%)")
        
    return result.wasSuccessful() and total_coverage >= target_coverage


def run_tests_without_coverage(test_dir):
    """Run tests without coverage measurement"""
    loader = unittest.TestLoader()
    suite = loader.discover(test_dir, pattern='test_*.py')
    
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result.wasSuccessful()


def run_specific_test(test_module, test_class=None, test_method=None):
    """Run a specific test module, class, or method"""
    if test_class:
        if test_method:
            suite = unittest.TestLoader().loadTestsFromName(
                f"{test_module}.{test_class}.{test_method}"
            )
        else:
            suite = unittest.TestLoader().loadTestsFromTestCase(
                getattr(__import__(test_module), test_class)
            )
    else:
        suite = unittest.TestLoader().loadTestsFromModule(
            __import__(test_module)
        )
        
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result.wasSuccessful()


def main():
    parser = argparse.ArgumentParser(description="Run tests with coverage reporting")
    
    parser.add_argument("--coverage", action="store_true",
                       help="Run with coverage measurement")
    parser.add_argument("--html", action="store_true",
                       help="Generate HTML coverage report")
    parser.add_argument("--test", help="Run specific test (module[.class[.method]])")
    parser.add_argument("--performance", action="store_true",
                       help="Run only performance tests")
    parser.add_argument("--unit", action="store_true",
                       help="Run only unit tests")
    parser.add_argument("--integration", action="store_true",
                       help="Run only integration tests")
    
    args = parser.parse_args()
    
    # Setup paths
    test_dir = Path(__file__).parent
    src_dir = test_dir.parent / "src"
    
    # Add src to path
    sys.path.insert(0, str(src_dir))
    
    # Change to test directory
    os.chdir(test_dir)
    
    success = True
    
    if args.test:
        # Run specific test
        parts = args.test.split('.')
        module = parts[0]
        test_class = parts[1] if len(parts) > 1 else None
        test_method = parts[2] if len(parts) > 2 else None
        
        success = run_specific_test(module, test_class, test_method)
        
    elif args.performance:
        # Run only performance tests
        print("Running performance tests...")
        success = run_specific_test("test_performance")
        
    elif args.unit:
        # Run only unit tests
        print("Running unit tests...")
        success = run_specific_test("test_production")
        
    elif args.coverage or args.html:
        # Run all tests with coverage
        success = run_tests_with_coverage(test_dir, src_dir, args.html)
        
    else:
        # Run all tests without coverage
        success = run_tests_without_coverage(test_dir)
        
    # Print summary
    print("\n" + "="*70)
    if success:
        print("✅ ALL TESTS PASSED")
    else:
        print("❌ SOME TESTS FAILED")
    print("="*70)
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()