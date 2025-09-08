#!/usr/bin/env python3
"""
Test runner script for Ring Attractor project.

This script provides a convenient interface for running different types of tests
with various configurations and reporting options.
"""

import sys
import os
import subprocess
import argparse
from pathlib import Path


def run_command(cmd, description):
    """Run a command and handle output"""
    print(f"\n{'='*60}")
    print(f"Running: {description}")
    print(f"Command: {' '.join(cmd)}")
    print('='*60)
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        print(result.stdout)
        if result.stderr:
            print("STDERR:", result.stderr)
        return True
    except subprocess.CalledProcessError as e:
        print(f"FAILED: {description}")
        print("STDOUT:", e.stdout)
        print("STDERR:", e.stderr)
        return False


def main():
    parser = argparse.ArgumentParser(description='Run Ring Attractor tests')
    
    # Test selection
    parser.add_argument('--unit', action='store_true', 
                       help='Run unit tests only')
    parser.add_argument('--integration', action='store_true',
                       help='Run integration tests only')
    parser.add_argument('--all', action='store_true', default=True,
                       help='Run all tests (default)')
    
    # Test options
    parser.add_argument('--coverage', action='store_true',
                       help='Run with coverage reporting')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Verbose output')
    parser.add_argument('--parallel', '-n', type=int, default=None,
                       help='Number of parallel workers')
    parser.add_argument('--fast', action='store_true',
                       help='Skip slow tests')
    parser.add_argument('--benchmark', action='store_true',
                       help='Run benchmark tests')
    
    # Specific test selection
    parser.add_argument('--test-file', type=str,
                       help='Run specific test file')
    parser.add_argument('--test-function', type=str,
                       help='Run specific test function')
    parser.add_argument('--test-class', type=str,
                       help='Run specific test class')
    
    # Output options
    parser.add_argument('--output-dir', type=str, default='test_results',
                       help='Directory for test outputs')
    parser.add_argument('--html-report', action='store_true',
                       help='Generate HTML coverage report')
    parser.add_argument('--junit-xml', action='store_true',
                       help='Generate JUnit XML report')
    
    args = parser.parse_args()
    
    # Ensure we're in the right directory
    project_root = Path(__file__).parent
    os.chdir(project_root)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Base pytest command
    pytest_cmd = ['python', '-m', 'pytest']
    
    # Add verbosity
    if args.verbose:
        pytest_cmd.extend(['-v', '-s'])
    
    # Add parallel execution
    if args.parallel:
        pytest_cmd.extend(['-n', str(args.parallel)])
    
    # Add coverage
    if args.coverage:
        pytest_cmd.extend([
            '--cov=.',
            '--cov-report=term-missing',
            f'--cov-report=html:{output_dir}/htmlcov'
        ])
        if args.html_report:
            pytest_cmd.append(f'--cov-report=html:{output_dir}/coverage_html')
    
    # Add JUnit XML
    if args.junit_xml:
        pytest_cmd.append(f'--junit-xml={output_dir}/junit.xml')
    
    # Add markers for test selection
    if args.fast:
        pytest_cmd.extend(['-m', 'not slow'])
    
    if args.benchmark:
        pytest_cmd.extend(['-m', 'benchmark'])
    
    # Determine which tests to run
    test_paths = []
    
    if args.unit and not args.integration:
        test_paths.append('tests/unit/')
    elif args.integration and not args.unit:
        test_paths.append('tests/integration/')
    elif args.test_file:
        test_paths.append(args.test_file)
    else:
        test_paths.append('tests/')
    
    # Add specific test selection
    if args.test_function:
        if not args.test_file:
            print("Error: --test-function requires --test-file")
            return 1
        test_paths[-1] += f'::{args.test_function}'
    
    if args.test_class:
        if not args.test_file:
            print("Error: --test-class requires --test-file")
            return 1
        test_paths[-1] += f'::{args.test_class}'
    
    # Add test paths to command
    pytest_cmd.extend(test_paths)
    
    # Print test configuration
    print("Ring Attractor Test Runner")
    print("="*40)
    print(f"Test paths: {test_paths}")
    print(f"Output directory: {output_dir}")
    print(f"Coverage: {args.coverage}")
    print(f"Parallel workers: {args.parallel or 'auto'}")
    print(f"Skip slow tests: {args.fast}")
    print(f"Benchmark tests: {args.benchmark}")
    
    # Check if test dependencies are installed
    try:
        import pytest
        print("[OK] pytest is installed")
    except ImportError:
        print("[ERROR] pytest not found. Install with: pip install -r requirements-test.txt")
        return 1
    
    # Run the tests
    success = run_command(pytest_cmd, "Running tests")
    
    if not success:
        print("\n[FAILED] Tests failed!")
        return 1
    
    print("\n[SUCCESS] All tests passed!")
    
    # Print summary information
    print(f"\nTest results saved to: {output_dir}")
    
    if args.coverage and args.html_report:
        print(f"Coverage report: {output_dir}/coverage_html/index.html")
    
    if args.junit_xml:
        print(f"JUnit XML: {output_dir}/junit.xml")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())