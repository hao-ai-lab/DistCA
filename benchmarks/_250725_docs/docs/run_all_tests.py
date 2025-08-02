#!/usr/bin/env python3
"""
Run all unit tests for the communication metadata system.

This script runs all unit tests and provides a summary of results.
"""

import os
import sys
import subprocess
import time
from pathlib import Path

def run_test_file(test_file):
    """Run a single test file and capture results."""
    print(f"\n{'='*60}")
    print(f"Running {test_file}")
    print(f"{'='*60}")
    
    start_time = time.time()
    
    try:
        result = subprocess.run(
            [sys.executable, test_file],
            capture_output=True,
            text=True,
            timeout=120  # 2 minute timeout
        )
        
        elapsed = time.time() - start_time
        
        if result.returncode == 0:
            print(result.stdout)
            return {"status": "PASSED", "time": elapsed, "output": result.stdout}
        else:
            print("STDOUT:")
            print(result.stdout)
            print("\nSTDERR:")
            print(result.stderr)
            return {"status": "FAILED", "time": elapsed, "error": result.stderr}
            
    except subprocess.TimeoutExpired:
        elapsed = time.time() - start_time
        print(f"❌ Test timed out after {elapsed:.1f} seconds")
        return {"status": "TIMEOUT", "time": elapsed, "error": "Test timed out"}
    
    except Exception as e:
        elapsed = time.time() - start_time
        print(f"❌ Error running test: {e}")
        return {"status": "ERROR", "time": elapsed, "error": str(e)}


def main():
    """Run all unit tests and provide summary."""
    print("🧪 Running All Communication Metadata Unit Tests")
    print("=" * 70)
    
    # Get the directory containing this script
    docs_dir = Path(__file__).parent
    unit_tests_dir = docs_dir / "unit_tests"
    
    # Find all test files
    test_files = []
    for file_path in unit_tests_dir.glob("test_*.py"):
        test_files.append(file_path)
    
    test_files.sort()  # Run in consistent order
    
    if not test_files:
        print("❌ No test files found in unit_tests directory")
        return 1
    
    print(f"Found {len(test_files)} test files:")
    for test_file in test_files:
        print(f"  📄 {test_file.name}")
    
    # Run all tests
    results = {}
    total_start_time = time.time()
    
    for test_file in test_files:
        test_name = test_file.name
        results[test_name] = run_test_file(str(test_file))
    
    total_elapsed = time.time() - total_start_time
    
    # Print summary
    print("\n" + "=" * 70)
    print("📊 TEST SUMMARY")
    print("=" * 70)
    
    passed = 0
    failed = 0
    errors = 0
    timeouts = 0
    
    print(f"{'Test File':<35} {'Status':<10} {'Time':<8}")
    print("-" * 55)
    
    for test_name, result in results.items():
        status = result["status"]
        elapsed = result["time"]
        
        if status == "PASSED":
            status_emoji = "✅"
            passed += 1
        elif status == "FAILED":
            status_emoji = "❌"
            failed += 1
        elif status == "TIMEOUT":
            status_emoji = "⏰"
            timeouts += 1
        else:  # ERROR
            status_emoji = "💥"
            errors += 1
        
        print(f"{test_name:<35} {status_emoji} {status:<8} {elapsed:>6.1f}s")
    
    print("-" * 55)
    print(f"Total: {len(test_files)} tests, {passed} passed, {failed} failed, {errors} errors, {timeouts} timeouts")
    print(f"Total time: {total_elapsed:.1f} seconds")
    
    # Show detailed failures
    if failed > 0 or errors > 0 or timeouts > 0:
        print(f"\n🔍 FAILURE DETAILS")
        print("=" * 70)
        
        for test_name, result in results.items():
            if result["status"] != "PASSED":
                print(f"\n❌ {test_name} ({result['status']}):")
                if "error" in result:
                    print(result["error"])
    
    # Overall result
    if failed == 0 and errors == 0 and timeouts == 0:
        print(f"\n🎉 ALL TESTS PASSED!")
        print("The communication metadata system is working correctly.")
        return 0
    else:
        print(f"\n💥 SOME TESTS FAILED!")
        print("Check the failure details above and fix the issues.")
        return 1


if __name__ == "__main__":
    # Change to the docs directory
    docs_dir = Path(__file__).parent
    os.chdir(docs_dir)
    
    # Add parent directories to Python path
    sys.path.insert(0, str(docs_dir.parent))
    
    exit_code = main()
    sys.exit(exit_code)