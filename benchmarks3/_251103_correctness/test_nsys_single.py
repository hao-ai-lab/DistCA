#!/usr/bin/env python3
"""
Test script to analyze a single nsys file and verify the analysis works.
Useful for debugging and testing individual files.
"""

import sys
import os
import subprocess
import argparse

def test_single_nsys_file(nsys_file: str, kernel_name: str = "ncclDevKernel_AllGather_RING_LL"):
    """Test analysis on a single nsys file."""
    
    if not os.path.exists(nsys_file):
        print(f"Error: File {nsys_file} not found!")
        return False
    
    print(f"Testing nsys file: {nsys_file}")
    print(f"Looking for kernel: {kernel_name}")
    print("-" * 60)
    
    # Test basic nsys stats command
    try:
        print("1. Testing basic nsys stats...")
        result = subprocess.run(['nsys', 'stats', nsys_file], 
                              capture_output=True, text=True, timeout=60)
        
        if result.returncode == 0:
            print("✓ nsys stats command works")
            
            # Check if kernel name appears in output
            if kernel_name in result.stdout:
                print(f"✓ Found '{kernel_name}' in output")
                
                # Show relevant lines
                lines = result.stdout.split('\n')
                relevant_lines = [line for line in lines if kernel_name in line]
                print(f"✓ Found {len(relevant_lines)} lines with kernel name:")
                for i, line in enumerate(relevant_lines):
                    print(f"  Line {i+1}: {line.strip()}")
            else:
                print(f"✗ Kernel '{kernel_name}' not found in output")
                print("Available kernels (first 10 lines with 'kernel' or 'Kernel'):")
                lines = result.stdout.split('\n')
                kernel_lines = [line for line in lines if 'kernel' in line.lower()][:10]
                for line in kernel_lines:
                    if line.strip():
                        print(f"  {line.strip()}")
        else:
            print(f"✗ nsys stats failed: {result.stderr}")
            return False
            
    except subprocess.TimeoutExpired:
        print("✗ nsys stats timed out")
        return False
    except FileNotFoundError:
        print("✗ nsys command not found")
        return False
    except Exception as e:
        print(f"✗ Error running nsys stats: {e}")
        return False
    
    print("\n2. Testing with simple analyzer...")
    
    # Test with simple analyzer
    try:
        # Import and test the simple analyzer
        sys.path.append(os.path.dirname(__file__))
        from simple_nccl_analyzer import extract_kernel_info_text, analyze_kernel_data
        
        # Extract data
        kernel_data = extract_kernel_info_text(nsys_file, kernel_name)
        
        if kernel_data:
            print(f"✓ Simple analyzer found {len(kernel_data)} entries")
            
            # Analyze data
            analysis = analyze_kernel_data(kernel_data)
            print("Analysis results:")
            for key, value in analysis.items():
                print(f"  {key}: {value}")
                
            # Show first few entries
            print("\nFirst few entries:")
            for i, entry in enumerate(kernel_data[:3]):
                print(f"  Entry {i+1}: {entry}")
        else:
            print("✗ Simple analyzer found no data")
            
    except Exception as e:
        print(f"✗ Error with simple analyzer: {e}")
        return False
    
    print("\n3. Summary:")
    print(f"File: {nsys_file}")
    print(f"File size: {os.path.getsize(nsys_file) / (1024*1024):.1f} MB")
    print(f"Kernel entries found: {len(kernel_data) if 'kernel_data' in locals() else 0}")
    
    return True

def main():
    parser = argparse.ArgumentParser(description='Test nsys analysis on a single file')
    parser.add_argument('nsys_file', help='Path to the .nsys-rep file to test')
    parser.add_argument('--kernel-name', default='ncclDevKernel_AllGather_RING_LL',
                       help='Kernel name to search for')
    
    args = parser.parse_args()
    
    success = test_single_nsys_file(args.nsys_file, args.kernel_name)
    
    if success:
        print("\n✓ Test completed successfully!")
        return 0
    else:
        print("\n✗ Test failed!")
        return 1

if __name__ == '__main__':
    sys.exit(main())







