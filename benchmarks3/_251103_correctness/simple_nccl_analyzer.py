#!/usr/bin/env python3
"""
Simple nsys profile analyzer for NCCL AllGather kernel timing.

This script provides a simpler approach that works with various nsys versions
and output formats.
"""

import os
import subprocess
import json
import re
from pathlib import Path
import sys
from typing import Dict, List, Tuple, Any
import argparse

def find_nsys_files(base_dir: str) -> List[str]:
    """Find all .nsys-rep files in the given directory recursively."""
    nsys_files = []
    for root, dirs, files in os.walk(base_dir):
        for file in files:
            if file.endswith('.nsys-rep'):
                nsys_files.append(os.path.join(root, file))
    return sorted(nsys_files)

def extract_kernel_info_text(nsys_file: str, kernel_name: str = "ncclDevKernel_AllGather_RING_LL") -> List[Dict[str, Any]]:
    """
    Extract kernel timing information using nsys stats text output.
    This approach is more compatible across different nsys versions.
    """
    print(f"Processing {nsys_file}...")
    
    try:
        # Try different nsys stats commands to get kernel information
        commands_to_try = [
            ['nsys', 'stats', '--report', 'gpukerntrace', nsys_file],
            ['nsys', 'stats', '--report', 'gpukernsum', nsys_file],
            ['nsys', 'stats', nsys_file]  # Default report
        ]
        
        kernel_data = []
        
        for cmd in commands_to_try:
            try:
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
                
                if result.returncode == 0 and kernel_name in result.stdout:
                    # Parse the text output
                    parsed_data = parse_nsys_text_output(result.stdout, kernel_name, nsys_file)
                    if parsed_data:
                        kernel_data.extend(parsed_data)
                        break  # Success with this command
                        
            except subprocess.TimeoutExpired:
                print(f"Timeout with command: {' '.join(cmd)}")
                continue
            except Exception as e:
                print(f"Error with command {' '.join(cmd)}: {e}")
                continue
        
        return kernel_data
        
    except Exception as e:
        print(f"Error processing {nsys_file}: {e}")
        return []

def parse_nsys_text_output(output: str, kernel_name: str, source_file: str) -> List[Dict[str, Any]]:
    """Parse nsys stats text output to extract kernel timing information."""
    kernel_data = []
    lines = output.split('\n')
    
    # Look for different output formats
    in_kernel_section = False
    headers = []
    
    for i, line in enumerate(lines):
        line = line.strip()
        
        # Skip empty lines
        if not line:
            continue
            
        # Look for kernel name in the line
        if kernel_name in line:
            # Try to extract timing information from this line
            timing_info = extract_timing_from_line(line, source_file)
            if timing_info:
                kernel_data.append(timing_info)
            
            # Also check if this is part of a table
            # Look for headers in nearby lines
            for j in range(max(0, i-5), min(len(lines), i+1)):
                header_line = lines[j].strip()
                if ('Time' in header_line or 'Duration' in header_line) and ('ns' in header_line or 'us' in header_line):
                    # This looks like a header line, try to parse the kernel line as a table row
                    table_info = parse_table_row(header_line, line, source_file)
                    if table_info and table_info not in kernel_data:
                        kernel_data.append(table_info)
    
    return kernel_data

def extract_timing_from_line(line: str, source_file: str) -> Dict[str, Any]:
    """Extract timing information from a line containing the kernel name."""
    # Common patterns to look for timing information
    patterns = [
        r'(\d+\.?\d*)\s*ns',  # nanoseconds
        r'(\d+\.?\d*)\s*us',  # microseconds
        r'(\d+\.?\d*)\s*ms',  # milliseconds
        r'(\d+\.?\d*)\s*s',   # seconds
    ]
    
    timing_info = {
        'kernel_name': 'ncclDevKernel_AllGather_RING_LL',
        'source_file': source_file,
        'raw_line': line.strip()
    }
    
    # Extract numeric values and their units
    for pattern in patterns:
        matches = re.findall(pattern, line)
        if matches:
            # Take the first numeric value found
            value = float(matches[0])
            if 'ns' in line:
                timing_info['duration_ns'] = value
                timing_info['duration_ms'] = value / 1e6
            elif 'us' in line:
                timing_info['duration_ns'] = value * 1e3
                timing_info['duration_ms'] = value / 1e3
            elif 'ms' in line:
                timing_info['duration_ns'] = value * 1e6
                timing_info['duration_ms'] = value
            elif 's' in line and 'ms' not in line and 'us' not in line and 'ns' not in line:
                timing_info['duration_ns'] = value * 1e9
                timing_info['duration_ms'] = value * 1e3
            break
    
    # Try to extract device information
    device_match = re.search(r'GPU\s*(\d+)', line, re.IGNORECASE)
    if device_match:
        timing_info['device'] = int(device_match.group(1))
    
    return timing_info

def parse_table_row(header_line: str, data_line: str, source_file: str) -> Dict[str, Any]:
    """Parse a table row based on header information."""
    headers = header_line.split()
    data_parts = data_line.split()
    
    timing_info = {
        'kernel_name': 'ncclDevKernel_AllGather_RING_LL',
        'source_file': source_file,
        'raw_line': data_line.strip()
    }
    
    # Map headers to data
    for i, header in enumerate(headers):
        if i < len(data_parts):
            value = data_parts[i]
            
            # Handle timing columns
            if 'time' in header.lower() or 'duration' in header.lower():
                try:
                    numeric_value = float(value)
                    if 'ns' in header.lower():
                        timing_info['duration_ns'] = numeric_value
                        timing_info['duration_ms'] = numeric_value / 1e6
                    elif 'us' in header.lower():
                        timing_info['duration_ns'] = numeric_value * 1e3
                        timing_info['duration_ms'] = numeric_value / 1e3
                    elif 'ms' in header.lower():
                        timing_info['duration_ns'] = numeric_value * 1e6
                        timing_info['duration_ms'] = numeric_value
                except ValueError:
                    pass
            
            # Handle device columns
            elif 'device' in header.lower() or 'gpu' in header.lower():
                try:
                    timing_info['device'] = int(value)
                except ValueError:
                    timing_info['device'] = value
    
    return timing_info

def analyze_kernel_data(all_data: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Analyze the collected kernel timing data."""
    if not all_data:
        return {}
    
    # Filter data with valid timing information
    valid_data = [d for d in all_data if 'duration_ms' in d]
    
    if not valid_data:
        return {
            'total_entries': len(all_data),
            'entries_with_timing': 0,
            'message': 'No timing data could be extracted'
        }
    
    durations_ms = [d['duration_ms'] for d in valid_data]
    
    analysis = {
        'total_entries': len(all_data),
        'entries_with_timing': len(valid_data),
        'timing_stats_ms': {
            'count': len(durations_ms),
            'mean': sum(durations_ms) / len(durations_ms),
            'min': min(durations_ms),
            'max': max(durations_ms),
            'total': sum(durations_ms)
        }
    }
    
    # Count by source file
    files = {}
    for d in valid_data:
        file = d['source_file']
        if file not in files:
            files[file] = []
        files[file].append(d['duration_ms'])
    
    analysis['by_file'] = {}
    for file, durations in files.items():
        analysis['by_file'][file] = {
            'count': len(durations),
            'mean_ms': sum(durations) / len(durations),
            'total_ms': sum(durations),
            'durations': durations
        }
    
    # Count by device (if available)
    devices = {}
    for d in valid_data:
        if 'device' in d:
            device = d['device']
            if device not in devices:
                devices[device] = []
            devices[device].append(d['duration_ms'])
    
    if devices:
        analysis['by_device'] = {}
        for device, durations in devices.items():
            analysis['by_device'][device] = {
                'count': len(durations),
                'mean_ms': sum(durations) / len(durations),
                'total_ms': sum(durations)
            }
    
    return analysis

def print_detailed_report(all_data: List[Dict[str, Any]], analysis: Dict[str, Any]):
    """Print a detailed report of the kernel analysis."""
    print("\n" + "=" * 80)
    print("NCCL AllGather Kernel Analysis Report")
    print("=" * 80)
    
    print(f"Total entries found: {analysis.get('total_entries', 0)}")
    print(f"Entries with timing data: {analysis.get('entries_with_timing', 0)}")
    
    if 'timing_stats_ms' in analysis:
        stats = analysis['timing_stats_ms']
        print(f"\nOverall Timing Statistics:")
        print(f"  Count: {stats['count']}")
        print(f"  Mean: {stats['mean']:.3f} ms")
        print(f"  Min: {stats['min']:.3f} ms")
        print(f"  Max: {stats['max']:.3f} ms")
        print(f"  Total: {stats['total']:.3f} ms")
    
    if 'by_file' in analysis:
        print(f"\nBy File:")
        for file, file_stats in analysis['by_file'].items():
            print(f"\n  {file}:")
            print(f"    Invocations: {file_stats['count']}")
            print(f"    Mean time: {file_stats['mean_ms']:.3f} ms")
            print(f"    Total time: {file_stats['total_ms']:.3f} ms")
            print(f"    Individual timings: {[f'{d:.3f}' for d in file_stats['durations']]}")
    
    if 'by_device' in analysis:
        print(f"\nBy Device:")
        for device, device_stats in analysis['by_device'].items():
            print(f"  Device {device}: {device_stats['count']} invocations, "
                  f"mean={device_stats['mean_ms']:.3f}ms, total={device_stats['total_ms']:.3f}ms")
    
    # Show raw data for debugging
    print(f"\n" + "-" * 60)
    print("Raw Data (for debugging):")
    print("-" * 60)
    for i, entry in enumerate(all_data):
        print(f"Entry {i+1}:")
        for key, value in entry.items():
            if key != 'raw_line':  # Skip long raw lines unless needed
                print(f"  {key}: {value}")
        print()

def main():
    parser = argparse.ArgumentParser(description='Simple nsys analyzer for NCCL AllGather kernel')
    parser.add_argument('--logs-dir', default='logs.v0-big', 
                       help='Directory containing logs (default: logs.v0-big)')
    parser.add_argument('--kernel-name', default='ncclDevKernel_AllGather_RING_LL',
                       help='Kernel name to search for')
    parser.add_argument('--show-raw', action='store_true',
                       help='Show raw output lines for debugging')
    
    args = parser.parse_args()
    
    # Get current script directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    logs_dir = os.path.join(script_dir, args.logs_dir)
    
    if not os.path.exists(logs_dir):
        print(f"Error: Directory {logs_dir} not found!")
        return 1
    
    print(f"Searching for nsys files in: {logs_dir}")
    nsys_files = find_nsys_files(logs_dir)
    
    if not nsys_files:
        print(f"No .nsys-rep files found in {logs_dir}")
        return 1
    
    print(f"Found {len(nsys_files)} nsys files to analyze:")
    for f in nsys_files:
        print(f"  - {f}")
    
    all_kernel_data = []
    
    for nsys_file in nsys_files:
        kernel_data = extract_kernel_info_text(nsys_file, args.kernel_name)
        all_kernel_data.extend(kernel_data)
    
    if not all_kernel_data:
        print(f"\nNo kernel data found for '{args.kernel_name}' in any files")
        return 1
    
    # Analyze the collected data
    analysis = analyze_kernel_data(all_kernel_data)
    
    # Print detailed report
    print_detailed_report(all_kernel_data, analysis)
    
    return 0

if __name__ == '__main__':
    sys.exit(main())







