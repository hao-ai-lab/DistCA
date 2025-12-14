#!/usr/bin/env python3
"""
Analyze nsys profile files to extract timing information for ncclDevKernel_AllGather_RING_LL kernel.

This script finds all .nsys-rep files in the logs.v0-big directory and extracts timing data
for the ncclDevKernel_AllGather_RING_LL kernel, showing results for each invocation,
device, and file.
"""

import os
import subprocess
import json
import csv
import pandas as pd
from pathlib import Path
import tempfile
import sys
from typing import Dict, List, Tuple
import argparse

def find_nsys_files(base_dir: str) -> List[str]:
    """Find all .nsys-rep files in the given directory recursively."""
    nsys_files = []
    for root, dirs, files in os.walk(base_dir):
        for file in files:
            if file.endswith('.nsys-rep'):
                nsys_files.append(os.path.join(root, file))
    return sorted(nsys_files)

def extract_kernel_data(nsys_file: str, kernel_name: str = "ncclDevKernel_AllGather_RING_LL") -> pd.DataFrame:
    """
    Extract timing data for a specific kernel from an nsys file.
    
    Args:
        nsys_file: Path to the .nsys-rep file
        kernel_name: Name of the kernel to search for
    
    Returns:
        DataFrame with kernel timing data
    """
    print(f"Processing {nsys_file}...")
    
    # Create temporary file for CSV output
    with tempfile.NamedTemporaryFile(mode='w+', suffix='.csv', delete=False) as temp_file:
        temp_csv_path = temp_file.name
    
    try:
        # Run nsys stats to export kernel data to CSV
        cmd = [
            'nsys', 'stats', 
            '--report', 'gpukernsum,gpukerntrace',
            '--format', 'csv',
            '--output', temp_csv_path,
            nsys_file
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        
        if result.returncode != 0:
            print(f"Error processing {nsys_file}: {result.stderr}")
            return pd.DataFrame()
        
        # Read the generated CSV files
        kernel_data = []
        
        # Look for kernel trace CSV file
        csv_files = [f for f in os.listdir(os.path.dirname(temp_csv_path)) 
                    if f.startswith(os.path.basename(temp_csv_path).replace('.csv', '')) and f.endswith('.csv')]
        
        for csv_file in csv_files:
            full_csv_path = os.path.join(os.path.dirname(temp_csv_path), csv_file)
            try:
                df = pd.read_csv(full_csv_path)
                
                # Filter for the specific kernel
                if 'Name' in df.columns or 'Kernel Name' in df.columns:
                    name_col = 'Name' if 'Name' in df.columns else 'Kernel Name'
                    kernel_df = df[df[name_col].str.contains(kernel_name, na=False)]
                    
                    if not kernel_df.empty:
                        kernel_df = kernel_df.copy()
                        kernel_df['Source_File'] = nsys_file
                        kernel_data.append(kernel_df)
            except Exception as e:
                print(f"Error reading CSV {full_csv_path}: {e}")
                continue
        
        # Cleanup temporary files
        for csv_file in csv_files:
            full_csv_path = os.path.join(os.path.dirname(temp_csv_path), csv_file)
            try:
                os.unlink(full_csv_path)
            except:
                pass
        
        if kernel_data:
            return pd.concat(kernel_data, ignore_index=True)
        else:
            return pd.DataFrame()
            
    except subprocess.TimeoutExpired:
        print(f"Timeout processing {nsys_file}")
        return pd.DataFrame()
    except Exception as e:
        print(f"Error processing {nsys_file}: {e}")
        return pd.DataFrame()
    finally:
        # Cleanup temp file
        try:
            os.unlink(temp_csv_path)
        except:
            pass

def analyze_kernel_timing(df: pd.DataFrame) -> Dict:
    """Analyze kernel timing data and generate statistics."""
    if df.empty:
        return {}
    
    analysis = {
        'total_invocations': len(df),
        'unique_devices': df['Device'].nunique() if 'Device' in df.columns else 'Unknown',
        'files_analyzed': df['Source_File'].nunique() if 'Source_File' in df.columns else 1,
    }
    
    # Duration analysis (try different possible column names)
    duration_cols = ['Duration (ns)', 'Duration', 'Total Time (ns)', 'Time (ns)']
    duration_col = None
    for col in duration_cols:
        if col in df.columns:
            duration_col = col
            break
    
    if duration_col:
        analysis['duration_stats'] = {
            'mean_ns': df[duration_col].mean(),
            'median_ns': df[duration_col].median(),
            'min_ns': df[duration_col].min(),
            'max_ns': df[duration_col].max(),
            'std_ns': df[duration_col].std(),
            'total_ns': df[duration_col].sum()
        }
        
        # Convert to more readable units
        analysis['duration_stats_ms'] = {
            'mean_ms': df[duration_col].mean() / 1e6,
            'median_ms': df[duration_col].median() / 1e6,
            'min_ms': df[duration_col].min() / 1e6,
            'max_ms': df[duration_col].max() / 1e6,
            'std_ms': df[duration_col].std() / 1e6,
            'total_ms': df[duration_col].sum() / 1e6
        }
    
    return analysis

def generate_detailed_report(df: pd.DataFrame, output_file: str = None):
    """Generate a detailed report of kernel invocations."""
    if df.empty:
        print("No data to generate report.")
        return
    
    report_lines = []
    report_lines.append("=" * 80)
    report_lines.append("NCCL AllGather Kernel Timing Analysis")
    report_lines.append("=" * 80)
    
    # Group by source file
    for source_file, file_df in df.groupby('Source_File'):
        report_lines.append(f"\nFile: {source_file}")
        report_lines.append("-" * 60)
        
        # Group by device if available
        if 'Device' in file_df.columns:
            for device, device_df in file_df.groupby('Device'):
                report_lines.append(f"\n  Device {device}:")
                
                duration_col = None
                for col in ['Duration (ns)', 'Duration', 'Total Time (ns)', 'Time (ns)']:
                    if col in device_df.columns:
                        duration_col = col
                        break
                
                if duration_col:
                    report_lines.append(f"    Invocations: {len(device_df)}")
                    report_lines.append(f"    Total time: {device_df[duration_col].sum() / 1e6:.3f} ms")
                    report_lines.append(f"    Average time: {device_df[duration_col].mean() / 1e6:.3f} ms")
                    report_lines.append(f"    Min time: {device_df[duration_col].min() / 1e6:.3f} ms")
                    report_lines.append(f"    Max time: {device_df[duration_col].max() / 1e6:.3f} ms")
                    
                    # Show individual invocations
                    report_lines.append("    Individual invocations:")
                    for idx, row in device_df.iterrows():
                        start_time = row.get('Start (ns)', row.get('Start', 'N/A'))
                        duration = row[duration_col]
                        report_lines.append(f"      Invocation {idx}: {duration / 1e6:.3f} ms (start: {start_time})")
                else:
                    report_lines.append(f"    Invocations: {len(device_df)} (no timing data available)")
        else:
            # No device information available
            duration_col = None
            for col in ['Duration (ns)', 'Duration', 'Total Time (ns)', 'Time (ns)']:
                if col in file_df.columns:
                    duration_col = col
                    break
            
            if duration_col:
                report_lines.append(f"  Invocations: {len(file_df)}")
                report_lines.append(f"  Total time: {file_df[duration_col].sum() / 1e6:.3f} ms")
                report_lines.append(f"  Average time: {file_df[duration_col].mean() / 1e6:.3f} ms")
                
                # Show individual invocations
                report_lines.append("  Individual invocations:")
                for idx, row in file_df.iterrows():
                    start_time = row.get('Start (ns)', row.get('Start', 'N/A'))
                    duration = row[duration_col]
                    report_lines.append(f"    Invocation {idx}: {duration / 1e6:.3f} ms (start: {start_time})")
    
    report_content = '\n'.join(report_lines)
    
    if output_file:
        with open(output_file, 'w') as f:
            f.write(report_content)
        print(f"Detailed report saved to: {output_file}")
    else:
        print(report_content)

def main():
    parser = argparse.ArgumentParser(description='Analyze nsys profiles for NCCL AllGather kernel timing')
    parser.add_argument('--logs-dir', default='logs.v0-big', 
                       help='Directory containing logs (default: logs.v0-big)')
    parser.add_argument('--kernel-name', default='ncclDevKernel_AllGather_RING_LL',
                       help='Kernel name to search for')
    parser.add_argument('--output-csv', help='Save raw data to CSV file')
    parser.add_argument('--output-report', help='Save detailed report to file')
    
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
    
    print(f"Found {len(nsys_files)} nsys files to analyze")
    
    all_kernel_data = []
    
    for nsys_file in nsys_files:
        kernel_df = extract_kernel_data(nsys_file, args.kernel_name)
        if not kernel_df.empty:
            all_kernel_data.append(kernel_df)
    
    if not all_kernel_data:
        print(f"No kernel data found for '{args.kernel_name}' in any files")
        return 1
    
    # Combine all data
    combined_df = pd.concat(all_kernel_data, ignore_index=True)
    
    print(f"\nFound {len(combined_df)} invocations of '{args.kernel_name}' across {len(nsys_files)} files")
    
    # Generate analysis
    analysis = analyze_kernel_timing(combined_df)
    
    print("\n" + "=" * 60)
    print("SUMMARY STATISTICS")
    print("=" * 60)
    print(f"Total invocations: {analysis.get('total_invocations', 0)}")
    print(f"Unique devices: {analysis.get('unique_devices', 'Unknown')}")
    print(f"Files analyzed: {analysis.get('files_analyzed', 0)}")
    
    if 'duration_stats_ms' in analysis:
        duration_stats = analysis['duration_stats_ms']
        print(f"\nTiming Statistics (milliseconds):")
        print(f"  Mean: {duration_stats['mean_ms']:.3f} ms")
        print(f"  Median: {duration_stats['median_ms']:.3f} ms") 
        print(f"  Min: {duration_stats['min_ms']:.3f} ms")
        print(f"  Max: {duration_stats['max_ms']:.3f} ms")
        print(f"  Std Dev: {duration_stats['std_ms']:.3f} ms")
        print(f"  Total: {duration_stats['total_ms']:.3f} ms")
    
    # Save raw data if requested
    if args.output_csv:
        combined_df.to_csv(args.output_csv, index=False)
        print(f"\nRaw data saved to: {args.output_csv}")
    
    # Generate detailed report
    generate_detailed_report(combined_df, args.output_report)
    
    return 0

if __name__ == '__main__':
    sys.exit(main())







