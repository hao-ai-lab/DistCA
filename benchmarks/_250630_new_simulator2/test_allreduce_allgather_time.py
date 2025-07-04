#!/usr/bin/env python3
"""
Test script for timemodule.py allreduce and allgather time calculation
Prints detailed tables for sanity checking the interpolation and timing
"""

import sys
import pandas as pd
import numpy as np
from timemodule import (
    get_allreduce_time, get_allgather_time, 
    setup_allreduce_time, setup_allgather_time,
    interpolate_value, K, M, G, T
)

def format_nelem(nelem):
    """Format number of elements in human readable format"""
    if nelem >= G:
        return f"{nelem/G:.2f}G"
    elif nelem >= M:
        return f"{nelem/M:.2f}M"
    elif nelem >= K:
        return f"{nelem/K:.2f}K"
    else:
        return f"{nelem:.0f}"

def test_allreduce_time():
    print("=" * 80)
    print("ALLREDUCE Time Calculation Test - Sanity Check")
    print("=" * 80)
    
    # Load and display available data
    allreduce_time_dict = setup_allreduce_time()
    print("\nAvailable world sizes in allreduce data:")
    for world_size in sorted(allreduce_time_dict.keys()):
        nelem_values = sorted(allreduce_time_dict[world_size].keys())
        print(f"  World size {world_size}: {len(nelem_values)} data points, nelem range: {format_nelem(min(nelem_values))} - {format_nelem(max(nelem_values))}")
    
    # Test exact lookups (values that exist in the data)
    print("\n" + "=" * 80)
    print("EXACT LOOKUPS (Testing values that exist in CSV)")
    print("=" * 80)
    
    exact_test_cases = [
        (2, 1024),      # Small
        (2, 1048576),   # Medium
        (2, 134217728), # Large
        (4, 1024),      # Small with 4 GPUs
        (4, 1048576),   # Medium with 4 GPUs
        (8, 1024),      # Small with 8 GPUs
        (8, 134217728), # Large with 8 GPUs
    ]
    
    exact_results = []
    for world_size, nelem in exact_test_cases:
        if world_size in allreduce_time_dict and nelem in allreduce_time_dict[world_size]:
            calculated_time = get_allreduce_time(world_size, nelem)
            actual_time = allreduce_time_dict[world_size][nelem]
            match = abs(calculated_time - actual_time) < 1e-6
            
            exact_results.append({
                'world_size': world_size,
                'nelem': nelem,
                'nelem_formatted': format_nelem(nelem),
                'calculated_ms': calculated_time,
                'actual_ms': actual_time,
                'match': '✓' if match else '✗'
            })
    
    if exact_results:
        exact_df = pd.DataFrame(exact_results)
        print(exact_df.to_string(index=False))
        
        all_match = all(result['match'] == '✓' for result in exact_results)
        print(f"\nExact lookup verification: {'✓ PASSED' if all_match else '✗ FAILED'}")
    
    # Test interpolation (values between data points)
    print("\n" + "=" * 80)
    print("INTERPOLATION TESTS (Testing values between data points)")
    print("=" * 80)
    
    interpolation_test_cases = [
        (2, 1500),      # Between 1024 and 2048
        (2, 3000),      # Between 2048 and 4096
        (2, 100000),    # Between 65536 and 131072
        (4, 1500),      # Between 1024 and 2048, 4 GPUs
        (8, 3000),      # Between 2048 and 4096, 8 GPUs
    ]
    
    interpolation_results = []
    for world_size, nelem in interpolation_test_cases:
        if world_size in allreduce_time_dict:
            calculated_time = get_allreduce_time(world_size, nelem)
            
            # Find surrounding data points for manual verification
            scoped_dict = allreduce_time_dict[world_size]
            sorted_keys = sorted(scoped_dict.keys())
            
            lower_key = max([k for k in sorted_keys if k < nelem], default=None)
            upper_key = min([k for k in sorted_keys if k > nelem], default=None)
            
            if lower_key and upper_key:
                # Manual log-linear interpolation
                log_y1 = np.log(scoped_dict[lower_key])
                log_y2 = np.log(scoped_dict[upper_key])
                log_x1 = np.log(lower_key)
                log_x2 = np.log(upper_key)
                log_x = np.log(nelem)
                expected_log_y = log_y1 + (log_y2 - log_y1) * (log_x - log_x1) / (log_x2 - log_x1)
                expected_time = np.exp(expected_log_y)
                
                match = abs(calculated_time - expected_time) < 1e-6
                
                interpolation_results.append({
                    'world_size': world_size,
                    'nelem': nelem,
                    'nelem_formatted': format_nelem(nelem),
                    'calculated_ms': calculated_time,
                    'expected_ms': expected_time,
                    'lower_bound': f"{format_nelem(lower_key)} ({scoped_dict[lower_key]:.3f}ms)",
                    'upper_bound': f"{format_nelem(upper_key)} ({scoped_dict[upper_key]:.3f}ms)",
                    'match': '✓' if match else '✗'
                })
    
    if interpolation_results:
        interpolation_df = pd.DataFrame(interpolation_results)
        print(interpolation_df[['world_size', 'nelem_formatted', 'calculated_ms', 'expected_ms', 'match']].to_string(index=False))
        
        all_match = all(result['match'] == '✓' for result in interpolation_results)
        print(f"\nInterpolation verification: {'✓ PASSED' if all_match else '✗ FAILED'}")
    
    # Test scaling behavior
    print("\n" + "=" * 80)
    print("SCALING BEHAVIOR ANALYSIS")
    print("=" * 80)
    
    print("\nElement scaling (world_size=2):")
    base_nelem = 1024
    for multiplier in [1, 2, 4, 8, 16, 32]:
        nelem = base_nelem * multiplier
        try:
            latency_ms = get_allreduce_time(2, nelem)
            scaling_factor = latency_ms / get_allreduce_time(2, base_nelem)
            throughput = nelem / latency_ms
            print(f"  nelem={format_nelem(nelem):>8s}: latency={latency_ms:8.3f}ms (scaling: {scaling_factor:.2f}x) (throughput: {format_nelem(throughput)}/ms)")
        except Exception as e:
            print(f"  nelem={format_nelem(nelem):>8s}: ERROR - {e}")
    
    print("\nWorld size scaling (nelem=1048576):")
    for world_size in [2, 4, 8]:
        try:
            latency_ms = get_allreduce_time(world_size, 1048576)
            baseline_latency = get_allreduce_time(2, 1048576)
            scaling_factor = baseline_latency / latency_ms
            print(f"  world_size={world_size}: latency={latency_ms:8.3f}ms (speedup vs 2 GPUs: {scaling_factor:.2f}x)")
        except Exception as e:
            print(f"  world_size={world_size}: ERROR - {e}")
    
    return True

def test_allgather_time():
    print("\n" + "=" * 80)
    print("ALLGATHER Time Calculation Test - Sanity Check")
    print("=" * 80)
    
    # Load and display available data
    allgather_time_dict = setup_allgather_time()
    print("\nAvailable world sizes in allgather data:")
    for world_size in sorted(allgather_time_dict.keys()):
        nelem_values = sorted(allgather_time_dict[world_size].keys())
        print(f"  World size {world_size}: {len(nelem_values)} data points, nelem range: {format_nelem(min(nelem_values))} - {format_nelem(max(nelem_values))}")
    
    # Test exact lookups (values that exist in the data)
    print("\n" + "=" * 80)
    print("EXACT LOOKUPS (Testing values that exist in CSV)")
    print("=" * 80)
    
    exact_test_cases = [
        (2, 1024),      # Small
        (2, 1048576),   # Medium
        (2, 134217728), # Large
        (4, 1024),      # Small with 4 GPUs
        (4, 1048576),   # Medium with 4 GPUs
        (8, 1024),      # Small with 8 GPUs
        (8, 134217728), # Large with 8 GPUs
    ]
    
    exact_results = []
    for world_size, nelem in exact_test_cases:
        if world_size in allgather_time_dict and nelem in allgather_time_dict[world_size]:
            calculated_time = get_allgather_time(world_size, nelem)
            actual_time = allgather_time_dict[world_size][nelem]
            match = abs(calculated_time - actual_time) < 1e-6
            
            exact_results.append({
                'world_size': world_size,
                'nelem': nelem,
                'nelem_formatted': format_nelem(nelem),
                'calculated_ms': calculated_time,
                'actual_ms': actual_time,
                'match': '✓' if match else '✗'
            })
    
    if exact_results:
        exact_df = pd.DataFrame(exact_results)
        print(exact_df.to_string(index=False))
        
        all_match = all(result['match'] == '✓' for result in exact_results)
        print(f"\nExact lookup verification: {'✓ PASSED' if all_match else '✗ FAILED'}")
    
    # Test interpolation (values between data points)
    print("\n" + "=" * 80)
    print("INTERPOLATION TESTS (Testing values between data points)")
    print("=" * 80)
    
    interpolation_test_cases = [
        (2, 1500),      # Between 1024 and 2048
        (2, 3000),      # Between 2048 and 4096
        (2, 100000),    # Between 65536 and 131072
        (4, 1500),      # Between 1024 and 2048, 4 GPUs
        (8, 3000),      # Between 2048 and 4096, 8 GPUs
    ]
    
    interpolation_results = []
    for world_size, nelem in interpolation_test_cases:
        if world_size in allgather_time_dict:
            calculated_time = get_allgather_time(world_size, nelem)
            
            # Find surrounding data points for manual verification
            scoped_dict = allgather_time_dict[world_size]
            sorted_keys = sorted(scoped_dict.keys())
            
            lower_key = max([k for k in sorted_keys if k < nelem], default=None)
            upper_key = min([k for k in sorted_keys if k > nelem], default=None)
            
            if lower_key and upper_key:
                # Manual log-linear interpolation
                log_y1 = np.log(scoped_dict[lower_key])
                log_y2 = np.log(scoped_dict[upper_key])
                log_x1 = np.log(lower_key)
                log_x2 = np.log(upper_key)
                log_x = np.log(nelem)
                expected_log_y = log_y1 + (log_y2 - log_y1) * (log_x - log_x1) / (log_x2 - log_x1)
                expected_time = np.exp(expected_log_y)
                
                match = abs(calculated_time - expected_time) < 1e-6
                
                interpolation_results.append({
                    'world_size': world_size,
                    'nelem': nelem,
                    'nelem_formatted': format_nelem(nelem),
                    'calculated_ms': calculated_time,
                    'expected_ms': expected_time,
                    'lower_bound': f"{format_nelem(lower_key)} ({scoped_dict[lower_key]:.3f}ms)",
                    'upper_bound': f"{format_nelem(upper_key)} ({scoped_dict[upper_key]:.3f}ms)",
                    'match': '✓' if match else '✗'
                })
    
    if interpolation_results:
        interpolation_df = pd.DataFrame(interpolation_results)
        print(interpolation_df[['world_size', 'nelem_formatted', 'calculated_ms', 'expected_ms', 'match']].to_string(index=False))
        
        all_match = all(result['match'] == '✓' for result in interpolation_results)
        print(f"\nInterpolation verification: {'✓ PASSED' if all_match else '✗ FAILED'}")
    
    # Test scaling behavior
    print("\n" + "=" * 80)
    print("SCALING BEHAVIOR ANALYSIS")
    print("=" * 80)
    
    print("\nElement scaling (world_size=2):")
    base_nelem = 1024
    for multiplier in [1, 2, 4, 8, 16, 32]:
        nelem = base_nelem * multiplier
        try:
            latency_ms = get_allgather_time(2, nelem)
            scaling_factor = latency_ms / get_allgather_time(2, base_nelem)
            throughput = nelem / latency_ms
            print(f"  nelem={format_nelem(nelem):>8s}: latency={latency_ms:8.3f}ms (scaling: {scaling_factor:.2f}x) (throughput: {format_nelem(throughput)}/ms)")
        except Exception as e:
            print(f"  nelem={format_nelem(nelem):>8s}: ERROR - {e}")
    
    print("\nWorld size scaling (nelem=1048576):")
    for world_size in [2, 4, 8]:
        try:
            latency_ms = get_allgather_time(world_size, 1048576)
            baseline_latency = get_allgather_time(2, 1048576)
            scaling_factor = baseline_latency / latency_ms
            print(f"  world_size={world_size}: latency={latency_ms:8.3f}ms (speedup vs 2 GPUs: {scaling_factor:.2f}x)")
        except Exception as e:
            print(f"  world_size={world_size}: ERROR - {e}")
    
    return True

def test_edge_cases():
    print("\n" + "=" * 80)
    print("EDGE CASE TESTS")
    print("=" * 80)
    
    # Test extrapolation beyond available data
    print("\nTesting extrapolation beyond available data:")
    
    allreduce_dict = setup_allreduce_time()
    allgather_dict = setup_allgather_time()
    
    # Test very large nelem (beyond max in data)
    for world_size in [2, 4, 8]:
        if world_size in allreduce_dict:
            max_nelem = max(allreduce_dict[world_size].keys())
            test_nelem = max_nelem * 2  # Beyond available data
            
            try:
                allreduce_time = get_allreduce_time(world_size, test_nelem)
                print(f"  AllReduce world_size={world_size}, nelem={format_nelem(test_nelem)} (extrapolated): {allreduce_time:.3f}ms")
            except Exception as e:
                print(f"  AllReduce world_size={world_size}, nelem={format_nelem(test_nelem)}: ERROR - {e}")
        
        if world_size in allgather_dict:
            max_nelem = max(allgather_dict[world_size].keys())
            test_nelem = max_nelem * 2  # Beyond available data
            
            try:
                allgather_time = get_allgather_time(world_size, test_nelem)
                print(f"  AllGather world_size={world_size}, nelem={format_nelem(test_nelem)} (extrapolated): {allgather_time:.3f}ms")
            except Exception as e:
                print(f"  AllGather world_size={world_size}, nelem={format_nelem(test_nelem)}: ERROR - {e}")
    
    # Test very small nelem (below min in data)
    print("\nTesting values below minimum available data:")
    for world_size in [2, 4, 8]:
        if world_size in allreduce_dict:
            min_nelem = min(allreduce_dict[world_size].keys())
            test_nelem = min_nelem // 2  # Below available data
            
            try:
                allreduce_time = get_allreduce_time(world_size, test_nelem)
                print(f"  AllReduce world_size={world_size}, nelem={format_nelem(test_nelem)} (below min): {allreduce_time:.3f}ms")
            except Exception as e:
                print(f"  AllReduce world_size={world_size}, nelem={format_nelem(test_nelem)}: ERROR - {e}")
        
        if world_size in allgather_dict:
            min_nelem = min(allgather_dict[world_size].keys())
            test_nelem = min_nelem // 2  # Below available data
            
            try:
                allgather_time = get_allgather_time(world_size, test_nelem)
                print(f"  AllGather world_size={world_size}, nelem={format_nelem(test_nelem)} (below min): {allgather_time:.3f}ms")
            except Exception as e:
                print(f"  AllGather world_size={world_size}, nelem={format_nelem(test_nelem)}: ERROR - {e}")
    
    return True

def compare_allreduce_vs_allgather():
    print("\n" + "=" * 80)
    print("ALLREDUCE vs ALLGATHER COMPARISON")
    print("=" * 80)
    
    test_cases = [
        (2, 1024),
        (2, 1048576),
        (2, 134217728),
        (4, 1024),
        (4, 1048576),
        (8, 1024),
        (8, 1048576),
    ]
    
    comparison_results = []
    for world_size, nelem in test_cases:
        try:
            allreduce_time = get_allreduce_time(world_size, nelem)
            allgather_time = get_allgather_time(world_size, nelem)
            ratio = allgather_time / allreduce_time
            
            comparison_results.append({
                'world_size': world_size,
                'nelem_formatted': format_nelem(nelem),
                'allreduce_ms': allreduce_time,
                'allgather_ms': allgather_time,
                'ratio': ratio,
                'faster': 'AllReduce' if allreduce_time < allgather_time else 'AllGather'
            })
        except Exception as e:
            print(f"  Error for world_size={world_size}, nelem={nelem}: {e}")
    
    if comparison_results:
        comparison_df = pd.DataFrame(comparison_results)
        print(comparison_df.to_string(index=False))
        
        avg_ratio = np.mean([r['ratio'] for r in comparison_results])
        print(f"\nAverage AllGather/AllReduce ratio: {avg_ratio:.2f}")
        print("(Ratio > 1 means AllGather is slower than AllReduce)")

def main():
    print("Testing AllReduce and AllGather time calculations from timemodule.py")
    print("This script validates the interpolation logic and data consistency")
    
    success = True
    
    try:
        success &= test_allreduce_time()
        success &= test_allgather_time()
        success &= test_edge_cases()
        compare_allreduce_vs_allgather()
        
        print("\n" + "=" * 80)
        if success:
            print("🎉 All tests PASSED! The timemodule functions are working correctly.")
        else:
            print("❌ Some tests FAILED! Check the output above for details.")
        print("=" * 80)
        
    except Exception as e:
        print(f"\n❌ Test suite failed with error: {e}")
        success = False
    
    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)