#!/usr/bin/env python3
"""
Test script for timemodule.py MLP time calculation
Prints detailed tables for sanity checking the FLOP calculations and timing
"""

import sys
import pandas as pd
from timemodule import get_mlp_time, K, M, G, T, hidden_size, num_qo_head, num_kv_head, head_dim, expert_dim, num_activate_experts, dtype_size, flops_per_ms

def format_flops(flops):
    """Format FLOPS in human readable format"""
    if flops >= T:
        return f"{flops/T:.2f}T"
    elif flops >= G:
        return f"{flops/G:.2f}G"
    elif flops >= M:
        return f"{flops/M:.2f}M"
    elif flops >= K:
        return f"{flops/K:.2f}K"
    else:
        return f"{flops:.0f}"

def test_mlp_time():
    print("=" * 80)
    print("MLP Time Calculation Test - Sanity Check")
    print("=" * 80)
    
    # Print configuration
    print("\nConfiguration:")
    print(f"  hidden_size: {hidden_size:,}")
    print(f"  num_qo_head: {num_qo_head}")
    print(f"  num_kv_head: {num_kv_head}")
    print(f"  head_dim: {head_dim}")
    print(f"  expert_dim: {expert_dim:,}")
    print(f"  num_activate_experts: {num_activate_experts}")
    print(f"  dtype_size: {dtype_size} bytes")
    print(f"  flops_per_ms: {format_flops(flops_per_ms)}")
    
    # Test cases: different tp, cp, and sequence length combinations
    test_cases = [
        (1, 1, 1024),
        (1, 1, 2048),
        (1, 1, 4096),
        (2, 1, 1024),
        (4, 1, 1024),
        (1, 2, 1024),
        (2, 2, 1024),
        (4, 4, 1024),
    ]
    
    results = []
    
    for tp, cp, seq_len in test_cases:
        latency_ms, component_flops = get_mlp_time(tp, cp, seq_len)
        
        results.append({
            'tp': tp,
            'cp': cp,
            'seq_len': seq_len,
            'latency_ms': latency_ms,
            'total_flops': component_flops['linear_flops'],
            'total_flops_formatted': format_flops(component_flops['linear_flops']),
            'q_proj': format_flops(component_flops['q_proj_flops']),
            'k_proj': format_flops(component_flops['k_proj_flops']),
            'v_proj': format_flops(component_flops['v_proj_flops']),
            'o_proj': format_flops(component_flops['o_proj_flops']),
            'mlp_fc1': format_flops(component_flops['mlp_fc1_flops']),
            'mlp_gate': format_flops(component_flops['mlp_gate_flops']),
            'mlp_activation': format_flops(component_flops['mlp_activation_flops']),
            'mlp_fc2': format_flops(component_flops['mlp_fc2_flops']),
        })
    
    # Create and print summary table
    df = pd.DataFrame(results)
    print("\n" + "=" * 80)
    print("SUMMARY TABLE")
    print("=" * 80)
    print(df[['tp', 'cp', 'seq_len', 'latency_ms', 'total_flops_formatted']].to_string(index=False))
    
    # Detailed component breakdown for first test case
    print("\n" + "=" * 80)
    print(f"DETAILED COMPONENT BREAKDOWN (tp={test_cases[0][0]}, cp={test_cases[0][1]}, seq_len={test_cases[0][2]})")
    print("=" * 80)
    
    latency_ms, component_flops = get_mlp_time(*test_cases[0])
    
    component_data = []
    for component, flops in component_flops.items():
        if component != 'linear_flops':  # Skip total
            component_data.append({
                'Component': component,
                'FLOPS': flops,
                'FLOPS_Formatted': format_flops(flops),
                'Percentage': f"{(flops / component_flops['linear_flops']) * 100:.1f}%"
            })
    
    component_df = pd.DataFrame(component_data)
    print(component_df.to_string(index=False))
    
    # Manual verification calculations
    print("\n" + "=" * 80)
    print("MANUAL VERIFICATION")
    print("=" * 80)
    
    tp, cp, seq_len = test_cases[0]
    print(f"Test case: tp={tp}, cp={cp}, seq_len={seq_len}")
    
    # Calculate expected values manually
    expected_q_proj = dtype_size * seq_len * hidden_size * num_qo_head * head_dim
    expected_k_proj = dtype_size * seq_len * hidden_size * num_kv_head * head_dim
    expected_v_proj = dtype_size * seq_len * hidden_size * num_kv_head * head_dim
    expected_o_proj = dtype_size * seq_len * hidden_size * num_qo_head * head_dim
    
    expected_mlp_fc1 = 2 * seq_len * hidden_size * (expert_dim * num_activate_experts)
    expected_mlp_gate = 2 * seq_len * hidden_size * (expert_dim * num_activate_experts)
    expected_mlp_activation = 4 * seq_len * (expert_dim * num_activate_experts)
    expected_mlp_fc2 = 2 * seq_len * hidden_size * (expert_dim * num_activate_experts)
    
    expected_total = (expected_q_proj + expected_k_proj + expected_v_proj + expected_o_proj +
                     expected_mlp_fc1 + expected_mlp_gate + expected_mlp_activation + expected_mlp_fc2)
    
    expected_latency = expected_total / (tp * cp * flops_per_ms)
    
    print(f"\nExpected calculations:")
    print(f"  Q proj: {dtype_size} * {seq_len} * {hidden_size} * {num_qo_head} * {head_dim} = {format_flops(expected_q_proj)}")
    print(f"  K proj: {dtype_size} * {seq_len} * {hidden_size} * {num_kv_head} * {head_dim} = {format_flops(expected_k_proj)}")
    print(f"  V proj: {dtype_size} * {seq_len} * {hidden_size} * {num_kv_head} * {head_dim} = {format_flops(expected_v_proj)}")
    print(f"  O proj: {dtype_size} * {seq_len} * {hidden_size} * {num_qo_head} * {head_dim} = {format_flops(expected_o_proj)}")
    print(f"  MLP FC1: 2 * {seq_len} * {hidden_size} * ({expert_dim} * {num_activate_experts}) = {format_flops(expected_mlp_fc1)}")
    print(f"  MLP Gate: 2 * {seq_len} * {hidden_size} * ({expert_dim} * {num_activate_experts}) = {format_flops(expected_mlp_gate)}")
    print(f"  MLP Activation: 4 * {seq_len} * ({expert_dim} * {num_activate_experts}) = {format_flops(expected_mlp_activation)}")
    print(f"  MLP FC2: 2 * {seq_len} * {hidden_size} * ({expert_dim} * {num_activate_experts}) = {format_flops(expected_mlp_fc2)}")
    print(f"  Total FLOPS: {format_flops(expected_total)}")
    print(f"  Expected latency: {expected_total} / ({tp} * {cp} * {format_flops(flops_per_ms)}) = {expected_latency:.6f} ms")
    
    # Verify against actual results
    actual_latency, actual_flops = get_mlp_time(tp, cp, seq_len)
    print(f"\nActual results:")
    print(f"  Total FLOPS: {format_flops(actual_flops['linear_flops'])}")
    print(f"  Actual latency: {actual_latency:.6f} ms")
    
    # Check if they match
    flops_match = abs(expected_total - actual_flops['linear_flops']) < 1e-6
    latency_match = abs(expected_latency - actual_latency) < 1e-9
    
    print(f"\nVerification:")
    print(f"  FLOPS match: {'✓' if flops_match else '✗'}")
    print(f"  Latency match: {'✓' if latency_match else '✗'}")
    
    if flops_match and latency_match:
        print(f"\n🎉 All calculations are CORRECT!")
    else:
        print(f"\n❌ There are discrepancies in the calculations!")
        return False
    
    # Test scaling behavior
    print("\n" + "=" * 80)
    print("SCALING BEHAVIOR ANALYSIS")
    print("=" * 80)
    
    print("\nSequence length scaling (tp=1, cp=1):")
    base_seq = 1024
    for multiplier in [1, 2, 4, 8]:
        seq_len = base_seq * multiplier
        latency_ms, _ = get_mlp_time(1, 1, seq_len)
        print(f"  seq_len={seq_len:5d}: latency={latency_ms:8.3f}ms (scaling factor: {latency_ms / get_mlp_time(1, 1, base_seq)[0]:.1f}x)")
    
    print("\nTensor parallelism scaling (cp=1, seq_len=1024):")
    for tp in [1, 2, 4, 8]:
        latency_ms, _ = get_mlp_time(tp, 1, 1024)
        print(f"  tp={tp}: latency={latency_ms:8.3f}ms (speedup: {get_mlp_time(1, 1, 1024)[0] / latency_ms:.1f}x)")
    
    print("\nContext parallelism scaling (tp=1, seq_len=1024):")
    for cp in [1, 2, 4, 8]:
        latency_ms, _ = get_mlp_time(1, cp, 1024)
        print(f"  cp={cp}: latency={latency_ms:8.3f}ms (speedup: {get_mlp_time(1, 1, 1024)[0] / latency_ms:.1f}x)")
    
    return True

if __name__ == "__main__":
    success = test_mlp_time()
    sys.exit(0 if success else 1) 