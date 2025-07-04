#!/usr/bin/env python3
"""
Expanded test script for timemodule.py MLP and Attention time calculation
Tests sequence lengths: 1k, 4k, 8k, 16k, 32k, 64k
Compares attention vs MLP times
"""

import sys
import pandas as pd
import numpy as np
from timemodule import (
    get_mlp_time, get_attn_time, setup_attn_time,
    K, M, G, T, hidden_size, num_qo_head, num_kv_head, 
    head_dim, expert_dim, num_activate_experts, dtype_size, flops_per_ms
)

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

def test_mlp_and_attn_time():
    print("=" * 100)
    print("EXPANDED MLP AND ATTENTION TIME TEST")
    print("=" * 100)
    
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
    
    # Test cases: different tp, cp combinations
    parallelism_configs = [
        (1, 1),
        (1, 2),
        (1, 4),
        (1, 8),
        (2, 1),
        (2, 2),
        (2, 4),
        (2, 8),
        (4, 1),
        (4, 2),
        (4, 4),
        (4, 8),
        (8, 1),
        (8, 2),
        (8, 4),
        (8, 8),
    ]
    
    # Sequence lengths to test
    sequence_lengths = [1*K, 4*K, 8*K, 16*K, 32*K, 64*K, 128*K, 256*K, 512*K, 1024*K, 2048*K, 4096*K, 8192*K]
    
    print("\n" + "=" * 100)
    print("ATTENTION VS MLP TIME COMPARISON")
    print("=" * 100)
    
    for tp, cp in parallelism_configs:
        print(f"\nParallelism Config: tp={tp}, cp={cp}")
        print("-" * 80)
        
        results = []
        
        for seq_len in sequence_lengths:
            try:
                # Get MLP time
                mlp_latency_ms, mlp_component_flops = get_mlp_time(tp, cp, seq_len)
                
                # Get attention time
                attn_latency_ms = get_attn_time(tp, cp, seq_len)
                
                # Calculate ratios
                total_latency = mlp_latency_ms + attn_latency_ms
                attn_percentage = (attn_latency_ms / total_latency) * 100
                mlp_percentage = (mlp_latency_ms / total_latency) * 100
                attn_mlp_ratio = attn_latency_ms / mlp_latency_ms
                
                results.append({
                    'seq_len': f"{seq_len//K}K",
                    'seq_len_raw': seq_len,
                    'attn_ms': attn_latency_ms,
                    'mlp_ms': mlp_latency_ms,
                    'total_ms': total_latency,
                    'attn_%': attn_percentage,
                    'mlp_%': mlp_percentage,
                    'attn/mlp': attn_mlp_ratio,
                    'mlp_flops': format_flops(mlp_component_flops['linear_flops'])
                })
                
            except Exception as e:
                raise e
                print(f"  Error for seq_len={seq_len//K}K: {e}")
                continue
        
        if results:
            df = pd.DataFrame(results)
            print(df[['seq_len', 'attn_ms', 'mlp_ms', 'total_ms', 'attn_%', 'mlp_%', 'attn/mlp']].to_string(
                index=False, 
                float_format='{:.3f}'.format
            ))
            
            # Analysis
            print(f"\nAnalysis for tp={tp}, cp={cp}:")
            avg_attn_pct = df['attn_%'].mean()
            avg_mlp_pct = df['mlp_%'].mean()
            print(f"  Average attention percentage: {avg_attn_pct:.1f}%")
            print(f"  Average MLP percentage: {avg_mlp_pct:.1f}%")
            
            if len(df) > 1:
                # Scaling analysis
                first_seq = df.iloc[0]
                last_seq = df.iloc[-1]
                seq_scaling = last_seq['seq_len_raw'] / first_seq['seq_len_raw']
                attn_scaling = last_seq['attn_ms'] / first_seq['attn_ms']
                mlp_scaling = last_seq['mlp_ms'] / first_seq['mlp_ms']
                
                print(f"  Sequence length scaling ({first_seq['seq_len']} -> {last_seq['seq_len']}):")
                print(f"    Sequence: {seq_scaling:.1f}x")
                print(f"    Attention: {attn_scaling:.1f}x")
                print(f"    MLP: {mlp_scaling:.1f}x")
    
    print("\n" + "=" * 100)
    print("DETAILED MLP COMPONENT BREAKDOWN (tp=1, cp=1)")
    print("=" * 100)
    
    detailed_results = []
    for seq_len in sequence_lengths:
        mlp_latency_ms, component_flops = get_mlp_time(1, 1, seq_len)
        
        detailed_results.append({
            'seq_len': f"{seq_len//K}K",
            'q_proj': format_flops(component_flops['q_proj_flops']),
            'k_proj': format_flops(component_flops['k_proj_flops']),
            'v_proj': format_flops(component_flops['v_proj_flops']),
            'o_proj': format_flops(component_flops['o_proj_flops']),
            'mlp_fc1': format_flops(component_flops['mlp_fc1_flops']),
            'mlp_gate': format_flops(component_flops['mlp_gate_flops']),
            'mlp_activation': format_flops(component_flops['mlp_activation_flops']),
            'mlp_fc2': format_flops(component_flops['mlp_fc2_flops']),
            'total': format_flops(component_flops['linear_flops']),
            'latency_ms': f"{mlp_latency_ms:.3f}"
        })
    
    detail_df = pd.DataFrame(detailed_results)
    print(detail_df.to_string(index=False))
    
    print("\n" + "=" * 100)
    print("PARALLELISM SCALING ANALYSIS")
    print("=" * 100)
    
    base_seq_len = 8*K  # Use 8K as base sequence length
    
    print(f"\nTensor Parallelism Scaling (cp=1, seq_len={base_seq_len//K}K):")
    print("-" * 60)
    
    tp_results = []
    base_mlp_time = None
    base_attn_time = None
    
    for tp in [1, 2, 4, 8]:
        try:
            mlp_latency_ms, _ = get_mlp_time(tp, 1, base_seq_len)
            attn_latency_ms = get_attn_time(tp, 1, base_seq_len)
            
            if base_mlp_time is None:
                base_mlp_time = mlp_latency_ms
                base_attn_time = attn_latency_ms
            
            mlp_speedup = base_mlp_time / mlp_latency_ms
            attn_speedup = base_attn_time / attn_latency_ms
            
            tp_results.append({
                'tp': tp,
                'mlp_ms': mlp_latency_ms,
                'attn_ms': attn_latency_ms,
                'mlp_speedup': mlp_speedup,
                'attn_speedup': attn_speedup,
                'mlp_efficiency': (mlp_speedup / tp) * 100,
                'attn_efficiency': (attn_speedup / tp) * 100
            })
        except Exception as e:
            print(f"Error for tp={tp}: {e}")
    
    if tp_results:
        tp_df = pd.DataFrame(tp_results)
        print(tp_df[['tp', 'mlp_ms', 'attn_ms', 'mlp_speedup', 'attn_speedup', 'mlp_efficiency', 'attn_efficiency']].to_string(
            index=False,
            float_format='{:.3f}'.format
        ))
    
    print(f"\nContext Parallelism Scaling (tp=1, seq_len={base_seq_len//K}K):")
    print("-" * 60)
    
    cp_results = []
    base_mlp_time = None
    base_attn_time = None
    
    for cp in [1, 2, 4, 8]:
        try:
            mlp_latency_ms, _ = get_mlp_time(1, cp, base_seq_len)
            attn_latency_ms = get_attn_time(1, cp, base_seq_len)
            
            if base_mlp_time is None:
                base_mlp_time = mlp_latency_ms
                base_attn_time = attn_latency_ms
            
            mlp_speedup = base_mlp_time / mlp_latency_ms
            attn_speedup = base_attn_time / attn_latency_ms
            
            cp_results.append({
                'cp': cp,
                'mlp_ms': mlp_latency_ms,
                'attn_ms': attn_latency_ms,
                'mlp_speedup': mlp_speedup,
                'attn_speedup': attn_speedup,
                'mlp_efficiency': (mlp_speedup / cp) * 100,
                'attn_efficiency': (attn_speedup / cp) * 100
            })
        except Exception as e:
            print(f"Error for cp={cp}: {e}")
    
    if cp_results:
        cp_df = pd.DataFrame(cp_results)
        print(cp_df[['cp', 'mlp_ms', 'attn_ms', 'mlp_speedup', 'attn_speedup', 'mlp_efficiency', 'attn_efficiency']].to_string(
            index=False,
            float_format='{:.3f}'.format
        ))
    
    print("\n" + "=" * 100)
    print("SEQUENCE LENGTH SCALING ANALYSIS")
    print("=" * 100)
    
    print("\nLatency vs Sequence Length (tp=1, cp=1):")
    print("-" * 50)
    
    scaling_results = []
    for seq_len in sequence_lengths:
        try:
            mlp_latency_ms, _ = get_mlp_time(1, 1, seq_len)
            attn_latency_ms = get_attn_time(1, 1, seq_len)
            
            scaling_results.append({
                'seq_len': f"{seq_len//K}K",
                'seq_len_raw': seq_len,
                'mlp_ms': mlp_latency_ms,
                'attn_ms': attn_latency_ms,
                'total_ms': mlp_latency_ms + attn_latency_ms,
                "attn/mlp ratio": attn_latency_ms / mlp_latency_ms,
            })
        except Exception as e:
            print(f"Error for seq_len={seq_len//K}K: {e}")
    
    if scaling_results:
        scaling_df = pd.DataFrame(scaling_results)
        print(scaling_df[['seq_len', 'mlp_ms', 'attn_ms', 'total_ms', "attn/mlp ratio"]].to_string(
            index=False,
            float_format='{:.3f}'.format
        ))
        
        # Calculate scaling factors
        if len(scaling_df) > 1:
            base_row = scaling_df.iloc[0]
            print(f"\nScaling factors (relative to {base_row['seq_len']}):")
            for i, row in scaling_df.iterrows():
                seq_factor = row['seq_len_raw'] / base_row['seq_len_raw']
                mlp_factor = row['mlp_ms'] / base_row['mlp_ms']
                attn_factor = row['attn_ms'] / base_row['attn_ms']
                print(f"  {row['seq_len']}: seq={seq_factor:.1f}x, mlp={mlp_factor:.1f}x, attn={attn_factor:.1f}x")
    
    print("\n" + "=" * 100)
    print("SUMMARY INSIGHTS")
    print("=" * 100)
    
    # Load attention data to verify availability
    try:
        attn_dict = setup_attn_time()
        available_configs = list(attn_dict.keys())
        print(f"\nAvailable attention configurations: {len(available_configs)}")
        print("Sample configurations:", available_configs[:5])
        
        # Check sequence length coverage
        if available_configs:
            sample_config = available_configs[0]
            available_seq_lens = list(attn_dict[sample_config].keys())
            print(f"Available sequence lengths for {sample_config}: {sorted(available_seq_lens)}")
            
    except Exception as e:
        print(f"Error loading attention data: {e}")
    
    return True

if __name__ == "__main__":
    success = test_mlp_and_attn_time()
    sys.exit(0 if success else 1) 