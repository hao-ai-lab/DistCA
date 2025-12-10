# %%
import json
import pandas as pd
import matplotlib.pyplot as plt
import os
from collections import defaultdict

# %%
# Analyze memory usage from JSONL files across all token sizes
root = "/mnt/weka/home/hao.zhang/jd/d2/benchmarks/_250915_small_exps/logs.v1-item_04"

def load_memory_data(num_tokens):
    """Load memory data from JSONL file for a specific token size"""
    try:
        jsonl_path = os.path.join(root, f"num_tokens_{num_tokens}", "mem-log", "mem.rank0.log.jsonl")
        
        memory_data = []
        with open(jsonl_path, 'r') as f:
            for line in f:
                data = json.loads(line.strip())
                memory_data.append(data)
        
        return memory_data
    except FileNotFoundError:
        print(f"No data found for num_tokens={num_tokens}")
        return None

# %%
# Load data for all token sizes
token_sizes = [16384, 32768, 65536, 131072, 262144]
all_memory_data = {}

print("=== Loading Memory Data from JSONL Files ===")
for tokens in token_sizes:
    data = load_memory_data(tokens)
    if data:
        all_memory_data[tokens] = data
        print(f"{tokens:,} tokens: {len(data)} measurements")
        
        # Show peak memory
        peak_memory = max(entry.get('allocated_peak', 0) for entry in data)
        print(f"  Peak memory: {peak_memory/1024:.2f} GB")
    else:
        print(f"{tokens:,} tokens: No data found")

# %%
# Create comprehensive analysis
if all_memory_data:
    # Extract peak memory usage for each token size
    token_list = sorted(all_memory_data.keys())
    peak_memory_gb = []
    current_memory_gb = []
    total_alloc_gb = []
    gpu_usage_gb = []
    
    for tokens in token_list:
        data = all_memory_data[tokens]
        
        # Get peak values across all measurements
        peak_mem = max(entry.get('allocated_peak', 0) for entry in data) / 1024  # Convert MB to GB
        current_mem = max(entry.get('allocated_cur', 0) for entry in data) / 1024
        total_mem = max(entry.get('total_alloc', 0) for entry in data) / 1024
        gpu_mem = max(entry.get('pynvml_gpu_memory_usage', 0) for entry in data) / 1024
        
        peak_memory_gb.append(peak_mem)
        current_memory_gb.append(current_mem)
        total_alloc_gb.append(total_mem)
        gpu_usage_gb.append(gpu_mem)
    
    # %%
    # Create the main chart: Memory vs Token Count
    plt.figure(figsize=(14, 10))
    
    # Plot different memory metrics
    plt.plot(token_list, peak_memory_gb, 'o-', linewidth=3, markersize=10, 
             label='Peak Allocated Memory', color='red', alpha=0.8)
    plt.plot(token_list, current_memory_gb, 's-', linewidth=3, markersize=8, 
             label='Current Allocated Memory', color='blue', alpha=0.8)
    plt.plot(token_list, gpu_usage_gb, '^-', linewidth=3, markersize=8, 
             label='Total GPU Memory Usage', color='green', alpha=0.8)
    
    # Add value labels
    for i, tokens in enumerate(token_list):
        plt.annotate(f'{peak_memory_gb[i]:.1f}GB', 
                    (tokens, peak_memory_gb[i]), 
                    textcoords="offset points", xytext=(0,15), 
                    ha='center', fontsize=10, fontweight='bold', color='red')
        plt.annotate(f'{gpu_usage_gb[i]:.1f}GB', 
                    (tokens, gpu_usage_gb[i]), 
                    textcoords="offset points", xytext=(0,-20), 
                    ha='center', fontsize=10, fontweight='bold', color='green')
    
    plt.xlabel('Number of Tokens per Batch', fontsize=14, fontweight='bold')
    plt.ylabel('Memory Usage (GB)', fontsize=14, fontweight='bold')
    plt.title('Peak Memory Usage vs Number of Tokens per Batch\n(Based on Actual Runtime Memory Measurements)', 
              fontsize=16, fontweight='bold', pad=20)
    
    # Format x-axis
    plt.xscale('log', base=2)
    plt.xticks(token_list, [f'{t//1024}K' for t in token_list], fontsize=12)
    plt.yticks(fontsize=12)
    
    plt.legend(fontsize=12, loc='upper left')
    plt.grid(True, alpha=0.3, which='both')
    plt.tight_layout()
    
    plt.savefig('memory_usage_vs_tokens_runtime.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("Runtime memory chart saved as 'memory_usage_vs_tokens_runtime.png'")
    
    # %%
    # Create detailed comparison table
    print("\n=== Detailed Memory Usage Analysis ===")
    print(f"{'Tokens':<12} {'Peak (GB)':<12} {'Current (GB)':<14} {'Total (GB)':<12} {'GPU (GB)':<10}")
    print("-" * 70)
    
    for i, tokens in enumerate(token_list):
        print(f"{tokens//1024}K{'':<8} {peak_memory_gb[i]:<12.2f} {current_memory_gb[i]:<14.2f} "
              f"{total_alloc_gb[i]:<12.2f} {gpu_usage_gb[i]:<10.2f}")
    
    # %%
    # Calculate scaling relationships
    print("\n=== Memory Scaling Analysis ===")
    
    # Calculate ratios between consecutive token sizes
    print("\nMemory scaling between token sizes:")
    for i in range(1, len(token_list)):
        token_ratio = token_list[i] / token_list[i-1]
        memory_ratio = peak_memory_gb[i] / peak_memory_gb[i-1]
        print(f"{token_list[i-1]//1024}K -> {token_list[i]//1024}K: "
              f"tokens ×{token_ratio:.1f}, memory ×{memory_ratio:.2f}")
    
    # Overall scaling
    total_token_ratio = token_list[-1] / token_list[0]
    total_memory_ratio = peak_memory_gb[-1] / peak_memory_gb[0]
    print(f"\nOverall scaling ({token_list[0]//1024}K -> {token_list[-1]//1024}K):")
    print(f"Tokens increased by: {total_token_ratio:.1f}x")
    print(f"Memory increased by: {total_memory_ratio:.2f}x")
    
    # Memory efficiency
    print(f"\nMemory efficiency:")
    for i, tokens in enumerate(token_list):
        mb_per_token = (peak_memory_gb[i] * 1024) / tokens
        print(f"{tokens//1024}K tokens: {mb_per_token:.3f} MB per token")

else:
    print("No memory data available for analysis")





