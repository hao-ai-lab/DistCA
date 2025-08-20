# %%
# Linear time
# This script computes "linear time" and "attention time" from the formulas on your slides,
# prints verbose intermediate values, builds a table, and plots a single chart (x: tokens, y: duration).
#
# Assumptions (matching the "Breaking Point Case Study"):
# - n_qo = 64 heads for Q/O
# - n_kv = 4  K/V heads
# - h_d   = 128 head_dim
# - d     = 4096 model hidden size
# - k     = 3  (so intermediate size k*d = 12288)
# - c     = 0.5 (common for SwiGLU-style gated MLP; you can change to 1 if desired)
#
# Formulas (from slides):
# Linear FLOPs (sum of components):
#   q:         2 t n_qo h_d d
#   kv:        4 t n_kv h_d d
#   o:         2 t n_qo h_d d
#   MLP gate:  2 t d (k d)      = 2 t k d^2
#   MLP fc1:   2 t d (k d)      = 2 t k d^2
#   MLP act:   4 t k d
#   MLP fc2:   2 t (c k d) d    = 2 t c k d^2
# Linear "time" := sum of FLOPs (arbitrary units)
#
# Attention FLOPs:
#   attn_flops(t) = 4 t^2 n_qo h_d
# Attention "time" := (2/3) * attn_flops(t)  (≈ FlashAttention runs at ~2/3 of peak H100 per the slide note)
#
# You can tweak params in the PARAMS block.

# %%

import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt

# %%

def compute_linear_vs_attention(model_name, n_qo, n_kv, h_d, d, k, n_experts=1, c=1, 
                               t_min=1024, t_max=65536, t_step=1024, 
                               attn_ratio=2/3, verbose=True, plot=True):
    """
    Compute and optionally plot linear time vs attention time for a given model configuration.
    
    Args:
        model_name: Name of the model for display
        n_qo: Number of Q/O heads
        n_kv: Number of K/V heads  
        h_d: Head dimension
        d: Model hidden size
        k: MLP expansion factor (intermediate_size / hidden_size)
        n_experts: Number of experts (1 for dense models)
        c: MLP gate factor (typically 1)
        t_min: Minimum tokens
        t_max: Maximum tokens
        t_step: Token step size
        attn_ratio: Attention efficiency ratio (default 2/3 for FlashAttention)
        verbose: Whether to print intermediate results
        plot: Whether to generate plot
        
    Returns:
        pandas.DataFrame: Table with tokens, linear_time, attn_time, and component breakdowns
    """
    TFLOPS = 1e12
    H100_peak_flops = 1 * TFLOPS
    
    # Helper functions
    def linear_flops(t):
        q   = 2 * t * n_qo * h_d * d
        kv  = 4 * t * n_kv * h_d * d
        o   = 2 * t * n_qo * h_d * d
        mlp_gate = 2 * t * k * d**2
        mlp_fc1  = 2 * t * k * d**2
        mlp_act  = 4 * t * k * d
        mlp_fc2  = 2 * t * c * k * d**2
        mlps = n_experts * (mlp_gate + mlp_fc1 + mlp_act + mlp_fc2)
        total = q + kv + o + mlps
        return {
            "q": q, "kv": kv, "o": o,
            "mlp_gate": mlp_gate, "mlp_fc1": mlp_fc1,
            "mlp_act": mlp_act, "mlp_fc2": mlp_fc2,
            "linear_time": total
        }

    def linear_time(t):
        return linear_flops(t)["linear_time"] / H100_peak_flops

    def attn_flops(t):
        return 4 * (t**2) * n_qo * h_d

    def attn_time(t):
        return attn_ratio * attn_flops(t) / H100_peak_flops
    
    # Print parameters if verbose
    if verbose:
        print("=== Parameters ===")
        print(f"Model: {model_name}")
        print(f"n_qo={n_qo}, n_kv={n_kv}, h_d={h_d}, d={d}, k={k}, c={c}")
        print(f"n_experts={n_experts}")
        print(f"Intermediate size k*d = {k*d}")
        print(f"Token sweep: t in [{t_min}, {t_max}] step {t_step}")
        print()

    # Build table
    tokens = np.arange(t_min, t_max + 1, t_step, dtype=int)
    rows = []
    for t in tokens:
        L = linear_flops(t)
        A = attn_flops(t)
        rows.append({
            "tokens": t,
            "linear_time": linear_time(t),
            "attn_flops": A,
            "attn_time": attn_time(t),
            # Component columns for inspection
            "q": L["q"], "kv": L["kv"], "o": L["o"],
            "mlp_gate": L["mlp_gate"], "mlp_fc1": L["mlp_fc1"],
            "mlp_act": L["mlp_act"], "mlp_fc2": L["mlp_fc2"],
        })

    df = pd.DataFrame(rows)
    
    if verbose:
        print("=== Head of the computed table ===")
        print(df.head(3))
        print()
        print("=== Tail of the computed table ===")
        print(df.tail(3))
        print()

    # Plot if requested
    if plot:
        plt.figure(figsize=(10, 6))
        plt.plot(df["tokens"], df["linear_time"], label="Linear time (sum of FLOPs)", linewidth=2)
        plt.plot(df["tokens"], df["attn_time"], label=f"Attention time ({attn_ratio:.2f} · attn FLOPs)", linewidth=2)
        
        # Find intersection point where attn_time = linear_time
        diff = np.abs(df["attn_time"] - df["linear_time"])
        min_idx = np.argmin(diff)
        intersection_tokens = df.iloc[min_idx]["tokens"]
        intersection_time = df.iloc[min_idx]["linear_time"]
        
        # Add vertical line at intersection
        plt.axvline(x=intersection_tokens, color='red', linestyle='--', alpha=0.7, 
                   label=f'Intersection at {intersection_tokens} tokens')
        
        # Optionally add a point marker at intersection
        plt.plot(intersection_tokens, intersection_time, 'ro', markersize=8, 
                label=f'Crossover point')
        
        plt.xlabel("tokens (t)")
        plt.ylabel("duration (arbitrary units)")
        plt.title(f"{model_name}: Linear time vs Attention time")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"{model_name}.png")
        plt.show()
    
    return df

# %%

# Qwen235B
# https://huggingface.co/Qwen/Qwen3-235B-A22B/blob/main/config.json
df_qwen = compute_linear_vs_attention(
    model_name="Qwen235B",
    n_qo=64,
    n_kv=4,
    h_d=128,
    d=4096,
    k=0.375,  # expert_size = 1536 = 4096 * 0.375
    n_experts=8,
    c=1
)

# %%

# Llama8B
# https://huggingface.co/meta-llama/Meta-Llama-3-8B/blob/main/config.json
df_llama8b = compute_linear_vs_attention(
    model_name="Llama8B",
    n_qo=32,  # num_attention_heads
    n_kv=8,   # num_key_value_heads
    h_d=128,  # hidden_size / num_attention_heads = 4096 / 32
    d=4096,   # hidden_size
    k=3.5,    # intermediate_size / hidden_size = 14336 / 4096
    n_experts=1,  # dense model
    c=1
)

# %%

# Llama70B
# https://huggingface.co/meta-llama/Meta-Llama-3-70B/blob/main/config.json
df_llama70b = compute_linear_vs_attention(
    model_name="Llama70B",
    n_qo=64,  # num_attention_heads
    n_kv=8,   # num_key_value_heads
    h_d=128,  # hidden_size / num_attention_heads = 8192 / 64
    d=8192,   # hidden_size
    k=3.5,    # intermediate_size / hidden_size = 28672 / 8192
    n_experts=1,  # dense model
    c=1,
    t_max=128*1024
)

# %%
# Mistral7B
# https://huggingface.co/mistralai/Mistral-7B-v0.1/blob/main/config.json
df_mistral7b = compute_linear_vs_attention(
    model_name="Mistral7B",
    n_qo=32,  # num_attention_heads
    n_kv=8,   # num_key_value_heads
    h_d=128,  # hidden_size / num_attention_heads = 4096 / 32
    d=4096,   # hidden_size
    k=3.5,    # intermediate_size / hidden_size = 14336 / 4096
    n_experts=1,  # dense model
    c=1
)

# %%
