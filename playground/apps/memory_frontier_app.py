#!/usr/bin/env python3
"""
Memory Frontier Explorer - Interactive webapp for exploring GPU memory configurations.

Explore what parallelism configurations (TP, PP, CP, DP), sequence lengths, and micro-batch
sizes will fit on your GPU cluster for various model architectures.

Usage:
    streamlit run memory_frontier_app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Any, Optional
import itertools


# ==============================================================================
# Configuration Data
# ==============================================================================

@dataclass
class ModelConfig:
    """Model architecture configuration."""
    name: str
    num_layers: int
    hidden_size: int
    num_attention_heads: int
    num_query_groups: int
    ffn_hidden_size: int
    vocab_size: int
    max_position_embeddings: int = 8192
    swiglu: bool = True
    untie_embeddings_and_output_weights: bool = True


@dataclass
class GPUConfig:
    """GPU type configuration."""
    name: str
    memory_gb: float
    bandwidth_gb_s: float  # Memory bandwidth
    tflops_bf16: float  # Peak BF16 TFLOPS


# Pre-defined model configurations
MODEL_CONFIGS = {
    "LLaMA-7B": ModelConfig(
        name="LLaMA-7B",
        num_layers=32,
        hidden_size=4096,
        num_attention_heads=32,
        num_query_groups=32,
        ffn_hidden_size=11008,
        vocab_size=32000,
    ),
    "LLaMA-8B": ModelConfig(
        name="LLaMA-8B",
        num_layers=32,
        hidden_size=4096,
        num_attention_heads=32,
        num_query_groups=8,
        ffn_hidden_size=14336,
        vocab_size=128256,
        max_position_embeddings=8192,
    ),
    "LLaMA-13B": ModelConfig(
        name="LLaMA-13B",
        num_layers=40,
        hidden_size=5120,
        num_attention_heads=40,
        num_query_groups=40,
        ffn_hidden_size=13824,
        vocab_size=32000,
    ),
    "LLaMA-70B": ModelConfig(
        name="LLaMA-70B",
        num_layers=80,
        hidden_size=8192,
        num_attention_heads=64,
        num_query_groups=8,
        ffn_hidden_size=28672,
        vocab_size=128256,
        max_position_embeddings=8192,
    ),
    "LLaMA-405B": ModelConfig(
        name="LLaMA-405B",
        num_layers=126,
        hidden_size=16384,
        num_attention_heads=128,
        num_query_groups=8,
        ffn_hidden_size=53248,
        vocab_size=128256,
        max_position_embeddings=131072,
    ),
    "Qwen-72B": ModelConfig(
        name="Qwen-72B",
        num_layers=80,
        hidden_size=8192,
        num_attention_heads=64,
        num_query_groups=8,
        ffn_hidden_size=24576,
        vocab_size=152064,
        max_position_embeddings=32768,
    ),
    "Mistral-7B": ModelConfig(
        name="Mistral-7B",
        num_layers=32,
        hidden_size=4096,
        num_attention_heads=32,
        num_query_groups=8,
        ffn_hidden_size=14336,
        vocab_size=32000,
    ),
    "DeepSeek-LLaMA-8B": ModelConfig(
        name="DeepSeek-LLaMA-8B",
        num_layers=32,
        hidden_size=4096,
        num_attention_heads=32,
        num_query_groups=8,
        ffn_hidden_size=14336,
        vocab_size=128256,
        max_position_embeddings=131072,
    ),
}

# GPU configurations
GPU_CONFIGS = {
    "NVIDIA H100 (80GB)": GPUConfig("H100-80GB", 80.0, 3350.0, 1979.0),
    "NVIDIA H100 (94GB)": GPUConfig("H100-94GB", 94.0, 3350.0, 1979.0),
    "NVIDIA A100 (80GB)": GPUConfig("A100-80GB", 80.0, 2039.0, 312.0),
    "NVIDIA A100 (40GB)": GPUConfig("A100-40GB", 40.0, 1555.0, 312.0),
    "NVIDIA H200 (141GB)": GPUConfig("H200-141GB", 141.0, 4800.0, 1979.0),
    "NVIDIA L40S (48GB)": GPUConfig("L40S-48GB", 48.0, 864.0, 362.0),
    "NVIDIA A10 (24GB)": GPUConfig("A10-24GB", 24.0, 600.0, 125.0),
    "Custom": GPUConfig("Custom", 80.0, 2000.0, 300.0),
}


# ==============================================================================
# Memory Calculation Functions
# ==============================================================================

NUM_BYTES_IN_GB = 1024 * 1024 * 1024


def compute_num_parameters(model: ModelConfig) -> int:
    """Compute total number of parameters in the model."""
    h = model.hidden_size
    L = model.num_layers
    V = model.vocab_size
    ffn = model.ffn_hidden_size
    n_heads = model.num_attention_heads
    n_kv_heads = model.num_query_groups
    head_dim = h // n_heads
    
    embedding_params = V * h
    q_params = h * h
    k_params = h * (n_kv_heads * head_dim)
    v_params = h * (n_kv_heads * head_dim)
    o_params = h * h
    attn_params = q_params + k_params + v_params + o_params
    
    if model.swiglu:
        mlp_params = 3 * h * ffn
    else:
        mlp_params = 2 * h * ffn
    
    ln_params = 4 * h
    params_per_layer = attn_params + mlp_params + ln_params
    transformer_params = L * params_per_layer
    final_ln_params = 2 * h
    output_params = V * h if model.untie_embeddings_and_output_weights else 0
    
    return embedding_params + transformer_params + final_ln_params + output_params


def compute_memory_estimate(
    model: ModelConfig,
    tp: int, pp: int, cp: int, dp: int,
    mbs: int, seq_len: int,
    num_microbatches: int = 1,
    use_distributed_optimizer: bool = True,
    recompute_granularity: str = "selective",
    sequence_parallel: bool = True,
) -> Dict[str, float]:
    """Compute memory estimate for a configuration."""
    
    total_params = compute_num_parameters(model)
    params_per_gpu = total_params / (tp * pp)
    
    # Weight memory (bf16 = 2 bytes)
    weight_memory = params_per_gpu * 2
    
    # Optimizer memory
    if use_distributed_optimizer:
        optimizer_bytes_per_param = 12 / dp
    else:
        optimizer_bytes_per_param = 12
    
    grad_bytes_per_param = 2
    optimizer_memory = params_per_gpu * optimizer_bytes_per_param
    weight_and_opt = params_per_gpu * (2 + optimizer_bytes_per_param + grad_bytes_per_param)
    
    # Activation memory
    s = seq_len
    b = mbs
    h = model.hidden_size
    L = model.num_layers // pp
    ffn = model.ffn_hidden_size
    
    if recompute_granularity == "selective" and sequence_parallel:
        per_layer_activation = (s * b * h) * (18 + 4 * (ffn / h))
    elif recompute_granularity == "full":
        per_layer_activation = s * b * h * 2
    else:
        per_layer_activation = (s * b * h) * (34 + 8 * (ffn / h))
    
    total_activation = per_layer_activation * L
    embedding_activation = 8 * s * b * pp
    embedding_dropout = s * b * h * pp
    total_activation += embedding_activation + embedding_dropout
    
    if pp > 1:
        scale = min(1, num_microbatches / pp)
        total_activation *= scale
    
    if pp == 1:
        output_activation = s * b * h * 4 * (1 + model.vocab_size / h)
        total_activation += output_activation
    
    if sequence_parallel:
        total_activation /= tp
    
    # Convert to GB
    weight_gb = weight_memory / NUM_BYTES_IN_GB
    optimizer_gb = optimizer_memory / NUM_BYTES_IN_GB
    weight_and_opt_gb = weight_and_opt / NUM_BYTES_IN_GB
    activation_gb = total_activation / NUM_BYTES_IN_GB
    total_gb = weight_and_opt_gb + activation_gb
    
    return {
        "weight_gb": weight_gb,
        "optimizer_gb": optimizer_gb,
        "weight_and_opt_gb": weight_and_opt_gb,
        "activation_gb": activation_gb,
        "total_gb": total_gb,
        "params_per_gpu_b": params_per_gpu / 1e9,
        "total_params_b": total_params / 1e9,
    }


def explore_configurations(
    model: ModelConfig,
    gpu: GPUConfig,
    num_gpus: int,
    tp_options: List[int],
    pp_options: List[int],
    cp_options: List[int],
    seq_len_options: List[int],
    mbs_options: List[int],
    use_distributed_optimizer: bool = True,
    recompute_granularity: str = "selective",
    sequence_parallel: bool = True,
    memory_buffer_gb: float = 2.0,
) -> pd.DataFrame:
    """Explore all valid configurations that fit in GPU memory."""
    
    results = []
    max_memory = gpu.memory_gb - memory_buffer_gb
    
    for tp, pp, cp in itertools.product(tp_options, pp_options, cp_options):
        # Check if parallelism is valid
        if tp * pp * cp > num_gpus:
            continue
        if num_gpus % (tp * pp * cp) != 0:
            continue
        
        # TP must divide num_attention_heads
        if model.num_attention_heads % tp != 0:
            continue
        
        # PP must divide num_layers
        if model.num_layers % pp != 0:
            continue
        
        dp = num_gpus // (tp * pp * cp)
        
        for seq_len, mbs in itertools.product(seq_len_options, mbs_options):
            # Skip if seq_len exceeds model max
            if seq_len > model.max_position_embeddings:
                continue
            
            estimate = compute_memory_estimate(
                model=model,
                tp=tp, pp=pp, cp=cp, dp=dp,
                mbs=mbs, seq_len=seq_len,
                num_microbatches=max(1, dp),  # Approximate
                use_distributed_optimizer=use_distributed_optimizer,
                recompute_granularity=recompute_granularity,
                sequence_parallel=sequence_parallel,
            )
            
            # Check if it fits
            if estimate["total_gb"] <= max_memory:
                results.append({
                    "TP": tp,
                    "PP": pp,
                    "CP": cp,
                    "DP": dp,
                    "Seq Len": seq_len,
                    "MBS": mbs,
                    "Weight (GB)": round(estimate["weight_gb"], 2),
                    "Optimizer (GB)": round(estimate["optimizer_gb"], 2),
                    "Activation (GB)": round(estimate["activation_gb"], 2),
                    "Total (GB)": round(estimate["total_gb"], 2),
                    "Headroom (GB)": round(max_memory - estimate["total_gb"], 2),
                    "Params/GPU (B)": round(estimate["params_per_gpu_b"], 3),
                    "Memory %": round(100 * estimate["total_gb"] / gpu.memory_gb, 1),
                    "Tokens/Step": seq_len * mbs * dp,
                })
    
    if not results:
        return pd.DataFrame()
    
    df = pd.DataFrame(results)
    # Sort by tokens per step (throughput proxy), then by memory usage
    df = df.sort_values(
        by=["Tokens/Step", "Total (GB)"],
        ascending=[False, True]
    )
    return df


# ==============================================================================
# Streamlit App
# ==============================================================================

def setup_page():
    """Configure the Streamlit page."""
    st.set_page_config(
        page_title="Memory Frontier Explorer",
        page_icon="🧠",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    
    # Custom CSS for a distinctive look
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600&family=Space+Grotesk:wght@400;600;700&display=swap');
    
    :root {
        --bg-primary: #0a0a0f;
        --bg-secondary: #12121a;
        --accent-cyan: #00d4ff;
        --accent-magenta: #ff00aa;
        --accent-yellow: #ffd000;
        --text-primary: #e8e8f0;
        --text-secondary: #888899;
    }
    
    .stApp {
        background: linear-gradient(135deg, #0a0a0f 0%, #1a1a2e 50%, #0a0a0f 100%);
    }
    
    .main-title {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 3rem;
        font-weight: 700;
        background: linear-gradient(90deg, #00d4ff, #ff00aa, #ffd000);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    
    .subtitle {
        font-family: 'JetBrains Mono', monospace;
        color: #888899;
        text-align: center;
        font-size: 1rem;
        margin-bottom: 2rem;
    }
    
    .metric-card {
        background: linear-gradient(145deg, #12121a, #1a1a2e);
        border: 1px solid #2a2a4a;
        border-radius: 12px;
        padding: 1.5rem;
        box-shadow: 0 4px 20px rgba(0, 212, 255, 0.1);
    }
    
    .stDataFrame {
        font-family: 'JetBrains Mono', monospace;
    }
    
    div[data-testid="stMetricValue"] {
        font-family: 'JetBrains Mono', monospace;
        font-size: 1.8rem;
        color: #00d4ff;
    }
    
    div[data-testid="stMetricLabel"] {
        font-family: 'Space Grotesk', sans-serif;
        color: #888899;
    }
    
    .stSelectbox label, .stMultiSelect label, .stSlider label, .stNumberInput label {
        font-family: 'Space Grotesk', sans-serif;
        color: #e8e8f0;
    }
    
    .highlight-row {
        background: linear-gradient(90deg, rgba(0,212,255,0.1), rgba(255,0,170,0.1));
    }
    
    div.stButton > button {
        background: linear-gradient(90deg, #00d4ff, #ff00aa);
        color: white;
        font-family: 'Space Grotesk', sans-serif;
        font-weight: 600;
        border: none;
        padding: 0.5rem 2rem;
        border-radius: 8px;
    }
    
    div.stButton > button:hover {
        background: linear-gradient(90deg, #ff00aa, #ffd000);
        color: white;
    }
    
    .sidebar .stSelectbox, .sidebar .stMultiSelect {
        background: #12121a;
    }
    
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #12121a 0%, #0a0a0f 100%);
        border-right: 1px solid #2a2a4a;
    }
    
    .info-box {
        background: rgba(0, 212, 255, 0.1);
        border-left: 3px solid #00d4ff;
        padding: 1rem;
        border-radius: 0 8px 8px 0;
        margin: 1rem 0;
    }
    
    .warning-box {
        background: rgba(255, 208, 0, 0.1);
        border-left: 3px solid #ffd000;
        padding: 1rem;
        border-radius: 0 8px 8px 0;
        margin: 1rem 0;
    }
    </style>
    """, unsafe_allow_html=True)


def render_sidebar():
    """Render the sidebar configuration panel."""
    st.sidebar.markdown("## 🎛️ Configuration")
    
    # Model selection
    st.sidebar.markdown("### Model")
    model_name = st.sidebar.selectbox(
        "Select Model",
        options=list(MODEL_CONFIGS.keys()),
        index=list(MODEL_CONFIGS.keys()).index("LLaMA-8B"),
    )
    model = MODEL_CONFIGS[model_name]
    
    # Show model info
    model_params = compute_num_parameters(model) / 1e9
    st.sidebar.markdown(f"""
    <div style='background: rgba(0,212,255,0.1); padding: 0.5rem; border-radius: 8px; font-family: JetBrains Mono; font-size: 0.85rem;'>
    📊 <b>{model_params:.2f}B</b> params | {model.num_layers} layers | {model.hidden_size} hidden
    </div>
    """, unsafe_allow_html=True)
    
    st.sidebar.markdown("---")
    
    # GPU selection
    st.sidebar.markdown("### GPU")
    gpu_name = st.sidebar.selectbox(
        "Select GPU Type",
        options=list(GPU_CONFIGS.keys()),
        index=0,
    )
    gpu = GPU_CONFIGS[gpu_name]
    
    if gpu_name == "Custom":
        custom_memory = st.sidebar.number_input(
            "Custom GPU Memory (GB)",
            min_value=8.0,
            max_value=256.0,
            value=80.0,
            step=8.0,
        )
        gpu = GPUConfig("Custom", custom_memory, 2000.0, 300.0)
    
    st.sidebar.markdown(f"""
    <div style='background: rgba(255,0,170,0.1); padding: 0.5rem; border-radius: 8px; font-family: JetBrains Mono; font-size: 0.85rem;'>
    💾 <b>{gpu.memory_gb:.0f} GB</b> memory | {gpu.tflops_bf16:.0f} TF BF16
    </div>
    """, unsafe_allow_html=True)
    
    num_gpus = st.sidebar.number_input(
        "Number of GPUs",
        min_value=1,
        max_value=2048,
        value=8,
        step=1,
    )
    
    memory_buffer = st.sidebar.slider(
        "Memory Buffer (GB)",
        min_value=0.0,
        max_value=10.0,
        value=2.0,
        step=0.5,
        help="Reserved memory for CUDA overhead, framework buffers, etc.",
    )
    
    st.sidebar.markdown("---")
    
    # Parallelism options
    st.sidebar.markdown("### Search Space")
    
    # Generate valid TP options (must divide attention heads)
    valid_tp = [t for t in [1, 2, 4, 8, 16] if model.num_attention_heads % t == 0 and t <= num_gpus]
    tp_options = st.sidebar.multiselect(
        "Tensor Parallel (TP)",
        options=valid_tp,
        default=valid_tp[:3] if len(valid_tp) >= 3 else valid_tp,
    )
    
    # Generate valid PP options (must divide layers)
    valid_pp = [p for p in [1, 2, 4, 8, 16, 32] if model.num_layers % p == 0 and p <= num_gpus]
    pp_options = st.sidebar.multiselect(
        "Pipeline Parallel (PP)",
        options=valid_pp,
        default=[p for p in valid_pp if p <= 4],
    )
    
    cp_options = st.sidebar.multiselect(
        "Context Parallel (CP)",
        options=[1, 2, 4, 8],
        default=[1],
    )
    
    st.sidebar.markdown("---")
    
    # Training config
    st.sidebar.markdown("### Training Config")
    
    max_seq = min(model.max_position_embeddings, 131072)
    seq_options = st.sidebar.multiselect(
        "Sequence Lengths",
        options=[512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072],
        default=[s for s in [2048, 4096, 8192] if s <= max_seq],
    )
    
    mbs_options = st.sidebar.multiselect(
        "Micro Batch Sizes",
        options=[1, 2, 4, 8, 16],
        default=[1, 2],
    )
    
    st.sidebar.markdown("---")
    
    # Advanced options
    with st.sidebar.expander("⚙️ Advanced Options"):
        use_distributed_opt = st.checkbox("Distributed Optimizer", value=True)
        sequence_parallel = st.checkbox("Sequence Parallelism", value=True)
        recompute = st.selectbox(
            "Activation Recomputation",
            options=["selective", "full", "none"],
            index=0,
        )
    
    return {
        "model": model,
        "model_name": model_name,
        "gpu": gpu,
        "num_gpus": num_gpus,
        "tp_options": tp_options or [1],
        "pp_options": pp_options or [1],
        "cp_options": cp_options or [1],
        "seq_len_options": seq_options or [4096],
        "mbs_options": mbs_options or [1],
        "use_distributed_optimizer": use_distributed_opt if 'use_distributed_opt' in dir() else True,
        "sequence_parallel": sequence_parallel if 'sequence_parallel' in dir() else True,
        "recompute_granularity": recompute if 'recompute' in dir() else "selective",
        "memory_buffer_gb": memory_buffer,
    }


def render_main_content(config: Dict[str, Any]):
    """Render the main content area."""
    
    # Header
    st.markdown('<h1 class="main-title">🧠 Memory Frontier Explorer</h1>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">Find optimal parallelism configurations for your GPU cluster</p>', unsafe_allow_html=True)
    
    # Quick stats
    col1, col2, col3, col4 = st.columns(4)
    
    model_params = compute_num_parameters(config["model"]) / 1e9
    
    with col1:
        st.metric("Model", config["model_name"], f"{model_params:.1f}B params")
    with col2:
        st.metric("GPU", config["gpu"].name, f"{config['gpu'].memory_gb:.0f} GB")
    with col3:
        st.metric("Cluster", f"{config['num_gpus']} GPUs", f"{config['num_gpus'] * config['gpu'].memory_gb:.0f} GB total")
    with col4:
        total_configs = (
            len(config["tp_options"]) * 
            len(config["pp_options"]) * 
            len(config["cp_options"]) * 
            len(config["seq_len_options"]) * 
            len(config["mbs_options"])
        )
        st.metric("Search Space", f"{total_configs} configs", "to evaluate")
    
    st.markdown("---")
    
    # Explore button
    if st.button("🔍 Explore Configurations", use_container_width=True):
        with st.spinner("Exploring configuration space..."):
            df = explore_configurations(
                model=config["model"],
                gpu=config["gpu"],
                num_gpus=config["num_gpus"],
                tp_options=config["tp_options"],
                pp_options=config["pp_options"],
                cp_options=config["cp_options"],
                seq_len_options=config["seq_len_options"],
                mbs_options=config["mbs_options"],
                use_distributed_optimizer=config["use_distributed_optimizer"],
                recompute_granularity=config["recompute_granularity"],
                sequence_parallel=config["sequence_parallel"],
                memory_buffer_gb=config["memory_buffer_gb"],
            )
            
            st.session_state["results_df"] = df
            st.session_state["config"] = config
    
    # Display results
    if "results_df" in st.session_state:
        df = st.session_state["results_df"]
        
        if df.empty:
            st.markdown("""
            <div class="warning-box">
            <b>⚠️ No valid configurations found!</b><br>
            Try increasing GPU count, reducing sequence length, or adjusting parallelism options.
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"### ✅ Found {len(df)} Valid Configurations")
            
            # Summary metrics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Best Throughput", f"{df['Tokens/Step'].max():,} tok/step")
            with col2:
                st.metric("Min Memory", f"{df['Total (GB)'].min():.1f} GB")
            with col3:
                st.metric("Max Seq Len", f"{df['Seq Len'].max():,}")
            with col4:
                st.metric("Avg Headroom", f"{df['Headroom (GB)'].mean():.1f} GB")
            
            st.markdown("---")
            
            # Filtering options
            col1, col2, col3 = st.columns(3)
            with col1:
                sort_by = st.selectbox(
                    "Sort by",
                    options=["Tokens/Step", "Total (GB)", "Seq Len", "Headroom (GB)", "Memory %"],
                    index=0,
                )
            with col2:
                sort_order = st.selectbox("Order", ["Descending", "Ascending"])
            with col3:
                top_n = st.slider("Show top N", 5, min(100, len(df)), min(25, len(df)))
            
            # Sort and display
            ascending = sort_order == "Ascending"
            df_display = df.sort_values(by=sort_by, ascending=ascending).head(top_n)
            
            # Highlight function for memory usage
            def highlight_memory(val):
                if val < 60:
                    return 'background-color: rgba(0, 212, 255, 0.2)'
                elif val < 80:
                    return 'background-color: rgba(255, 208, 0, 0.2)'
                else:
                    return 'background-color: rgba(255, 0, 170, 0.2)'
            
            styled_df = df_display.style.applymap(
                highlight_memory, subset=["Memory %"]
            ).format({
                "Weight (GB)": "{:.2f}",
                "Optimizer (GB)": "{:.2f}",
                "Activation (GB)": "{:.2f}",
                "Total (GB)": "{:.2f}",
                "Headroom (GB)": "{:.2f}",
                "Params/GPU (B)": "{:.3f}",
                "Memory %": "{:.1f}%",
                "Tokens/Step": "{:,}",
                "Seq Len": "{:,}",
            })
            
            st.dataframe(styled_df, use_container_width=True, height=600)
            
            # Download button
            csv = df.to_csv(index=False)
            st.download_button(
                label="📥 Download Results as CSV",
                data=csv,
                file_name=f"memory_frontier_{config['model_name']}_{config['num_gpus']}gpus.csv",
                mime="text/csv",
            )
            
            st.markdown("---")
            
            # Visualization
            st.markdown("### 📊 Memory vs Throughput Analysis")
            
            import plotly.express as px
            import plotly.graph_objects as go
            
            # Create scatter plot
            fig = px.scatter(
                df,
                x="Total (GB)",
                y="Tokens/Step",
                color="Seq Len",
                size="MBS",
                hover_data=["TP", "PP", "CP", "DP", "Headroom (GB)"],
                color_continuous_scale="viridis",
                title="Memory Usage vs Throughput (color=Seq Len, size=MBS)",
            )
            
            # Add GPU memory limit line
            fig.add_vline(
                x=config["gpu"].memory_gb - config["memory_buffer_gb"],
                line_dash="dash",
                line_color="red",
                annotation_text=f"GPU Limit ({config['gpu'].memory_gb - config['memory_buffer_gb']:.0f} GB)",
            )
            
            fig.update_layout(
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(20,20,35,0.8)",
                font=dict(family="JetBrains Mono"),
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Parallelism distribution
            col1, col2 = st.columns(2)
            
            with col1:
                fig_tp = px.histogram(
                    df, x="TP", 
                    title="Distribution of TP Values",
                    color_discrete_sequence=["#00d4ff"],
                )
                fig_tp.update_layout(
                    template="plotly_dark",
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(20,20,35,0.8)",
                )
                st.plotly_chart(fig_tp, use_container_width=True)
            
            with col2:
                fig_pp = px.histogram(
                    df, x="PP",
                    title="Distribution of PP Values",
                    color_discrete_sequence=["#ff00aa"],
                )
                fig_pp.update_layout(
                    template="plotly_dark",
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(20,20,35,0.8)",
                )
                st.plotly_chart(fig_pp, use_container_width=True)
    
    else:
        # Initial state - show instructions
        st.markdown("""
        <div class="info-box">
        <b>📌 How to use:</b><br>
        1. Select your model architecture from the sidebar<br>
        2. Choose your GPU type and cluster size<br>
        3. Configure the search space (TP, PP, CP, sequence lengths, batch sizes)<br>
        4. Click <b>Explore Configurations</b> to find valid setups<br>
        <br>
        The tool will evaluate all combinations and show configurations that fit in GPU memory.
        </div>
        """, unsafe_allow_html=True)
        
        # Show example configurations for quick start
        st.markdown("### 🚀 Quick Start Examples")
        
        examples = [
            ("LLaMA-8B on 8×H100", "LLaMA-8B", "H100-80GB", 8, "TP=2, PP=1, seq=8K works great"),
            ("LLaMA-70B on 16×A100", "LLaMA-70B", "A100-80GB", 16, "TP=4, PP=4 or TP=8, PP=2"),
            ("LLaMA-405B on 64×H100", "LLaMA-405B", "H100-80GB", 64, "TP=8, PP=8 for full model"),
        ]
        
        cols = st.columns(3)
        for i, (title, model, gpu, gpus, note) in enumerate(examples):
            with cols[i]:
                st.markdown(f"""
                <div style='background: rgba(0,0,0,0.3); border: 1px solid #2a2a4a; border-radius: 12px; padding: 1rem;'>
                <b style='color: #00d4ff;'>{title}</b><br>
                <span style='color: #888; font-size: 0.9rem;'>{note}</span>
                </div>
                """, unsafe_allow_html=True)


def main():
    setup_page()
    config = render_sidebar()
    render_main_content(config)


if __name__ == "__main__":
    main()

