#!/usr/bin/env python3
"""
Lightweight memory estimation script for distributed training configurations.

This script estimates GPU memory usage without requiring full Megatron initialization.
It loads model configuration once and calculates memory for multiple parallelism configs.

Usage:
    python estimate_memory.py --model llama-8b --configs '[{"tp": 2, "pp": 4, "cp": 1, "mbs": 1, "seq_len": 4096}]'
    
    # Or use the convenience flags:
    python estimate_memory.py --model llama-8b --tp 2 --pp 4 --cp 1 --mbs 1 --seq-len 4096
    
    # Multiple configurations at once:
    python estimate_memory.py --model llama-8b --configs '[
        {"tp": 2, "pp": 4, "cp": 1, "mbs": 1, "seq_len": 4096},
        {"tp": 4, "pp": 2, "cp": 2, "mbs": 2, "seq_len": 8192}
    ]'
"""

# Suppress SyntaxWarnings from apex library (LaTeX docstrings with unescaped backslashes)
import warnings
warnings.filterwarnings("ignore", category=SyntaxWarning, module="apex")

import argparse
import json
import sys
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Union


# ==============================================================================
# Model Configurations
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
    
    
# Pre-defined model configurations
MODEL_CONFIGS = {
    "llama-7b": ModelConfig(
        name="llama-7b",
        num_layers=32,
        hidden_size=4096,
        num_attention_heads=32,
        num_query_groups=32,
        ffn_hidden_size=11008,
        vocab_size=32000,
    ),
    "llama-8b": ModelConfig(
        name="llama-8b",
        num_layers=32,
        hidden_size=4096,
        num_attention_heads=32,
        num_query_groups=8,
        ffn_hidden_size=14336,
        vocab_size=128256,
        max_position_embeddings=8192,
    ),
    "llama-70b": ModelConfig(
        name="llama-70b",
        num_layers=80,
        hidden_size=8192,
        num_attention_heads=64,
        num_query_groups=8,
        ffn_hidden_size=28672,
        vocab_size=128256,
        max_position_embeddings=8192,
    ),
    "llama-405b": ModelConfig(
        name="llama-405b",
        num_layers=126,
        hidden_size=16384,
        num_attention_heads=128,
        num_query_groups=8,
        ffn_hidden_size=53248,
        vocab_size=128256,
        max_position_embeddings=131072,
    ),
    "deepseek-llama-8b": ModelConfig(
        name="deepseek-llama-8b",
        num_layers=32,
        hidden_size=4096,
        num_attention_heads=32,
        num_query_groups=8,
        ffn_hidden_size=14336,
        vocab_size=128256,
        max_position_embeddings=131072,
    ),
    "qwen-72b": ModelConfig(
        name="qwen-72b",
        num_layers=80,
        hidden_size=8192,
        num_attention_heads=64,
        num_query_groups=8,
        ffn_hidden_size=24576,
        vocab_size=152064,
        max_position_embeddings=32768,
    ),
}


@dataclass 
class ParallelConfig:
    """Parallelism and batch configuration."""
    tp: int = 1  # Tensor Parallel
    pp: int = 1  # Pipeline Parallel
    cp: int = 1  # Context Parallel
    dp: int = 1  # Data Parallel (computed from world_size if not specified)
    mbs: int = 1  # Micro Batch Size
    seq_len: int = 4096  # Sequence Length
    num_microbatches: int = 1  # Number of microbatches (for gradient accumulation)
    world_size: Optional[int] = None  # Total GPUs (optional, used to infer DP)
    use_distributed_optimizer: bool = True
    recompute_granularity: str = "selective"  # none, selective, full
    sequence_parallel: bool = True
    virtual_pipeline_model_parallel_size: Optional[int] = None
    
    def __post_init__(self):
        if self.world_size is not None and self.dp == 1:
            # Infer DP from world_size
            self.dp = self.world_size // (self.tp * self.pp * self.cp)


@dataclass
class MemoryEstimate:
    """Memory estimation result."""
    config: ParallelConfig
    weight_memory_gb: float
    optimizer_memory_gb: float  
    weight_and_optimizer_gb: float
    activation_memory_gb: float
    total_memory_gb: float
    num_params_billions: float
    params_per_gpu_billions: float
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "tp": self.config.tp,
            "pp": self.config.pp,
            "cp": self.config.cp,
            "dp": self.config.dp,
            "mbs": self.config.mbs,
            "seq_len": self.config.seq_len,
            "num_microbatches": self.config.num_microbatches,
            "weight_memory_gb": round(self.weight_memory_gb, 2),
            "optimizer_memory_gb": round(self.optimizer_memory_gb, 2),
            "weight_and_optimizer_gb": round(self.weight_and_optimizer_gb, 2),
            "activation_memory_gb": round(self.activation_memory_gb, 2),
            "total_memory_gb": round(self.total_memory_gb, 2),
            "num_params_billions": round(self.num_params_billions, 4),
            "params_per_gpu_billions": round(self.params_per_gpu_billions, 4),
        }


# ==============================================================================
# Memory Calculation Functions (Pure Python, no Megatron dependency)
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
    
    # Embedding parameters
    embedding_params = V * h
    
    # Per-layer parameters
    # Attention: Q, K, V projections and output projection
    # Q: h -> n_heads * head_dim = h
    # K: h -> n_kv_heads * head_dim  
    # V: h -> n_kv_heads * head_dim
    # O: h -> h
    q_params = h * h
    k_params = h * (n_kv_heads * head_dim)
    v_params = h * (n_kv_heads * head_dim)
    o_params = h * h
    attn_params = q_params + k_params + v_params + o_params
    
    # MLP parameters (with SwiGLU: gate + up + down)
    if model.swiglu:
        # SwiGLU has gate_proj, up_proj (both h -> ffn), and down_proj (ffn -> h)
        mlp_params = 3 * h * ffn
    else:
        mlp_params = 2 * h * ffn
    
    # LayerNorm parameters (2 per layer: attention and MLP)
    ln_params = 4 * h  # 2 * (weight + bias-ish, though RMSNorm has no bias)
    
    # Total per layer
    params_per_layer = attn_params + mlp_params + ln_params
    
    # Total transformer parameters
    transformer_params = L * params_per_layer
    
    # Final LayerNorm
    final_ln_params = 2 * h
    
    # Output embedding (if untied)
    output_params = V * h if model.untie_embeddings_and_output_weights else 0
    
    total = embedding_params + transformer_params + final_ln_params + output_params
    return total


def compute_weight_and_optimizer_memory(
    model: ModelConfig,
    config: ParallelConfig,
) -> tuple[float, float, float]:
    """
    Compute weight and optimizer state memory.
    
    Returns: (weight_memory_bytes, optimizer_memory_bytes, total_bytes)
    """
    total_params = compute_num_parameters(model)
    
    # Parameters per GPU after parallelism
    # Simplification: assume even distribution across TP and PP
    params_per_gpu = total_params / (config.tp * config.pp)
    
    # Weight memory (bf16/fp16 = 2 bytes per parameter)
    weight_bytes_per_param = 2
    weight_memory = params_per_gpu * weight_bytes_per_param
    
    # Optimizer state memory
    # With distributed optimizer: optimizer states are sharded across DP
    # Without distributed optimizer: each GPU has full optimizer state
    # Adam: 2 states (momentum + variance) * 4 bytes each = 8 bytes
    # Plus master weights in fp32: 4 bytes
    # Total optimizer: 12 bytes per parameter
    
    if config.use_distributed_optimizer:
        # Optimizer states sharded across DP
        optimizer_bytes_per_param = 12 / config.dp
    else:
        optimizer_bytes_per_param = 12
    
    # Gradients (same precision as weights)
    grad_bytes_per_param = 2
    
    # Total per parameter
    total_bytes_per_param = weight_bytes_per_param + optimizer_bytes_per_param + grad_bytes_per_param
    
    optimizer_memory = params_per_gpu * optimizer_bytes_per_param
    total_memory = params_per_gpu * total_bytes_per_param
    
    return weight_memory, optimizer_memory, total_memory


def compute_activation_memory(
    model: ModelConfig,
    config: ParallelConfig,
) -> float:
    """
    Compute activation memory for training.
    
    Based on formulas from "Reducing Activation Recomputation in Large Transformer Models"
    (https://arxiv.org/pdf/2205.05198.pdf)
    """
    s = config.seq_len
    b = config.mbs
    h = model.hidden_size
    L = model.num_layers // config.pp  # Layers on this pipeline stage
    ffn = model.ffn_hidden_size
    
    # Per-layer activation memory with selective recomputation
    # From the paper, with sequence parallelism and selective recompute:
    # ~= sbh * (18 + 4 * ffn/h) per layer
    if config.recompute_granularity == "selective" and config.sequence_parallel:
        per_layer_activation = (s * b * h) * (18 + 4 * (ffn / h))
    elif config.recompute_granularity == "full":
        # With full recomputation, only need to store layer inputs
        per_layer_activation = s * b * h * 2  # Input activations only
    else:
        # No recomputation - need all intermediate activations
        # Approximate: much larger
        per_layer_activation = (s * b * h) * (34 + 8 * (ffn / h))
    
    total_activation = per_layer_activation * L
    
    # Input embeddings (pipeline stages in flight)
    embedding_activation = 8 * s * b * config.pp
    
    # Dropout in embedding
    embedding_dropout = s * b * h * config.pp
    
    total_activation += embedding_activation + embedding_dropout
    
    # Interleaved pipeline parallelism penalty
    if config.virtual_pipeline_model_parallel_size is not None:
        vpp = config.virtual_pipeline_model_parallel_size
        penalty = 1 + (config.pp - 1) / (config.pp * vpp)
        total_activation *= penalty
    elif config.pp > 1:
        # Non-interleaved: scale by min(num_microbatches, pp) / pp
        scale = min(1, config.num_microbatches / config.pp)
        total_activation *= scale
    
    # Output layer (only on pp=1 or last stage, simplified here)
    if config.pp == 1:
        output_activation = s * b * h * 4 * (1 + model.vocab_size / h)
        total_activation += output_activation
    
    # Divide by TP (sequence parallelism)
    if config.sequence_parallel:
        total_activation /= config.tp
    
    return total_activation


def estimate_memory_for_config(
    model: ModelConfig,
    config: ParallelConfig,
) -> MemoryEstimate:
    """Estimate memory for a single configuration."""
    
    weight_mem, opt_mem, weight_and_opt_mem = compute_weight_and_optimizer_memory(model, config)
    activation_mem = compute_activation_memory(model, config)
    
    total_params = compute_num_parameters(model)
    params_per_gpu = total_params / (config.tp * config.pp)
    
    return MemoryEstimate(
        config=config,
        weight_memory_gb=weight_mem / NUM_BYTES_IN_GB,
        optimizer_memory_gb=opt_mem / NUM_BYTES_IN_GB,
        weight_and_optimizer_gb=weight_and_opt_mem / NUM_BYTES_IN_GB,
        activation_memory_gb=activation_mem / NUM_BYTES_IN_GB,
        total_memory_gb=(weight_and_opt_mem + activation_mem) / NUM_BYTES_IN_GB,
        num_params_billions=total_params / 1e9,
        params_per_gpu_billions=params_per_gpu / 1e9,
    )


def estimate_memory_batch(
    model: Union[str, ModelConfig],
    configs: List[Union[Dict, ParallelConfig]],
) -> List[MemoryEstimate]:
    """
    Estimate memory for multiple configurations.
    
    Args:
        model: Model name (e.g., "llama-8b") or ModelConfig object
        configs: List of ParallelConfig objects or dicts with config params
        
    Returns:
        List of MemoryEstimate objects
    """
    # Resolve model
    if isinstance(model, str):
        if model not in MODEL_CONFIGS:
            available = ", ".join(MODEL_CONFIGS.keys())
            raise ValueError(f"Unknown model: {model}. Available: {available}")
        model = MODEL_CONFIGS[model]
    
    results = []
    for cfg in configs:
        if isinstance(cfg, dict):
            cfg = ParallelConfig(**cfg)
        estimate = estimate_memory_for_config(model, cfg)
        results.append(estimate)
    
    return results


# ==============================================================================
# Display Functions
# ==============================================================================

def print_estimate(estimate: MemoryEstimate, model_name: str = ""):
    """Print a single memory estimate."""
    cfg = estimate.config
    print(f"\n{'='*70}")
    if model_name:
        print(f"Model: {model_name}")
    print(f"Config: TP={cfg.tp}, PP={cfg.pp}, CP={cfg.cp}, DP={cfg.dp}")
    print(f"Batch:  mbs={cfg.mbs}, seq_len={cfg.seq_len}, num_microbatches={cfg.num_microbatches}")
    print(f"{'='*70}")
    print(f"  Total Parameters:     {estimate.num_params_billions:.4f} B")
    print(f"  Params per GPU:       {estimate.params_per_gpu_billions:.4f} B")
    print(f"{'='*70}")
    print(f"  Weight Memory:        {estimate.weight_memory_gb:>8.2f} GB")
    print(f"  Optimizer Memory:     {estimate.optimizer_memory_gb:>8.2f} GB")
    print(f"  Weight + Optimizer:   {estimate.weight_and_optimizer_gb:>8.2f} GB")
    print(f"  Activation Memory:    {estimate.activation_memory_gb:>8.2f} GB")
    print(f"  -----------------------------------------")
    print(f"  TOTAL MEMORY:         {estimate.total_memory_gb:>8.2f} GB")
    print(f"{'='*70}")


def print_estimates_table(estimates: List[MemoryEstimate], model_name: str = ""):
    """Print estimates in a compact table format."""
    if model_name:
        print(f"\nModel: {model_name}")
        if model_name in MODEL_CONFIGS:
            m = MODEL_CONFIGS[model_name]
            print(f"  Layers={m.num_layers}, Hidden={m.hidden_size}, Heads={m.num_attention_heads}")
    
    print(f"\n{'='*100}")
    header = (
        f"{'TP':>3} {'PP':>3} {'CP':>3} {'DP':>3} | "
        f"{'MBS':>4} {'SeqLen':>7} | "
        f"{'Weight':>8} {'Optim':>8} {'Activ':>8} {'TOTAL':>8} | "
        f"{'Params/GPU':>10}"
    )
    print(header)
    print(f"{'='*100}")
    
    for est in estimates:
        cfg = est.config
        row = (
            f"{cfg.tp:>3} {cfg.pp:>3} {cfg.cp:>3} {cfg.dp:>3} | "
            f"{cfg.mbs:>4} {cfg.seq_len:>7} | "
            f"{est.weight_memory_gb:>7.2f}G {est.optimizer_memory_gb:>7.2f}G "
            f"{est.activation_memory_gb:>7.2f}G {est.total_memory_gb:>7.2f}G | "
            f"{est.params_per_gpu_billions:>9.4f}B"
        )
        print(row)
    
    print(f"{'='*100}")


# ==============================================================================
# CLI
# ==============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Estimate GPU memory for distributed training configurations",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single config with flags:
  python estimate_memory.py --model llama-8b --tp 2 --pp 4 --mbs 1 --seq-len 4096
  
  # Multiple configs with JSON:
  python estimate_memory.py --model llama-70b --configs '[
    {"tp": 4, "pp": 4, "cp": 1, "mbs": 1, "seq_len": 4096},
    {"tp": 8, "pp": 2, "cp": 2, "mbs": 2, "seq_len": 8192}
  ]'
  
  # Sweep over configurations:
  python estimate_memory.py --model llama-8b --sweep
  
Available models: """ + ", ".join(MODEL_CONFIGS.keys())
    )
    
    parser.add_argument("--model", type=str, default="llama-8b",
                        help="Model name (e.g., llama-8b, llama-70b)")
    parser.add_argument("--list-models", action="store_true",
                        help="List available model configurations")
    
    # Single config mode
    parser.add_argument("--tp", type=int, default=1, help="Tensor Parallel size")
    parser.add_argument("--pp", type=int, default=1, help="Pipeline Parallel size")
    parser.add_argument("--cp", type=int, default=1, help="Context Parallel size")
    parser.add_argument("--dp", type=int, default=1, help="Data Parallel size")
    parser.add_argument("--mbs", type=int, default=1, help="Micro batch size")
    parser.add_argument("--seq-len", type=int, default=4096, help="Sequence length")
    parser.add_argument("--num-microbatches", type=int, default=1, 
                        help="Number of microbatches for gradient accumulation")
    parser.add_argument("--world-size", type=int, default=None,
                        help="Total number of GPUs (used to infer DP if not specified)")
    
    # Optimizer settings
    parser.add_argument("--no-distributed-optimizer", action="store_true",
                        help="Disable distributed optimizer (default: enabled)")
    parser.add_argument("--recompute", type=str, default="selective",
                        choices=["none", "selective", "full"],
                        help="Activation recomputation granularity")
    parser.add_argument("--no-sequence-parallel", action="store_true",
                        help="Disable sequence parallelism (default: enabled)")
    
    # Multi-config mode
    parser.add_argument("--configs", type=str, default=None,
                        help="JSON list of config dicts (overrides single config flags)")
    parser.add_argument("--configs-file", type=str, default=None,
                        help="Path to JSON file with config list")
    
    # Sweep mode
    parser.add_argument("--sweep", action="store_true",
                        help="Run a sweep over common configurations")
    
    # Output
    parser.add_argument("--json", action="store_true",
                        help="Output results as JSON")
    parser.add_argument("--table", action="store_true",
                        help="Output results as compact table (default for multi-config)")
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # List models mode
    if args.list_models:
        print("Available models:")
        for name, cfg in MODEL_CONFIGS.items():
            print(f"  {name}:")
            print(f"    Layers={cfg.num_layers}, Hidden={cfg.hidden_size}, "
                  f"Heads={cfg.num_attention_heads}, FFN={cfg.ffn_hidden_size}")
            params = compute_num_parameters(cfg) / 1e9
            print(f"    Parameters: {params:.2f}B")
        return
    
    # Validate model
    if args.model not in MODEL_CONFIGS:
        print(f"Error: Unknown model '{args.model}'")
        print(f"Available: {', '.join(MODEL_CONFIGS.keys())}")
        sys.exit(1)
    
    model = MODEL_CONFIGS[args.model]
    
    # Determine configs to run
    configs = []
    
    if args.configs:
        # Parse JSON configs
        configs = json.loads(args.configs)
    elif args.configs_file:
        with open(args.configs_file) as f:
            configs = json.load(f)
    elif args.sweep:
        # Generate sweep configurations
        # Common configurations for the model
        configs = [
            {"tp": 1, "pp": 1, "cp": 1, "dp": 1, "mbs": 1, "seq_len": 4096},
            {"tp": 2, "pp": 1, "cp": 1, "dp": 1, "mbs": 1, "seq_len": 4096},
            {"tp": 2, "pp": 2, "cp": 1, "dp": 1, "mbs": 1, "seq_len": 4096},
            {"tp": 4, "pp": 2, "cp": 1, "dp": 1, "mbs": 1, "seq_len": 4096},
            {"tp": 4, "pp": 4, "cp": 1, "dp": 1, "mbs": 1, "seq_len": 4096},
            {"tp": 8, "pp": 2, "cp": 1, "dp": 1, "mbs": 1, "seq_len": 4096},
            {"tp": 8, "pp": 4, "cp": 2, "dp": 1, "mbs": 1, "seq_len": 8192},
            # Vary sequence length
            {"tp": 4, "pp": 2, "cp": 1, "dp": 1, "mbs": 1, "seq_len": 2048},
            {"tp": 4, "pp": 2, "cp": 1, "dp": 1, "mbs": 1, "seq_len": 8192},
            {"tp": 4, "pp": 2, "cp": 1, "dp": 1, "mbs": 1, "seq_len": 16384},
            # Vary micro batch size
            {"tp": 4, "pp": 2, "cp": 1, "dp": 1, "mbs": 2, "seq_len": 4096},
            {"tp": 4, "pp": 2, "cp": 1, "dp": 1, "mbs": 4, "seq_len": 4096},
        ]
    else:
        # Single config from CLI args
        configs = [{
            "tp": args.tp,
            "pp": args.pp,
            "cp": args.cp,
            "dp": args.dp,
            "mbs": args.mbs,
            "seq_len": args.seq_len,
            "num_microbatches": args.num_microbatches,
            "world_size": args.world_size,
            "use_distributed_optimizer": not args.no_distributed_optimizer,
            "recompute_granularity": args.recompute,
            "sequence_parallel": not args.no_sequence_parallel,
        }]
    
    # Add common settings to all configs if not present
    for cfg in configs:
        if "use_distributed_optimizer" not in cfg:
            cfg["use_distributed_optimizer"] = not args.no_distributed_optimizer
        if "recompute_granularity" not in cfg:
            cfg["recompute_granularity"] = args.recompute
        if "sequence_parallel" not in cfg:
            cfg["sequence_parallel"] = not args.no_sequence_parallel
    
    # Run estimation
    estimates = estimate_memory_batch(model, configs)
    
    # Output results
    if args.json:
        output = {
            "model": args.model,
            "model_params_billions": compute_num_parameters(model) / 1e9,
            "estimates": [est.to_dict() for est in estimates]
        }
        print(json.dumps(output, indent=2))
    elif args.table or len(estimates) > 1:
        print_estimates_table(estimates, args.model)
    else:
        print_estimate(estimates[0], args.model)


if __name__ == "__main__":
    main()

