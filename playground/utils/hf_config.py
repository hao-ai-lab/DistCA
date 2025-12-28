"""
Utility module to convert HuggingFace model configs to Megatron CLI arguments.

This module provides functions to load a HuggingFace model config and generate
the corresponding Megatron command-line arguments for training.

Example usage:
    from utils.hf_config import get_megatron_args_from_hf_model
    
    # Generate args for a HuggingFace model
    model_args = get_megatron_args_from_hf_model(
        model_name_or_path="deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
        seq_length=4096,
        pp=2,  # Override num_layers = pp * layers_per_stage
    )
"""

import os
import logging
from typing import Optional, List, Dict, Any, Union, TYPE_CHECKING
from dataclasses import dataclass, field

if TYPE_CHECKING:
    from transformers import PretrainedConfig

logger = logging.getLogger(__name__)


@dataclass
class MegatronModelArgs:
    """Configuration for Megatron model arguments derived from HuggingFace config."""
    
    # Model architecture
    num_layers: int = 32
    hidden_size: int = 4096
    num_attention_heads: int = 32
    num_query_groups: Optional[int] = None  # For GQA, None means same as num_attention_heads
    ffn_hidden_size: Optional[int] = None  # None means 4 * hidden_size
    seq_length: int = 4096
    max_position_embeddings: int = 4096
    vocab_size: int = 128256
    
    # Normalization
    normalization: str = "RMSNorm"  # LayerNorm or RMSNorm
    layernorm_epsilon: float = 1e-5
    
    # Position embedding
    position_embedding_type: str = "rope"  # learned_absolute, rope, none
    rotary_base: int = 10000
    rotary_percent: float = 1.0
    use_rope_scaling: bool = False
    
    # Activation
    swiglu: bool = True  # Use SwiGLU activation (common in LLaMA-style models)
    
    # Embeddings
    untie_embeddings_and_output_weights: bool = True
    
    # Additional model-specific settings
    attention_dropout: float = 0.0
    hidden_dropout: float = 0.0
    
    def to_cli_args(self) -> List[str]:
        """Convert the configuration to Megatron CLI argument list."""
        args = [
            "--num-layers", str(self.num_layers),
            "--hidden-size", str(self.hidden_size),
            "--num-attention-heads", str(self.num_attention_heads),
            "--seq-length", str(self.seq_length),
            "--max-position-embeddings", str(self.max_position_embeddings),
            "--vocab-size", str(self.vocab_size),
            "--normalization", self.normalization,
            "--position-embedding-type", self.position_embedding_type,
        ]
        
        # FFN hidden size (only add if different from default 4*hidden_size)
        if self.ffn_hidden_size is not None:
            args.extend(["--ffn-hidden-size", str(self.ffn_hidden_size)])
        
        # GQA (Group Query Attention)
        if self.num_query_groups is not None and self.num_query_groups != self.num_attention_heads:
            args.extend([
                "--group-query-attention",
                "--num-query-groups", str(self.num_query_groups),
            ])
        
        # RoPE settings
        if self.position_embedding_type == "rope":
            args.extend(["--rotary-base", str(self.rotary_base)])
            if self.rotary_percent != 1.0:
                args.extend(["--rotary-percent", str(self.rotary_percent)])
            if self.use_rope_scaling:
                args.append("--use-rope-scaling")
        
        # SwiGLU activation
        if self.swiglu:
            args.append("--swiglu")
        
        # Untie embeddings
        if self.untie_embeddings_and_output_weights:
            args.append("--untie-embeddings-and-output-weights")
        
        return args


def get_hf_config(model_name_or_path: str, trust_remote_code: bool = True, cache_dir: Optional[str] = None) -> 'PretrainedConfig':
    """
    Load HuggingFace model config, with fallback to download if not cached.
    
    Args:
        model_name_or_path: HuggingFace model name or local path
        trust_remote_code: Whether to trust remote code
        cache_dir: Optional cache directory for downloaded models
    
    Returns:
        HuggingFace PretrainedConfig object
    """
    from transformers import AutoConfig
    
    try:
        # First try with local_files_only to use cached version
        hf_config = AutoConfig.from_pretrained(
            model_name_or_path, 
            local_files_only=True,
            trust_remote_code=trust_remote_code,
        )
        logger.info(f"Loaded HF config from cache: {model_name_or_path}")
    except Exception as e:
        logger.info(f"Local cache not found for {model_name_or_path}, downloading... Error: {e}")
        # Fallback to downloading with cache_dir specified
        hf_config = AutoConfig.from_pretrained(
            model_name_or_path, 
            cache_dir=cache_dir or "./models/",
            trust_remote_code=trust_remote_code,
        )
    
    return hf_config


def hf_config_to_megatron_args(
    hf_config,
    seq_length: Optional[int] = None,
    num_layers_override: Optional[int] = None,
) -> MegatronModelArgs:
    """
    Convert HuggingFace config to MegatronModelArgs.
    
    Args:
        hf_config: HuggingFace PretrainedConfig object
        seq_length: Override sequence length (default: use model's max_position_embeddings)
        num_layers_override: Override number of layers (useful for testing with fewer layers)
    
    Returns:
        MegatronModelArgs dataclass with model configuration
    """
    # Extract architecture info
    architectures = getattr(hf_config, "architectures", [])
    is_llama = any("Llama" in arch for arch in architectures)
    is_qwen = any("Qwen" in arch for arch in architectures)
    
    # Basic model dimensions
    num_layers = num_layers_override if num_layers_override is not None else hf_config.num_hidden_layers
    hidden_size = hf_config.hidden_size
    num_attention_heads = hf_config.num_attention_heads
    
    # GQA (Group Query Attention) - common in LLaMA 2/3 and similar models
    num_key_value_heads = getattr(hf_config, "num_key_value_heads", num_attention_heads)
    
    # FFN hidden size (intermediate_size in HF config)
    ffn_hidden_size = getattr(hf_config, "intermediate_size", 4 * hidden_size)
    
    # Sequence length
    max_position_embeddings = getattr(hf_config, "max_position_embeddings", 4096)
    if seq_length is None:
        seq_length = max_position_embeddings
    
    # Vocab size
    vocab_size = hf_config.vocab_size
    
    # Normalization
    rms_norm_eps = getattr(hf_config, "rms_norm_eps", 1e-5)
    # LLaMA and Qwen use RMSNorm
    normalization = "RMSNorm" if (is_llama or is_qwen) else "LayerNorm"
    
    # RoPE settings
    rope_theta = getattr(hf_config, "rope_theta", 10000)
    use_rope_scaling = getattr(hf_config, "rope_scaling", None) is not None
    
    # Tied embeddings
    tie_word_embeddings = getattr(hf_config, "tie_word_embeddings", False)
    
    # Dropout
    attention_dropout = getattr(hf_config, "attention_dropout", 0.0)
    hidden_dropout = getattr(hf_config, "hidden_dropout", 0.0)
    
    # LLaMA-style models use SwiGLU
    # Check for hidden_act to determine activation
    hidden_act = getattr(hf_config, "hidden_act", "silu")
    swiglu = hidden_act in ["silu", "swiglu"]
    
    return MegatronModelArgs(
        num_layers=num_layers,
        hidden_size=hidden_size,
        num_attention_heads=num_attention_heads,
        num_query_groups=num_key_value_heads,
        ffn_hidden_size=ffn_hidden_size,
        seq_length=seq_length,
        max_position_embeddings=max_position_embeddings,
        vocab_size=vocab_size,
        normalization=normalization,
        layernorm_epsilon=rms_norm_eps,
        position_embedding_type="rope",
        rotary_base=int(rope_theta),
        use_rope_scaling=use_rope_scaling,
        swiglu=swiglu,
        untie_embeddings_and_output_weights=not tie_word_embeddings,
        attention_dropout=attention_dropout,
        hidden_dropout=hidden_dropout,
    )


def get_megatron_args_from_hf_model(
    model_name_or_path: str,
    seq_length: Optional[int] = None,
    num_layers_override: Optional[int] = None,
    trust_remote_code: bool = True,
    cache_dir: Optional[str] = None,
) -> List[str]:
    """
    Generate Megatron CLI arguments from a HuggingFace model name or path.
    
    This is the main entry point for converting HF models to Megatron args.
    
    Args:
        model_name_or_path: HuggingFace model name (e.g., "meta-llama/Llama-2-7b-hf") 
                           or local path to model
        seq_length: Override sequence length (default: use model's max_position_embeddings)
        num_layers_override: Override number of layers (useful for testing with fewer layers)
        trust_remote_code: Whether to trust remote code for custom models
        cache_dir: Optional cache directory for downloaded models
    
    Returns:
        List of Megatron CLI arguments as strings
    
    Example:
        >>> args = get_megatron_args_from_hf_model(
        ...     "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
        ...     seq_length=2048,
        ...     num_layers_override=8,
        ... )
        >>> print(args)
        ['--num-layers', '8', '--hidden-size', '4096', ...]
    """
    hf_config = get_hf_config(model_name_or_path, trust_remote_code, cache_dir)
    megatron_args = hf_config_to_megatron_args(hf_config, seq_length, num_layers_override)
    return megatron_args.to_cli_args()


def get_megatron_args_dict_from_hf_model(
    model_name_or_path: str,
    seq_length: Optional[int] = None,
    num_layers_override: Optional[int] = None,
    max_position_embeddings_override: Optional[int] = None,
    trust_remote_code: bool = True,
    cache_dir: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Generate Megatron arguments as a dictionary from a HuggingFace model.
    
    Useful when you need to access individual config values.
    
    Args:
        model_name_or_path: HuggingFace model name or local path
        seq_length: Training sequence length (can be shorter than max_position_embeddings).
                    For RoPE models, you can train with seq_length < max_position_embeddings
                    for efficiency while keeping the model capable of longer sequences.
        num_layers_override: Override number of layers (useful for testing with fewer layers)
        max_position_embeddings_override: Override max_position_embeddings separately.
                                         If None and seq_length is provided, defaults to seq_length.
                                         If None and seq_length is None, uses model's original value.
        trust_remote_code: Whether to trust remote code
        cache_dir: Optional cache directory
    
    Returns:
        Dictionary with Megatron configuration values
    
    Note on seq_length vs max_position_embeddings:
        - seq_length: Actual sequence length used during training (affects memory, speed)
        - max_position_embeddings: Maximum sequence length the model can handle (architectural limit)
        - For RoPE models, you can train with seq_length < max_position_embeddings for efficiency
        - For learned absolute embeddings, max_position_embeddings determines embedding table size
    """
    hf_config = get_hf_config(
        model_name_or_path=model_name_or_path,
        trust_remote_code=trust_remote_code,
        cache_dir=cache_dir,
    )
    # Override hf_config attributes for specific logics.
    # This ensures the hf_config object itself reflects the overrides,
    # which is useful when the hf_config is accessed later.
    # Only override if values are provided (not None)
    if num_layers_override is not None:
        hf_config.num_hidden_layers = num_layers_override
        logger.info(f"Overriding hf_config.num_hidden_layers to {num_layers_override}")
    
    # Handle max_position_embeddings override
    # By default, if seq_length is provided, we match max_position_embeddings to it (backward compatibility).
    # However, you can explicitly set max_position_embeddings_override to keep the original value
    # or set a different value, allowing seq_length < max_position_embeddings.
    # This is useful for training efficiency while keeping the model capable of longer sequences.
    if max_position_embeddings_override is not None:
        hf_config.max_position_embeddings = max_position_embeddings_override
        logger.info(f"Overriding hf_config.max_position_embeddings to {max_position_embeddings_override}")
    elif seq_length is not None:
        # Default: match max_position_embeddings to seq_length for backward compatibility
        hf_config.max_position_embeddings = seq_length
        logger.info(f"Setting hf_config.max_position_embeddings to {seq_length} (matching seq_length)")
        logger.info(f"  To train with seq_length < max_position_embeddings, set max_position_embeddings_override "
                   f"to the desired value (e.g., model's original max_position_embeddings)")

    megatron_args = hf_config_to_megatron_args(
        hf_config=hf_config,
        seq_length=seq_length,
        num_layers_override=num_layers_override,
    )
    
    model_config_dict = {
        "num_layers": megatron_args.num_layers,
        "hidden_size": megatron_args.hidden_size,
        "num_attention_heads": megatron_args.num_attention_heads,
        "num_query_groups": megatron_args.num_query_groups,
        "ffn_hidden_size": megatron_args.ffn_hidden_size,
        "seq_length": megatron_args.seq_length,
        "max_position_embeddings": megatron_args.max_position_embeddings,
        "vocab_size": megatron_args.vocab_size,
        "normalization": megatron_args.normalization,
        "position_embedding_type": megatron_args.position_embedding_type,
        "rotary_base": megatron_args.rotary_base,
        "swiglu": megatron_args.swiglu,
        "untie_embeddings_and_output_weights": megatron_args.untie_embeddings_and_output_weights,
    }
    return dict(
        hf_config=hf_config,
        megatron_args=megatron_args,
        model_config_dict=model_config_dict,
    )


# Pre-defined model configurations for common models
# These can be used when you don't have network access to download configs
KNOWN_MODEL_CONFIGS = {
    "llama-7b": MegatronModelArgs(
        num_layers=32,
        hidden_size=4096,
        num_attention_heads=32,
        num_query_groups=32,
        ffn_hidden_size=11008,
        max_position_embeddings=4096,
        vocab_size=32000,
        rotary_base=10000,
    ),
    "llama-8b": MegatronModelArgs(
        num_layers=32,
        hidden_size=4096,
        num_attention_heads=32,
        num_query_groups=8,  # GQA
        ffn_hidden_size=14336,
        max_position_embeddings=8192,
        vocab_size=128256,
        rotary_base=500000,
    ),
    "llama-70b": MegatronModelArgs(
        num_layers=80,
        hidden_size=8192,
        num_attention_heads=64,
        num_query_groups=8,  # GQA
        ffn_hidden_size=28672,
        max_position_embeddings=8192,
        vocab_size=128256,
        rotary_base=500000,
    ),
    "deepseek-llama-8b": MegatronModelArgs(
        num_layers=32,
        hidden_size=4096,
        num_attention_heads=32,
        num_query_groups=8,  # GQA
        ffn_hidden_size=14336,
        max_position_embeddings=131072,
        vocab_size=128256,
        rotary_base=500000,
    ),
}


def get_known_model_args(
    model_key: str,
    seq_length: Optional[int] = None,
    num_layers_override: Optional[int] = None,
) -> List[str]:
    """
    Get Megatron CLI args for a known model configuration.
    
    Useful when you don't have network access or want to use a predefined config.
    
    Args:
        model_key: Key from KNOWN_MODEL_CONFIGS (e.g., "llama-8b", "deepseek-llama-8b")
        seq_length: Override sequence length
        num_layers_override: Override number of layers
    
    Returns:
        List of Megatron CLI arguments
    """
    if model_key not in KNOWN_MODEL_CONFIGS:
        available = ", ".join(KNOWN_MODEL_CONFIGS.keys())
        raise ValueError(f"Unknown model key: {model_key}. Available: {available}")
    
    # Create a copy to avoid modifying the shared config
    from dataclasses import replace
    config = replace(KNOWN_MODEL_CONFIGS[model_key])
    
    # Apply overrides
    if seq_length is not None:
        config.seq_length = seq_length
    if num_layers_override is not None:
        config.num_layers = num_layers_override
    
    return config.to_cli_args()


def get_known_model_config(
    model_key: str,
    seq_length: Optional[int] = None,
    num_layers_override: Optional[int] = None,
) -> MegatronModelArgs:
    """
    Get a copy of a known model configuration.
    
    Args:
        model_key: Key from KNOWN_MODEL_CONFIGS (e.g., "llama-8b", "deepseek-llama-8b")
        seq_length: Override sequence length
        num_layers_override: Override number of layers
    
    Returns:
        MegatronModelArgs dataclass (copy of the known config with overrides applied)
    """
    if model_key not in KNOWN_MODEL_CONFIGS:
        available = ", ".join(KNOWN_MODEL_CONFIGS.keys())
        raise ValueError(f"Unknown model key: {model_key}. Available: {available}")
    
    # Create a copy to avoid modifying the shared config
    from dataclasses import replace
    config = replace(KNOWN_MODEL_CONFIGS[model_key])
    
    # Apply overrides
    if seq_length is not None:
        config.seq_length = seq_length
    if num_layers_override is not None:
        config.num_layers = num_layers_override
    
    return config


if __name__ == "__main__":
    # Test the module
    import sys
    
    if len(sys.argv) > 1:
        model_name = sys.argv[1]
    else:
        model_name = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
    
    print(f"Generating Megatron args for: {model_name}")
    print("-" * 60)
    
    try:
        args = get_megatron_args_from_hf_model(
            model_name,
            seq_length=2048,
            num_layers_override=8,
        )
        print("CLI args:")
        print(" ".join(args))
        print()
        
        args_dict = get_megatron_args_dict_from_hf_model(
            model_name,
            seq_length=2048,
            num_layers_override=8,
        )
        print("Config dict:")
        for k, v in args_dict.items():
            print(f"  {k}: {v}")
    except Exception as e:
        print(f"Error loading HF model: {e}")
        print("\nUsing known config for llama-8b instead:")
        args = get_known_model_args("llama-8b", seq_length=2048, num_layers_override=8)
        print(" ".join(args))

