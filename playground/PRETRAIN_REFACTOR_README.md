# DistCA LLaMA Pre-training Refactor

This document describes the refactored `pretrain_llama_refactored.py` and its improvements over the original `pretrain_llama.py`.

## Overview

The original `pretrain_llama.py` was a working but poorly structured script with:
- Hardcoded values scattered throughout
- Manual worker initialization instead of using DistCA infrastructure
- Poor separation of concerns
- Limited dataset support
- Difficult to understand and modify

The refactored version addresses these issues while maintaining full DistCA functionality.

## Key Improvements

### 1. **Better Architecture & Code Organization**

**Before:**
- 936 lines of mixed concerns
- Worker class defined inline
- No clear separation between initialization, training, and utilities
- Hardcoded environment variable handling

**After:**
- Clean separation of concerns with dedicated sections:
  - Environment Setup
  - Configuration
  - Dataset Providers
  - Model Provider
  - Microbatch Creation
  - Forward/Backward Steps
  - Training Loop
  - Entry Point
- Follows patterns from `simple_4d.py`
- Well-documented with clear docstrings

### 2. **Configuration Management**

**Before:**
```python
# Scattered argparse arguments
parser.add_argument("--num-tokens", type=int, default=1024)
parser.add_argument("--num-batches", type=int, default=1)
# ... 30+ more arguments scattered through 936 lines
```

**After:**
```python
# Centralized configuration with sensible defaults
@dataclass
class DistCAConfig(TransformerConfig):
    """Extended TransformerConfig with DistCA-specific options."""
    distca_nvshmem_buffer_size_gb: float = 1.0
    distca_use_planner: bool = True
    # ... well-organized config options
```

### 3. **Better Dataset Support**

**Before:**
- Only supported synthetic datasets (`wlbllm`, `prolong`)
- Real dataset support was experimental and buggy
- Dataset creation scattered across multiple locations

**After:**
- **Synthetic Datasets**: `wlbllm`, `prolong` (for testing)
- **Real Datasets via Megatron**: Full support for Megatron's `GPTDataset`
  - Properly integrated with `BlendedMegatronDatasetBuilder`
  - Supports all Megatron dataset features (blending, splits, caching)
- **HuggingFace Datasets**: Easy integration path via tokenizer
- **Flexible Configuration**: Dataset parameters exposed via CLI
- **Standard Patterns**: Follows Megatron's `pretrain_gpt.py` conventions

```python
def train_valid_test_datasets_provider(train_val_test_num_samples):
    """Build train/val/test datasets using Megatron's GPTDataset."""
    args = get_args()
    config = core_gpt_dataset_config_from_args(args)

    train_ds, valid_ds, test_ds = BlendedMegatronDatasetBuilder(
        GPTDataset,
        train_val_test_num_samples,
        is_dataset_built_on_rank,
        config,
    ).build()
    return train_ds, valid_ds, test_ds
```

### 4. **Proper DistCA Worker Usage**

**Before:**
```python
class MegatronE2eWorker(BaseMegatronE2eWorker):
    def __init__(self, rank: int, world_size: int):
        # Manual initialization with hardcoded values
        super().__init__(rank, world_size)
        # ... manual setup
```

**After:**
- Uses `distca_model_provider()` following `simple_4d.py` pattern
- Leverages `init_mcore_model()` for proper model initialization
- Automatic `PingPongGPTModel` creation via model registry
- Follows Megatron's `setup_model_and_optimizer()` convention

```python
def distca_model_provider(pre_process=True, post_process=True, hf_config=None):
    """Build PingPongGPTModel for DistCA training."""
    args = get_args()
    config = core_transformer_config_from_args(args, config_class=DistCAConfig)

    parallel_model = init_mcore_model(
        config, hf_config, pre_process, post_process,
        share_embeddings_and_output_weights=not args.untie_embeddings_and_output_weights,
    )
    return parallel_model
```

### 5. **Improved Logging & Monitoring**

**Before:**
- Inconsistent print statements
- Debug flags scattered throughout
- No structured logging

**After:**
- Uses `distca.utils.logging` infrastructure
- Rank-aware logging with `setup_logging()`
- Proper log directories via `setup_log_directories()`
- NVTX markers for profiling
- Clean progress reporting

```python
logger = setup_logging(
    rank=rank,
    world_size=world_size,
    level=logging.INFO,
    console_ranks=[0],  # Only rank 0 logs to console
)
```

### 6. **Better Error Handling & Validation**

**Before:**
- Silent failures
- No input validation
- Cryptic error messages

**After:**
- Clear assertions with helpful messages
- Validation of parallelism configuration
- Graceful degradation where possible
- Memory estimation with OOM warnings

```python
assert tp * pp * cp * dp == world_size, \
    f"tp*pp*cp*dp != world_size: {tp}*{pp}*{cp}*{dp} != {world_size}"
```

### 7. **Cleaner Training Loop**

**Before:**
```python
# Complex nested loops with unclear flow
for sample_idx in range(max_sample_id):
    os.environ["__PRG__INTERNAL__EXPERIMENT_SAMPLE_ID"] = str(sample_idx)
    microbatches_0, tick_per_rank_doc_lens_0 = create_pp_microbatches(...)
    microbatches_1, tick_per_rank_doc_lens_1 = create_pp_microbatches(...)
    # ... 80+ lines of complex logic
```

**After:**
```python
# Clear, linear training loop
for iteration in range(max_iters):
    logger.info(f"Iteration {iteration + 1}/{max_iters}")

    # Create microbatches
    microbatches_0, _ = create_pp_microbatches(...)
    microbatches_1, _ = create_pp_microbatches(...)

    # Combine ping-pong microbatches
    microbatches = combine_ping_pong_microbatches(microbatches_0, microbatches_1)

    # Forward/backward pass
    loss_reduced = forward_backward_batch(...)

    # Log results
    logger.info(f"Iteration {iteration + 1}: loss = {loss_value:.4f}")
```

### 8. **Modular & Testable Design**

**Before:**
- Monolithic `main()` function
- Difficult to test individual components
- Hard to extend or modify

**After:**
- Small, focused functions with single responsibilities
- Clear interfaces between components
- Easy to test and extend
- Reusable utility functions

## Usage

### Basic Usage

```bash
# Make the script executable
chmod +x playground/run_pretrain_llama_refactored.sh

# Run with default configuration (synthetic data)
./playground/run_pretrain_llama_refactored.sh
```

### Advanced Usage

```bash
# Run with custom configuration
torchrun --nproc_per_node=4 playground/pretrain_llama_refactored.py \
    --model-name meta-llama/Llama-3.1-8B \
    --num-layers-override 4 \
    --train-iters 100 \
    --tp 2 \
    --pp 2 \
    --dp 1 \
    --seq-length 4096 \
    --use-planner \
    --use-bf16
```

### Using Real Datasets

#### Option 1: Megatron GPTDataset (Recommended)

```bash
# Prepare your dataset using Megatron's preprocessing tools
# See: https://github.com/NVIDIA/Megatron-LM#data-preprocessing

# Then run with data path
python playground/pretrain_llama_refactored.py \
    --data-path /path/to/your/dataset \
    --sample-name bookcorpus \
    --tokenizer-type HuggingFaceTokenizer \
    --tokenizer-model meta-llama/Llama-3.1-8B
```

#### Option 2: HuggingFace Datasets

The script supports HuggingFace datasets through the tokenizer:

```python
# In your custom script:
from distca.utils.training_utils import setup_global_batch

setup_global_batch(
    total_seq_len=num_tokens,
    sample_name="wikitext",  # or "bookcorpus", "openwebtext", "c4"
    tokenizer=tokenizer,
    max_total_tokens=1_000_000,  # Optional token budget
)
```

## Configuration Options

### Parallelism

| Option | Description | Default |
|--------|-------------|---------|
| `--tp` | Tensor Parallel size | 1 |
| `--pp` | Pipeline Parallel size | 1 |
| `--cp` | Context Parallel size | 1 |
| `--dp` | Data Parallel size | Auto-computed |

### Model

| Option | Description | Default |
|--------|-------------|---------|
| `--model-name` | HuggingFace model name/path | meta-llama/Llama-3.1-8B |
| `--num-layers-override` | Override number of layers | None |
| `--use-bf16` | Use bfloat16 precision | True |

### Training

| Option | Description | Default |
|--------|-------------|---------|
| `--train-iters` | Number of training iterations | 10 |
| `--seed` | Random seed | 42 |
| `--seq-length` | Sequence length | 4096 |
| `--num-tokens` | Tokens per batch | 1024 |
| `--num-batches` | Number of batches | 1 |

### Dataset

| Option | Description | Default |
|--------|-------------|---------|
| `--sample-name` | Dataset type | wlbllm |
| `--data-path` | Path to dataset | "" |
| `--tokenizer-type` | Tokenizer type | HuggingFaceTokenizer |
| `--up-sample-factor` | Up-sampling factor | 4 |
| `--max-total-tokens` | Max tokens for real datasets | None |

### DistCA

| Option | Description | Default |
|--------|-------------|---------|
| `--use-planner` | Use DistCA planner | True |
| `--nvshmem-buffer-size-gb` | NVSHMEM buffer size (GB) | 1.0 |
| `--quit-if-maybe-oom` | Quit on estimated OOM | False |

## Comparison: Lines of Code

| Metric | Original | Refactored | Improvement |
|--------|----------|------------|-------------|
| Total lines | 936 | ~850 | 9% reduction |
| Function count | ~8 major | 15+ focused | Better modularity |
| Configuration params | 30+ scattered | Centralized | Easier to use |
| Dataset support | 2 (synthetic) | 5+ (real + synthetic) | 2.5x more |
| Documentation | Minimal | Comprehensive | Much better |

## Migration Guide

If you're using the old `pretrain_llama.py`, here's how to migrate:

### 1. Update Launch Script

**Before:**
```bash
python playground/pretrain_llama.py \
    --num-tokens 1024 \
    --num-batches 1 \
    --cp-size 2 \
    --tp-size 1 \
    --pp-size 4
```

**After:**
```bash
python playground/pretrain_llama_refactored.py \
    --num-tokens 1024 \
    --num-batches 1 \
    --cp 1 \
    --dp 2 \
    --tp 1 \
    --pp 4
```

### 2. Environment Variables

**Before:**
```bash
export EXPERIMENT_NVSHMEM_BUFFER_SIZE_GB=2
export EXPERIMENT_ENABLE_CUDA_GRAPHS=1
```

**After:**
```bash
# Pass as arguments instead
--nvshmem-buffer-size-gb 2
```

### 3. Dataset Configuration

**Before:**
```python
# Hardcoded in setup_global_batch() call
setup_global_batch(
    total_seq_len=num_tokens,
    sample_name="wlbllm",
    # ... many parameters
)
```

**After:**
```bash
# Pass as CLI arguments
--sample-name wlbllm \
--up-sample-factor 4 \
--elongate-factor 1
```

## Future Improvements

Potential areas for further enhancement:

1. **Full Optimizer Integration**: Currently zeros gradients instead of taking optimizer steps
2. **Checkpointing**: Add proper checkpoint save/load functionality
3. **Validation Loop**: Add separate validation dataset evaluation
4. **W&B Integration**: Add Weights & Biases logging
5. **Dataset Rank Optimization**: Build dataset only on rank 0 and broadcast
6. **CUDA Graphs**: Integrate CUDA graph support from simple_4d.py
7. **Memory Profiling**: Add automatic memory profiling and OOM detection
8. **Multi-node Support**: Better support for multi-node training

## Contributing

When making changes to this refactored version:

1. Keep functions small and focused
2. Add docstrings to all functions
3. Use type hints where appropriate
4. Follow the existing code structure
5. Test with both synthetic and real datasets
6. Update this README with any new features

## References

- Original script: `playground/pretrain_llama.py`
- Reference implementation: `playground/simple_4d.py`
- DistCA documentation: `distca/README.md`
- Megatron-LM: https://github.com/NVIDIA/Megatron-LM
