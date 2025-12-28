"""
Simplified version to illustrate the creation of microbatches.

This module refactors the complex microbatch creation logic from 
test_megatron_e2e_pipeline_with_cp.py into a clean, easy-to-understand format.

REFACTORING SUMMARY:
--------------------
The original code (lines 237-368 and 554-656 in test_megatron_e2e_pipeline_with_cp.py)
had the following issues:
1. Complex nested logic with unclear variable names
2. Mixed concerns (creation + combination + backward metadata assignment)
3. Minimal documentation
4. Hard to understand the overall flow

This refactored version:
1. Separates concerns into clear functions:
   - create_single_microbatch_set(): Creates one set of microbatches
   - combine_microbatches_for_ping_pong(): Combines two sets
   - create_ping_pong_microbatches(): Main entry point
2. Uses a MicrobatchConfig dataclass for all parameters
3. Adds extensive documentation with examples
4. Explains the "why" behind each step
5. Maintains backward compatibility with create_pp_microbatches()

OVERVIEW:
---------
The goal is to create microbatches for pipeline-parallel training with ping-pong scheduling.
The process involves:
1. Creating two sets of microbatches (for ping-pong)
2. Combining them into ping-pong microbatches
3. Each microbatch contains metadata for attention computation and communication

KEY CONCEPTS:
------------
- Pipeline Parallelism (PP): Splits model layers across GPUs
- Tensor Parallelism (TP): Splits tensors within a layer
- Data Parallelism (DP): Replicates model across GPUs
- CP (Contiguous Parallelism): A layout strategy for attention computation
- Ping-Pong: Alternating between two dispatchers to overlap computation and communication

EXAMPLE USAGE:
-------------
    # Configuration
    config = MicrobatchConfig(
        num_microbatch=4,
        pp_degree=2,
        as_rank=0,
        as_world_size=8,
        total_seq_len=2048,
        num_seqs=10,
        max_cp_degree=2,
        hidden_size_q_tp=1024,
        hidden_size_k_tp=1024,
        element_size=2,  # bfloat16
        num_head_in_dtype=32,
        tp_size=2,
        dp_size=4,
        num_token_per_rank=512,
        num_batches=1,
        use_planner=False,
    )
    
    # Create ping-pong microbatches
    microbatches = create_ping_pong_microbatches(config)
"""

import torch
import math
import random
import os
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass

# Import required classes and functions
# NOTE: These imports match the structure used in test_megatron_e2e_pipeline_with_cp.py
# When using this file, ensure you're in the same Python path context (tests/ directory)
from megatron.core.packed_seq_params import PackedSeqParams
from distca.runtime.megatron.packed_seq_params import (
    PingPangSingleStepPackedSeqParams,
    PingPangPackedSeqParams,
)
from distca.runtime.compute_metadata import get_attn_metadata
from distca.planner.planner import cp_list_to_mlp_list
from distca.utils.test_util import create_qkv_dispatch_pipeline_tick


# ============================================================================
# STANDALONE PIPELINE DOCUMENT LENGTH CREATION FUNCTIONS
# ============================================================================
# These functions are moved from distca.utils.test_util to make this file
# self-contained for refactoring and experimentation.

def debug_print(*args, **kwargs):
    """Simple debug print function controlled by environment variable."""
    if os.getenv("D2_DEBUG_PRINT", "0") == "1":
        print(*args, **kwargs)


@dataclass
class MicrobatchConfig:
    """
    Configuration for creating microbatches.
    
    Attributes:
        num_microbatch: Number of microbatches in the pipeline
        pp_degree: Pipeline parallelism degree (number of pipeline stages)
        as_rank: Current rank in the attention shard group
        as_world_size: Total number of ranks in attention shard group
        total_seq_len: Total sequence length (tokens per rank)
        num_seqs: Number of sequences in the batch
        max_cp_degree: Maximum contiguous parallelism degree
        hidden_size_q_tp: Hidden size for query (per TP rank)
        hidden_size_k_tp: Hidden size for key (per TP rank)
        element_size: Size of each element in bytes (e.g., 2 for bfloat16)
        num_head_in_dtype: Number of attention heads (in dtype units)
        tp_size: Tensor parallelism size
        dp_size: Data parallelism size
        num_token_per_rank: Number of tokens per rank
        num_batches: Number of batches per tick (None = use default)
        use_planner: Whether to use the planner for sequence length distribution
    """
    num_microbatch: int
    pp_degree: int
    as_rank: int
    as_world_size: int
    total_seq_len: int
    num_seqs: int
    max_cp_degree: int
    hidden_size_q_tp: int
    hidden_size_k_tp: int
    element_size: int
    num_head_in_dtype: int
    tp_size: int
    dp_size: int
    num_token_per_rank: int
    num_batches: Optional[int] = None
    use_planner: bool = False


def create_list(n: int, s: int, min_val: int, t: int) -> list[int] | None:
    """
    Generates a list of n integers that sum to s.

    Each integer in the list must be:
    - in range of [min_val, s).
    - Divisible by t.

    Returns:
        A list of integers meeting the criteria, or None if no solution exists.
    """
    # --- 1. Input Validation ---
    assert n > 0
    assert s % t == 0

    # --- 2. Determine Valid Range for Numbers ---
    if t > 1:
        min_val = math.ceil((min_val + 1) / t) * t
    assert s >= n * min_val

    # --- 3. Construct the List ---
    # Start with a list where every element is the minimum possible value.
    result = [min_val] * n
    remainder = (s - min_val * n) // t
    remain_result = torch.rand((n,))
    remain_result = (remain_result / remain_result.sum() * remainder).to(torch.int)
    # handle rounding error
    if torch.sum(remain_result).item() != remainder:
        diff = remainder - sum(remain_result)
        if diff > 0:
            remain_result[-1] += diff
        else:
            idx = 0
            while remain_result[idx] <= diff: idx += 1
            remain_result[idx] += diff
            assert remain_result[idx] > 0
    remain_result = (remain_result * t).tolist()
    for rid in range(n):
        result[rid] += remain_result[rid]
    assert sum(result) == s

    return result


def _block_reverse_list(l: list, d: int):
    """
    Blockwise reverse a list:
    return l[-d:0] + l[-2d:-d] + ...
    This is because the backward is the flip of forward in the pp dimension, but
    keep the order in the dp dimension.
    """
    return [item for i in range(len(l), 0, -d) for item in l[max(0, i - d):i]]


class SimpleBatchGenerator:
    """Simple batch generator for testing without planner dependencies."""
    def __init__(self, total_token_on_rank: int, num_docs: int, tp_size: int, seed: int = 42):
        self.total_token_on_rank = total_token_on_rank
        self.num_docs = num_docs
        self.tp_size = tp_size
        self.seed = seed
        random.seed(seed)
        torch.manual_seed(seed)
    
    def get_next_batch(self, num_batches: int) -> list[list[int]]:
        """
        Generate num_batches random batches.
        Each batch is a list of document lengths that sum to total_token_on_rank.
        """
        batches = []
        for _ in range(num_batches):
            num_docs_in_batch = random.randint(1, self.num_docs)
            doc_lens = create_list(
                num_docs_in_batch, 
                self.total_token_on_rank, 
                self.tp_size, 
                1
            )
            batches.append(doc_lens)
        return batches


def create_pipeline_doclens(
    ref_doc_lens: Optional[list[list[int]]],
    add_dummy: bool,
    is_backward: bool,
    world_size: int,
    total_token_on_rank: int,
    num_docs: int,
    tp_size: int,
    dp_size: int,
    num_batches: int = None,
    use_planner: bool = False,
    batch_generator: Optional[SimpleBatchGenerator] = None,
    return_original_doclen: bool = False,
) -> list[list[int]]:
    """
    Create document lengths for a pipeline parallel microbatch (one pp-tick).
    
    This function creates the document lengths needed for one pipeline tick.
    In pipeline parallelism, multiple microbatches flow through different stages
    simultaneously, so we need to track document lengths for all stages.

    For a forward tick, its sequence length follows:
        [new_microbatch, last_tick_seq_len[:PP_stage_-1]]

        The new_microbatch is either generated, or a dummy one. (controlled by add_dummy)
        A special case is that, the first tick does not have a previous one.
        In this way, we make all stages' microbatch dummy, except for the first one.

    For a backward tick, its sequence length is the reverse of a forward tick.
    
    Args:
        ref_doc_lens: Document lengths from previous tick (None for first tick)
        add_dummy: Whether to create dummy microbatches (for drain-out phase)
        is_backward: Whether this is a backward pass
        world_size: Total number of ranks (dp_size * pp_degree)
        total_token_on_rank: Total tokens per rank
        num_docs: Number of documents to generate
        tp_size: Tensor parallelism size
        dp_size: Data parallelism size
        num_batches: Number of batches per tick (None = use dp_size)
        use_planner: Whether to use batch_generator (if provided)
        batch_generator: SimpleBatchGenerator instance for getting batches
        return_original_doclen: Whether to return original doc lengths separately
    
    Returns:
        List of document lengths per rank, or tuple if return_original_doclen=True
    """
    if is_backward:
        return _block_reverse_list(ref_doc_lens, dp_size)
    
    # Create new microbatches for the first PP stage
    if add_dummy:
        # Each rank only gets one dummy document, which is of `tp_size`
        # to avoid Sequence Parallel having an issue.
        pp_head_new_doc_len = [[total_token_on_rank // dp_size] for _ in range(dp_size)]
    else:
        assert total_token_on_rank % tp_size == 0, "Sequence Parallel requires total token divisible by tp_size"
        if use_planner and batch_generator is not None:
            # Use batch generator to get real batches
            debug_print(f"In create_pipeline_doclens, before calling get_next_batch")
            num_batches = num_batches or dp_size 
            pp_head_new_doc_len = batch_generator.get_next_batch(num_batches)
        else:
            # Generate random batches
            pp_head_new_doc_len = [
                create_list(random.randint(1, num_docs), total_token_on_rank, tp_size, 1)
                for _ in range(dp_size)
            ]
    debug_print(f"In create_pipeline_doclens, after calling get_next_batch: pp_head_new_doc_len is: {pp_head_new_doc_len}")

    # Get existing microbatch seqlens from previous stages
    if ref_doc_lens is not None:
        if num_batches == None:
            num_batches = dp_size
        # Not the first microbatch
        if ref_doc_lens[-1] == [tp_size]:
            other_pp_doc_len = ref_doc_lens[:-dp_size]
        else:
            other_pp_doc_len = ref_doc_lens[:-num_batches]
    else:
        # First tick: create dummy lengths for all non-head stages
        dummy_fwd_num = world_size - dp_size
        other_pp_doc_len = [[total_token_on_rank // dp_size] for _ in range(dummy_fwd_num)]
    
    tick_per_rank_doc_len = pp_head_new_doc_len + other_pp_doc_len
    debug_print(f"In create_pipeline_doclens, finally tick_per_rank_doc_len is: {tick_per_rank_doc_len}")

    if return_original_doclen:
        debug_print("In create_pipeline_doclens, return_original_doclen is True, returning (tick_per_rank_doc_len, pp_head_new_doc_len)")
        return (tick_per_rank_doc_len, pp_head_new_doc_len)
    return tick_per_rank_doc_len




def create_single_microbatch_set(
    config: MicrobatchConfig,
    return_seq_lens: bool = False
) -> Tuple[List[Dict], Optional[List]]:
    """
    Create a single set of microbatches for pipeline parallelism.
    
    This function creates microbatches for one "dispatcher" in the ping-pong scheme.
    Each microbatch contains metadata needed for forward and backward passes.
    
    Pipeline Parallelism Overview:
    ------------------------------
    In pipeline parallelism, we need:
    - num_microbatch forward microbatches (the actual work)
    - (pp_degree - 1) dummy microbatches (to drain the pipeline)
    - Total: num_microbatch + pp_degree - 1 microbatches
    
    For example, with num_microbatch=4 and pp_degree=2:
    - Tick 0: Forward MB0 on PP rank 0
    - Tick 1: Forward MB1 on PP rank 0, Forward MB0 on PP rank 1
    - Tick 2: Forward MB2 on PP rank 0, Forward MB1 on PP rank 1
    - Tick 3: Forward MB3 on PP rank 0, Forward MB2 on PP rank 1
    - Tick 4: Dummy on PP rank 0, Forward MB3 on PP rank 1 (drain)
    
    Returns:
        Tuple of (microbatches, original_seq_lens)
        - microbatches: List of microbatch dictionaries, each containing:
            - position_ids: Tensor of position IDs for this rank
            - packed_seq_params: PingPangSingleStepPackedSeqParams with all metadata
        - original_seq_lens: List of original sequence lengths per tick (if return_seq_lens=True)
    """
    # Initialize tracking variables
    tick_per_rank_doc_lens = None  # Document lengths per rank, updated each tick
    bwd_metadata = []  # Store backward metadata for later assignment
    microbatches = []  # The microbatches we're creating
    all_original_seq_lens = []  # Track original sequence lengths
    
    # Total number of pipeline ticks = num_microbatch + pp_degree - 1
    # This includes the drain-out phase
    num_ticks = config.num_microbatch + config.pp_degree - 1
    
    print(f"Creating {num_ticks} microbatches (num_microbatch={config.num_microbatch}, "
          f"pp_degree={config.pp_degree}, drain_ticks={config.pp_degree - 1})")
    
    # Step 1: Create metadata for each pipeline tick
    for tick_idx in range(num_ticks):
        # Determine if this is a drain-out tick (needs dummy forward)
        is_drain_tick = tick_idx >= config.num_microbatch
        
        # Create QKV dispatch metadata for this tick
        # This determines how attention computation is distributed across ranks
        (
            fa_fwd_params,           # Forward attention parameters per rank
            fa_bwd_params,           # Backward attention parameters per rank
            qkv_fwd_fa2a_metadata,   # QKV forward all-to-all communication metadata
            qkv_bwd_fa2a_metadata,   # QKV backward all-to-all communication metadata
            attn_out_fwd_fa2a_metadata,  # Attention output forward all-to-all metadata
            attn_out_qkv_bwd_fa2a_metadata,  # Attention output backward all-to-all metadata
            tick_per_rank_doc_lens,  # Updated document lengths per rank (CP layout)
            original_tick_per_rank_doc_lens,  # Original document lengths (DP layout)
        ) = create_qkv_dispatch_pipeline_tick(
            world_size=config.as_world_size,
            total_num_token=config.total_seq_len,
            num_docs=config.num_seqs,
            max_cp_degree=config.max_cp_degree,
            hidden_size_q=config.hidden_size_q_tp,
            hidden_size_k=config.hidden_size_k_tp,
            element_size=config.element_size,
            softmax_lse_size=config.num_head_in_dtype,
            ref_doc_lens=tick_per_rank_doc_lens,  # Pass previous tick's doc lens
            add_dummy=is_drain_tick,
            tp_size=config.tp_size,
            dp_size=config.dp_size,
            num_token_per_rank=config.num_token_per_rank,
            num_batches=config.num_batches,
            use_planner=config.use_planner,
            return_original_doclen=return_seq_lens,
        )
        
        if return_seq_lens:
            all_original_seq_lens.append(original_tick_per_rank_doc_lens)
        
        # Step 2: Convert CP layout to MLP layout
        # CP (Contiguous Parallelism) layout: Documents may span multiple ranks
        # MLP layout: Each rank has its own token allocation
        # 
        # Example CP -> MLP conversion:
        # CP: [[256, 768], [512, 10, 502]]  (doc spans ranks)
        # MLP: [[256, 128, 128], [256, 256], [512], [10, 502]]  (split across ranks)
        tick_per_rank_doc_lens_mlp = cp_list_to_mlp_list(
            tick_per_rank_doc_lens,
            as_world_size=config.as_world_size,
            num_token_per_rank=config.num_token_per_rank
        )
        
        # Calculate number of tokens for this rank
        this_rank_num_tokens = sum(tick_per_rank_doc_lens_mlp[config.as_rank])
        
        # Step 3: Create backward packed sequence parameters
        # These are used during the backward pass
        bwd_packed_seq_params = PackedSeqParams(
            qkv_format="thd",
            **fa_bwd_params[config.as_rank]
        )
        
        # Step 4: Create MLP packed sequence parameters
        # These are used for RoPE (Rotary Position Embedding) computation
        tensor_doc_lens = torch.tensor(
            tick_per_rank_doc_lens_mlp[config.as_rank],
            dtype=torch.int32
        )
        mlp_packed_seq_params = get_attn_metadata(
            tensor_doc_lens,
            get_packed_seq_params=True
        )
        
        # Step 5: Create forward packed sequence parameters with all metadata
        # This is the main metadata structure for forward pass
        ping_pang_params = PingPangSingleStepPackedSeqParams(
            qkv_format="thd",
            **fa_fwd_params[config.as_rank],  # Forward attention parameters
            qkv_fwd_metadata=qkv_fwd_fa2a_metadata.get_slice(config.as_rank),
            attn_out_fwd_metadata=attn_out_fwd_fa2a_metadata.get_slice(config.as_rank),
            mlp_packed_seq_params=mlp_packed_seq_params,
        )
        
        # Step 6: Create the microbatch dictionary
        position_ids_local = torch.arange(this_rank_num_tokens, dtype=torch.int64)
        microbatch = {
            "position_ids": position_ids_local,
            "packed_seq_params": ping_pang_params,
        }
        microbatches.append(microbatch)
        
        # Store backward metadata for later assignment
        # We'll assign it to the correct microbatch in the next step
        bwd_metadata.append((
            qkv_bwd_fa2a_metadata.get_slice(config.as_rank),
            attn_out_qkv_bwd_fa2a_metadata.get_slice(config.as_rank),
            bwd_packed_seq_params
        ))
    
    # Step 7: Assign backward metadata to the correct microbatches
    # 
    # In pipeline parallelism with 1F1B (1 Forward 1 Backward) scheduling:
    # - Forward of microbatch i happens at forward tick t
    # - Backward of microbatch i happens at backward tick t'
    # 
    # The relationship between forward and backward ticks:
    # - At the end of PP warmup, forward tick is (pp_degree - 1), backward tick is 0
    # - So: t - t' = pp_degree - 1
    # 
    # For microbatch i computed on pp_rank at forward tick t:
    # - It needs (pp_degree - 1 - pp_rank) forward steps to start backward
    # - It needs (pp_degree - 1 - pp_rank) backward steps to compute on this pp_rank
    # - Since 1F1B alternates forward/backward: backward_tick = t' + 2 * (pp_degree - 1 - pp_rank)
    # - Substituting t' = t - (pp_degree - 1): backward_tick = t + pp_degree - 1 - pp_rank * 2
    # 
    # Therefore: bwd_metadata_idx = forward_tick + pp_degree - 1 - pp_rank * 2
    pp_rank = config.as_rank // config.dp_size
    dp_rank = config.as_rank % config.dp_size
    
    for microbatch_idx, microbatch in enumerate(microbatches):
        # Calculate which backward metadata corresponds to this microbatch
        bwd_metadata_idx = (
            microbatch_idx + config.pp_degree - 1 - pp_rank * 2
        ) % len(bwd_metadata)
        
        # Assign backward metadata to this microbatch
        qkv_bwd_metadata, attn_out_bwd_metadata, bwd_packed_seq_params = (
            bwd_metadata[bwd_metadata_idx]
        )
        packed_seq_params = microbatch["packed_seq_params"]
        packed_seq_params.qkv_bwd_metadata = qkv_bwd_metadata
        packed_seq_params.attn_out_bwd_metadata = attn_out_bwd_metadata
        packed_seq_params.bwd_packed_seq_params = bwd_packed_seq_params
    
    if return_seq_lens:
        return microbatches, all_original_seq_lens
    return microbatches


def combine_microbatches_for_ping_pong(
    microbatches_0: List[Dict],
    microbatches_1: List[Dict],
    seed: Optional[int] = None
) -> List[Dict]:
    """
    Combine two sets of microbatches into ping-pong microbatches.
    
    Ping-Pong Overview:
    -------------------
    Ping-pong scheduling uses two "dispatchers" that alternate:
    - Dispatcher 0 processes one set of microbatches
    - Dispatcher 1 processes another set of microbatches
    - While one dispatcher computes, the other can communicate
    - This overlaps computation and communication for better efficiency
    
    This function combines the two sets by:
    1. Pairing corresponding microbatches from each set
    2. Creating a PingPangPackedSeqParams that contains both
    3. Concatenating position_ids and creating input_ids
    4. Setting dispatcher IDs (0 and 1)
    
    Args:
        microbatches_0: First set of microbatches (dispatcher 0)
        microbatches_1: Second set of microbatches (dispatcher 1)
        seed: Random seed for generating input_ids (for reproducibility)
    
    Returns:
        List of combined ping-pong microbatches, each containing:
            - input_ids: Combined token IDs
            - position_ids: Combined position IDs
            - packed_seq_params: PingPangPackedSeqParams with both dispatchers' metadata
    """
    if seed is not None:
        torch.manual_seed(seed)
    
    combined_microbatches = []
    
    for mb_0, mb_1 in zip(microbatches_0, microbatches_1):
        # Extract packed sequence parameters from both microbatches
        mb_0_psp = mb_0["packed_seq_params"]
        mb_1_psp = mb_1["packed_seq_params"]
        
        # Extract MLP layout parameters (used for RoPE)
        mb_0_mlp_psp = mb_0_psp.mlp_packed_seq_params
        mb_1_mlp_psp = mb_1_psp.mlp_packed_seq_params
        
        # Set dispatcher IDs to distinguish them
        mb_0_psp.dispatcher_id = 0
        mb_1_psp.dispatcher_id = 1
        
        # Create combined ping-pong parameters
        # This contains metadata for both dispatchers
        ping_pong_params = PingPangPackedSeqParams(
            seq_params=[mb_0_psp, mb_1_psp],
            mlp_layout_seq_params=[mb_0_mlp_psp, mb_1_mlp_psp],
            max_seqlen_q=max(
                mb_0_mlp_psp.max_seqlen_q,
                mb_1_mlp_psp.max_seqlen_q
            ),
            max_seqlen_kv=max(
                mb_0_mlp_psp.max_seqlen_kv,
                mb_1_mlp_psp.max_seqlen_kv
            ),
        )
        
        # Combine position IDs
        position_ids = torch.concat([
            mb_0["position_ids"],
            mb_1["position_ids"]
        ])
        
        # Create input_ids (token IDs)
        # In real usage, these would come from the data loader
        num_tokens = position_ids.numel()
        input_ids = torch.randint(10, 1000, (num_tokens,))
        
        # Create the combined microbatch
        combined_mb = {
            "input_ids": input_ids,
            "position_ids": position_ids,
            "packed_seq_params": ping_pong_params,
        }
        combined_microbatches.append(combined_mb)
    
    return combined_microbatches


def create_ping_pong_microbatches(
    config: MicrobatchConfig,
    seed: Optional[int] = None,
    return_seq_lens: bool = False
) -> Tuple[List[Dict], Optional[List]]:
    """
    Main function to create ping-pong microbatches.
    
    This is the high-level interface that orchestrates the entire process:
    1. Create first set of microbatches (dispatcher 0)
    2. Create second set of microbatches (dispatcher 1)
    3. Combine them into ping-pong microbatches
    
    Args:
        config: MicrobatchConfig with all parameters
        seed: Random seed for reproducibility
        return_seq_lens: Whether to return original sequence lengths
    
    Returns:
        Tuple of (microbatches, seq_lens)
        - microbatches: List of ping-pong microbatches ready for training
        - seq_lens: List of sequence lengths per tick (if return_seq_lens=True)
    
    Example:
        >>> config = MicrobatchConfig(...)
        >>> microbatches, seq_lens = create_ping_pong_microbatches(config)
        >>> # Use microbatches in forward_backward_batch()
    """
    # Create first set of microbatches (dispatcher 0)
    microbatches_0, seq_lens_0 = create_single_microbatch_set(
        config,
        return_seq_lens=return_seq_lens
    )
    
    # Create second set of microbatches (dispatcher 1)
    microbatches_1, seq_lens_1 = create_single_microbatch_set(
        config,
        return_seq_lens=return_seq_lens
    )
    
    # Combine into ping-pong microbatches
    combined_microbatches = combine_microbatches_for_ping_pong(
        microbatches_0,
        microbatches_1,
        seed=seed
    )
    
    if return_seq_lens:
        seq_lens = [seq_lens_0, seq_lens_1]
        return combined_microbatches, seq_lens
    
    return combined_microbatches


# ============================================================================
# LEGACY COMPATIBILITY WRAPPER
# ============================================================================

def create_pp_microbatches(
    num_microbatch: int,
    pp_degree: int,
    as_rank: int,
    as_world_size: int,
    total_seq_len: int,
    num_seqs: int,
    max_cp_degree: int,
    hidden_size_q_tp: int,
    hidden_size_k_tp: int,
    element_size: int,
    num_head_in_dtype: int,
    tp_size: int,
    dp_size: int,
    num_token_per_rank: int,
    num_batches: int = None,
    use_planner: bool = False,
    return_seq_lens: bool = False
):
    """
    Legacy wrapper function for backward compatibility.
    
    This function maintains the same interface as the original
    create_pp_microbatches() but uses the simplified implementation.
    
    For new code, use create_ping_pong_microbatches() with MicrobatchConfig instead.
    """
    config = MicrobatchConfig(
        num_microbatch=num_microbatch,
        pp_degree=pp_degree,
        as_rank=as_rank,
        as_world_size=as_world_size,
        total_seq_len=total_seq_len,
        num_seqs=num_seqs,
        max_cp_degree=max_cp_degree,
        hidden_size_q_tp=hidden_size_q_tp,
        hidden_size_k_tp=hidden_size_k_tp,
        element_size=element_size,
        num_head_in_dtype=num_head_in_dtype,
        tp_size=tp_size,
        dp_size=dp_size,
        num_token_per_rank=num_token_per_rank,
        num_batches=num_batches,
        use_planner=use_planner,
    )
    
    return create_single_microbatch_set(config, return_seq_lens=return_seq_lens)


# ============================================================================
# STANDALONE EXAMPLES
# ============================================================================

def example_basic_usage():
    """Basic example: Create document lengths for a few pipeline ticks."""
    print("=" * 80)
    print("Example 1: Basic Pipeline Document Length Creation")
    print("=" * 80)
    
    # Configuration
    world_size = 6  # 2 DP ranks * 3 PP stages
    dp_size = 2
    pp_degree = 3
    total_token_on_rank = 1024
    num_docs = 5
    tp_size = 2
    
    print(f"\nConfiguration:")
    print(f"  world_size={world_size}, dp_size={dp_size}, pp_degree={pp_degree}")
    print(f"  total_token_on_rank={total_token_on_rank}, num_docs={num_docs}, tp_size={tp_size}")
    
    # Simulate a few pipeline ticks
    ref_doc_lens = None
    num_ticks = 5
    
    for tick in range(num_ticks):
        is_drain = tick >= 3  # Assume 3 real microbatches
        print(f"\n--- Tick {tick} (drain={is_drain}) ---")
        
        tick_doc_lens = create_pipeline_doclens(
            ref_doc_lens=ref_doc_lens,
            add_dummy=is_drain,
            is_backward=False,
            world_size=world_size,
            total_token_on_rank=total_token_on_rank,
            num_docs=num_docs,
            tp_size=tp_size,
            dp_size=dp_size,
            use_planner=False,
        )
        
        print(f"  Document lengths per rank: {tick_doc_lens}")
        print(f"  Length: {len(tick_doc_lens)} (should be {world_size})")
        
        # Update for next tick
        ref_doc_lens = tick_doc_lens


def example_with_batch_generator():
    """Example using SimpleBatchGenerator for reproducible batches."""
    print("\n" + "=" * 80)
    print("Example 2: Using Batch Generator for Reproducible Batches")
    print("=" * 80)
    
    # Configuration
    world_size = 4  # 2 DP ranks * 2 PP stages
    dp_size = 2
    pp_degree = 2
    total_token_on_rank = 512
    num_docs = 3
    tp_size = 2
    
    # Create batch generator
    batch_gen = SimpleBatchGenerator(
        total_token_on_rank=total_token_on_rank,
        num_docs=num_docs,
        tp_size=tp_size,
        seed=42  # For reproducibility
    )
    
    print(f"\nConfiguration:")
    print(f"  world_size={world_size}, dp_size={dp_size}, pp_degree={pp_degree}")
    print(f"  Using batch generator with seed=42")
    
    # Simulate pipeline ticks
    ref_doc_lens = None
    num_ticks = 4
    
    for tick in range(num_ticks):
        is_drain = tick >= 2  # 2 real microbatches
        print(f"\n--- Tick {tick} (drain={is_drain}) ---")
        
        tick_doc_lens, original_doc_lens = create_pipeline_doclens(
            ref_doc_lens=ref_doc_lens,
            add_dummy=is_drain,
            is_backward=False,
            world_size=world_size,
            total_token_on_rank=total_token_on_rank,
            num_docs=num_docs,
            tp_size=tp_size,
            dp_size=dp_size,
            use_planner=True,
            batch_generator=batch_gen,
            return_original_doclen=True,
        )
        
        print(f"  All ranks doc lens: {tick_doc_lens}")
        print(f"  Original (head stage) doc lens: {original_doc_lens}")
        
        ref_doc_lens = tick_doc_lens


def example_backward_pass():
    """Example showing backward pass document length generation."""
    print("\n" + "=" * 80)
    print("Example 3: Backward Pass Document Lengths")
    print("=" * 80)
    
    # First create forward pass lengths
    world_size = 6
    dp_size = 2
    total_token_on_rank = 1024
    num_docs = 5
    tp_size = 2
    
    print(f"\nForward pass:")
    forward_doc_lens = create_pipeline_doclens(
        ref_doc_lens=None,
        add_dummy=False,
        is_backward=False,
        world_size=world_size,
        total_token_on_rank=total_token_on_rank,
        num_docs=num_docs,
        tp_size=tp_size,
        dp_size=dp_size,
    )
    print(f"  Forward doc lens: {forward_doc_lens}")
    
    # Now create backward pass lengths
    print(f"\nBackward pass (reversed):")
    backward_doc_lens = create_pipeline_doclens(
        ref_doc_lens=forward_doc_lens,
        add_dummy=False,
        is_backward=True,
        world_size=world_size,
        total_token_on_rank=total_token_on_rank,
        num_docs=num_docs,
        tp_size=tp_size,
        dp_size=dp_size,
    )
    print(f"  Backward doc lens: {backward_doc_lens}")
    
    print(f"\nVerification:")
    print(f"  Forward length: {len(forward_doc_lens)}")
    print(f"  Backward length: {len(backward_doc_lens)}")
    print(f"  Should be equal: {len(forward_doc_lens) == len(backward_doc_lens)}")


def example_integration_with_microbatch_creation():
    """Example showing how to use create_pipeline_doclens in microbatch creation."""
    print("\n" + "=" * 80)
    print("Example 4: Integration with Microbatch Creation")
    print("=" * 80)
    
    # This shows how you could modify create_single_microbatch_set to use
    # the standalone create_pipeline_doclens instead of create_qkv_dispatch_pipeline_tick
    
    config = MicrobatchConfig(
        num_microbatch=3,
        pp_degree=2,
        as_rank=0,
        as_world_size=4,
        total_seq_len=1024,
        num_seqs=5,
        max_cp_degree=2,
        hidden_size_q_tp=1024,
        hidden_size_k_tp=1024,
        element_size=2,
        num_head_in_dtype=32,
        tp_size=2,
        dp_size=2,
        num_token_per_rank=512,
        num_batches=1,
        use_planner=False,
    )
    
    print(f"\nConfiguration:")
    print(f"  num_microbatch={config.num_microbatch}, pp_degree={config.pp_degree}")
    print(f"  world_size={config.as_world_size}, dp_size={config.dp_size}")
    
    # Simulate the loop from create_single_microbatch_set
    tick_per_rank_doc_lens = None
    num_ticks = config.num_microbatch + config.pp_degree - 1
    
    print(f"\nSimulating {num_ticks} pipeline ticks:")
    for tick_idx in range(num_ticks):
        is_drain_tick = tick_idx >= config.num_microbatch
        
        # This is what create_pipeline_doclens does internally
        tick_per_rank_doc_lens, original_doc_lens = create_pipeline_doclens(
            ref_doc_lens=tick_per_rank_doc_lens,
            add_dummy=is_drain_tick,
            is_backward=False,
            world_size=config.as_world_size,
            total_token_on_rank=config.total_seq_len,
            num_docs=config.num_seqs,
            tp_size=config.tp_size,
            dp_size=config.dp_size,
            return_original_doclen=True,
        )
        
        print(f"\n  Tick {tick_idx} (drain={is_drain_tick}):")
        print(f"    All ranks: {tick_per_rank_doc_lens}")
        print(f"    Head stage: {original_doc_lens}")


if __name__ == "__main__":
    """Run examples when script is executed directly."""
    print("\n" + "=" * 80)
    print("Pipeline Parallel Microbatch Creation - Standalone Examples")
    print("=" * 80)
    
    # Run all examples
    example_basic_usage()
    example_with_batch_generator()
    example_backward_pass()
    example_integration_with_microbatch_creation()
    
    print("\n" + "=" * 80)
    print("All examples completed!")
    print("=" * 80)
    print("\nTo enable debug prints, set environment variable:")
    print("  export D2_DEBUG_PRINT=1")
    print("=" * 80)
