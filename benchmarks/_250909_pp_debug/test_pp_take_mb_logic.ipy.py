import sys
import torch
import test_util

from global_batch_provider import setup_global_batch
from d2.runtime.megatron_patch.packed_seq_params import PingPangSingleStepPackedSeqParams



from dataclasses import dataclass
@dataclass
class PackedSeqParams:
    '''
    parameters to TEDotProductAttention and fused rope kernels for the
    `thd` (packed) sequence format
    '''

    qkv_format: str = None
    cu_seqlens_q: 'Tensor' = None
    cu_seqlens_kv: 'Tensor' = None
    cu_seqlens_q_padded: 'Tensor' = None
    cu_seqlens_kv_padded: 'Tensor' = None
    max_seqlen_q: 'Tensor' = None
    max_seqlen_kv: 'Tensor' = None

import importlib
importlib.reload(test_util)

def create_pp_microbatches(
    num_microbatch: int, pp_degree: int, as_rank: int,
    as_world_size: int, total_seq_len: int, num_seqs: int,
    max_cp_degree: int, hidden_size_q_tp: int,
    hidden_size_k_tp: int, element_size: int,
    num_head_in_dtype: int, tp_size: int, dp_size: int,
    num_token_per_rank: int,
    num_batches: int = None,
    use_planner: bool=False,
    return_seq_lens: bool=False
):

    tick_per_rank_doc_lens = None
    bwd_metadata = []
    microbatches = []
    if use_planner:
        print("Enable planner. Get real batch.")
    else:
        print("No planner. Use random batch.")


    all_original_seq_lens = []
    for i in range(num_microbatch + pp_degree - 1):
        print(f"⚪ =========== [Prepare to get microbatch = {i}] ======== ")
        # For the last few ticks (drain-out ticks)
        # add a dummy forward microbatch at PP rank 0.
        add_dummy_forward = i >= num_microbatch
        print(f"🟡 tick_per_rank_doc_lens: {tick_per_rank_doc_lens}")
        (
            fa_fwd_params, fa_bwd_params,
            qkv_fwd_fa2a_metadata, qkv_bwd_fa2a_metadata,
            attn_out_fwd_fa2a_metadata, attn_out_qkv_bwd_fa2a_metadata,
            tick_per_rank_doc_lens, original_tick_per_rank_doc_lens,
        ) = test_util.create_qkv_dispatch_pipeline_tick(
            as_world_size, total_seq_len, num_seqs, max_cp_degree,
            hidden_size_q_tp, hidden_size_k_tp, element_size, num_head_in_dtype,
            ref_doc_lens=tick_per_rank_doc_lens,
            add_dummy=add_dummy_forward,
            tp_size=tp_size,
            dp_size=dp_size,
            num_token_per_rank=num_token_per_rank,
            num_batches=num_batches,
            use_planner=use_planner,
            return_original_doclen=return_seq_lens,
        )
        print(f"🟡 fa_fwd_params: {fa_fwd_params}")
        all_original_seq_lens.append(original_tick_per_rank_doc_lens)
        
        # For MLP-CP, we need to transfer List[List[int]] from CP layout back to DP, so each rank knows its number of tokens.
        #   Example1 DP case:
        # tick_per_rank_doc_lens cp list: List[List[int]] = [[8], [8], [8], [8], [256, 256],[128, 384],[512], [10, 502] ]
        # tick_per_rank_doc_lens mlp list : [[8], [8], [8], [8], [256, 256],[128, 384],[512], [10, 502] ]
        #   Example2 CP case:
        # tick_per_rank_doc_lens cp list: List[List[int]] = [[8], [8], [8], [8], [256, 768],[512, 10, 502] ]
        # tick_per_rank_doc_lens mlp list: [[8], [8], [8], [8], [256, 128, 128], [256, 256], [512], [10, 502]]

        tick_per_rank_doc_lens_after_cp_transfer = test_util.cp_list_to_mlp_list(tick_per_rank_doc_lens, as_world_size, num_token_per_rank)
        print(f"🟡 tick_per_rank_doc_lens_after_cp_transfer: {tick_per_rank_doc_lens_after_cp_transfer}")
        
        this_rank_num_tokens = sum(tick_per_rank_doc_lens_after_cp_transfer[as_rank])
        print(f"🟡 this_rank_num_tokens: {this_rank_num_tokens}")

        bwd_packed_seq_params = PackedSeqParams(
            qkv_format="thd", **fa_bwd_params[as_rank]
        )
        tensor_doc_lens = torch.tensor(tick_per_rank_doc_lens_after_cp_transfer[as_rank], dtype=torch.int32)
        mlp_packed_seq_params = test_util.get_attn_metadata(tensor_doc_lens, get_packed_seq_params=True)

        # Create packed_params. Note that we do not add backward params here.
        ping_pang_params = PingPangSingleStepPackedSeqParams(
            qkv_format="thd",
            **fa_fwd_params[as_rank],
            qkv_fwd_metadata=qkv_fwd_fa2a_metadata.get_slice(as_rank),
            attn_out_fwd_metadata=attn_out_fwd_fa2a_metadata.get_slice(as_rank),
            mlp_packed_seq_params=mlp_packed_seq_params,
        )
        print(f"🟡 [bid = {i}] ping_pang_params", ping_pang_params)

        # NOTE: we init input_ids at the end after creating dispatching strategy
        # and seq lens of each iteration. This is to ensure that each
        # rank has the same randomly initialized strategy.
        position_ids_local = torch.arange(this_rank_num_tokens)
        microbatch = {
            "position_ids": position_ids_local,
            "packed_seq_params": ping_pang_params,
        }
        microbatches.append(microbatch)

        # store the corresponding bwd metadata (for later ticks)
        bwd_metadata.append(
            (qkv_bwd_fa2a_metadata.get_slice(as_rank), attn_out_qkv_bwd_fa2a_metadata.get_slice(as_rank), bwd_packed_seq_params)
        )

    pp_rank = as_rank // dp_size
    dp_rank = as_rank % dp_size

    # put bwd metadata to the corresponding side
    for i, microbatch in enumerate(microbatches):
        # When mb_i is computed on pp_rank at forward tick t, assume the backward right after this forward is at tick t'.
        # mb_i requires another (pp_degree - 1 - pp_rank) forward steps to start mb_i backward,
        # and another (pp_degree - 1 - pp_rank) backward steps to compute mb_i backward on this pp_rank.
        # Since in 1F1B, every forward follows a backward, so backward tick
        # t' + 2 * (pp_degree - 1 - pp_rank) is the one to compute mb_i on this pp_rank.
        # Note that at the end of PP warmup, forward tick is pp_degree - 1, while backward tick
        # is 0. Therefore, t - t' = pp_degree - 1, and thus
        # t' + 2 * (pp_degree - 1 - pp_rank) == t + pp_degree - 1 - pp_rank * 2
        bwd_metadata_idx = (i + pp_degree - 1 - pp_rank * 2) % len(bwd_metadata)
        qkv_bwd_metadata, attn_out_bwd_metadata, bwd_packed_seq_params = bwd_metadata[bwd_metadata_idx]
        packed_seq_params = microbatch["packed_seq_params"]
        packed_seq_params.qkv_bwd_metadata = qkv_bwd_metadata
        packed_seq_params.attn_out_bwd_metadata = attn_out_bwd_metadata
        packed_seq_params.bwd_packed_seq_params = bwd_packed_seq_params
    
    ret = microbatches
    if return_seq_lens:
        ret = (microbatches, all_original_seq_lens)
    return ret


# %%

num_batches = 1
num_microbatch = 2
pp_size = 2
dp_size = 2 # ???
dpcp_size = 2
tp_size = 8
as_rank = 0
as_world_size = pp_size * dpcp_size
total_seq_len = 128 * 1024
num_seqs = 1
hidden_size_q_tp = 128 * 4
hidden_size_k_tp = 128
element_size = 2
num_head_in_dtype = 16
num_token_per_rank = 1024
use_planner = True

setup_global_batch(
    total_seq_len=total_seq_len,
    should_add_debug_cases=True,
    sample_name='wlbllm',
)

microbatches_0 = create_pp_microbatches(
    num_microbatch, pp_size, as_rank,
    as_world_size, total_seq_len, num_seqs, dpcp_size,
    hidden_size_q_tp, hidden_size_k_tp, element_size, num_head_in_dtype,
    tp_size, dp_size, 
    num_token_per_rank, num_batches, use_planner,  
)