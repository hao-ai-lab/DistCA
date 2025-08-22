from test_util import ParallelConfig, init_worker_torch_distributed
from d2.simulator.optimizers.samples import sample_wlbllm_docs_upsample, batch_documents
from typing import Iterable, List, Optional
from transformers import AutoConfig

class MockAutoConfig:
    def __init__(self):
        self.hidden_size = 128
        self.num_attention_heads = 8
        self.num_key_value_heads = 16


args = {
        "world_size": 8,
        "total_seq_len": 100,
        "num_seqs": 4,
        "max_cp_degree": 1,
        "hidden_size_q": 128,
        "hidden_size_k": 128,
        "element_size": 2,  # e.g., float16
        "softmax_lse_size": 8, # e.g., num_heads
        "ref_seq_lens": None,
        "add_dummy": False,
        "tp_size": 2,
        "dp_size": 2,
        "pp_size": 2,
        "hf_config": MockAutoConfig(),
    }


# for i in range(5):
#     results = create_qkv_dispatch_pipeline_tick_planned(**args)
#     print(results)


GLOBAL_BATCH: Optional[Iterable[List[int]]] = None

def setup_global_batch(
    total_seq_len, 
    up_sample_factor=2,
    elongate_factor=1,
    filter_threshold=64 * 1024,
    filter_ratio=0.90,
    should_add_debug_cases=False,
):
    global GLOBAL_BATCH
    if GLOBAL_BATCH is not None:
        return

    GLOBAL_BATCH = batch_documents(
        sample_wlbllm_docs_upsample(
            size=10000,
            filter_threshold=filter_threshold,
            filter_ratio=filter_ratio,
            upsample_long_factor=up_sample_factor,
            elongate_factor=elongate_factor,
        ), max_ctx_length=total_seq_len
    )

    
    if should_add_debug_cases:
        GLOBAL_BATCH = list(GLOBAL_BATCH)
        # DP2 case
        manual_case = [
            [total_seq_len],
            [total_seq_len // 8] * 8,
            [total_seq_len],
            [total_seq_len // 8] * 8,
        ]
        GLOBAL_BATCH = manual_case * 4 + GLOBAL_BATCH 
    GLOBAL_BATCH = [
            [total_seq_len/4, total_seq_len/4] * 2,
            [total_seq_len/2, total_seq_len/4, total_seq_len/8, total_seq_len/8],
        ]
    GLOBAL_BATCH = GLOBAL_BATCH * 100
    GLOBAL_BATCH = iter(GLOBAL_BATCH)
    return

def run_batch_setup_example():
    print("--- 场景1: 基本调用 ---")
    sequence_length = 32 * 1024
    setup_global_batch(total_seq_len=sequence_length)
    
    try:
        first_batch = next(GLOBAL_BATCH)
        sec_batch = next(GLOBAL_BATCH)
        thi_batch = next(GLOBAL_BATCH)
        four_batch = next(GLOBAL_BATCH)
        fif_batch = next(GLOBAL_BATCH)
        print(f" {len(first_batch)} 个文档。\n")
    except StopIteration:
        print("数据批次为空。\n")

ITERATION_ID = 0
iterated_samples = []
def get_next_batch(dp_size):
    global GLOBAL_BATCH
    global ITERATION_ID
    global iterated_samples
    # get dp_size number of batches 
    batches = []
    for _ in range(dp_size):    
        batches.append(next(GLOBAL_BATCH))
    ITERATION_ID += 1
    iterated_samples.append(batches)
    return batches

sequence_length = 10 * 1024
setup_global_batch(total_seq_len=sequence_length)

# batch = get_next_batch(2)
# breakpoint()

import torch

from test_util import gen_seq_lens

def create_pipeline_seqlens(
    ref_seq_lens: Optional[torch.Tensor],
    add_dummy: bool,
    is_backward: bool,
    world_size: int,
    total_seq_len: int,
    num_seqs: int,
    max_cp_degree: int,
    tp_size: int,
    dp_size: int,
):
    """
    For a forward tick, its sequence length follows:
        [new_microbatch, last_tick_seq_len[:PP_stage_-1]]

        The new_microbatch is either generated, or a dummy one. (controlled by add_dummy)
        A special case is that, the first tick does not have a previous one.
        In this way, we make all stages' microbatch dummy, except for the first one.

    For a backward tick, its sequence length is the reverse of a forward tick.
    """
    if is_backward:
        num_seqs = ref_seq_lens.shape[1]
        return ref_seq_lens.reshape(-1, dp_size, num_seqs).flip(0).reshape(-1, num_seqs)
    # Create new microbatches for the first PP stage
    if add_dummy:
        # should be at least 'tp_size' tokens
        _dummy_tokens = max(1, tp_size // max_cp_degree)
        # NOTE: we should avoid this repeat in a real case. The same applies for repeat below.
        new_seqlen = gen_seq_lens(dp_size, 1, _dummy_tokens).long().reshape(dp_size, 1).repeat(1, num_seqs)
    else:
        assert total_seq_len % max_cp_degree == 0
        _num_tokens_shard = total_seq_len // (max_cp_degree)
        new_seqlen = gen_seq_lens(dp_size, num_seqs, _num_tokens_shard).long()
    
    # Change to torch tensor for later concat.
    new_seqlen_list = get_next_batch(dp_size, )
    # Change to even 
    from copy import deepcopy
    result = deepcopy(new_seqlen_list)
    for l in result:
        odd_indices = [i for i, num in enumerate(l) if num % 2 != 0]
        for i in range(0, len(odd_indices), 2):
            idx1 = odd_indices[i]
            idx2 = odd_indices[i+1]
            l[idx1] += 1
            l[idx2] -= 1
    new_seqlen_list = result
    for seq in new_seqlen_list:
        for s in seq:
            assert(s % 2 == 0)
    # pad real sequence and dummy sequence.
    pad_num = tp_size
    max_cols = max(len(x) for x in new_seqlen_list)
    new_seqlen_list = [x + [pad_num] * (max_cols - len(x)) for x in new_seqlen_list]
    
    new_seqlen = torch.tensor(new_seqlen_list, dtype=torch.long) # Padded Tensor.

    #  new_seqlen : shape : [dp, num_seqs]  We should sample seq_len here.
    # And add the sampled seq_len to the batch. 
    # Next step is based on the previous batch, move the batch. 
    # Get existing microbatch seqlens
    if ref_seq_lens is not None:
        # Not the first microbatch
        prev_seqlen = ref_seq_lens[:-dp_size]
    else:
        dummy_fwd_num = world_size - dp_size
        _dummy_tokens = max(1, tp_size // max_cp_degree)
        prev_seqlen = gen_seq_lens(dummy_fwd_num, 1, _dummy_tokens).long().reshape(dummy_fwd_num, 1).repeat(1, num_seqs)
        prev_seqlen *= max_cp_degree
    import torch.nn.functional as F
    new_cols = new_seqlen.shape[1]
    prev_cols = prev_seqlen.shape[1]
    max_num_cols = max(new_cols, prev_cols)
    pad_new = max_num_cols - new_cols
    pad_prev = max_num_cols - prev_cols
    # Pad previous batch and new batch.
    if pad_new > 0:
        padding = (torch.ones(new_seqlen.shape[0], pad_new) * pad_num).int()
        new_seqlen = torch.cat([new_seqlen, padding], dim=1)
    
    if pad_prev > 0:
        padding = (torch.ones(prev_seqlen.shape[0], pad_prev) * pad_num).int()
        prev_seqlen = torch.cat([prev_seqlen, padding], dim=1)
    
    seq_len = torch.cat([new_seqlen, prev_seqlen], dim=0)

    assert torch.all(seq_len.sum(-1) % tp_size == 0), f"tot_seqlen_on_rank % tp_size should be 0 for sequence parallel, seq_sum : {seq_len.sum(-1)}, {seq_len.sum(-1) % tp_size}"

    # expected : total_seq_len
    for rank_seq in seq_len:
        current_token = sum(rank_seq)
        gap = current_token - total_seq_len
        if gap < 0 :
            continue

        max_index = rank_seq.argmax().item()
        rank_seq[max_index] -= gap
        assert sum(rank_seq) == total_seq_len, f"current_token : {sum(rank_seq)}, max_index : {max_index}, rank_seq : {rank_seq}"
    
    return seq_len


from d2.planner.planner import Planner, batch_to_items_class
from d2.runtime.fast_alltoall_metadata import compute_e2e_fa2a_metadata, compute_backward_resend_e2e_metadata
from test_util import create_raw_qkv_dispatch
def create_qkv_dispatch_pipeline_tick_planned(
    world_size: int, total_seq_len: int, num_seqs: int, max_cp_degree: int,
    hidden_size_q: int, hidden_size_k: int,
    element_size: int, # dtype's size
    softmax_lse_size: int, # size of the softmax_lse tensor, should be num_heads
    ref_seq_lens: Optional[torch.Tensor],
    add_dummy: bool,
    tp_size: int, dp_size: int, pp_size: int,
    hf_config: Optional[AutoConfig] = None,
):
    create_pp_seqlen_kwargs = dict(
        world_size=world_size,
        total_seq_len=total_seq_len,
        num_seqs=num_seqs,
        max_cp_degree=max_cp_degree,
    )
    # Change this to junda's document sampler.
    seq_lens = create_pipeline_seqlens(
        ref_seq_lens, add_dummy, is_backward=False,
        **create_pp_seqlen_kwargs,
        tp_size=tp_size,
        dp_size=dp_size,
    )
    print("Current tick seq_lens is:")
    print(seq_lens)
    # Create parallel config for the planner.
    parallel_config = ParallelConfig(
        tensor_model_parallel_size=tp_size,
        pipeline_model_parallel_size=pp_size,
    )
    planner_ws = tp_size * dp_size * pp_size
    planner = Planner(
            world_size=planner_ws,  # Should be the total number of GPU. Not Attention server world size.
            parallel_config = parallel_config,
            tolerance_factor = 0.1,
            model_config = None)

    # Post processing seq_len. Remove padded zero.
    batch = seq_lens.tolist()

    print("batch =", batch)
    
    items = batch_to_items_class(batch,model_config=hf_config)
    
    (
        mlp_num_seqs,
        mlp_q_dispatch_fwd,
        mlp_seq_len,
        kv_to_q_mapping,
        kv_to_q_rank,
        kv_context_size,
        q_to_num_kv_seq,
        q_to_num_kv_tokens
    ) = planner.plan_to_raw_qkv_dispatch(items, verbose=False)

    (_, _, _, _,
     fa_fwd_params, fa2a_fwd_metadata) = compute_e2e_fa2a_metadata(
        mlp_seq_len, mlp_num_seqs, mlp_q_dispatch_fwd,
        kv_to_q_mapping, kv_to_q_rank,
        kv_context_size, q_to_num_kv_seq, q_to_num_kv_tokens,
        hidden_size_q, hidden_size_k,
        element_size, softmax_lse_size
    )
    (qkv_fwd_fa2a_metadata, _, attn_out_fwd_fa2a_metadata, _,
    ) = fa2a_fwd_metadata

    reversed_seqlens = create_pipeline_seqlens(
        seq_lens, add_dummy=False, is_backward=True,
        **create_pp_seqlen_kwargs,
        tp_size=tp_size,
        dp_size=dp_size,
    )
    # NOTE: those begin with bwd_ is mostly the flip of the original value.
    print("reversed_seqlens is : ", reversed_seqlens)
    (bwd_mlp_seq_len, bwd_mlp_num_seqs, mlp_q_dispatch_bwd,
     bwd_kv_to_q_mapping, bwd_kv_to_q_rank, bwd_kv_context_size,
     bwd_q_to_num_kv_seq, bwd_q_to_num_kv_tokens
    ) = create_raw_qkv_dispatch(
        world_size, total_seq_len, num_seqs, max_cp_degree,
        return_mlp_no_shard_seq_lens=False,
        seq_lens=reversed_seqlens,
    )

    (fa_bwd_params, attn_out_qkv_bwd_fa2a_metadata, qkv_bwd_fa2a_metadata
     ) = compute_backward_resend_e2e_metadata(
        bwd_mlp_seq_len, bwd_mlp_num_seqs, mlp_q_dispatch_bwd,
        bwd_kv_to_q_mapping, bwd_kv_to_q_rank, bwd_kv_context_size,
        bwd_q_to_num_kv_seq, bwd_q_to_num_kv_tokens,
        hidden_size_q, hidden_size_k, element_size, softmax_lse_size,
    )
    return (fa_fwd_params, fa_bwd_params,
            qkv_fwd_fa2a_metadata, qkv_bwd_fa2a_metadata,
            attn_out_fwd_fa2a_metadata, attn_out_qkv_bwd_fa2a_metadata,
            seq_lens,)


if __name__ == "__main__":
    import torch
    from typing import Optional

    world_size = 32
    
    dp_size = 2
    tp_size = 8
    pp_size = 2
    as_world_size = dp_size * pp_size
    num_seqs = 4
    max_cp_degree = 2
    total_seq_len = 10 * 1024



    # for i in range(5):
    #     print(f"\n---  {i+1} th call ---")
    #     ref_seq_lens = create_pipeline_seqlens(
    #         ref_seq_lens=ref_seq_lens,
    #         add_dummy=add_dummy,
    #         is_backward=is_backward,
    #         world_size=world_size,
    #         total_seq_len=total_seq_len,
    #         num_seqs=num_seqs,
    #         max_cp_degree=max_cp_degree,
    #         tp_size=tp_size,
    #         dp_size=dp_size,
    #     )
    #     reversed_seq_lens = create_pipeline_seqlens(
    #         ref_seq_lens=ref_seq_lens,
    #         add_dummy=add_dummy,
    #         is_backward=True,
    #         world_size=world_size,
    #         total_seq_len=total_seq_len,
    #         num_seqs=num_seqs,
    #         max_cp_degree=max_cp_degree,
    #         tp_size=tp_size,
    #         dp_size=dp_size,
    #     )

    ref_seq_lens =  None
    add_dummy = True
    is_backward = False

    tick_seq_lens = None
    bwd_metadata = []
    microbatches = []
    num_microbatch = 5

    for i in range(num_microbatch + pp_size - 1):
        # For the last few ticks (drain-out ticks)
        # add a dummy forward microbatch at PP rank 0.
        add_dummy_forward = i >= num_microbatch

        # Update the tick_seq_lens for the next iteration.
        (
            fa_fwd_params, fa_bwd_params,
            qkv_fwd_fa2a_metadata, qkv_bwd_fa2a_metadata,
            attn_out_fwd_fa2a_metadata, attn_out_qkv_bwd_fa2a_metadata,
            ref_seq_lens,
        ) = create_qkv_dispatch_pipeline_tick_planned(
            world_size=as_world_size,
            total_seq_len=total_seq_len,
            num_seqs=num_seqs,
            max_cp_degree=max_cp_degree,
            hidden_size_q=128,
            hidden_size_k=128,
            element_size=2,
            softmax_lse_size=8,
            ref_seq_lens=ref_seq_lens,
            add_dummy=add_dummy,
            tp_size=tp_size,
            dp_size=dp_size,
            pp_size=pp_size,
            hf_config=MockAutoConfig(),
        )
    print("create_qkv_dispatch_pipeline_tick_planned result:")
