"""
Debug example:
NVTE_ALLOW_NONDETERMINISTIC_ALGO=0 torchrun --nnodes 1 --nproc_per_node 8 test_megatron_e2e_pipeline_planner.py --num-gpus-per-node 8 --pp-size 2 --num-microbatch 2 --tp-size 2
NVTE_ALLOW_NONDETERMINISTIC_ALGO=0 torchrun --nnodes 1 --nproc_per_node 4 test_megatron_e2e_pipeline_planner.py --num-gpus-per-node 4 --pp-size 2 --num-microbatch 2 --tp-size 2
"""
import argparse
from functools import partial
import os
import time

import megatron.core.parallel_state as mpu
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.pipeline_parallel.schedules import get_forward_backward_func
import torch
from transformers import AutoConfig

from d2.runtime.megatron_patch.packed_seq_params import arg_to_cuda, PingPangSingleStepPackedSeqParams, PingPangPackedSeqParams
from d2.runtime.inplace_metadata import mlp_layout_packed_params
from d2.runtime.megatron_patch.forward_backward_func import forward_backward_pipelining_without_interleaving as forward_backward_func

from test_util import ParallelConfig, init_worker_torch_distributed
from test_megatron_e2e import MegatronE2eWorker as BaseMegatronE2eWorker, set_random_seed
from megatron_test_utils import (
    gptmodel_forward, make_batch_generator, unwrap_model,
)
from typing import Optional

# PP Megatron Worker. Need to add optimizer.
class MegatronE2eWorker(BaseMegatronE2eWorker):
    def __init__(self, rank: int, world_size: int):
        super().__init__(rank, world_size)
        local_rank = int(os.getenv("LOCAL_RANK"))
        torch.cuda.set_device(local_rank)
        torch.set_default_device(torch.device("cuda", local_rank))

    def forward_backward_batch(self, microbatches: list[dict], forward_only: bool=False,
                               mode: str="ping_pong", with_dummy: bool=True):

        microbatches = [{
            k: arg_to_cuda(v) for k, v in microbatch.items()
        } for microbatch in microbatches]
        if "orig" in mode:
            for mb in microbatches:
                psp = mb["packed_seq_params"]
                if isinstance(psp, PingPangSingleStepPackedSeqParams):
                    mb["packed_seq_params"] = mb["packed_seq_params"].mlp_packed_seq_params
                else:
                    assert isinstance(psp, PackedSeqParams)

        # forward_backward_func = get_forward_backward_func()
        pp_size = self.tf_config.pipeline_model_parallel_size
        pp_rank = mpu.get_pipeline_model_parallel_rank()
        n_micro_batch = len(microbatches) - pp_size + 1
        # thd layout
        total_seqlen = microbatches[0]['input_ids'].shape[0]

        def loss_func(output):
            # NOTE: this is a dummy loss function.
            loss = output.mean()
            return loss, {'loss': loss}

        def forward_step(batch_iter, model):
            batch = next(batch_iter)
            torch.cuda.nvtx.range_push("forward_step")
            input_ids = batch['input_ids']
            position_ids = batch['position_ids']
            attention_mask = None
            packed_seq_params = batch['packed_seq_params']
            # returns "hidden_states" if not model.post_process (not the last layer)
            # returns "logits" when label is None.
            output = gptmodel_forward(
                model, input_ids, attention_mask, position_ids, self.tf_config.sequence_parallel,
                packed_seq_params, labels=input_ids.unsqueeze(0),
            )
            torch.cuda.nvtx.range_pop()
            return output, loss_func

        def dummy_backward_step(model, dummy_bwd_iter, skip: bool):
            next_iter_args = next(dummy_bwd_iter)
            if skip:
                return
            unwrap_model(model).dummy_backward(next_iter_args)

        assert len(self.train_module) == 1, "only support one module"

        # shift bwd metadata since the order it runs is different from the
        # corresponding dummy forward's.
        dummy_bwd_packed_seq_params = [
            microbatch['packed_seq_params'] for microbatch in
            (microbatches[-pp_size + pp_rank + 1:][:pp_size - pp_rank - 1] + microbatches[:pp_rank])
        ]
        dummy_bwd_packed_seq_params = dummy_bwd_packed_seq_params[pp_rank:] + dummy_bwd_packed_seq_params[:pp_rank]

        assert mode in ["ping_pong", "orig_reimpl", "single_sided"]

        for module in self.train_module:
            debug = (mode != "ping_pong")
            debug_fwd_impl = mode if debug else None
            unwrap_model(module).set_debug(debug=debug, debug_fwd_impl=debug_fwd_impl)
            unwrap_model(module).train()
        assert len(self.train_module) == 1, "only support one module"

        dummy_bwd_packed_seq_params_iter = iter(dummy_bwd_packed_seq_params)
        batch_generator = make_batch_generator(
            microbatches if with_dummy else microbatches[pp_rank:],
            vpp_size=len(self.train_module)
        )
        # if mpu.get_pipeline_model_parallel_world_size() > 1:

        torch.cuda.synchronize()
        from d2.runtime.attn_kernels.ops import nvshmem_barrier_all
        nvshmem_barrier_all()
        if with_dummy:
            losses_reduced = forward_backward_func(
                forward_step_func=forward_step,
                data_iterator=batch_generator,
                model=self.train_module,
                num_microbatches=n_micro_batch,
                seq_length=total_seqlen,  # no use, since variable_seq_lengths=True
                micro_batch_size=1,  # no use when input_shapes was set
                forward_only=forward_only,
                dummy_bwd_func=partial(
                    dummy_backward_step,
                    dummy_bwd_iter=dummy_bwd_packed_seq_params_iter,
                    skip="orig" in mode,
                ),
            )
        else:
            losses_reduced = get_forward_backward_func()(
                forward_step_func=forward_step,
                data_iterator=batch_generator,
                model=self.train_module,
                num_microbatches=n_micro_batch,
                seq_length=total_seqlen,  # no use, since variable_seq_lengths=True
                micro_batch_size=1,  # no use when input_shapes was set
                forward_only=forward_only,
            )
        grad_sample = unwrap_model(self.train_module[0]).decoder.layers[-1].self_attention.linear_proj.weight.main_grad.clone()

        # when testing numerical correctness, instead of running optimizer step, reset grads.
        for tm in self.train_module:
            for param in unwrap_model(tm).parameters():
                param.main_grad.zero_()
        return losses_reduced, grad_sample


def init_megatron_e2e_test(
    hidden_size_q: int, hidden_size_kv: int, num_heads: int, num_tokens: int,
    world_size: int, max_cp_degree: int, tp_size: int, pp_size: int,
    dtype, worker_cls=MegatronE2eWorker
):
    token_bytes_q = hidden_size_q * dtype.itemsize // tp_size
    token_bytes_kv = hidden_size_kv * dtype.itemsize // tp_size
    max_tokens_query = num_tokens * (world_size // tp_size)
    max_tokens_key_value = num_tokens * (world_size // tp_size)
    buffer_size = (
        token_bytes_q * max_tokens_query * 3 +
        # lse_norm. TODO: the factor of 2 might be removed
        num_heads * torch.float32.itemsize * 2 * max_tokens_query +
        token_bytes_kv * max_tokens_key_value * max_cp_degree * 2
    )
    print(f'{buffer_size=}', flush=True)
    parallel_config = ParallelConfig(
        tensor_model_parallel_size=tp_size,
        pipeline_model_parallel_size=pp_size,
    )

    worker = init_worker_torch_distributed(
        world_size, buffer_size, worker_cls, parallel_config
    )
    print("Communication groups initialized")
    return worker



# We need a new method to get batch document length. 
# Similar to Junda' s DP Sequence length sampling.
# Each tick, will have DP * List. Each List is the sequence length on that rank.(MLP DP for now. CP is a little different.)
from d2.simulator.optimizers.samples import sample_wlbllm_docs_upsample, batch_documents
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

    GLOBAL_BATCH = iter(GLOBAL_BATCH)
    return





# 1. Figure out junda's dataloader.
# num_microbatch, 
# world_size, 
# parallel_config,
# total_seq_len,
# model_config,


# 

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
    ) = planner.plan_to_raw_qkv_dispatch(items, verbose=True)

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


from test_util import gen_seq_lens
from typing import Iterable, List, Optional

ITERATION_ID = 0
iterated_samples = []
GLOBAL_BATCH: Optional[Iterable[List[int]]] = None

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






# def create_pipeline_seqlens(
#     ref_seq_lens: Optional[torch.Tensor],
#     add_dummy: bool,
#     is_backward: bool,
#     world_size: int,
#     total_seq_len: int,
#     num_seqs: int,
#     max_cp_degree: int,
#     tp_size: int,
#     dp_size: int,
# ):
#     """
#     For a forward tick, its sequence length follows:
#         [new_microbatch, last_tick_seq_len[:PP_stage_-1]]

#         The new_microbatch is either generated, or a dummy one. (controlled by add_dummy)
#         A special case is that, the first tick does not have a previous one.
#         In this way, we make all stages' microbatch dummy, except for the first one.

#     For a backward tick, its sequence length is the reverse of a forward tick.
#     """
#     if is_backward:
#         return ref_seq_lens.reshape(-1, dp_size, num_seqs).flip(0).reshape(-1, num_seqs)
#     # Create new microbatches for the first PP stage
#     if add_dummy:
#         # should be at least 'tp_size' tokens
#         _dummy_tokens = max(1, tp_size // max_cp_degree)
#         # NOTE: we should avoid this repeat in a real case. The same applies for repeat below.
#         new_seqlen = gen_seq_lens(dp_size, 1, _dummy_tokens).long().reshape(dp_size, 1).repeat(1, num_seqs)
#     else:
#         assert total_seq_len % max_cp_degree == 0
#         _num_tokens_shard = total_seq_len // (max_cp_degree)
#         new_seqlen = gen_seq_lens(dp_size, num_seqs, _num_tokens_shard).long()

#     # Change to torch tensor for later concat.
#     # new_seqlen = get_next_batch(dp_size), dtype=torch.long
#     # max_cols = max(len(x) for x in new_seqlen)
#     # new_seqlen = [x + [0] * (max_cols - len(x)) for x in new_seqlen]
#     # new_seqlen = torch.tensor(new_seqlen, dtype=torch.long) # Padded Tensor.



#     # gen_seq_lens(dp_size, num_seqs, _num_tokens_shard).long()
#     # new_seqlen *= max_cp_degree
#     print("Next tick sequence length is :", new_seqlen)
#     #  new_seqlen : shape : [dp, num_seqs]  We should sample seq_len here.
#     # And add the sampled seq_len to the batch. 
#     # Next step is based on the previous batch, move the batch. 
#     # Get existing microbatch seqlens
#     if ref_seq_lens is not None:
#         # Not the first microbatch
#         prev_seqlen = ref_seq_lens[:-dp_size]
#     else:
#         dummy_fwd_num = world_size - dp_size
#         _dummy_tokens = max(1, tp_size // max_cp_degree)
#         prev_seqlen = gen_seq_lens(dummy_fwd_num, 1, _dummy_tokens).long().reshape(dummy_fwd_num, 1).repeat(1, num_seqs)
#         prev_seqlen *= max_cp_degree
#     seq_len = torch.cat([new_seqlen, prev_seqlen], dim=0)
#     assert torch.all(seq_len.sum(-1) % tp_size == 0), "tot_seqlen_on_rank % tp_size should be 0 for sequence parallel"
#     return seq_len


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
    pad_num = max(1, tp_size // max_cp_degree)
    pad_num *= max_cp_degree
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



def create_pp_microbatches(num_microbatch: int, pp_degree: int, as_rank: int,
                           as_world_size: int, total_seq_len: int, num_seqs: int,
                           max_cp_degree: int, hidden_size_q_tp: int,
                           hidden_size_k_tp: int, element_size: int,
                           num_head_in_dtype: int, tp_size: int, dp_size: int, 
                           hf_config: Optional[AutoConfig] = None,  # add this input for planner.
                           ):
    tick_seq_lens = None
    bwd_metadata = []
    microbatches = []
    for i in range(num_microbatch + pp_degree - 1):
        # For the last few ticks (drain-out ticks)
        # add a dummy forward microbatch at PP rank 0.
        add_dummy_forward = i >= num_microbatch

        # Update the tick_seq_lens for the next iteration.
        (
            fa_fwd_params, fa_bwd_params,
            qkv_fwd_fa2a_metadata, qkv_bwd_fa2a_metadata,
            attn_out_fwd_fa2a_metadata, attn_out_qkv_bwd_fa2a_metadata,
            tick_seq_lens,
        ) = create_qkv_dispatch_pipeline_tick_planned(
            as_world_size, total_seq_len, num_seqs, max_cp_degree,
            hidden_size_q_tp, hidden_size_k_tp, element_size, num_head_in_dtype,
            ref_seq_lens=tick_seq_lens,
            add_dummy=add_dummy_forward,
            tp_size=tp_size,
            dp_size=dp_size,
            pp_size= pp_degree,
            hf_config=hf_config,
        )
        this_rank_num_tokens = tick_seq_lens[as_rank].sum().item()

        (cu_seqlens_q, cu_seqlens_kv, max_seqlen_q, max_seqlen_kv, *_) = fa_bwd_params
        bwd_packed_seq_params = PackedSeqParams(
            cu_seqlens_q=cu_seqlens_q[as_rank],
            cu_seqlens_kv=cu_seqlens_kv[as_rank],
            max_seqlen_q=max_seqlen_q[as_rank].item(),
            max_seqlen_kv=max_seqlen_kv[as_rank].item(),
        )
        mlp_packed_seq_params = mlp_layout_packed_params(tick_seq_lens[as_rank][:num_seqs])
        (cu_seqlens_q, cu_seqlens_kv, max_seqlen_q, max_seqlen_kv, *_) = fa_fwd_params

        # Create packed_params. Note that we do not add backward params here.
        ping_pang_params = PingPangSingleStepPackedSeqParams(
            qkv_format="thd",
            cu_seqlens_q=cu_seqlens_q[as_rank],
            cu_seqlens_kv=cu_seqlens_kv[as_rank],
            max_seqlen_q=max_seqlen_q[as_rank].item(),
            max_seqlen_kv=max_seqlen_kv[as_rank].item(),
            qkv_fwd_metadata=qkv_fwd_fa2a_metadata.get_slice(as_rank),
            attn_out_fwd_metadata=attn_out_fwd_fa2a_metadata.get_slice(as_rank),
            mlp_packed_seq_params=mlp_packed_seq_params,
        )

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
    return microbatches





def test(args):
    seed = args.seed
    # test scale
    num_tokens = args.num_tokens
    max_cp_degree = args.cp_degree
    num_seqs = args.num_seqs
    total_seq_len = args.num_tokens
    # parallelization
    tp_size = args.tp_size
    pp_size = args.pp_size
    world_size = args.num_nodes * args.num_gpus_per_node
    assert world_size % (tp_size * pp_size) == 0
    dp_size = world_size // (tp_size * pp_size)

    dtype = torch.bfloat16
    element_size = dtype.itemsize

    model_path = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
    hf_config = AutoConfig.from_pretrained(model_path)
    hidden_size_q = hf_config.hidden_size

    hidden_size_kv = hidden_size_q
    if hasattr(hf_config, "num_key_value_heads"):
        hidden_size_kv = (hidden_size_kv * hf_config.num_key_value_heads //
                          hf_config.num_attention_heads)

    worker: MegatronE2eWorker = init_megatron_e2e_test(
        hidden_size_q, hidden_size_kv, hf_config.num_attention_heads, num_tokens,
        world_size, max_cp_degree, tp_size, pp_size,
        dtype, MegatronE2eWorker
    )
    worker.set_config(dtype=dtype)
    worker.init(model_path, seed=seed)
    # set again to potentially adapt to the ray launch case.
    set_random_seed(seed, set_megatron=False)

    as_world_size = worker.as_world_size
    as_rank = worker.as_rank

    # Check rank correctness
    dp_rank = mpu.get_data_parallel_rank()
    pp_rank = mpu.get_pipeline_model_parallel_rank()
    assert as_rank == dp_rank + pp_rank * dp_size

    # Set global batch to be fetched later.
    sequence_length = args.num_tokens
    setup_global_batch(total_seq_len=sequence_length)

    hidden_size_q_tp = hidden_size_q // tp_size
    hidden_size_k_tp = hidden_size_kv // tp_size
    num_head_in_dtype = (hf_config.num_attention_heads *
                         torch.float32.itemsize // element_size // tp_size)

    microbatches_0 = create_pp_microbatches(
        args.num_microbatch, pp_size, as_rank,
        as_world_size, total_seq_len, num_seqs, max_cp_degree,
        hidden_size_q_tp, hidden_size_k_tp, element_size, num_head_in_dtype,
        tp_size, dp_size, hf_config
    )
    microbatches_1 = create_pp_microbatches(
        args.num_microbatch, pp_size, as_rank,
        as_world_size, total_seq_len, num_seqs, max_cp_degree,
        hidden_size_q_tp, hidden_size_k_tp, element_size, num_head_in_dtype,
        tp_size, dp_size, hf_config
    )
    set_random_seed(seed, set_megatron=True)
    microbatches = []
    orig_impl_microbatches = []
    for mb_0, mb_1 in zip(microbatches_0, microbatches_1):
        mb_0_psp = mb_0["packed_seq_params"]
        mb_1_psp = mb_1["packed_seq_params"]
        mb_0_mlp_psp = mb_0_psp.mlp_packed_seq_params
        mb_1_mlp_psp = mb_1_psp.mlp_packed_seq_params
        mb_0_psp.dispatcher_id = 0
        mb_1_psp.dispatcher_id = 1
        ping_pong_params = PingPangPackedSeqParams(
            seq_params = [mb_0_psp, mb_1_psp],
            mlp_layout_seq_params = [mb_0_mlp_psp, mb_1_mlp_psp],
            max_seqlen_q = max(mb_0_mlp_psp.max_seqlen_q, mb_1_mlp_psp.max_seqlen_q),
            max_seqlen_kv = max(mb_0_mlp_psp.max_seqlen_kv, mb_1_mlp_psp.max_seqlen_kv),
        )
        num_tokens = sum(mb["position_ids"].numel() for mb in [mb_0, mb_1])
        input_ids = torch.randint(10, 1000, (num_tokens,))
        mb = {
            "input_ids": input_ids,
            "position_ids": torch.concat([mb_0["position_ids"], mb_1["position_ids"]]),
            "packed_seq_params": ping_pong_params,
        }
        microbatches.append(mb)
        packed_seq_params = PackedSeqParams(
            qkv_format="thd",
            cu_seqlens_q = torch.concat([
                mb_0_mlp_psp.cu_seqlens_q, mb_1_mlp_psp.cu_seqlens_q[1:] + mb_0_mlp_psp.cu_seqlens_q[-1]
            ]),
            cu_seqlens_kv = torch.concat([
                mb_0_mlp_psp.cu_seqlens_kv, mb_1_mlp_psp.cu_seqlens_kv[1:] + mb_0_mlp_psp.cu_seqlens_kv[-1]
            ]),
            max_seqlen_q = ping_pong_params.max_seqlen_q,
            max_seqlen_kv = ping_pong_params.max_seqlen_kv,
        )
        orig_mb = {
            "input_ids": mb["input_ids"],
            "position_ids": mb["position_ids"],
            "packed_seq_params": packed_seq_params,
        }
        orig_impl_microbatches.append(orig_mb)

    time.sleep(2)
    loss_orig_reimpl, grad_orig_reimpl = worker.forward_backward_batch(
        microbatches=orig_impl_microbatches,
        forward_only=False,
        mode="orig_reimpl",
        with_dummy=True,
    )
    loss_orig, grad_orig = worker.forward_backward_batch(
        microbatches=orig_impl_microbatches,
        forward_only=False,
        mode="orig_reimpl",
        with_dummy=False,
    )
    print("finish baseline")
    for _ in range(3):
        time.sleep(2)
        loss_reduced, grad_sample = worker.forward_backward_batch(
            microbatches=microbatches,
            forward_only=False,
            mode="ping_pong",
            with_dummy=True,
        )
        print(f"{loss_reduced=}, {loss_orig_reimpl=}, {loss_orig=}")
    torch.cuda.synchronize()
    torch.testing.assert_close(grad_orig_reimpl, grad_orig)
    if worker.as_rank == 1:
        torch.testing.assert_close(grad_orig_reimpl, grad_sample, rtol=1.1e-3, atol=1.1e-3)
    print(f"{worker.rank} finish pingpong")

    print("=" * 20 + "forward_backward_batch attention server, done")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-tokens", type=int, default=1024)
    parser.add_argument("--cp-degree", type=int, default=2)
    parser.add_argument("--num-seqs", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-nodes", type=int, default=1)
    parser.add_argument("--num-gpus-per-node", type=int, default=4)
    parser.add_argument("--tp-size", type=int, default=1)
    parser.add_argument("--pp-size", type=int, default=4)
    parser.add_argument("--num-microbatch", type=int, default=5)
    args = parser.parse_args()
    test(args)
