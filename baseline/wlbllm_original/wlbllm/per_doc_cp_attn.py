import time
import torch
from itertools import accumulate
import os
import rich
import torch.distributed as dist
import warnings
from torch.distributed import (
    get_world_size,
    get_rank,
)

from torch.cuda.nvtx import range_push as nvtx_range_push
from torch.cuda.nvtx import range_pop  as nvtx_range_pop
from torch.cuda.nvtx import range  as nvtx_range


from wlbllm.attn_module import (
    flash_attn_varlen_func,
    _flash_attn_varlen_forward,
    _flash_attn_varlen_backward,
)

from wlbllm.utils import (
    kv_shuffle_for_per_doc_cp,
    kv_unshuffle_for_per_doc_cp,
    compute_per_doc_cp_shard_doc_len,
)

import wlbllm.registry

def cat_slices(tensor, starts, lens):
            if starts.numel() == 1:
                s, l = int(starts[0]), int(lens[0])
                return tensor[s:s+l]
            return torch.cat([tensor[int(s):int(s)+int(l)]
                              for s, l in zip(starts.tolist(), lens.tolist())],
                             dim=0)   

import rich
def debug_print(*args, **kwargs):
    if os.getenv("D2_DEBUG_PRINT", "0") == "1":
        if torch.distributed.is_initialized():
            rank = torch.distributed.get_rank()
            rich.print(f"[Rank {rank}]", *args, **kwargs)
    return


fake_lse = None

class PerDocumentCPAttention(torch.autograd.Function):
    """
    Attention with per‑document context parallelism.

    - local_q / k / v should contain every shard that this rank owns:
    [Doc1-Shard1, Doc2-Shard1, ..., DocN-Shard1, Doc1-Shard2, Doc2-Shard2, ..., DocN-Shard2]

    - cu_seqlens_q / kv are built in the same order as the local qkv tensor
    - q_len_left is the number of tokens of the concatenated front chunks (Shard1's) of the documents
    """

    @staticmethod
    def forward(
        ctx,
        local_q, local_k, local_v,
        cu_seqlens_q_list, cu_seqlens_kv_list,
        max_seqlen_q_list, max_seqlen_kv_list,
        doc_lens,
        doc_shards,
        kv_idx_list,                 # global offsets of the K shards after shuffle
        dropout_p,
        softmax_scale,
        attn_mask_type,
        cp_group,
        cp_stream,
        allgather_events: 'Optional[List[torch.cuda.Event]]',
        allreduce_events: 'Optional[List[torch.cuda.Event]]',
        attn_events: 'Optional[List[torch.cuda.Event]]',
    ):
        # TODO(HACK): obviously cursor don't know how to use warning to do log the first time...
        print("👻 Inside PerDocumentCPAttention.forward()")
        should_sync_time_flash_attn = os.getenv("WLBLLM_SYNC_TIME_FLASH_ATTN", "0") == "1"
        should_sync_time_perdocattn = os.getenv("WLBLLM_SYNC_TIME_PERDOC_ATTN", "0") == "1"
        ENABLE_SHUFFLE = os.getenv("WLBLLM_ENABLE_SHUFFLE", "1") == "1"

        if should_sync_time_perdocattn:
            torch.cuda.synchronize()
            torch.distributed.barrier()
            start_time__fwd = time.time()

        global fake_lse

        nvtx_range_push("wlbllm.PerDocumentCPAttention.fwd")
        assert attn_mask_type == "causal", "Only causal attention is supported"
        assert cp_group is not None, "cp_group must be provided"
        assert cp_stream is not None, "cp_stream must be provided"

        nvtx_range_push("wlbllm.PerDocumentCPAttention.fwd.prepare")
        cp_size = get_world_size(cp_group)
        rank = get_rank(cp_group)
        world_rank = torch.distributed.get_rank()
        context_length = local_q.shape[0] * cp_size
        
        # allgather kv, then shuffle back to global order
        if allgather_events is not None:
            allgather_events[0].record()

        if should_sync_time_perdocattn:
            torch.cuda.synchronize()
            torch.distributed.barrier()
            start_time__gather = time.time()

        if cp_size > 1:
            gather_k_list = [torch.empty_like(local_k) for _ in range(cp_size)]
            gather_v_list = [torch.empty_like(local_v) for _ in range(cp_size)]
            with torch.cuda.stream(cp_stream):
                local_k = local_k.contiguous()
                local_v = local_v.contiguous()
                dist.all_gather(gather_k_list, local_k, group=cp_group)
                dist.all_gather(gather_v_list, local_v, group=cp_group)
                print("🟡 All gather local_k.shape =", local_k.shape)

            start_time__shuffle = time.time()
            
            with nvtx_range("wlbllm.PerDocumentCPAttention.fwd.shuffle"):

                if ENABLE_SHUFFLE:
                    start_time__shuffle = time.time()
                    k_global, v_global = kv_shuffle_for_per_doc_cp(context_length, gather_k_list, gather_v_list, doc_lens, doc_shards, cp_size)
                    end_time__shuffle = time.time()
                    duration_ms__shuffle = (end_time__shuffle - start_time__shuffle) * 1000
                    debug_print(f"🟡 PerDocumentCPAttention kv_shuffle_for_per_doc_cp time: {duration_ms__shuffle} ms")
                    # print("k_global.shape =", k_global.shape)
                    # print("v_global.shape =", v_global.shape)
                    # context_length = max(max(i) for i in cu_seqlens_kv_list)
                    # print("🟡 kv_idx_list =", kv_idx_list)
                else:
                    # Simply using a random global tensor for testing. This avoids a significant performance issue introduced by the shuffle logic.
                    context_length = wlbllm.registry.get("global_tensor_length")
                    # print("🟡 context_length =", context_length)
                    nkvheads = local_k.shape[1]
                    # print("🟡 nkvheads =", nkvheads)
                    d_head = local_k.shape[2]
                    # print("🟡 d_head =", d_head)
                    k_global = torch.randn(context_length, nkvheads, d_head, device=local_k.device, dtype=local_k.dtype)
                    v_global = torch.randn(context_length, nkvheads, d_head, device=local_v.device, dtype=local_v.dtype)
            
            end_time__shuffle = time.time()
            duration_ms__shuffle = (end_time__shuffle - start_time__shuffle) * 1000
            debug_print(f"🟡 PerDocumentCPAttention kv_shuffle_for_per_doc_cp time: {duration_ms__shuffle} ms")
        else:
            k_global = local_k.contiguous()
            v_global = local_v.contiguous() if local_v is not None else None

        if allgather_events is not None:
            allgather_events[1].record()
        nvtx_range_pop()

        if should_sync_time_perdocattn:
            torch.cuda.synchronize()
            torch.distributed.barrier()
            end_time__gather = time.time()
            duration_ms__gather = (end_time__gather - start_time__gather) * 1000
            debug_print(f"🟡 PerDocumentCPAttention allgather time (with barrier): {duration_ms__gather} ms")
        
        
        nvtx_range_push("wlbllm.PerDocumentCPAttention.fwd.flash_attn")


        if attn_events is not None:
            attn_events[0].record()
        # compute fwd of two chunks
        outputs, lses = [], []
        local_ks, local_vs = [], []
        q_chunks = local_q.chunk(2, dim=0)
        for chunk_id in range(2):
            local_k_list = []
            local_v_list = []
            for start, end in kv_idx_list[chunk_id]:
                local_k_list.append(k_global[start:end, :, :])
                local_v_list.append(v_global[start:end, :, :])

            local_k = torch.cat(local_k_list, dim=0)
            local_v = torch.cat(local_v_list, dim=0)
            local_ks.append(local_k)
            local_vs.append(local_v)

            
            rank = torch.distributed.get_rank()
            # if rank % 8 == 0:
            #     debug_print("🟡 Inside WLBLLM's PerDocumentCPAttention.forward(). Printing this may mean we have performance issue.")
            #     debug_print(f"  - q_chunks[chunk_id].shape = {q_chunks[chunk_id].shape}")
            #     debug_print(f"  - local_k.shape = {local_k.shape}")
            #     debug_print(f"  - local_v.shape = {local_v.shape}")
            #     debug_print(f"  - cu_seqlens_q_list[chunk_id] = {cu_seqlens_q_list[chunk_id]}")
            #     debug_print(f"  - cu_seqlens_kv_list[chunk_id] = {cu_seqlens_kv_list[chunk_id]}")
            #     debug_print(f"  - max_seqlen_q_list[chunk_id] = {max_seqlen_q_list[chunk_id]}")
            #     debug_print(f"  - max_seqlen_kv_list[chunk_id] = {max_seqlen_kv_list[chunk_id]}")

            
            # TODO:(Hack) PerDocumentCPAttention performance degrade significantly when returning LSE.
            # We create an env var to disable it only for performance testing.
            
            
            if should_sync_time_flash_attn:
                torch.cuda.synchronize()
                start_time = time.time()

            out, lse, _ = flash_attn_varlen_func(
                q=q_chunks[chunk_id],
                k=local_k,
                v=local_v,
                cu_seqlens_q=cu_seqlens_q_list [chunk_id],
                cu_seqlens_k=cu_seqlens_kv_list[chunk_id],
                max_seqlen_q=max_seqlen_q_list [chunk_id],
                max_seqlen_k=max_seqlen_kv_list[chunk_id],
                dropout_p=0.0,
                softmax_scale=softmax_scale,
                causal=True,
                return_attn_probs=True,
                deterministic=False, # do not turn on this flag - performance will degrade!
            )
            
            if should_sync_time_flash_attn:
                torch.cuda.synchronize()
                end_time = time.time()
                duration_ms = (end_time - start_time) * 1000
                if world_rank % 8 == 0:
                    debug_print(f"🟡 PerDocumentCPAttention FlashAttnVarlenFunc time: {duration_ms} ms")
            outputs.append(out)
            lses.append(lse)

        final_out = torch.cat(outputs, dim=0)
        if attn_events is not None:
            attn_events[1].record()
        nvtx_range_pop()

        nvtx_range_push("wlbllm.PerDocumentCPAttention.fwd.save_for_backward")

        ctx.save_for_backward(
            local_q,
            k_global, v_global,
            *outputs,       
            *lses,
            *local_ks,
            *local_vs,   
            *cu_seqlens_q_list, *cu_seqlens_kv_list,
            *max_seqlen_q_list, *max_seqlen_kv_list,
        )
        ctx.kv_idx_list      = kv_idx_list
        ctx.q_chunk_sizes  = [c.shape[0] for c in q_chunks]
        ctx.dropout_p      = dropout_p
        ctx.doc_lens       = doc_lens
        ctx.doc_shards     = doc_shards
        ctx.softmax_scale  = softmax_scale
        ctx.attn_mask_type = attn_mask_type
        ctx.cp_group       = cp_group
        ctx.cp_stream      = cp_stream

        nvtx_range_pop()

        nvtx_range_pop()
        if should_sync_time_perdocattn:
            torch.cuda.synchronize()
            torch.distributed.barrier()
            end_time__fwd = time.time()
            duration_ms__fwd = (end_time__fwd - start_time__fwd) * 1000
            debug_print(f"🟡 PerDocumentCPAttention total forward time (with barrier): {duration_ms__fwd} ms")
            
        return final_out

    @staticmethod
    def backward(ctx, d_out_cat):
        """
        Backward pass for PerDocumentCPAttention.
        """
        ENABLE_SHUFFLE = os.getenv("WLBLLM_ENABLE_SHUFFLE", "1") == "1"

        nvtx_range_push("wlbllm.PerDocumentCPAttention.bwd")

        (
            local_q,
            gathered_k, gathered_v,
            out_L, out_R, 
            lse_L, lse_R,
            k_L, k_R,
            v_L, v_R,
            cu_q_L, cu_q_R, cu_k_L, cu_k_R,
            maxq_L, maxq_R, maxk_L, maxk_R,
            ) = ctx.saved_tensors

        cp_group   = ctx.cp_group
        kv_idx_list = ctx.kv_idx_list 
        doc_lens = ctx.doc_lens
        doc_shards = ctx.doc_shards
        (qlen_L, qlen_R) = ctx.q_chunk_sizes
        world_size = get_world_size(cp_group)
        rank    = get_rank(cp_group)

        # split grad_out into two chunks
        dq_local = torch.zeros_like(local_q)
        dk_global = torch.zeros_like(gathered_k)
        dv_global = torch.zeros_like(gathered_v)
        # debug_print(f"💜 dq_local = {dq_local.shape}")
        # debug_print(f"💜 dk_global = {dk_global.shape}")
        # debug_print(f"💜 dv_global = {dv_global.shape}")

        # split incoming d_out into the two logical chunks
        d_out_L, d_out_R   = d_out_cat.split([qlen_L, qlen_R], dim=0)

        cp_size = get_world_size(cp_group)
        chunk_size = d_out_L.size(0)
        context_length = chunk_size * 2 * cp_size

        for i, (d_out, q_len, out, lse, kv_k, kv_v, cu_q, cu_k, max_q, max_k) in enumerate([
            (d_out_L, qlen_L, out_L, lse_L, k_L, v_L, cu_q_L, cu_k_L, maxq_L, maxk_L),
            (d_out_R, qlen_R, out_R, lse_R, k_R, v_R, cu_q_R, cu_k_R, maxq_R, maxk_R),
        ]):
            if i == 0:
                chunk_index = rank
            else:
                chunk_index = 2 * cp_size - 1 - rank

            dq_chunk = torch.zeros_like(local_q[:q_len])
            dk_chunk = torch.zeros_like(kv_k)
            dv_chunk = torch.zeros_like(kv_v)
            # debug_print(f"💜 dq_chunk = {dq_chunk.shape}")
            # debug_print(f"💜 dk_chunk = {dk_chunk.shape}")
            # debug_print(f"💜 dv_chunk = {dv_chunk.shape}")

            _ = _flash_attn_varlen_backward(
                d_out,
                local_q[ sum(ctx.q_chunk_sizes[:i]) : sum(ctx.q_chunk_sizes[:i+1]) ],
                kv_k, kv_v,
                out,
                lse,
                dq_chunk, dk_chunk, dv_chunk,
                cu_q, cu_k, int(max_q), int(max_k),
                0.0, ctx.softmax_scale, True, -1, -1, 0.0, None, False, None
            )

            dq_local[ sum(ctx.q_chunk_sizes[:i]) : sum(ctx.q_chunk_sizes[:i+1]) ] = dq_chunk
            local_idx = 0
            for start, end in kv_idx_list[i]:
                chunk_len = end - start
                dk_global[start:end] += dk_chunk[local_idx:local_idx + chunk_len] 
                dv_global[start:end] += dv_chunk[local_idx:local_idx + chunk_len] 
                local_idx += chunk_len

        if ENABLE_SHUFFLE:
            start_time__unshuffle = time.time()
            dk_global, dv_global = kv_unshuffle_for_per_doc_cp(context_length, dk_global, dv_global, doc_lens, doc_shards, cp_size)
            end_time__unshuffle = time.time()
            duration_ms__unshuffle = (end_time__unshuffle - start_time__unshuffle) * 1000
            debug_print(f"🟡 PerDocumentCPAttention kv_unshuffle_for_per_doc_cp time: {duration_ms__unshuffle} ms")
 
        # TODO: Fix GQA here...
        # now do reduce_scatter for dk/dv
        shape = list(dk_global.shape)
        shape[0] = shape[0] // cp_size
        # debug_print(f"💜 shape =", shape)

        # dk_local = torch.empty_like(dq_local)
        # dv_local = torch.empty_like(dq_local)
        dk_local = torch.empty(shape, device=dk_global.device, dtype=dk_global.dtype)
        dv_local = torch.empty(shape, device=dv_global.device, dtype=dv_global.dtype)
        # debug_print(f"💜 dk_global =", dk_global.shape, dk_global.dtype)
        # debug_print(f"💜 dv_global =", dv_global.shape, dv_global.dtype)
        # debug_print(f"💜 dk_local =", dk_local.shape, dk_local.dtype)
        # debug_print(f"💜 dv_local =", dv_local.shape, dv_local.dtype)

        with torch.cuda.stream(ctx.cp_stream):
            dk_global = dk_global.contiguous()
            dv_global = dv_global.contiguous()

            dist.reduce_scatter_tensor(dk_local, dk_global,
                                    op=dist.ReduceOp.SUM, group=cp_group)
            dist.reduce_scatter_tensor(dv_local, dv_global,
                                    op=dist.ReduceOp.SUM, group=cp_group)

        nvtx_range_pop()

        return (
            dq_local,        # grad w.r.t. local_q
            dk_local,        # grad w.r.t. local_k
            dv_local,        # grad w.r.t. local_v
            None,  # cu_seqlens_q_list
            None,  # cu_seqlens_kv_list
            None,  # max_seqlen_q_list
            None,  # max_seqlen_kv_list
            None,  # doc_lens
            None,  # doc_shards
            None,  # kv_idx_list
            None,  # dropout_p
            None,  # softmax_scale
            None,  # attn_mask_type
            None,  # cp_group
            None,  # cp_stream
            None,  # allgather_events
            None,  # allreduce_events
            None,  # attn_events
        )
