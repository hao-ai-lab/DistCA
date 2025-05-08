import torch
from itertools import accumulate

import torch.distributed as dist

from torch.distributed import (
    get_world_size,
    get_rank,
)

from torch.cuda.nvtx import range_push as nvtx_range_push
from torch.cuda.nvtx import range_pop  as nvtx_range_pop


from flash_attn.flash_attn_interface import (
    flash_attn_varlen_func,
    _flash_attn_varlen_forward,
    _flash_attn_varlen_backward
)

from utils import (
    kv_shuffle_for_per_doc_cp,
    kv_unshuffle_for_per_doc_cp,
    compute_per_doc_cp_shard_doc_len,
)

def cat_slices(tensor, starts, lens):
            if starts.numel() == 1:
                s, l = int(starts[0]), int(lens[0])
                return tensor[s:s+l]
            return torch.cat([tensor[int(s):int(s)+int(l)]
                              for s, l in zip(starts.tolist(), lens.tolist())],
                             dim=0)   

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
        cp_stream
    ):
        nvtx_range_push("PerDocumentCPAttention.fwd")
        assert attn_mask_type == "causal", "Only causal attention is supported"
        assert cp_group is not None, "cp_group must be provided"
        assert cp_stream is not None, "cp_stream must be provided"

        cp_size = get_world_size(cp_group)
        rank = get_rank(cp_group)
        context_length = local_q.shape[0] * cp_size
        
        # allgather kv, then shuffle back to global order
        gather_k_list = [torch.empty_like(local_q) for _ in range(cp_size)]
        gather_v_list = [torch.empty_like(local_q) for _ in range(cp_size)]
        with torch.cuda.stream(cp_stream):
            local_k = local_k.contiguous()
            local_v = local_v.contiguous()
            dist.all_gather(gather_k_list, local_k, group=cp_group)
            dist.all_gather(gather_v_list, local_v, group=cp_group)

        k_global, v_global = kv_shuffle_for_per_doc_cp(context_length, gather_k_list, gather_v_list, doc_lens, doc_shards, cp_size)

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
                return_attn_probs=True
            )
            
            outputs.append(out)
            lses.append(lse)

        final_out = torch.cat(outputs, dim=0)
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
        return final_out

    @staticmethod
    def backward(ctx, d_out_cat):
        """
        Backward pass for PerDocumentCPAttention.
        """
        nvtx_range_push("PerDocumentCPAttention.bwd")

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

            _ = _flash_attn_varlen_backward(
                d_out,
                local_q[ sum(ctx.q_chunk_sizes[:i]) : sum(ctx.q_chunk_sizes[:i+1]) ],
                kv_k, kv_v,
                out,
                lse,
                dq_chunk, dk_chunk, dv_chunk,
                cu_q, cu_k, int(max_q), int(max_k),
                0.0, ctx.softmax_scale, True, (-1,-1), None, False, None
            )

            dq_local[ sum(ctx.q_chunk_sizes[:i]) : sum(ctx.q_chunk_sizes[:i+1]) ] = dq_chunk
            local_idx = 0
            for start, end in kv_idx_list[i]:
                chunk_len = end - start
                dk_global[start:end] += dk_chunk[local_idx:local_idx + chunk_len] 
                dv_global[start:end] += dv_chunk[local_idx:local_idx + chunk_len] 
                local_idx += chunk_len

        dk_global, dv_global = kv_unshuffle_for_per_doc_cp(context_length, dk_global, dv_global, doc_lens, doc_shards, cp_size)
 
        # now do reduce_scatter for dk/dv
        dk_local = torch.empty_like(dq_local)
        dv_local = torch.empty_like(dq_local)
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
        )

