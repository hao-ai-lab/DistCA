import argparse
import math
import os
import time

import torch

from d2.runtime.compute_metadata import from_planner_output, backward_from_planner_output
from d2.runtime.fast_alltoall_metadata import FastAlltoAllMetadata

from test_util import create_random_shard_info
from test_fa2a_metadata import (
    simulate_fa2a, simulate_fa2a_send_qkv, simulate_fa2a_send_qkv_rev,
    simulate_fa2a_recv_qkv, simulate_fa2a_recv_qkv_rev
)


def simulate_all2all(
    q: list[torch.Tensor], k: list[torch.Tensor], v: list[torch.Tensor],
    metadata: FastAlltoAllMetadata,
    element_size: int, hidden_q: int, hidden_k: int,
    is_from_linear_layout: bool,
):
    world_size = len(q)
    has_kv = k is not None
    if has_kv:
        assert world_size == len(k) == len(v)
        pad_memcpy_arg_for_no_kv = ()
    else:
        assert v is None
        pad_memcpy_arg_for_no_kv = (None, None)

    # sender transfer sz
    tot_send_bytes = metadata.fa2a_metadata[1].sum(dim=1) // element_size
    max_send_bytes = int(torch.max(tot_send_bytes).item())
    tot_recv_bytes = metadata.fa2a_metadata[3].sum(dim=1) // element_size
    max_recv_bytes = int(torch.max(tot_recv_bytes).item())
    dtype = q[0].dtype
    device = q[0].device
    src_buffer = torch.empty(
        (world_size, max_send_bytes), dtype=dtype, device=device
    )
    dst_buffer = torch.empty(
        (world_size, max_recv_bytes), dtype=dtype, device=device
    )
    for rank in range(world_size):
        metadata_local = metadata.get_slice(rank)
        max_cp = metadata_local.kv_replica_mask.shape[1] if has_kv else None

        fwd_args = ()
        if is_from_linear_layout:
            fn = simulate_fa2a_send_qkv
            fwd_args = (max_cp, metadata_local.kv_replica_mask)
        else:
            fn = simulate_fa2a_send_qkv_rev
        assert q[rank].shape == metadata_local.tensor_shape[0].send_shape, f"{q[rank].shape} vs {metadata_local.tensor_shape[0].send_shape}"
        # the first dim in metadata_local is the max cp

        if has_kv:
            assert k[rank].shape == metadata_local.tensor_shape[1].send_shape[-2:]
            assert v[rank].shape == k[rank].shape
            local_k = k[rank].flatten()
            local_v = v[rank].flatten()
            k_seqlens = metadata_local.seq_lens[1].send_seqlens
        else:
            local_k = local_v = None
            k_seqlens = None

        src_buffer[rank] = fn(
            q[rank].flatten(), local_k, local_v,
            src_buffer[rank],
            *metadata_local.send_memcpy_metadata,
            *pad_memcpy_arg_for_no_kv,
            metadata_local.seq_lens[0].send_seqlens,
            k_seqlens, hidden_q, hidden_k, element_size,
            *fwd_args,
        )
    dst_buffer = simulate_fa2a(
        src_buffer, dst_buffer, metadata.fa2a_metadata, element_size
    )
    dst_qs = []
    dst_ks = []
    dst_vs = []
    for rank in range(world_size):
        metadata_local = metadata.get_slice(rank)
        q_shape = metadata_local.tensor_shape[0].recv_shape
        dst_q = torch.empty(q_shape, dtype=q[rank].dtype, device=q[rank].device)
        if has_kv:
            k_shape = metadata_local.tensor_shape[1].recv_shape
            dst_k = torch.empty(k_shape, dtype=k[rank].dtype, device=k[rank].device)
            dst_v = torch.empty(k_shape, dtype=v[rank].dtype, device=v[rank].device)
        else:
            dst_k = dst_v = None

        bwd_args = ()
        max_cp = metadata_local.kv_replica_mask.shape[1] if has_kv else None
        if is_from_linear_layout:
            fn = simulate_fa2a_recv_qkv
            dst_q: torch.Tensor = dst_q.flatten()
            if has_kv:
                dst_k: torch.Tensor = dst_k.flatten()
                dst_v: torch.Tensor = dst_v.flatten()
        else:
            bwd_args = (max_cp, metadata_local.kv_replica_mask)
            fn = simulate_fa2a_recv_qkv_rev
            dst_q: torch.Tensor = dst_q.flatten()
            if has_kv:
                dst_k: torch.Tensor = dst_k.flatten(start_dim=1)
                dst_v: torch.Tensor = dst_v.flatten(start_dim=1)

        if has_kv:
            k_recv_seqlens = metadata_local.seq_lens[1].recv_seqlens
        else:
            k_recv_seqlens = None

        dst_q, dst_k, dst_v = fn(
            dst_q, dst_k, dst_v,
            dst_buffer[rank],
            *metadata_local.recv_memcpy_metadata,
            *pad_memcpy_arg_for_no_kv,
            metadata_local.seq_lens[0].recv_seqlens,
            k_recv_seqlens,
            hidden_q, hidden_k, element_size,
            *bwd_args,
        )
        if has_kv:
            dst_k = dst_k.reshape(k_shape)
            dst_v = dst_v.reshape(k_shape)
        dst_qs.append(dst_q.reshape(q_shape))
        dst_ks.append(dst_k)
        dst_vs.append(dst_v)
    return dst_qs, dst_ks, dst_vs


def seq_mask_to_token_mask(mask: torch.Tensor, seqlens: torch.Tensor):
    num_seqs, cp_degree = mask.shape
    assert seqlens.shape == (num_seqs,)
    num_token: int = seqlens.sum().item()
    token_mask = torch.zeros((cp_degree, num_token), dtype=mask.dtype, device=mask.device)
    cur_token = 0
    for seq in range(num_seqs):
        seqlen = seqlens[seq].item()
        for cp in range(cp_degree):
            token_mask[cp, cur_token:cur_token + seqlen] = mask[seq, cp]
        cur_token += seqlen
    return token_mask


def test(args):
    seed = args.seed
    num_doc = args.num_doc
    max_num_shard = args.max_num_shard
    max_shard_len = args.max_shard_len
    min_shard_len = args.min_shard_len
    simulate = args.simulate_world_size > 0
    world_size = args.simulate_world_size if simulate else int(os.environ.get("WORLD_SIZE"))
    # TODO: if not simulate, launch workers to run all-to-all
    assert simulate, "test with real op not supported yet"
    assert world_size >= max_num_shard

    hidden_size_q = args.hidden_size_q
    hidden_size_k = args.hidden_size_k
    num_head = args.num_head
    element_size = torch.bfloat16.itemsize
    INT_4_BYTES = 16
    lse_size = math.ceil(num_head * torch.float32.itemsize / INT_4_BYTES) * INT_4_BYTES // element_size

    scheduler_output, src_num_token = create_random_shard_info(
        seed, world_size, num_doc, max_num_shard, max_shard_len, min_shard_len
    )

    tik = time.time()
    fwd_qkv_metadata, bwd_qkv_metadata, fwd_attn_out_metadata, bwd_attn_out_metadata = from_planner_output(
        world_size, scheduler_output, hidden_size_q, hidden_size_k, lse_size, element_size,
        is_pipeline_tick=False
    )
    tok = time.time()
    print("gen metadata time:", tok - tik)

    src_qs = [
        torch.rand((src_num_token[rank], hidden_size_q), dtype=torch.bfloat16)
        for rank in range(world_size)
    ]
    src_ks = [
        torch.rand((src_num_token[rank], hidden_size_k), dtype=torch.bfloat16)
        for rank in range(world_size)
    ]
    src_vs = [
        torch.rand((src_num_token[rank], hidden_size_k), dtype=torch.bfloat16)
        for rank in range(world_size)
    ]
    # 1. without pp, send with fwd_qkv_metadata and bwd_qkv_metadata, examine if received is the same as sent
    dst_qs, dst_ks, dst_vs = simulate_all2all(
        src_qs, src_ks, src_vs, fwd_qkv_metadata, element_size, hidden_size_q, hidden_size_k,
        is_from_linear_layout=True,
    )
    rev_qs, rev_ks, rev_vs = simulate_all2all(
        dst_qs, dst_ks, dst_vs, bwd_qkv_metadata, element_size, hidden_size_q, hidden_size_k,
        is_from_linear_layout=False,
    )
    # verify answer
    torch.testing.assert_close(src_qs, rev_qs)
    print("forward without pipeline, qs are correct after sending forward and back.")

    kv_mask = fwd_qkv_metadata.kv_replica_mask
    for i in range(world_size):
        mask = kv_mask[i]
        rank_fwd_metadata = fwd_qkv_metadata.get_slice(i)
        torch.testing.assert_close(mask, bwd_qkv_metadata.kv_replica_mask[i])
        token_mask = seq_mask_to_token_mask(mask, rank_fwd_metadata.seq_lens[1].send_seqlens)
        rev_k: torch.Tensor = rev_ks[i]
        assert rev_k.shape[:2] == token_mask.shape
        token_mask = token_mask.unsqueeze(-1)
        expand_src_k = src_ks[i].unsqueeze(0) * token_mask
        torch.testing.assert_close(expand_src_k, rev_k)

        rev_v = rev_vs[i]
        expand_src_v = src_vs[i].unsqueeze(0) * token_mask
        torch.testing.assert_close(expand_src_v, rev_v)
    print("forward without pipeline, kv are correct after sending forward and back")

    # 2. without pp, send with fwd_attn_out_metadata and bwd_att_out_metadata, examine if received is the same as sent
    attn_out_attn_layout = dst_qs
    hidden_attn_out = hidden_size_q
    recv_attn_out, _, _ = simulate_all2all(
        attn_out_attn_layout, None, None, fwd_attn_out_metadata,
        element_size, hidden_attn_out, None, is_from_linear_layout=False
    )
    torch.testing.assert_close(src_qs, recv_attn_out)
    print("forward without pipeline, attn out forward is correct")
    # let grad equal itself for simplicity.
    attn_out_grad_linear_layout = recv_attn_out
    attn_out_grad_attn_layout, _, _ = simulate_all2all(
        attn_out_grad_linear_layout, None, None, bwd_attn_out_metadata,
        element_size, hidden_attn_out, None, is_from_linear_layout=True
    )
    torch.testing.assert_close(attn_out_grad_attn_layout, attn_out_attn_layout)
    print("forward without pipeline, attn out backward is correct")

    # 3. with PP, resend qkv and test correctness
    # Randomly generate input
    attn_outs_attn_layout = [torch.rand_like(q) for q in dst_qs]
    lse_attn_layout = [torch.rand_like(q)[:, :lse_size] for q in dst_qs]
    # For attn out, simulate an reference answer.
    recv_attn_outs_ref, _, _ = simulate_all2all(
        attn_outs_attn_layout, None, None, fwd_attn_out_metadata,
        element_size, hidden_attn_out, None, is_from_linear_layout=False
    )
    # update metadata to the fused version
    hidden_attn_out_merged = hidden_size_q + lse_size
    fwd_qkv_metadata, _, fwd_attn_out_metadata, _ = from_planner_output(
        world_size, scheduler_output, hidden_size_q, hidden_size_k, lse_size, element_size,
        is_pipeline_tick=True
    )
    # We use the same scheduler output for backward to verify correctness. It should be different
    qkv_resend_and_out_grad_linear_to_attn, qkv_grad_attn_to_linear = backward_from_planner_output(
        world_size, scheduler_output, hidden_size_q, hidden_size_k, lse_size, element_size
    )
    merged_attn_outs_lse_attn_layout = [
        torch.concat([ao, l], dim=1) for ao, l in zip(attn_outs_attn_layout, lse_attn_layout)
    ]
    merged_attn_outs_lse_linear_layout, _, _ = simulate_all2all(
        merged_attn_outs_lse_attn_layout, None, None, fwd_attn_out_metadata,
        element_size, hidden_attn_out_merged, None, is_from_linear_layout=False
    )
    # test the new attn outs AttnLayout -> LinearLayout
    attn_outs_linear_layout = [
        t[:, :hidden_attn_out] for t in merged_attn_outs_lse_linear_layout
    ]
    torch.testing.assert_close(recv_attn_outs_ref, attn_outs_linear_layout)
    print("with pipeline, attn outs & lse forward is correct.")
    # backward AttnOutGrad and AttnOut and lse and qkv
    attn_out_grad_linear_layout = attn_outs_linear_layout
    merged_attn_out_grad_resend_q_linear_layout = [
        torch.concat([aog, aol, q], dim=1) for aog, aol, q in
        zip(attn_out_grad_linear_layout, merged_attn_outs_lse_linear_layout, src_qs)
    ]
    hidden_size_q_merged = hidden_size_q * 3 + lse_size
    merged_attn_out_grad_resend_q_attn_layout, k_resends, v_resends = simulate_all2all(
        merged_attn_out_grad_resend_q_linear_layout, src_ks, src_vs, qkv_resend_and_out_grad_linear_to_attn,
        element_size, hidden_size_q_merged, hidden_size_k, is_from_linear_layout=True,
    )
    torch.testing.assert_close(k_resends, dst_ks)
    torch.testing.assert_close(v_resends, dst_vs)
    print("backward resend kv is correct")
    attn_out_grad_attn_layout = [
        t[:, :hidden_attn_out] for t in merged_attn_out_grad_resend_q_attn_layout
    ]
    torch.testing.assert_close(attn_out_grad_attn_layout, attn_outs_attn_layout)
    print("backward attn out grad is correct")
    attn_out_and_lse_attn_layout = [
        t[:, hidden_attn_out:hidden_attn_out + hidden_attn_out_merged]
        for t in merged_attn_out_grad_resend_q_attn_layout
    ]
    torch.testing.assert_close(attn_out_and_lse_attn_layout, merged_attn_outs_lse_attn_layout)
    print("backward resend attn out and lse is correct")
    resend_qs = [
        t[:, -hidden_size_q:] for t in merged_attn_out_grad_resend_q_attn_layout
    ]
    torch.testing.assert_close(resend_qs, dst_qs)
    print("backward resend q is correct")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--simulate-world-size", type=int, default=2)
    parser.add_argument("--num-doc", type=int, default=4)
    parser.add_argument("--max-num-shard", type=int, default=2)
    parser.add_argument("--max-shard-len", type=int, default=128)
    parser.add_argument("--min-shard-len", type=int, default=2)
    parser.add_argument("--hidden-size-q", type=int, default=512)
    parser.add_argument("--hidden-size-k", type=int, default=128)
    parser.add_argument("--num-head", type=int, default=2)
    args = parser.parse_args()

    test(args)
