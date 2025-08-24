import argparse
import os
import random

import torch

from d2.runtime.compute_metadata import from_shard_info
from d2.runtime.fast_alltoall_metadata import FastAlltoAllMetadata
from d2.runtime.shard_info import ShardInfo

from test_util import set_random_seed
from test_fa2a_metadata import (
    simulate_fa2a, simulate_fa2a_send_qkv, simulate_fa2a_send_qkv_rev,
    simulate_fa2a_recv_qkv, simulate_fa2a_recv_qkv_rev
)

def create_random_shard_info(
    seed: int, world_size: int, num_doc: int,
    max_num_shard: int, max_shard_len: int, min_shard_len: int=8
):
    set_random_seed(seed)
    scheduler_output = []
    num_shards = torch.randint(1, max_num_shard + 1, (num_doc,)).tolist()
    has_shard_src = [False] * world_size
    has_shard_dst = [False] * world_size
    src_num_token = [0] * world_size
    for doc_id in range(num_doc):
        num_shard = num_shards[doc_id]
        doc_schedule = []
        for shard_id in range(num_shard):
            rid = random.randint(0, world_size - 1)
            d_rid = random.randint(0, world_size - 1)
            has_shard_src[rid] = True
            has_shard_dst[d_rid] = True
            shard_len = random.randint(min_shard_len, max_shard_len)
            src_num_token[rid] += shard_len
            doc_schedule.append(
                ShardInfo(rid=rid, dispatch_rid=d_rid, logical_sid=shard_id, shard_len=shard_len)
            )
        scheduler_output.append(doc_schedule)
    for rank in range(world_size):
        if not has_shard_src[rank]:
            scheduler_output.append([ShardInfo(rid=rank, dispatch_rid=rank, logical_sid=0, shard_len=min_shard_len)])
            has_shard_src[rank] = True
            has_shard_dst[rank] = True
            src_num_token[rank] += min_shard_len
        if not has_shard_dst[rank]:
            scheduler_output.append([ShardInfo(rid=rank, dispatch_rid=rank, logical_sid=0, shard_len=min_shard_len)])
            has_shard_src[rank] = True
            has_shard_dst[rank] = True
            src_num_token[rank] += min_shard_len
    return scheduler_output, src_num_token


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


def create_redispatch_info(
    seed: int, fwd_dispatch: list[list[ShardInfo]], world_size: int,
    max_num_shard: int, max_shard_len: int, min_shard_len: int=8,
):
    """
    TODO: fix the doc length from the given fwd_dispatch, create a new bwd dispatch.
    However, it should be guaranteed that each src and dst must have at least one shard.
    """
    raise NotImplementedError("")


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
    is_pp = args.is_pp
    num_head = args.num_head
    element_size = torch.bfloat16.itemsize
    lse_size = num_head * torch.float32.itemsize // element_size

    scheduler_output, src_num_token = create_random_shard_info(
        seed, world_size, num_doc, max_num_shard, max_shard_len, min_shard_len
    )

    assert not is_pp, "not supported yet"
    if is_pp:
        scheduler_output_bwd = create_redispatch_info(seed, scheduler_output, world_size)
    else:
        scheduler_output_bwd = None
    fwd_qkv_metadata, bwd_qkv_metadata, fwd_attn_out_metadata, bwd_attn_out_metadata = from_shard_info(
        world_size, scheduler_output, hidden_size_q, hidden_size_k, lse_size, element_size,
        is_pp
    )
    fwd_qkv_metadata: FastAlltoAllMetadata

    src_q = [
        torch.rand((src_num_token[rank], hidden_size_q), dtype=torch.bfloat16)
        for rank in range(world_size)
    ]
    src_k = [
        torch.rand((src_num_token[rank], hidden_size_k), dtype=torch.bfloat16)
        for rank in range(world_size)
    ]
    src_v = [
        torch.rand((src_num_token[rank], hidden_size_k), dtype=torch.bfloat16)
        for rank in range(world_size)
    ]
    # 1. without pp, send with fwd_qkv_metadata and bwd_qkv_metadata, examine if received is the same as sent
    dst_qs, dst_ks, dst_vs = simulate_all2all(
        src_q, src_k, src_v, fwd_qkv_metadata, element_size, hidden_size_q, hidden_size_k,
        is_from_linear_layout=True,
    )
    rev_qs, rev_ks, rev_vs = simulate_all2all(
        dst_qs, dst_ks, dst_vs, bwd_qkv_metadata, element_size, hidden_size_q, hidden_size_k,
        is_from_linear_layout=False,
    )
    # verify answer
    torch.testing.assert_close(src_q, rev_qs)
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
        expand_src_k = src_k[i].unsqueeze(0) * token_mask
        torch.testing.assert_close(expand_src_k, rev_k)

        rev_v = rev_vs[i]
        expand_src_v = src_v[i].unsqueeze(0) * token_mask
        torch.testing.assert_close(expand_src_v, rev_v)
    print("forward without pipeline, kv are correct after sending forward and back")

    # 2. without pp, send with fwd_attn_out_metadata and bwd_att_out_metadata, examine if received is the same as sent
    attn_out_attn_layout = dst_qs
    hidden_attn_out = hidden_size_q
    recv_attn_out, _, _ = simulate_all2all(
        attn_out_attn_layout, None, None, fwd_attn_out_metadata,
        element_size, hidden_attn_out, None, is_from_linear_layout=False
    )
    torch.testing.assert_close(src_q, recv_attn_out)
    print("forward without pipeline, attn out forward is correct")
    bwd_attn_out_grad, _, _ = simulate_all2all(
        recv_attn_out, None, None, bwd_attn_out_metadata,
        element_size, hidden_attn_out, None, is_from_linear_layout=True
    )
    torch.testing.assert_close(bwd_attn_out_grad, attn_out_attn_layout)
    print("forward without pipeline, attn out backward is correct")
    # with pp, send with the four metadata, examine if received is the same as sent.


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
    parser.add_argument("--is-pp", type=bool, default=False)
    args = parser.parse_args()

    test(args)
