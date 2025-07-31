"""Test whether forward and backward metadata generation works correctly."""

import math

import torch

from d2.runtime.inplace_metadata import (
    Metadata, compute_attn_layout_seqlens, compute_metadata, compute_metadata_kv,
    exclusive_cumsum
)

from test_util import gen_seq_lens


@torch.no_grad()
def orchestrate_simulate(tensor: torch.Tensor, output_tensor: torch.Tensor, metadata: Metadata):
    assert tensor.dim() == 3    # (world_size, num_tokens, hidden_dim)
    world_size = tensor.shape[0]
    # handle sending rank-by-rank:
    for src_rank in range(world_size):
        dst_rank = metadata.dst_rank[src_rank]
        dst_offset = metadata.dst_offset[src_rank]
        seq_lens = metadata.seq_len[src_rank]
        acu_tokens = 0
        for j, rs in enumerate(dst_rank):
            seq_len = seq_lens[j]
            seq = tensor[src_rank][acu_tokens:acu_tokens + seq_len]
            if dst_rank.dim() == 1:
                rank = rs
                if rank >= 0:
                    try:
                        output_tensor[rank][dst_offset[j]: dst_offset[j] + seq_len] = seq
                    except RuntimeError as e:
                        print(f"{src_rank=}, {rank=}, {dst_offset[j]=}, {dst_offset[j] + seq_len=}, {seq_len=}, {output_tensor.shape, seq.shape, acu_tokens, tensor.shape}")
                        raise e
            else:
                for k, rank in enumerate(rs):
                    if rank >= 0:
                        output_tensor[rank][dst_offset[j][k]: dst_offset[j][k] + seq_len] = seq
            acu_tokens += seq_len
    return output_tensor


def test_query_dispatch(args):
    torch.manual_seed(args.seed)
    world_size = args.world_size
    num_seqs = args.num_seqs
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    hidden_size = args.hidden_size
    total_seq_len = args.num_tokens
    seq_len = gen_seq_lens(world_size, num_seqs, total_seq_len).long().to(device)
    global_dispatch = torch.randint(0, world_size, (world_size, num_seqs),
                                    device=device)

    tensor = torch.rand((world_size, total_seq_len, hidden_size), device=device) + 1    # + 1 to avoid zeros
    fwd_metadata, rev_metadata = compute_metadata(seq_len, global_dispatch)

    # forward
    max_recv_tokens = fwd_metadata.num_recv_tokens.max()
    output_tensor = torch.zeros((world_size, max_recv_tokens, hidden_size),
                                device=device, dtype=tensor.dtype)
    output_tensor = orchestrate_simulate(tensor, output_tensor, fwd_metadata)

    # reverse
    rev_metadata.num_recv_tokens.max()
    rev_tensor = torch.zeros((world_size, total_seq_len, hidden_size),
                             device=device, dtype=output_tensor.dtype)
    rev_tensor = orchestrate_simulate(output_tensor, rev_tensor, rev_metadata)
    rev_tensor = rev_tensor.reshape(world_size, total_seq_len, hidden_size)
    torch.testing.assert_close(tensor, rev_tensor)
    print("test metadata passed")


def create_qkv_dispatch(world_size: int, total_seq_len: int, num_seqs: int, max_cp_degree: int):
    """NOTE: this is currently a dispatch tensor of not consider the 2CP optimization."""
    # init sequence
    assert total_seq_len % (max_cp_degree) == 0
    _num_tokens_shard = total_seq_len // (max_cp_degree)
    seq_lens = gen_seq_lens(world_size, num_seqs, _num_tokens_shard).long()
    # make sure each sequence is divisible by max_cp_degree.
    seq_lens *= max_cp_degree

    # init cp degree for each sequence
    log_cp_num = torch.randint(0, int(math.log2(max_cp_degree)) + 1, (world_size, num_seqs))
    cp_num = torch.pow(2, log_cp_num)

    # init cp send dstination.
    cp_dst_helper = torch.rand((world_size, num_seqs, world_size)).argsort(dim=2)
    cp_dst = cp_dst_helper[:, :, :max_cp_degree]
    mask = torch.arange(max_cp_degree).expand(world_size, num_seqs, max_cp_degree)
    cp_num_expanded = cp_num.unsqueeze(-1)
    mask = mask >= cp_num_expanded
    cp_dst[mask] = -1

    # q_global_dispatch tensor:
    num_cp_shards = cp_num.sum(dim=1)
    pad_len = torch.max(num_cp_shards)
    cp_seq_lens = torch.zeros(world_size, pad_len, dtype=torch.int64)
    cp_query_dst = torch.ones(world_size, pad_len, dtype=torch.int64) * -1
    kv_to_q_mapping = torch.ones((world_size, pad_len, max_cp_degree, 2), dtype=torch.int64) * -1
    kv_to_q_rank = torch.ones((world_size, pad_len, max_cp_degree), dtype=torch.int64) * -1
    kv_context_size = torch.zeros((world_size, pad_len), dtype=torch.int64)
    num_kv_to_q = torch.zeros((world_size, pad_len), dtype=torch.int64)

    # cumulative number of cp shards before this one.
    num_cul_cp_shards = exclusive_cumsum(cp_num, dim=1)

    for i in range(world_size):
        cp_seq_lens_local = []
        cp_query_dst_local = []
        kv_to_q_mapping_local = []
        kv_to_q_rank_local = []
        kv_context_size_local = []
        num_kv_to_q_local = []

        for j in range(num_seqs):
            num_cp = int((cp_num[i, j]).item())
            seq_len = seq_lens[i, j]
            seq_shard_len = seq_len // num_cp

            cp_seq_lens_local.append(seq_shard_len.reshape(1,).repeat(num_cp))
            cp_query_dst_local.append(cp_dst[i, j, :num_cp].flatten())
            #### Compute kv_to_q_mapping.
            row_indices = torch.arange(num_cp).view(-1, 1)
            col_indices = torch.arange(max_cp_degree).view(1, -1)
            mask = col_indices < (num_cp - row_indices)
            kv_to_q_mapping_seq = torch.empty((num_cp, max_cp_degree, 2), dtype=torch.int64)
            # All q shards are on this node (TODO: we are testing MLP-DP. For MLP-CP, this is different).
            kv_to_q_mapping_seq[..., 0] = torch.where(mask, i, -1)
            vals_ch1 = row_indices + col_indices + num_cul_cp_shards[i, j]
            kv_to_q_mapping_seq[..., 1] = torch.where(mask, vals_ch1, -1)
            kv_to_q_mapping_local.append(kv_to_q_mapping_seq)
            #### Compute kv_to_q_rank (Index of this KV to the query's dst).
            kv_to_q_rank_seq = torch.arange(num_cp).view(-1, 1).repeat(1, max_cp_degree) * mask + (mask.int() - 1)
            kv_to_q_rank_local.append(kv_to_q_rank_seq)
            #### Compute kv context size (For this kv, how many tokens are in the context).
            kv_context_size_seq = torch.arange(num_cp) * seq_shard_len
            kv_context_size_local.append(kv_context_size_seq)
            #### Compute num_kv_to_q (For this kv, how many shards are in the context).
            num_kv_to_q_seq = torch.arange(num_cp) + 1
            num_kv_to_q_local.append(num_kv_to_q_seq)

        cp_seq_lens_local = torch.cat(cp_seq_lens_local, dim=0)
        cp_query_dst_local = torch.cat(cp_query_dst_local, dim=0)
        kv_to_q_mapping_local = torch.cat(kv_to_q_mapping_local, dim=0)
        kv_to_q_rank_local = torch.cat(kv_to_q_rank_local, dim=0)
        kv_context_size_local = torch.cat(kv_context_size_local, dim=0)
        num_kv_to_q_local = torch.cat(num_kv_to_q_local, dim=0)
        # shape check:
        seq_shards = cp_seq_lens_local.shape[0]
        assert cp_seq_lens_local.shape == (seq_shards,)
        assert cp_query_dst_local.shape == (seq_shards,)
        assert kv_to_q_mapping_local.shape == (seq_shards, max_cp_degree, 2)
        assert kv_to_q_rank_local.shape == (seq_shards, max_cp_degree)
        assert kv_context_size_local.shape == (seq_shards,)
        assert num_kv_to_q_local.shape == (seq_shards,)

        cp_seq_lens[i, :seq_shards] = cp_seq_lens_local
        cp_query_dst[i, :seq_shards] = cp_query_dst_local
        kv_to_q_mapping[i, :seq_shards] = kv_to_q_mapping_local
        kv_to_q_rank[i, :seq_shards] = kv_to_q_rank_local
        kv_context_size[i, :seq_shards] = kv_context_size_local
        num_kv_to_q[i, :seq_shards] = num_kv_to_q_local

    num_total_kv_to_q = kv_context_size + cp_seq_lens

    fwd_q_metadata, rev_q_metadata, intermediates = compute_metadata(
        cp_seq_lens, cp_query_dst, return_intermediate=True
    )
    _, q_seq_to_dst, _ = intermediates
    fwd_k_metadata, rev_k_metadata = compute_metadata_kv(
        kv_to_q_mapping, kv_to_q_rank, kv_context_size, num_kv_to_q,
        num_total_kv_to_q, cp_seq_lens, num_cp_shards, cp_query_dst,
        q_seq_to_dst.squeeze(2), pad_len
    )
    attention_metadata = compute_attn_layout_seqlens(
        cp_seq_lens, num_total_kv_to_q, cp_query_dst
    )
    return fwd_q_metadata, rev_q_metadata, fwd_k_metadata, rev_k_metadata, attention_metadata


def test_qkv_dispatch(args):
    world_size = args.world_size
    num_seqs = args.num_seqs
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    hidden_size = args.hidden_size
    total_seq_len = args.num_tokens
    max_cp_degree: int = args.max_seq_shard
    torch.manual_seed(args.seed)
    (fwd_q_metadata, rev_q_metadata, fwd_k_metadata, rev_k_metadata, _) = create_qkv_dispatch(
        world_size, total_seq_len, num_seqs, max_cp_degree
    )

    tensor = torch.rand((world_size, total_seq_len, hidden_size), device=device) + 1    # + 1 to avoid zeros
    # forward
    max_recv_tokens = fwd_q_metadata.num_recv_tokens.max()
    output_tensor = torch.zeros((world_size, max_recv_tokens, hidden_size),
                                device=device, dtype=tensor.dtype)
    output_tensor = orchestrate_simulate(tensor, output_tensor, fwd_q_metadata)

    # reverse
    rev_q_metadata.num_recv_tokens.max()
    rev_tensor = torch.zeros((world_size, total_seq_len, hidden_size),
                             device=device, dtype=output_tensor.dtype)
    rev_tensor = orchestrate_simulate(output_tensor, rev_tensor, rev_q_metadata)
    rev_tensor = rev_tensor.reshape(world_size, total_seq_len, hidden_size)
    torch.testing.assert_close(tensor, rev_tensor)

    # Test that key-value is correctly sent.
    tensor = torch.rand((world_size, total_seq_len, hidden_size), device=device) + 1
    max_recv_tokens_kv = fwd_k_metadata.num_recv_tokens.max()
    output_tensor = torch.zeros((world_size, max_recv_tokens_kv, hidden_size),
                                device=device, dtype=tensor.dtype)
    output_tensor = orchestrate_simulate(tensor, output_tensor, fwd_k_metadata)
    # reverse
    rev_tensor = torch.zeros((world_size, total_seq_len * max_cp_degree, hidden_size), device=device)
    rev_tensor = orchestrate_simulate(output_tensor, rev_tensor, rev_k_metadata)
    rev_tensor = rev_tensor.reshape(world_size, total_seq_len, max_cp_degree, hidden_size)
    rev_tensor_mask = rev_tensor != 0
    rev_tensor_dedup = rev_tensor.sum(dim=2) // rev_tensor_mask.sum(dim=2)
    torch.testing.assert_close(tensor, rev_tensor_dedup)
    torch.testing.assert_close(rev_tensor_mask * rev_tensor_dedup.unsqueeze(2), rev_tensor)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="Test metadata generation")
    parser.add_argument('--world_size', type=int, default=4)
    parser.add_argument('--num_seqs', type=int, default=4, help='Number of sequences per rank')
    parser.add_argument('--max_seq_shard', type=int, default=2, help='Number of shards per sequence')
    parser.add_argument('--num_tokens', type=int, default=1024, help='Total number of tokens per rank')
    parser.add_argument('--hidden_size', type=int, default=128, help='Hidden size of the tensor')
    parser.add_argument('--seed', type=int, default=0, help='Random seed for reproducibility')
    args = parser.parse_args()
    # test_query_dispatch(args)
    test_qkv_dispatch(args)
