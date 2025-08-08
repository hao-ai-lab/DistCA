"""Test whether forward and backward metadata generation works correctly."""

import random
from typing import List

import rich
import torch
from test_util import (create_qkv_dispatch, create_qkv_dispatch_2cp,
                       gen_seq_lens, orchestrate_simulate)

from d2.runtime.inplace_metadata import Metadata, compute_metadata
from d2.runtime.shard_info import ShardInfo, plan_to_metadata


def test_query_dispatch(args):
    rich.print("🟢 Testing query dispatch")

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
    rich.print("🟢 Test query dispatch passed")


def test_qkv_dispatch(args):
    world_size = args.world_size
    num_seqs = args.num_seqs
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    hidden_size = args.hidden_size
    total_seq_len = args.num_tokens
    #max_cp_degree: int = args.max_seq_shard
    
    torch.manual_seed(args.seed)

    rich.print("⚪ Testing qkv dispatch")

    (fwd_q_metadata, rev_q_metadata, fwd_k_metadata, rev_k_metadata, _), sequence_plans = test_plan_to_metadata(args)
    max_cp_degree:int = max(len(s) for s in sequence_plans)
    rich.print(f"max_cp_degree: {max_cp_degree}")
    
    # Test that Query is correctly sent.
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
    rev_tensor = rev_tensor.reshape(world_size, max_cp_degree, total_seq_len, hidden_size)

    rev_tensor_mask = rev_tensor != 0
    rev_tensor_dedup = rev_tensor.sum(dim=1) / rev_tensor_mask.int().sum(dim=1)

    torch.testing.assert_close(rev_tensor_mask * tensor.unsqueeze(1), rev_tensor)
    torch.testing.assert_close(tensor, rev_tensor_dedup)
    rich.print("🟢 Test qkv dispatch passed")


def get_sequence_plans_fixed():
    world_size = 8
    token_per_rank = 64
    sequence_plans: List[List[ShardInfo]] = [
    [
        ShardInfo(rid=4, dispatch_rid=4, logical_sid=0, shard_len=16),
        ShardInfo(rid=5, dispatch_rid=5, logical_sid=1, shard_len=16),
        ShardInfo(rid=6, dispatch_rid=6, logical_sid=2, shard_len=16),
        ShardInfo(rid=7, dispatch_rid=7, logical_sid=3, shard_len=16),
        ShardInfo(rid=7, dispatch_rid=7, logical_sid=4, shard_len=16),
        ShardInfo(rid=6, dispatch_rid=6, logical_sid=5, shard_len=16),
        ShardInfo(rid=5, dispatch_rid=5, logical_sid=6, shard_len=16),
        ShardInfo(rid=4, dispatch_rid=4, logical_sid=7, shard_len=16)
    ],
    [
        ShardInfo(rid=4, dispatch_rid=4, logical_sid=0, shard_len=16),
        ShardInfo(rid=5, dispatch_rid=5, logical_sid=1, shard_len=16),
        ShardInfo(rid=6, dispatch_rid=6, logical_sid=2, shard_len=16),
        ShardInfo(rid=7, dispatch_rid=7, logical_sid=3, shard_len=16),
        ShardInfo(rid=7, dispatch_rid=7, logical_sid=4, shard_len=16),
        ShardInfo(rid=6, dispatch_rid=6, logical_sid=5, shard_len=16),
        ShardInfo(rid=5, dispatch_rid=5, logical_sid=6, shard_len=16),
        ShardInfo(rid=4, dispatch_rid=4, logical_sid=7, shard_len=16)
    ],
    [
        ShardInfo(rid=0, dispatch_rid=0, logical_sid=0, shard_len=32),
        ShardInfo(rid=1, dispatch_rid=1, logical_sid=1, shard_len=32),
        ShardInfo(rid=1, dispatch_rid=1, logical_sid=2, shard_len=32),
        ShardInfo(rid=0, dispatch_rid=0, logical_sid=3, shard_len=32)
    ],
    [
        ShardInfo(rid=2, dispatch_rid=2, logical_sid=0, shard_len=32),
        ShardInfo(rid=3, dispatch_rid=3, logical_sid=1, shard_len=32),
        ShardInfo(rid=3, dispatch_rid=3, logical_sid=2, shard_len=32),
        ShardInfo(rid=2, dispatch_rid=2, logical_sid=3, shard_len=32)
    ]
    ]
    return sequence_plans


def test_plan_to_metadata(args):
    world_size = args.world_size
    num_seqs = args.num_seqs
    total_seq_len = args.num_tokens
    sequence_plans = get_sequence_plans(world_size, num_seqs, total_seq_len)
    #sequence_plans = get_sequence_plans_fixed()
    print(sequence_plans)
    result = plan_to_metadata(args.world_size, sequence_plans)

    return result, sequence_plans

# Generate Random plans for testing.
def get_sequence_plans(
    world_size: int,
    num_seqs: int,
    total_len: int
) -> List[List[ShardInfo]]:
    all_shards = []
    for rid in range(world_size):
        num_shards_on_rank = random.randint(2, world_size * 2)
        ratios = torch.rand(num_shards_on_rank) + 0.1
        ratios /= ratios.sum()
        shard_lens_on_rank = (ratios * total_len).round().int()
        error = shard_lens_on_rank.sum() - total_len
        shard_lens_on_rank[-1] -= error
        for shard_len_tensor in shard_lens_on_rank:
            shard_len = int(shard_len_tensor.item())
            if shard_len <= 0:
                continue
            shard = ShardInfo(
                rid=rid,
                dispatch_rid=-1,
                logical_sid=-1,
                shard_len=shard_len
            )
            all_shards.append(shard)
    if len(all_shards) < num_seqs:
        raise ValueError(
            f"Total number of generated shards ({len(all_shards)}) is not enough to assign to each sequence ({num_seqs})."
            "Consider increasing world_size or adjusting the number of shards generated per rank."
        )
    random.shuffle(all_shards)
    shards_by_sequence = [[] for _ in range(num_seqs)]
    for i, shard in enumerate(all_shards):
        assigned_sid = i % num_seqs
        shard.dispatch_rid = random.randint(0, world_size - 1)
        shards_by_sequence[assigned_sid].append(shard)
    for seq_id in range(num_seqs):
        current_seq_shards = shards_by_sequence[seq_id]
        current_seq_shards.sort(key=lambda s: s.rid)
        for logical_id, shard in enumerate(current_seq_shards):
            shard.logical_sid = logical_id
    return shards_by_sequence



if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="Test metadata generation")
    parser.add_argument('--world_size', type=int, default=8)
    parser.add_argument('--num_seqs', type=int, default=4, help='Number of sequences per rank')
    parser.add_argument('--max_seq_shard', type=int, default=8, help='Number of shards per sequence')
    parser.add_argument('--num_tokens', type=int, default=64, help='Total number of tokens per rank')
    parser.add_argument('--hidden_size', type=int, default=128, help='Hidden size of the tensor')
    parser.add_argument('--seed', type=int, default=0, help='Random seed for reproducibility')
    args = parser.parse_args()
    rich.print(f"Testing {__file__} with args =", args)

    world_size = args.world_size
    total_seq_len = args.num_tokens
    num_seqs = args.num_seqs

    test_query_dispatch(args)
    test_qkv_dispatch(args)
