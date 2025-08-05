"""Test whether forward and backward metadata generation works correctly."""

import torch
import rich

from d2.runtime.inplace_metadata import (
    Metadata, compute_metadata
)

from test_util import (
    gen_seq_lens, 
    create_qkv_dispatch, 
    orchestrate_simulate, 
    create_qkv_dispatch_balanced_flops,
)

def test_query_dispatch(args):
    rich.print("⚪ Testing query dispatch")

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
    max_cp_degree: int = args.max_seq_shard
    torch.manual_seed(args.seed)

    rich.print("⚪ Testing qkv dispatch")

    # TODO: Make sure the metadata is correctly defined via the interface.
    (fwd_q_metadata, rev_q_metadata, fwd_k_metadata, rev_k_metadata, _) = create_qkv_dispatch_balanced_flops(
        world_size, total_seq_len, num_seqs, max_cp_degree,
        verbose=True,
    )

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


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="Test metadata generation")
    parser.add_argument('--world_size', type=int, default=2)
    parser.add_argument('--num_seqs', type=int, default=4, help='Number of sequences per rank')
    parser.add_argument('--max_seq_shard', type=int, default=4, help='Number of shards per sequence')
    parser.add_argument('--num_tokens', type=int, default=1024, help='Total number of tokens per rank')
    parser.add_argument('--hidden_size', type=int, default=128, help='Hidden size of the tensor')
    parser.add_argument('--seed', type=int, default=0, help='Random seed for reproducibility')
    args = parser.parse_args()
    rich.print(f"Testing {__file__} with args =", args)
    
    create_qkv_dispatch_balanced_flops(
        world_size=args.world_size, 
        total_seq_len=args.num_tokens, 
        num_seqs=args.num_seqs, 
        max_cp_degree=args.max_seq_shard,
        return_intermediate=True,
        return_mlp_no_shard_seq_lens=True,
    )
    test_query_dispatch(args)
    test_qkv_dispatch(args)
