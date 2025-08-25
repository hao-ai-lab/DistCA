"""TODO: deprecate this file."""
import torch

from test_util import (
    create_qkv_dispatch_2cp, create_fast_a2a_metadata_from_qkv_dispatch,
    orchestrate_simulate
)
from test_fa2a_metadata import simulate_qkv_a2a


def test(args):
    world_size = args.world_size
    num_seqs = args.num_seqs
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    hidden_size_q = args.hidden_size_q
    hidden_size_k = args.hidden_size_k
    total_seq_len = args.num_tokens
    max_cp_degree: int = args.max_seq_shard
    torch.manual_seed(args.seed)
    (fwd_q_metadata, rev_q_metadata, fwd_k_metadata, rev_k_metadata,
     _, intermediates
     ) = create_qkv_dispatch_2cp(
        world_size, total_seq_len, num_seqs, max_cp_degree, return_intermediate=True
    )

    tensor_q = torch.rand((world_size, total_seq_len, hidden_size_q), device=device) + 1    # + 1 to avoid zeros
    tensor_k = torch.rand((world_size, total_seq_len, hidden_size_k), device=device) + 1
    tensor_v = torch.rand((world_size, total_seq_len, hidden_size_k), device=device) + 1
    max_recv_tokens_q = int(fwd_q_metadata.num_recv_tokens.max().item())
    max_recv_tokens_k = int(fwd_k_metadata.num_recv_tokens.max().item())
    output_tensor_q = torch.zeros((world_size, max_recv_tokens_q, hidden_size_q),
                                device=device, dtype=tensor_q.dtype)
    output_tensor_k = torch.zeros((world_size, max_recv_tokens_k, hidden_size_k),
                                device=device, dtype=tensor_k.dtype)
    output_tensor_v = output_tensor_k.clone()

    # ground truth.
    output_tensor_q = orchestrate_simulate(tensor_q, output_tensor_q, fwd_q_metadata)
    output_tensor_k = orchestrate_simulate(tensor_k, output_tensor_k, fwd_k_metadata)
    output_tensor_v = orchestrate_simulate(tensor_v, output_tensor_v, fwd_k_metadata)
    print("correct answer done.")
    rev_tensor_q = torch.zeros((world_size, total_seq_len, hidden_size_q),
                             device=device, dtype=output_tensor_q.dtype)
    rev_tensor_k = torch.zeros((world_size, total_seq_len * max_cp_degree, hidden_size_k), device=device)
    rev_tensor_v = torch.zeros((world_size, total_seq_len * max_cp_degree, hidden_size_k), device=device)

    rev_tensor_q = orchestrate_simulate(output_tensor_q, rev_tensor_q, rev_q_metadata)
    rev_tensor_k = orchestrate_simulate(output_tensor_k, rev_tensor_k, rev_k_metadata)
    rev_tensor_v = orchestrate_simulate(output_tensor_v, rev_tensor_v, rev_k_metadata)

    rev_tensor_q = rev_tensor_q.reshape(world_size, total_seq_len, hidden_size_q)
    rev_tensor_k = rev_tensor_k.reshape(world_size, max_cp_degree, total_seq_len, hidden_size_k)
    rev_tensor_v = rev_tensor_v.reshape(world_size, max_cp_degree, total_seq_len, hidden_size_k)
    print("rev correct answer done")

    (q_tokens_to_dst_per_dispatch, q_seq_to_dst,
     _, kv_dst_global_seq_id) = intermediates
    element_size = tensor_q.element_size()
    qkv_fwd_fa2a_metadata, qkv_rev_fa2a_metadata = \
        create_fast_a2a_metadata_from_qkv_dispatch(
            fwd_q_metadata, rev_q_metadata, fwd_k_metadata, rev_k_metadata,
            intermediates, element_size, hidden_size_q, hidden_size_k,
            total_seq_len,
        )

    fa2a_q, fa2a_k, fa2a_v = simulate_qkv_a2a(
        tensor_q, tensor_k, tensor_v,
        qkv_fwd_fa2a_metadata, fwd_q_metadata, fwd_k_metadata,
        rev_q_metadata, rev_k_metadata,
        element_size, hidden_size_q, hidden_size_k,
        max_cp_degree, fwd_k_metadata.dst_rank >= 0,
        is_fwd=True
    )
    torch.testing.assert_close(output_tensor_q, fa2a_q)
    torch.testing.assert_close(output_tensor_k, fa2a_k)
    torch.testing.assert_close(output_tensor_v, fa2a_v)
    print("pass forward send qkv")
    fa2a_rev_q, fa2a_rev_k, fa2a_rev_v = simulate_qkv_a2a(
        fa2a_q, fa2a_k, fa2a_v,
        qkv_rev_fa2a_metadata, rev_q_metadata, rev_k_metadata,
        fwd_q_metadata, fwd_k_metadata,
        element_size, hidden_size_q, hidden_size_k,
        max_cp_degree, fwd_k_metadata.dst_rank >= 0,
        is_fwd=False
    )
    torch.testing.assert_close(rev_tensor_q, fa2a_rev_q)
    torch.testing.assert_close(rev_tensor_k, fa2a_rev_k.reshape_as(rev_tensor_k))
    torch.testing.assert_close(rev_tensor_v, fa2a_rev_v.reshape_as(rev_tensor_v))
    print("pass reverse recv qkv")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--world_size', type=int, default=2)
    parser.add_argument('--num_seqs', type=int, default=4)
    parser.add_argument('--num_tokens', type=int, default=1024)
    parser.add_argument('--hidden_size_q', type=int, default=256)
    parser.add_argument('--hidden_size_k', type=int, default=128)
    parser.add_argument('--max_seq_shard', type=int, default=4)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()
    test(args)

"""
python test_fa2a_metadata_2cp.py --world_size 2 --max_seq_shard 4
python test_fa2a_metadata_2cp.py --world_size 4 --max_seq_shard 8
"""