from typing import Sequence

import torch
from torch import Tensor

from d2.runtime.inplace_metadata import Metadata
from d2.runtime.fast_alltoall_metadata import (
    FastAlltoAllMetadata
)

from test_util import (
    create_qkv_dispatch_2cp, create_fast_a2a_metadata_from_qkv_dispatch,
    orchestrate_simulate
)
from test_fa2a_metadata import simulate_qkv_a2a


from test_comm_metadata_balanced_flops import (
    create_qkv_dispatch_with_custom_mapping,
)

def test_create_qkv_dispatch_balanced_flops(
    world_size_, total_seq_len_, num_seqs_, max_cp_degree_, 
    verbose=False, return_intermediate=False,
):
    K = 1024
    total_seq_len = 16 * K
    assert total_seq_len == total_seq_len_, f"This test forces total_seq_len = 16K"

    from d2.planner.equal_flops import (
        batch_to_items, 
        plan_relocation,
        item_to_intermediate_tensors,
    )

    items = batch_to_items([
        [16 * K] * 1,
        [8 * K] * 2,
        [4 * K] * 4,
        [2 * K] * 8, 
    ])
    items = plan_relocation(items, verbose=False, plot=False)

    world_info, (items, info_mapping, info_list), (seq_lens, cp_num, cp_dst, seq_shard_lens) = item_to_intermediate_tensors(items)    

    world_size = world_info["world_size"]
    num_seqs = world_info["num_seqs"]
    max_cp_degree = world_info["max_cp_degree"]

    assert world_size == world_size_, f"This test forces world_size = {world_size}"
    assert num_seqs == num_seqs_, f"This test forces num_seqs = {num_seqs}"
    assert max_cp_degree == max_cp_degree_, f"This test forces max_cp_degree = {max_cp_degree}"
    

    ret = create_qkv_dispatch_with_custom_mapping(
        world_size, 
        seq_lens,
        cp_num,
        cp_dst,
        seq_shard_lens,
        verbose=verbose, return_intermediate=return_intermediate,
    )
    return ret


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
     ) = test_create_qkv_dispatch_balanced_flops(
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
    parser.add_argument('--world_size', type=int, default=4)
    parser.add_argument('--num_seqs', type=int, default=8)
    parser.add_argument('--num_tokens', type=int, default=16 * 1024)
    parser.add_argument('--hidden_size_q', type=int, default=256)
    parser.add_argument('--hidden_size_k', type=int, default=128)
    parser.add_argument('--max_seq_shard', type=int, default=6)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()
    test(args)

"""
python test_fa2a_metadata_balanced_flops.py
"""