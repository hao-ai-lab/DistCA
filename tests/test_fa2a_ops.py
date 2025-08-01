"""
NOTE: this is not well maintained.
"""
import torch
from torch import Tensor

from d2.runtime.inplace_metadata import Metadata
from d2.runtime.fast_alltoall_metadata import (
    compute_forward_qkv_a2a_layout_meatadata,
    compute_reverse_a2a_layout_metadata,
    FastAlltoAllMetadata
)

from d2.runtime.attn_kernels.ops import fast_a2a_memcpy_non_cp

from test_comm_metadata import orchestrate_simulate
from test_util import create_qkv_dispatch
from test_fa2a_metadata import (
    simulate_fa2a_send_copy_non_cp, simulate_fa2a_send_copy_cp,
    simulate_fa2a_recv_copy_non_cp, simulate_fa2a_recv_copy_kv_grad,
    simulate_fa2a_send_qkv_copy, simulate_fa2a_send_qkv_copy_rev,
    simulate_fa2a_recv_copy_qkv, simulate_fa2a_recv_copy_qkv_rev,
    simulate_fa2a, simulate_qkv_a2a
)

def get_seq_len_slice(metadata: Metadata, rank):
    return metadata.seq_len[rank][:metadata.num_seqs[rank]]


def simulate_qkv_a2a(
    q: Tensor, k: Tensor, v: Tensor,
    metadata: FastAlltoAllMetadata,
    # Only to get the seq len and max num recv tokens
    fwd_q_metadata: Metadata, fwd_kv_metadata: Metadata,
    rev_q_metadata: Metadata, rev_kv_metadata: Metadata,
    element_size: int, hidden_q: int, hidden_k: int,
    max_cp: int, kv_comm_mask: Tensor,
    is_fwd: bool
):
    world_size = q.shape[0]
    assert k.shape[0] == world_size
    assert v.shape[0] == world_size
    # sender transfer sz
    tot_send_bytes = metadata.fa2a_metadata[1].sum(dim=1) // element_size
    max_send_bytes = int(torch.max(tot_send_bytes).item())
    tot_recv_bytes = metadata.fa2a_metadata[3].sum(dim=1) // element_size
    max_recv_bytes = int(torch.max(tot_recv_bytes).item())

    # Flatten the tensors
    q, k, v = (t.flatten(start_dim=1) for t in (q, k, v))

    src_buffer = torch.zeros(
        (world_size, max_send_bytes), dtype=q.dtype, device=q.device
    )
    dst_buffer = torch.zeros(
        (world_size, max_recv_bytes), dtype=q.dtype, device=q.device
    )
    for rank in range(world_size):
        metadata_slice = metadata.get_slice(rank)
        q_src_offsets, k_src_offsets, v_src_offsets = metadata_slice.send_memcpy_metadata
        q_src_seqlen = get_seq_len_slice(fwd_q_metadata, rank)
        kv_src_seqlen = get_seq_len_slice(fwd_kv_metadata, rank)

        args = (
            q[rank], k[rank], v[rank], src_buffer[rank],
            q_src_offsets, k_src_offsets, v_src_offsets,
            q_src_seqlen, kv_src_seqlen,
            hidden_q, hidden_k, element_size,
        )
        if is_fwd:
            args += (max_cp, kv_comm_mask[rank])
            copy_fn = simulate_fa2a_send_qkv_copy
        else:
            copy_fn = simulate_fa2a_send_qkv_copy_rev
        src_buffer[rank] = copy_fn(*args)

    dst_buffer = simulate_fa2a(
        src_buffer, dst_buffer, metadata.fa2a_metadata, element_size
    )

    num_recv_tokens_q = fwd_q_metadata.num_recv_tokens[:, -1]
    max_num_recv_tokens_q = int(torch.max(num_recv_tokens_q).item())
    num_recv_tokens_k = fwd_kv_metadata.num_recv_tokens[:, -1]
    max_num_recv_tokens_k = int(torch.max(num_recv_tokens_k).item())
    dst_q = torch.zeros(
        (world_size, max_num_recv_tokens_q * hidden_q),
        device=q.device, dtype=q.dtype
    )
    if is_fwd:
        dst_k = torch.zeros(
            (world_size, max_num_recv_tokens_k * hidden_k),
            device=k.device, dtype=k.dtype
        )
        dst_v = dst_k.clone()
    else:
        dst_k = torch.zeros(
            (world_size, max_cp, max_num_recv_tokens_q * hidden_k),
            device=k.device, dtype=k.dtype
        )
        dst_v = dst_k.clone()
    for rank in range(world_size):
        metadata_slice = metadata.get_slice(rank)
        q_recv_offsets, k_recv_offsets, v_recv_offsets = metadata_slice.recv_memcpy_metadata
        rev_seqlen_q = get_seq_len_slice(rev_q_metadata, rank)
        rev_seqlen_kv = get_seq_len_slice(rev_kv_metadata, rank)

        args = (
            dst_q[rank], dst_k[rank], dst_v[rank],
            dst_buffer[rank], q_recv_offsets, 
            k_recv_offsets, v_recv_offsets,
            rev_seqlen_q, rev_seqlen_kv,
            hidden_q, hidden_k, element_size,
        )
        if not is_fwd:
            args += (max_cp, kv_comm_mask[rank])
            copy_fn = simulate_fa2a_recv_copy_qkv_rev
        else:
            copy_fn = simulate_fa2a_recv_copy_qkv

        q_slice, k_slice, v_slice = copy_fn(*args)
        dst_q[rank] = q_slice
        dst_k[rank] = k_slice
        dst_v[rank] = v_slice
    dst_q = dst_q.reshape(world_size, -1, hidden_q)
    dst_k = dst_k.reshape(world_size, -1, hidden_k)
    dst_v = dst_v.reshape(world_size, -1, hidden_k)
    return dst_q, dst_k, dst_v


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
     ) = create_qkv_dispatch(
        world_size, total_seq_len, num_seqs, max_cp_degree, return_intermediate=True
    )

    tensor_q = torch.rand((world_size, total_seq_len, hidden_size_q), device=device) + 1    # + 1 to avoid zeros
    tensor_k = torch.rand((world_size, total_seq_len, hidden_size_k), device=device) + 1
    tensor_v = torch.rand((world_size, total_seq_len, hidden_size_k), device=device) + 1

    (q_tokens_to_dst_per_dispatch, q_seq_to_dst,
     _, kv_dst_global_seq_id) = intermediates
    element_size = tensor_q.element_size()
    bytes_q = element_size * hidden_size_q
    bytes_k = element_size * hidden_size_k
    recver_transfer_sz_q = (
        fwd_q_metadata.num_recv_tokens * bytes_q
    )[..., :-1]
    recver_transfer_sz_kv = (
        fwd_k_metadata.num_recv_tokens * bytes_k
    )[..., :-1]
    num_recv_seqs_q = rev_q_metadata.num_seqs
    num_recv_seqs_kv = rev_k_metadata.num_seqs
    num_send_tokens_kv = rev_k_metadata.num_recv_tokens[..., :-1]

    qkv_fwd_fa2a_metadata = compute_forward_qkv_a2a_layout_meatadata(
        q_tokens_to_dst_per_dispatch.squeeze(2), q_seq_to_dst,
        recver_transfer_sz_q, recver_transfer_sz_kv,
        num_recv_seqs_q, num_recv_seqs_kv,
        fwd_k_metadata.dst_rank, fwd_k_metadata.seq_len,
        kv_dst_global_seq_id, num_send_tokens_kv,
        bytes_q, bytes_k,
    )

    qkv_rev_fa2a_metadata = compute_reverse_a2a_layout_metadata(
        qkv_fwd_fa2a_metadata,
    )

    # fa2a_q, fa2a_k, fa2a_v = simulate_qkv_a2a(
    #     tensor_q, tensor_k, tensor_v,
    #     qkv_fwd_fa2a_metadata, fwd_q_metadata, fwd_k_metadata,
    #     rev_q_metadata, rev_k_metadata,
    #     element_size, hidden_size_q, hidden_size_k,
    #     max_cp_degree, fwd_k_metadata.dst_rank >= 0,
    #     is_fwd=True
    # )
    tot_send_bytes = qkv_fwd_fa2a_metadata.fa2a_metadata[1].sum(dim=1) // element_size
    max_send_bytes = int(torch.max(tot_send_bytes).item())
    tot_recv_bytes = qkv_fwd_fa2a_metadata.fa2a_metadata[3].sum(dim=1) // element_size
    max_recv_bytes = int(torch.max(tot_recv_bytes).item())

    dtype = tensor_q.dtype
    device = tensor_q.device
    src_buffer = torch.zeros(
        (world_size, max_send_bytes), dtype=dtype, device=device
    )
    dst_buffer = torch.zeros(
        (world_size, max_recv_bytes), dtype=dtype, device=device
    )
    # use rank 0 for the test
    rank = 0
    metadata_slice = qkv_fwd_fa2a_metadata.get_slice(rank)
    q_src_offsets, k_src_offsets, v_src_offsets = metadata_slice.send_memcpy_metadata
    q_src_seqlen = get_seq_len_slice(fwd_q_metadata, rank)
    kv_src_seqlen = get_seq_len_slice(fwd_k_metadata, rank)
    q_shard = tensor_q[rank]
    k_shard = tensor_k[rank]
    v_shard = tensor_v[rank]
    src_shard = src_buffer[rank]
    src_shard_cor = src_shard.clone()
    src_shard_cor = simulate_fa2a_send_copy_non_cp(
        q_shard.flatten(), src_shard_cor, q_src_offsets, q_src_seqlen,
        hidden_size_q, element_size
    )
    src_shard_out = src_shard.clone()
    fast_a2a_memcpy_non_cp(
        q_shard.cuda(), q_src_offsets.long().cuda(), q_src_seqlen.long().cuda(),
        to_nvshmem=True, buffer=src_shard_out.cuda(), use_buffer=True
    )
    torch.cuda.synchronize()
    torch.testing.assert_close(src_shard_cor, src_shard_out)

    # fa2a_rev_q, fa2a_rev_k, fa2a_rev_v = simulate_qkv_a2a(
    #     fa2a_q, fa2a_k, fa2a_v,
    #     qkv_rev_fa2a_metadata, rev_q_metadata, rev_k_metadata,
    #     fwd_q_metadata, fwd_k_metadata,
    #     element_size, hidden_size_q, hidden_size_k,
    #     max_cp_degree, fwd_k_metadata.dst_rank >= 0,
    #     is_fwd=False
    # )


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--world_size', type=int, default=2)
    parser.add_argument('--num_seqs', type=int, default=4)
    parser.add_argument('--num_tokens', type=int, default=128)
    parser.add_argument('--hidden_size_q', type=int, default=256)
    parser.add_argument('--hidden_size_k', type=int, default=128)
    parser.add_argument('--max_seq_shard', type=int, default=2)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()
    test(args)
