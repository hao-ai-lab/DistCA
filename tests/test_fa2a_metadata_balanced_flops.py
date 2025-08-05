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


def simulate_fa2a_copy_non_cp(
    q: Tensor, buffer: Tensor, q_offsets: Tensor, q_seqlen: Tensor,
    hidden_size_q: int, element_size: int, is_send: bool=True
):

    q_seq_offset = 0
    for seq_id, seq_len in enumerate(q_seqlen):
        q_offset_bytes = q_offsets[seq_id]
        q_dst_offset = q_offset_bytes // element_size
        q_size = seq_len * hidden_size_q
        if is_send:
            buffer[q_dst_offset:q_dst_offset + q_size] = (
                q[q_seq_offset:q_seq_offset + q_size]
            )
        else:
            q[q_seq_offset:q_seq_offset + q_size] = (
                buffer[q_dst_offset:q_dst_offset + q_size]
            )
        q_seq_offset += q_size
    return buffer if is_send else q


def simulate_fa2a_copy_cp(
    k: Tensor, buffer: Tensor,
    k_offsets: Tensor, k_seqlen: Tensor,
    hidden_size_k: int, element_size: int, max_cp: int,
    send_mask: Tensor, is_send: bool=True
):
    """
    Simulate the fa2a send: MLP layout (CP) -> a2a layout.
    MLP layout (CP) is in form of: (seq_len,), but has multiple copies
    to the a2a layout.
    """
    for cp_id in range(max_cp):
        k_seq_offset = 0
        for seq_id, seq_len in enumerate(k_seqlen):
            k_size = seq_len * hidden_size_k
            if not send_mask[seq_id, cp_id]:
                k_seq_offset += k_size
                continue
            k_offset_bytes = k_offsets[cp_id, seq_id]
            k_dst_offset = k_offset_bytes // element_size
            if is_send:
                buffer[k_dst_offset:k_dst_offset + k_size] = (
                    k[k_seq_offset:k_seq_offset + k_size]
                )
            else:
                k[cp_id, k_seq_offset:k_seq_offset + k_size] = (
                    buffer[k_dst_offset:k_dst_offset + k_size]
                )
            k_seq_offset += k_size
    return buffer if is_send else k


def simulate_fa2a_send_qkv(
    q: Tensor, k: Tensor, v: Tensor,
    dst_tensor: Tensor, q_offset: Tensor, k_offset: Tensor, v_offset: Tensor,
    q_seqlen: Tensor, k_seqlen: Tensor,
    hidden_size_q: int, hidden_size_k: int, element_size: int, max_cp: int,
    send_mask: Tensor,
):
    """QKV (MLP) -> a2a send layout"""
    dst_tensor = simulate_fa2a_copy_non_cp(
        q, dst_tensor, q_offset, q_seqlen, hidden_size_q, element_size, is_send=True
    )
    dst_tensor = simulate_fa2a_copy_cp(
        k, dst_tensor, k_offset, k_seqlen, hidden_size_k, element_size,
        max_cp, send_mask, is_send=True,
    )
    dst_tensor = simulate_fa2a_copy_cp(
        v, dst_tensor, v_offset, k_seqlen, hidden_size_k, element_size,
        max_cp, send_mask, is_send=True,
    )
    return dst_tensor


def simulate_fa2a_send_qkv_rev(
    q: Tensor, k: Tensor, v: Tensor,
    dst_tensor: Tensor, q_offset: Tensor, k_offset: Tensor, v_offset: Tensor,
    q_seqlen: Tensor, k_seqlen: Tensor,
    hidden_size_q: int, hidden_size_k: int, element_size: int
):
    """QKV grad (ATTN) -> a2a send layout"""
    dst_tensor = simulate_fa2a_copy_non_cp(
        q, dst_tensor, q_offset, q_seqlen, hidden_size_q, element_size, is_send=True
    )
    dst_tensor = simulate_fa2a_copy_non_cp(
        k, dst_tensor, k_offset, k_seqlen, hidden_size_k, element_size, is_send=True
    )
    dst_tensor = simulate_fa2a_copy_non_cp(
        v, dst_tensor, v_offset, k_seqlen, hidden_size_k, element_size, is_send=True
    )
    return dst_tensor


def simulate_fa2a_recv_qkv(
    q: Tensor, k: Tensor, v: Tensor,
    src_tensor: Tensor, q_offset: Tensor, k_offset: Tensor, v_offset: Tensor,
    q_seqlen: Tensor, k_seqlen: Tensor,
    hidden_size_q: int, hidden_size_k: int, element_size: int
):
    """a2a recv layout -> QKV (ATTN)"""
    q = simulate_fa2a_copy_non_cp(
        q, src_tensor, q_offset, q_seqlen, hidden_size_q, element_size, is_send=False
    )
    k = simulate_fa2a_copy_non_cp(
        k, src_tensor, k_offset, k_seqlen, hidden_size_k, element_size, is_send=False
    )
    v = simulate_fa2a_copy_non_cp(
        v, src_tensor, v_offset, k_seqlen, hidden_size_k, element_size, is_send=False
    )
    return q, k, v


def simulate_fa2a_recv_qkv_rev(
    q: Tensor, k: Tensor, v: Tensor,
    src_tensor: Tensor, q_offset: Tensor, k_offset: Tensor, v_offset: Tensor,
    q_seqlen: Tensor, k_seqlen: Tensor,
    hidden_size_q: int, hidden_size_k: int, element_size: int,
    max_cp: int, send_mask: Tensor
):
    """a2a recv layout -> QKV grad (MLP)"""
    q = simulate_fa2a_copy_non_cp(
        q, src_tensor, q_offset, q_seqlen, hidden_size_q, element_size, is_send=False
    )
    k = simulate_fa2a_copy_cp(
        k, src_tensor, k_offset, k_seqlen, hidden_size_k, element_size,
        max_cp, send_mask, is_send=False,
    )
    v = simulate_fa2a_copy_cp(
        v, src_tensor, v_offset, k_seqlen, hidden_size_k, element_size,
        max_cp, send_mask, is_send=False,
    )
    return q, k, v


def simulate_fa2a(send_buffer: Tensor, recv_buffer: Tensor,
                  fa2a_metadata: Sequence[Tensor], element_size: int):
    """Simulate all2all from send buffer to recv buffer."""
    world_size = send_buffer.shape[0]
    assert recv_buffer.shape[0] == world_size
    (sender_send_disp, sender_transfer_sz, sender_recv_disp,
     recver_transfer_sz) = fa2a_metadata
    for src_rank in range(world_size):
        for dst_rank in range(world_size):
            send_offset = sender_send_disp[src_rank, dst_rank] // element_size
            recv_offset = sender_recv_disp[src_rank, dst_rank] // element_size
            size = sender_transfer_sz[src_rank, dst_rank] // element_size
            recv_buffer[dst_rank][recv_offset:recv_offset + size] = (
                send_buffer[src_rank][send_offset:send_offset + size]
            )
    return recv_buffer


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
    """
    Simulate the whole QKV A2A communication: copy to the send buffer,
    simulate the all2all, and copy back from the recv buffer.
    """
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
        torch.testing.assert_close(q_src_seqlen, metadata_slice.seq_lens[0].send_seqlens)
        torch.testing.assert_close(kv_src_seqlen, metadata_slice.seq_lens[1].send_seqlens)
        # NOTE: fwd and rev in the arg is not the actual fwd/rev, but instead
        # relative to this communication.
        num_seqs_send = fwd_kv_metadata.num_seqs[rank] if is_fwd else rev_kv_metadata.num_seqs[rank]
        torch.testing.assert_close(metadata_slice.kv_replica_mask,
                                   (kv_comm_mask[rank][:num_seqs_send]).to(
                                        metadata_slice.kv_replica_mask.dtype
                                    ))

        args = (
            q[rank], k[rank], v[rank], src_buffer[rank],
            q_src_offsets, k_src_offsets, v_src_offsets,
            q_src_seqlen, kv_src_seqlen,
            hidden_q, hidden_k, element_size,
        )
        if is_fwd:
            args += (max_cp, kv_comm_mask[rank])
            copy_fn = simulate_fa2a_send_qkv
        else:
            copy_fn = simulate_fa2a_send_qkv_rev
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

        if is_fwd:
            expected_shape_q = (num_recv_tokens_q[rank], hidden_q)
            expected_shape_k = (num_recv_tokens_k[rank], hidden_k)
        else:
            expected_shape_q = (max_num_recv_tokens_q, hidden_q)
            expected_shape_k = (max_cp, max_num_recv_tokens_q, hidden_k)
        assert expected_shape_q == metadata_slice.tensor_shape[0].recv_shape
        assert expected_shape_k == metadata_slice.tensor_shape[1].recv_shape, f"{rank} {expected_shape_k} != {metadata_slice.tensor_shape[1].recv_shape}"
        torch.testing.assert_close(rev_seqlen_q, metadata_slice.seq_lens[0].recv_seqlens)
        torch.testing.assert_close(rev_seqlen_kv, metadata_slice.seq_lens[1].recv_seqlens)
        # NOTE: fwd and rev in the arg is not the actual fwd/rev, but instead
        # relative to this communication.
        num_seqs_send = fwd_kv_metadata.num_seqs[rank] if is_fwd else rev_kv_metadata.num_seqs[rank]
        torch.testing.assert_close(metadata_slice.kv_replica_mask,
                                   kv_comm_mask[rank][:num_seqs_send].to(
                                       metadata_slice.kv_replica_mask.dtype
                                   ))

        args = (
            dst_q[rank], dst_k[rank], dst_v[rank],
            dst_buffer[rank], q_recv_offsets, 
            k_recv_offsets, v_recv_offsets,
            rev_seqlen_q, rev_seqlen_kv,
            hidden_q, hidden_k, element_size,
        )
        if not is_fwd:
            args += (max_cp, kv_comm_mask[rank])
            copy_fn = simulate_fa2a_recv_qkv_rev
        else:
            copy_fn = simulate_fa2a_recv_qkv

        q_slice, k_slice, v_slice = copy_fn(*args)
        dst_q[rank] = q_slice
        dst_k[rank] = k_slice
        dst_v[rank] = v_slice
    dst_q = dst_q.reshape(world_size, -1, hidden_q)
    dst_k = dst_k.reshape(world_size, -1, hidden_k)
    dst_v = dst_v.reshape(world_size, -1, hidden_k)
    return dst_q, dst_k, dst_v


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