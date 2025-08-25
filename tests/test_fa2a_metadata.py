"""TODO: deprecate this file."""

from typing import Sequence

import torch
from torch import Tensor

from d2.runtime.inplace_metadata import Metadata
from d2.runtime.fast_alltoall_metadata import (
    FastAlltoAllMetadata
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
    if k is not None:
        assert v is not None
        dst_tensor = simulate_fa2a_copy_cp(
            k, dst_tensor, k_offset, k_seqlen, hidden_size_k, element_size,
            max_cp, send_mask, is_send=True,
        )
        dst_tensor = simulate_fa2a_copy_cp(
            v, dst_tensor, v_offset, k_seqlen, hidden_size_k, element_size,
            max_cp, send_mask, is_send=True,
        )
    else:
        assert v is None
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
    if k is not None:
        assert v is not None
        dst_tensor = simulate_fa2a_copy_non_cp(
            k, dst_tensor, k_offset, k_seqlen, hidden_size_k, element_size, is_send=True
        )
        dst_tensor = simulate_fa2a_copy_non_cp(
            v, dst_tensor, v_offset, k_seqlen, hidden_size_k, element_size, is_send=True
        )
    else:
        assert v is None
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
    if k is not None:
        assert v is not None
        k = simulate_fa2a_copy_non_cp(
            k, src_tensor, k_offset, k_seqlen, hidden_size_k, element_size, is_send=False
        )
        v = simulate_fa2a_copy_non_cp(
            v, src_tensor, v_offset, k_seqlen, hidden_size_k, element_size, is_send=False
        )
    else:
        assert v is None
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
    if k is not None:
        assert v is not None
        k = simulate_fa2a_copy_cp(
            k, src_tensor, k_offset, k_seqlen, hidden_size_k, element_size,
            max_cp, send_mask, is_send=False,
        )
        v = simulate_fa2a_copy_cp(
            v, src_tensor, v_offset, k_seqlen, hidden_size_k, element_size,
            max_cp, send_mask, is_send=False,
        )
    else:
        assert v is None
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

