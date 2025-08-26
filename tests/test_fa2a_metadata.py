"""TODO: deprecate this file."""

from typing import Sequence

import torch
from torch import Tensor

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
