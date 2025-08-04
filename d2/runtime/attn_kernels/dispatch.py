import torch
from torch import Tensor

from d2.runtime.attn_kernels.ops import (
    FastDispatcherWrapper, fast_a2a_memcpy_non_cp,
    fast_a2a_memcpy_cp, fast_a2a,
)


# NOTE: currently, we use the original q,k,v as the output of pre_fast_a2a memcpy.
# Instead, we should use a signal tensor for this.

def pre_fast_a2a_qkv(
    q: Tensor, k: Tensor, v: Tensor, kv_dispatch_mask: Tensor,
    q_seq_tokens: Tensor, k_seq_tokens: Tensor,
    q_send_buffer_offset: Tensor, k_send_buffer_offset: Tensor, v_send_buffer_offset: Tensor,
    is_fwd: bool, instance_id: int=None,
):
    # copy in advance
    to_nvshmem = True
    fast_a2a_memcpy_non_cp(
        q.contiguous(), q_send_buffer_offset, q_seq_tokens, to_nvshmem,
        instance_id=instance_id,
    )
    if is_fwd:
        fast_a2a_memcpy_cp(
            k.contiguous(), kv_dispatch_mask, k_send_buffer_offset, k_seq_tokens, to_nvshmem,
            instance_id=instance_id,
        )
        fast_a2a_memcpy_cp(
            v.contiguous(), kv_dispatch_mask, v_send_buffer_offset, k_seq_tokens, to_nvshmem,
            instance_id=instance_id,
        )
    else:
        fast_a2a_memcpy_non_cp(
            k.contiguous(), k_send_buffer_offset, k_seq_tokens, to_nvshmem,
            instance_id=instance_id,
        )
        fast_a2a_memcpy_non_cp(
            v.contiguous(), v_send_buffer_offset, k_seq_tokens, to_nvshmem,
            instance_id=instance_id,
        )
    return q, k, v


def post_fast_a2a_qkv(
    recv_q: Tensor, recv_k: Tensor, recv_v: Tensor, kv_dispatch_mask: Tensor,
    q_recv_seq_tokens: Tensor, k_recv_seq_tokens: Tensor,
    q_recv_buffer_offset: Tensor, k_recv_buffer_offset: Tensor, v_recv_buffer_offset: Tensor,
    is_fwd: bool, switch_buffer: bool = True, instance_id: int=None
):
    to_nvshmem = False
    fast_a2a_memcpy_non_cp(
        recv_q, q_recv_buffer_offset, q_recv_seq_tokens, to_nvshmem,
        instance_id=instance_id,
    )
    if is_fwd:
        fast_a2a_memcpy_non_cp(
            recv_k, k_recv_buffer_offset, k_recv_seq_tokens, to_nvshmem,
            instance_id=instance_id,
        )
        fast_a2a_memcpy_non_cp(
            recv_v, v_recv_buffer_offset, k_recv_seq_tokens, to_nvshmem,
            instance_id=instance_id,
        )
    else:
        fast_a2a_memcpy_cp(
            recv_k, kv_dispatch_mask, k_recv_buffer_offset, k_recv_seq_tokens, to_nvshmem,
            instance_id=instance_id,
        )
        fast_a2a_memcpy_cp(
            recv_v, kv_dispatch_mask, v_recv_buffer_offset, k_recv_seq_tokens, to_nvshmem,
            instance_id=instance_id,
        )
    if switch_buffer:
        FastDispatcherWrapper.switch_buffer()
    return recv_q, recv_k, recv_v


def fast_a2a_qkv(
    q: Tensor, k: Tensor, v: Tensor, kv_dispatch_mask: Tensor,
    recv_q: Tensor, recv_k: Tensor, recv_v: Tensor,
    q_seq_tokens: Tensor, k_seq_tokens: Tensor,
    q_recv_seq_tokens: Tensor, k_recv_seq_tokens: Tensor,
    q_send_buffer_offset: Tensor, k_send_buffer_offset: Tensor, v_send_buffer_offset: Tensor,
    q_recv_buffer_offset: Tensor, k_recv_buffer_offset: Tensor, v_recv_buffer_offset: Tensor,
    sender_send_disp: Tensor, sender_transfer_sz: Tensor,
    sender_recv_disp: Tensor, recver_transfer_sz: Tensor,
    my_rank_send_offset: int, my_rank_recv_offset: int, my_rank_send_sz: int,
    is_fwd: bool,
    switch_buffer: bool = True,
    instance_id: int=None
):
    # copy in advance
    q, k, v = pre_fast_a2a_qkv(
        q, k, v, kv_dispatch_mask, q_seq_tokens, k_seq_tokens,
        q_send_buffer_offset, k_send_buffer_offset, v_send_buffer_offset,
        is_fwd=is_fwd, instance_id=instance_id,
    )
    # all2all
    fast_a2a(
        sender_send_disp, sender_transfer_sz,
        sender_recv_disp, recver_transfer_sz,
        my_rank_send_offset, my_rank_recv_offset, my_rank_send_sz,
        instance_id=instance_id,
    )
    # copy back
    recv_q, recv_k, recv_v = post_fast_a2a_qkv(
        recv_q, recv_k, recv_v, kv_dispatch_mask,
        q_recv_seq_tokens, k_recv_seq_tokens,
        q_recv_buffer_offset, k_recv_buffer_offset, v_recv_buffer_offset,
        is_fwd=is_fwd, switch_buffer=switch_buffer, instance_id=instance_id,
    )
    return recv_q, recv_k, recv_v


def pre_fast_a2a_attn_out(
    q: Tensor, q_seq_tokens: Tensor, q_send_buffer_offset: Tensor,
    instance_id: int=None,
):
    to_nvshmem = True
    q = fast_a2a_memcpy_non_cp(
        q.contiguous(), q_send_buffer_offset, q_seq_tokens, to_nvshmem,
        instance_id=instance_id,
    )
    return q


def post_fast_a2a_attn_out(
    recv_q: Tensor, q_recv_seq_tokens: Tensor, q_recv_buffer_offset: Tensor,
    switch_buffer: bool = True, instance_id: int=None,
):
    to_nvshmem = False
    fast_a2a_memcpy_non_cp(
        recv_q, q_recv_buffer_offset, q_recv_seq_tokens, to_nvshmem,
        instance_id=instance_id,
    )
    if switch_buffer:
        FastDispatcherWrapper.switch_buffer()
    return recv_q


def fast_a2a_attn_out(
    q: Tensor, recv_q: Tensor,
    q_seq_tokens: Tensor, q_recv_seq_tokens: Tensor,
    q_send_buffer_offset: Tensor, q_recv_buffer_offset: Tensor,
    sender_send_disp: torch.Tensor, sender_transfer_sz: torch.Tensor,
    sender_recv_disp: torch.Tensor, recver_transfer_sz: torch.Tensor,
    my_rank_send_offset: int, my_rank_recv_offset: int, my_rank_send_sz: int,
    switch_buffer: bool = True, instance_id: int=None,
):
    # copy in advance
    q = pre_fast_a2a_attn_out(
        q, q_seq_tokens, q_send_buffer_offset,
        instance_id=instance_id,
    )
    # all2all
    fast_a2a(
        sender_send_disp, sender_transfer_sz,
        sender_recv_disp, recver_transfer_sz,
        my_rank_send_offset, my_rank_recv_offset, my_rank_send_sz,
        instance_id=instance_id,
    )
    # copy back
    recv_q = post_fast_a2a_attn_out(
        recv_q, q_recv_seq_tokens, q_recv_buffer_offset,
        switch_buffer=switch_buffer, instance_id=instance_id
    )
    return recv_q
