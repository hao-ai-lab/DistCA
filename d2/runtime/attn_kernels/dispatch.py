import torch
from torch import Tensor
import torch.nn.functional as F

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
    fast_a2a_memcpy_non_cp(
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


#### Functions for PP and grad ckpt:
#### during forward, attention out will be sent with softmax_lse.
#### during backward, attn_out_grad, attn_out, softmax_lse, and qkv are all sent.
_CUDA_INT4_BYTES = 16


def size_pad_by_int4(hidden_size: int, itemsize: int):
    """
    Args:
        hidden_size: num elements
        itemsize: of each element
    Returns:
        hidden_size_pad: padded num elements
        pad_size: padding size, in number of elements
    """
    hidden_bytes = hidden_size * itemsize
    if hidden_bytes % _CUDA_INT4_BYTES != 0:
        hidden_bytes += _CUDA_INT4_BYTES - (hidden_bytes % _CUDA_INT4_BYTES)
    hidden_size_pad = hidden_bytes // itemsize
    pad_size = hidden_size_pad - hidden_size
    return hidden_size_pad, pad_size


def pre_fast_a2a_attn_out_grad_resend_qkv(
    attn_out_grad: Tensor, attn_out: Tensor, lse_norm: Tensor,
    q: Tensor, k: Tensor, v: Tensor, kv_dispatch_mask: Tensor,
    q_seq_tokens: Tensor, k_seq_tokens: Tensor,
    q_send_buffer_offset: Tensor, k_send_buffer_offset: Tensor, v_send_buffer_offset: Tensor,
    instance_id: int=None,
):
    # This is used for attention output's backward. However, it's actually a forward qkv send
    # aiming to provide a new attention layout during grad_compute, different from fwd.
    is_fwd = True
    assert attn_out.ndim == 2
    assert lse_norm.ndim == 2 and lse_norm.shape[0] == attn_out.shape[0]
    assert q.ndim == 2 and q.shape[0] == attn_out.shape[0]
    assert q.dtype == attn_out.dtype == lse_norm.dtype
    assert attn_out_grad.shape == attn_out.shape

    # create merged_q
    merged_q = torch.concat([attn_out_grad, attn_out, lse_norm, q], dim=1)
    # Padding for int4. TODO(yonghao): 
    _, pad_len = size_pad_by_int4(merged_q.shape[1], q.itemsize)
    if pad_len > 0:
        merged_q = F.pad(merged_q, (0, pad_len), mode='constant', value=0)
    return pre_fast_a2a_qkv(merged_q, k, v, kv_dispatch_mask, q_seq_tokens, k_seq_tokens,
                            q_send_buffer_offset, k_send_buffer_offset, v_send_buffer_offset,
                            is_fwd=is_fwd, instance_id=instance_id)


def post_fast_a2a_attn_out_grad_resend_qkv(
    recv_attn_out_shape: list[int], recv_lse_shape: list[int], recv_q_shape: list[int],
    recv_k: Tensor, recv_v: Tensor,
    kv_dispatch_mask: Tensor, q_recv_seq_tokens: Tensor, k_recv_seq_tokens: Tensor,
    q_recv_buffer_offset: Tensor, k_recv_buffer_offset: Tensor, v_recv_buffer_offset: Tensor,
    is_fwd: bool, switch_buffer: bool = True, instance_id: int=None
):
    is_fwd = True
    assert len(recv_attn_out_shape) == 2
    assert len(recv_lse_shape) == 2 and recv_lse_shape[0] == recv_attn_out_shape[0]
    assert len(recv_q_shape) == 2 and recv_q_shape[0] == recv_attn_out_shape[0]

    # layout: attn_out_grad, attn_out, softmax_lse, q
    recv_q_splits = (
        recv_attn_out_shape[1], recv_attn_out_shape[1], recv_lse_shape[1],
        recv_q_shape[1]
    )
    recv_merged_q_hidden = sum(recv_q_splits)
    recv_merged_q_hidden, padding = size_pad_by_int4(recv_merged_q_hidden, recv_k.itemsize)

    recv_merged_q = recv_k.new_empty((recv_attn_out_shape[0], recv_merged_q_hidden))

    recv_merged_q, recv_k, recv_v = post_fast_a2a_qkv(
        recv_merged_q, recv_k, recv_v, kv_dispatch_mask,
        q_recv_seq_tokens, k_recv_seq_tokens,
        q_recv_buffer_offset, k_recv_buffer_offset, v_recv_buffer_offset,
        is_fwd, switch_buffer=switch_buffer, instance_id=instance_id
    )
    recv_attn_out_grad, recv_attn_out, recv_lse, recv_q, _ = torch.split(
        recv_merged_q, [*recv_q_splits, padding], dim=1
    )
    return (
        recv_attn_out_grad, recv_attn_out, recv_lse,
        recv_q, recv_k, recv_v
    )


def pre_fast_a2a_attn_out_with_lse(
    attn_out: Tensor, softmax_lse: Tensor,
    send_seqlens: Tensor, send_memcpy_metadata: Tensor, dispatcher_id: int
):
    assert softmax_lse.shape[0] == attn_out.shape[0]
    merged_attn_out = torch.concat([attn_out, softmax_lse], dim=1)
    _, pad_len = size_pad_by_int4(merged_attn_out.shape[1], attn_out.itemsize)
    if pad_len > 0:
        merged_attn_out = F.pad(merged_attn_out, (0, pad_len), mode='constant', value=0)
    return pre_fast_a2a_attn_out(
        merged_attn_out, send_seqlens, send_memcpy_metadata[0],
        instance_id=dispatcher_id,
    )
