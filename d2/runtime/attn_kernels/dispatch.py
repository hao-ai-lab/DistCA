import torch
from torch import Tensor
import torch.nn.functional as F

from d2.runtime.attn_kernels.ops import (
    DispatcherWrapper, a2a_memcpy_non_cp,
    a2a_memcpy_cp, pre_a2a_grad_acc

)
from d2.runtime.utils import size_pad_by_int4


def pre_a2a_qkv(
    q: Tensor, k: Tensor, v: Tensor, kv_dispatch_mask: Tensor,
    q_seq_tokens: Tensor, k_seq_tokens: Tensor,
    q_send_buffer_offset: Tensor, k_send_buffer_offset: Tensor, v_send_buffer_offset: Tensor,
    is_fwd: bool, instance_id: int=None,
    # TODO: reorder args to make it more logical
    kv_grad_copy_shard_mask: Tensor=None,
    # this includes (num_copies, copy_start_id, shard_tokens)
    pre_a2a_grad_acc_args = None,
):
    """pre_a2a_grad_acc_args is effective only when it's a backward."""
    # copy in advance
    to_nvshmem = True
    a2a_memcpy_non_cp(
        q.contiguous(), q_send_buffer_offset, q_seq_tokens, to_nvshmem,
        instance_id=instance_id,
    )
    if is_fwd:
        a2a_memcpy_cp(
            k.contiguous(), kv_dispatch_mask, k_send_buffer_offset, k_seq_tokens, to_nvshmem,
            instance_id=instance_id,
        )
        a2a_memcpy_cp(
            v.contiguous(), kv_dispatch_mask, v_send_buffer_offset, k_seq_tokens, to_nvshmem,
            instance_id=instance_id,
        )
    else:
        assert kv_grad_copy_shard_mask is not None, "kv grad copy requires a dedup mask."
        k = k.contiguous()
        v = v.contiguous()
        if pre_a2a_grad_acc_args is not None:
            pre_a2a_grad_acc(k, *pre_a2a_grad_acc_args)
            pre_a2a_grad_acc(v, *pre_a2a_grad_acc_args)

        a2a_memcpy_non_cp(
            k, k_send_buffer_offset, k_seq_tokens, to_nvshmem,
            shard_do_copy_mask=kv_grad_copy_shard_mask,
            instance_id=instance_id,
        )
        a2a_memcpy_non_cp(
            v, v_send_buffer_offset, k_seq_tokens, to_nvshmem,
            shard_do_copy_mask=kv_grad_copy_shard_mask,
            instance_id=instance_id,
        )
    return q, k, v


def post_a2a_qkv(
    recv_q: Tensor, recv_k: Tensor, recv_v: Tensor, kv_dispatch_mask: Tensor,
    q_recv_seq_tokens: Tensor, k_recv_seq_tokens: Tensor,
    q_recv_buffer_offset: Tensor, k_recv_buffer_offset: Tensor, v_recv_buffer_offset: Tensor,
    is_fwd: bool, switch_buffer: bool = True, instance_id: int=None
):
    """NOTE: dispatcher is released here"""
    to_nvshmem = False
    a2a_memcpy_non_cp(
        recv_q, q_recv_buffer_offset, q_recv_seq_tokens, to_nvshmem,
        instance_id=instance_id,
    )
    if is_fwd:
        a2a_memcpy_non_cp(
            recv_k, k_recv_buffer_offset, k_recv_seq_tokens, to_nvshmem,
            instance_id=instance_id,
        )
        a2a_memcpy_non_cp(
            recv_v, v_recv_buffer_offset, k_recv_seq_tokens, to_nvshmem,
            instance_id=instance_id,
        )
    else:
        a2a_memcpy_cp(
            recv_k, kv_dispatch_mask, k_recv_buffer_offset, k_recv_seq_tokens, to_nvshmem,
            instance_id=instance_id,
        )
        a2a_memcpy_cp(
            recv_v, kv_dispatch_mask, v_recv_buffer_offset, k_recv_seq_tokens, to_nvshmem,
            instance_id=instance_id,
        )
    if switch_buffer:
        DispatcherWrapper.switch_buffer()
    if instance_id is not None:
        DispatcherWrapper.release(instance_id)
    return recv_q, recv_k, recv_v


def pre_a2a_attn_out(
    q: Tensor, q_seq_tokens: Tensor, q_send_buffer_offset: Tensor,
    instance_id: int=None,
):
    to_nvshmem = True
    a2a_memcpy_non_cp(
        q.contiguous(), q_send_buffer_offset, q_seq_tokens, to_nvshmem,
        instance_id=instance_id,
    )
    return q


def post_a2a_attn_out(
    recv_q: Tensor, q_recv_seq_tokens: Tensor, q_recv_buffer_offset: Tensor,
    switch_buffer: bool = True, instance_id: int=None,
):
    """NOTE: dispatcher is released here"""
    to_nvshmem = False
    a2a_memcpy_non_cp(
        recv_q, q_recv_buffer_offset, q_recv_seq_tokens, to_nvshmem,
        instance_id=instance_id,
    )
    if switch_buffer:
        DispatcherWrapper.switch_buffer()
    if instance_id is not None:
        DispatcherWrapper.release(instance_id)
    return recv_q


#### Functions for PP and grad ckpt:
#### during forward, attention out will be sent with softmax_lse.
#### during backward, attn_out_grad, attn_out, softmax_lse, and qkv are all sent.
def _concat_with_uint8_and_pad(tensors: list[Tensor], dim: int):
    tensor = torch.concat([t.view(torch.uint8) for t in tensors], dim=dim)
    pad_bytes, pad_len = size_pad_by_int4(tensor.shape[dim], tensor.itemsize)
    if pad_len > 0:
        tensor = F.pad(tensor, (0, pad_len), mode='constant', value=0)
    assert tensor.shape[dim] == pad_bytes
    return tensor.contiguous()


def pre_a2a_attn_out_grad_resend_qkv(
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
    assert q.dtype == attn_out.dtype
    assert attn_out_grad.shape == attn_out.shape

    # create merged_q
    merged_q = _concat_with_uint8_and_pad([attn_out_grad, attn_out, lse_norm, q], dim=1)
    return pre_a2a_qkv(merged_q, k, v, kv_dispatch_mask, q_seq_tokens, k_seq_tokens,
                       q_send_buffer_offset, k_send_buffer_offset, v_send_buffer_offset,
                       is_fwd=is_fwd, instance_id=instance_id)


def post_a2a_attn_out_grad_resend_qkv(
    recv_attn_out_shape: list[int], recv_lse_shape: list[int], recv_q_shape: list[int],
    recv_lse_dtype: torch.dtype,
    recv_k: Tensor, recv_v: Tensor,
    kv_dispatch_mask: Tensor, q_recv_seq_tokens: Tensor, k_recv_seq_tokens: Tensor,
    q_recv_buffer_offset: Tensor, k_recv_buffer_offset: Tensor, v_recv_buffer_offset: Tensor,
    is_fwd: bool, switch_buffer: bool = True, instance_id: int=None
):
    """NOTE: dispatcher is returned internally in post_a2a_qkv"""
    is_fwd = True
    assert len(recv_attn_out_shape) == 2
    assert len(recv_lse_shape) == 2 and recv_lse_shape[0] == recv_attn_out_shape[0]
    assert len(recv_q_shape) == 2 and recv_q_shape[0] == recv_attn_out_shape[0]

    # layout: attn_out_grad, attn_out, softmax_lse, q
    recv_q_splits = (
        recv_attn_out_shape[1], recv_attn_out_shape[1], recv_lse_shape[1],
        recv_q_shape[1]
    )
    recv_q_dtypes = [recv_k.dtype] * len(recv_q_splits)
    recv_q_dtypes[2] = recv_lse_dtype

    recv_q_bytes = [s * d.itemsize for s, d in zip(recv_q_splits, recv_q_dtypes)]
    merged_q_bytes = sum(recv_q_bytes)
    merged_q_bytes, pad_bytes = size_pad_by_int4(merged_q_bytes, torch.uint8.itemsize)
    recv_merged_q = recv_k.new_empty(
        (recv_attn_out_shape[0], merged_q_bytes), dtype=torch.uint8
    )

    recv_merged_q, recv_k, recv_v = post_a2a_qkv(
        recv_merged_q, recv_k, recv_v, kv_dispatch_mask,
        q_recv_seq_tokens, k_recv_seq_tokens,
        q_recv_buffer_offset, k_recv_buffer_offset, v_recv_buffer_offset,
        is_fwd, switch_buffer=switch_buffer, instance_id=instance_id
    )

    tensors = torch.split(
        recv_merged_q, [*recv_q_bytes, pad_bytes], dim=1
    )[:-1]
    recv_attn_out_grad, recv_attn_out, recv_lse, recv_q = [
        t.view(dtype).contiguous() for t, dtype in zip(tensors, recv_q_dtypes)
    ]
    return (
        recv_attn_out_grad, recv_attn_out, recv_lse,
        recv_q, recv_k, recv_v
    )


def pre_a2a_attn_out_with_lse(
    attn_out: Tensor, softmax_lse: Tensor,
    send_seqlens: Tensor, send_memcpy_metadata: Tensor, dispatcher_id: int
):
    assert softmax_lse.shape[0] == attn_out.shape[0]
    merged_attn_out = _concat_with_uint8_and_pad([attn_out, softmax_lse], dim=1)
    return pre_a2a_attn_out(
        merged_attn_out, send_seqlens, send_memcpy_metadata,
        instance_id=dispatcher_id,
    )
