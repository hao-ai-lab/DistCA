from dataclasses import dataclass
import math
from typing import Tuple

from flash_attn.flash_attn_interface import (
    _wrapped_flash_attn_varlen_forward, _wrapped_flash_attn_varlen_backward,
    flash_attn_varlen_func
)
from megatron.core.packed_seq_params import PackedSeqParams
import torch
from torch import Tensor

from d2.runtime.attn_kernels.dispatch import (
    # fwd send attn_out, bwd send qkv grad
    post_fast_a2a_attn_out, pre_fast_a2a_attn_out_grad_resend_qkv, pre_fast_a2a_attn_out_with_lse, pre_fast_a2a_qkv,
    # bwd recv attn_out_grad and qkv, fwd recv qkv
    post_fast_a2a_attn_out_grad_resend_qkv, post_fast_a2a_qkv,
)
from d2.runtime.fast_alltoall_metadata import FastAlltoAllMetadata


@dataclass
class FlashAttnArgs:
    num_heads_q: int
    num_heads_kv: int
    head_dim: int
    dropout_p: float = 0.0
    softmax_scale: float = None
    causal: bool = True
    window_size: Tuple[int, int] = (-1, -1)
    softcap: float = 0.0
    alibi_slopes = None
    deterministic: bool = False
    return_attn_probs: bool = False
    block_table=None


def _qkv_to_attn_out_fwd(q: Tensor, k: Tensor, v: Tensor,
                         fa_params: PackedSeqParams, fa_args: FlashAttnArgs,):
    assert fa_args.causal
    assert fa_args.window_size == (-1, -1)
    assert fa_args.alibi_slopes is None
    assert fa_args.return_attn_probs
    assert fa_args.block_table is None

    # reshape q,k,v from the received shape (2 dim)
    q = q.reshape(q.shape[0], fa_args.num_heads_q, fa_args.head_dim)
    k = k.reshape(k.shape[0], fa_args.num_heads_kv, fa_args.head_dim)
    v = v.reshape(v.shape[0], fa_args.num_heads_kv, fa_args.head_dim)

    softmax_scale = fa_args.softmax_scale
    if softmax_scale is None:
        softmax_scale = q.shape[-1] ** (-0.5)

    head_size_og = q.size(2)
    if head_size_og % 8 != 0:
        q = torch.nn.functional.pad(q, [0, 8 - head_size_og % 8])
        k = torch.nn.functional.pad(k, [0, 8 - head_size_og % 8])
        v = torch.nn.functional.pad(v, [0, 8 - head_size_og % 8])
    # rng_states seems only to be debug used. Skip it.
    out_padded, softmax_lse, S_dmask, rng_state = _wrapped_flash_attn_varlen_forward(
        q, k, v,
        fa_params.cu_seqlens_q,
        fa_params.cu_seqlens_kv,
        fa_params.max_seqlen_q,
        fa_params.max_seqlen_kv,
        fa_args.dropout_p,
        softmax_scale,
        causal=fa_args.causal,
        window_size_left=fa_args.window_size[0],
        window_size_right=fa_args.window_size[1],
        softcap=fa_args.softcap,
        alibi_slopes=fa_args.alibi_slopes,
        return_softmax=fa_args.return_attn_probs and fa_args.dropout_p > 0,
        block_table=fa_args.block_table,
    )
    # remove padding, if necessary; then reshape
    out = out_padded[..., :head_size_og]
    out = out.reshape(out.shape[0], fa_args.num_heads_q * fa_args.head_dim)
    return out, softmax_lse


def _qkv_to_attn_out_bwd(
    q: Tensor, k: Tensor, v: Tensor, attn_out: Tensor,
    attn_out_grad: Tensor, softmax_lse: Tensor,
    fa_args: FlashAttnArgs, cu_seqlen_q: Tensor, cu_seqlen_kv: Tensor,
    max_seqlen_q: int, max_seqlen_kv: int,
):
    num_heads_q = fa_args.num_heads_q
    num_heads_kv = fa_args.num_heads_kv
    head_dim = fa_args.head_dim
    q = q.reshape(q.shape[0], num_heads_q, head_dim)
    k = k.reshape(k.shape[0], num_heads_kv, head_dim)
    v = v.reshape(v.shape[0], num_heads_kv, head_dim)
    out = attn_out.reshape(attn_out.shape[0], num_heads_q, head_dim)
    dout = attn_out_grad.reshape(
        attn_out_grad.shape[0], num_heads_q, head_dim
    )

    softmax_scale = fa_args.softmax_scale
    if softmax_scale is None:
        softmax_scale = q.shape[-1] ** (-0.5)

    # maybe padding
    out_padded = out
    dout_padded = dout
    if head_dim % 8 != 0:
        q = torch.nn.functional.pad(q, [0, 8 - head_dim % 8])
        k = torch.nn.functional.pad(k, [0, 8 - head_dim % 8])
        v = torch.nn.functional.pad(v, [0, 8 - head_dim % 8])
        out_padded = torch.nn.functional.pad(out_padded, [0, 8 - head_dim % 8])
        dout_padded = torch.nn.functional.pad(dout_padded, [0, 8 - head_dim % 8])

    dq, dk, dv = torch.empty_like(q), torch.empty_like(k), torch.empty_like(v)
    _wrapped_flash_attn_varlen_backward(
        dout_padded, q, k, v, out_padded, softmax_lse, dq, dk, dv,
        cu_seqlen_q, cu_seqlen_kv, max_seqlen_q, max_seqlen_kv,
        fa_args.dropout_p, softmax_scale, fa_args.causal,
        fa_args.window_size[0], fa_args.window_size[1], fa_args.softcap, fa_args.alibi_slopes,
        fa_args.deterministic,
    )
    dq = dq[..., : dout.shape[-1]]
    dk = dk[..., : dout.shape[-1]]
    dv = dv[..., : dout.shape[-1]]
    dq = dq.reshape(dq.shape[0], num_heads_q * head_dim)
    dk = dk.reshape(dk.shape[0], num_heads_kv * head_dim)
    dv = dv.reshape(dv.shape[0], num_heads_kv * head_dim)
    return dq, dk, dv


class FusedCommAttn(torch.autograd.Function):
    """
    Fused post-recv + core-attention + pre-send kernel.
    """
    @staticmethod
    def forward(
        ctx, signal: Tensor,
        fwd_qkv_metadata: FastAlltoAllMetadata,
        bwd_qkv_metadata: FastAlltoAllMetadata,
        fwd_attn_out_metadata: FastAlltoAllMetadata,
        bwd_attn_out_qkv_metadata: FastAlltoAllMetadata,
        fwd_fa_params: PackedSeqParams,
        bwd_fa_params: PackedSeqParams,
        dispatcher_id: int,
        flash_attn_args: FlashAttnArgs,
    ):
        # Step 1: receive QKV tensors
        switch_buffer = dispatcher_id is None
        saved_tensors = []

        recv_q_shape = fwd_qkv_metadata.tensor_shape[0].recv_shape
        recv_k_shape = fwd_qkv_metadata.tensor_shape[1].recv_shape
        recv_q = torch.empty(recv_q_shape, dtype=signal.dtype, device=signal.device)
        recv_k = torch.empty(recv_k_shape, dtype=signal.dtype, device=signal.device)
        recv_v = torch.empty_like(recv_k)
        print(123, flush=True)
        post_fast_a2a_qkv(
            recv_q, recv_k, recv_v, None,
            fwd_qkv_metadata.seq_lens[0].recv_seqlens, fwd_qkv_metadata.seq_lens[1].recv_seqlens,
            *fwd_qkv_metadata.recv_memcpy_metadata,
            is_fwd=True,
            switch_buffer=switch_buffer,
            instance_id=dispatcher_id,
        )

        # Save metadata and tensors for pre-dispatch-qkv-bwd
        # Do not store the KV-replica here, because it's a qkv backward.
        saved_tensors.extend([
            # q_grad, k_grad send shape
            bwd_qkv_metadata.seq_lens[0].send_seqlens,
            bwd_qkv_metadata.seq_lens[1].send_seqlens,
            # q_grad, k_grad, v_grad communication
            *bwd_qkv_metadata.send_memcpy_metadata,
        ])
        ctx.dispatcher_id = dispatcher_id
        # Step 2. call FA fwd.
        attn_out, softmax_lse = _qkv_to_attn_out_fwd(
            recv_q, recv_k, recv_v, fwd_fa_params, flash_attn_args,
        )
        saved_tensors.extend([
            bwd_fa_params.cu_seqlens_q, bwd_fa_params.cu_seqlens_kv,
            bwd_fa_params.max_seqlen_q, bwd_fa_params.max_seqlen_kv,
        ])
        ctx.fa_args = flash_attn_args

        # Step 3: pre-dispatch attn out
        assert attn_out.shape == recv_q.shape
        softmax_lse = softmax_lse.T.contiguous().view(attn_out.dtype)
        print(456, flush=True)
        attn_out = pre_fast_a2a_attn_out_with_lse(
            attn_out, softmax_lse, fwd_attn_out_metadata.seq_lens[0].send_seqlens,
            fwd_attn_out_metadata.send_memcpy_metadata[0],
            dispatcher_id,
        )
        print(789, flush=True)
        signal = torch.empty((1,), device=attn_out.device, dtype=attn_out.dtype)

        saved_tensors.extend([
            bwd_attn_out_qkv_metadata.seq_lens[0].recv_seqlens,
            bwd_attn_out_qkv_metadata.seq_lens[1].recv_seqlens,
            *bwd_attn_out_qkv_metadata.recv_memcpy_metadata,
        ])
        ctx.attn_out_shape = attn_out.shape
        ctx.softmax_lse_shape = softmax_lse.shape
        ctx.bwd_q_shape = recv_q.shape
        ctx.bwd_k_shape = bwd_attn_out_qkv_metadata.tensor_shape[1].recv_shape
        ctx.save_for_backward(*saved_tensors)
        return signal

    @staticmethod
    def backward(ctx, signal_grad: Tensor):
        dispatcher_id = ctx.dispatcher_id
        switch_buffer = ctx.dispatcher_id is None
        (bwd_qkv_grad_send_seqlens_q, bwd_qkv_grad_send_seqlens_k,
         bwd_q_grad_send_offset, bwd_k_grad_send_offset, bwd_v_grad_send_offset,
         bwd_attn_cu_seqlens_q, bwd_attn_cu_seqlens_kv,
         bwd_attn_max_seqlen_q, bwd_attn_max_seqlen_kv,
         bwd_attn_out_qkv_recv_seqlens_q, bwd_attn_out_qkv_recv_seqlens_k,
         bwd_attn_out_qkv_recv_q_offset, bwd_attn_out_qkv_recv_k_offset,
         bwd_attn_out_qkv_recv_v_offset
         ) = ctx.saved_tensors

        # Step 1: post-dispatch merged_q, k, v
        recv_k_shape = ctx.bwd_k_shape
        recv_k = torch.empty(
            recv_k_shape, dtype=signal_grad.dtype, device=signal_grad.device
        )
        recv_v = torch.empty_like(recv_k)
        (recv_attn_out_grad, recv_attn_out, recv_lse, recv_q, recv_k, recv_v
         ) = post_fast_a2a_attn_out_grad_resend_qkv(
            ctx.attn_out_shape, ctx.softmax_lse_shape, ctx.bwd_q_shape,
            recv_k, recv_v,
            None,
            bwd_attn_out_qkv_recv_seqlens_q, bwd_attn_out_qkv_recv_seqlens_k,
            bwd_attn_out_qkv_recv_q_offset, bwd_attn_out_qkv_recv_k_offset,
            bwd_attn_out_qkv_recv_v_offset,
            is_fwd=True,
            switch_buffer=switch_buffer,
            instance_id=dispatcher_id,
        )
        recv_lse = recv_lse.view(recv_attn_out_grad.dtype).T
        # Step 2: call FA bwd.
        dq, dk, dv = _qkv_to_attn_out_bwd(
            recv_q, recv_k, recv_v, recv_attn_out, recv_attn_out_grad,
            recv_lse, ctx.fa_args, bwd_attn_cu_seqlens_q, bwd_attn_cu_seqlens_kv,
            bwd_attn_max_seqlen_q, bwd_attn_max_seqlen_kv,
        )
        # Step 3: pre-dispatch q_grad, k_grad, v_grad
        dq, dk, dv = pre_fast_a2a_qkv(
            dq, dk, dv, None, bwd_qkv_grad_send_seqlens_q, bwd_qkv_grad_send_seqlens_k,
            bwd_q_grad_send_offset, bwd_k_grad_send_offset, bwd_v_grad_send_offset,
            is_fwd=False, instance_id=dispatcher_id
        )
        signal_grad = torch.empty((1,), device=recv_q.device, dtype=recv_q.dtype)
        return signal_grad


class post_a2a_attn_out_with_lse(torch.autograd.Function):
    """
    Post a2a attention out with lse.
    """
    @staticmethod
    def forward(ctx, signal: Tensor,
                q: Tensor, k: Tensor, v: Tensor,
                metadata: FastAlltoAllMetadata,
                bwd_attn_out_qkv_metadata: FastAlltoAllMetadata,
                dispatcher_id: int,
    ):
        switch_buffer = dispatcher_id is None
        recv_shape = metadata.tensor_shape[0].recv_shape
        recv_attn_out = torch.empty(
            recv_shape, dtype=signal.dtype, device=signal.device
        )
        post_fast_a2a_attn_out(
            recv_attn_out,
            metadata.seq_lens[0].recv_seqlens,
            *metadata.recv_memcpy_metadata,
            switch_buffer=switch_buffer,
            instance_id=dispatcher_id,
        )
        hidden_size = math.prod(q.shape[1:])
        lse_size = torch.float32.itemsize // signal.itemsize
        attn_out = recv_attn_out[:, :hidden_size].unsqueeze(1)
        softmax_lse = recv_attn_out[:, hidden_size:hidden_size + lse_size]
        ctx.save_for_backward(
            q, k, v, attn_out, softmax_lse,
            # q_grad, k_grad send shape
            bwd_attn_out_qkv_metadata.seq_lens[0].send_seqlens,
            bwd_attn_out_qkv_metadata.seq_lens[1].send_seqlens,
            # q_grad, k_grad, v_grad communication
            *bwd_attn_out_qkv_metadata.send_memcpy_metadata,
        )
        ctx.dispatcher_id = dispatcher_id
        return attn_out

    @staticmethod
    def backward(ctx, grad_attn_out: Tensor):
        q, k, v, attn_out, softmax_lse, *send_metadata = ctx.saved_tensors

        pre_fast_a2a_attn_out_grad_resend_qkv(
            grad_attn_out, attn_out, softmax_lse, q, k, v,
            None,
            *send_metadata,
            instance_id=ctx.dispatcher_id
        )

        signal_grad = grad_attn_out.new_empty((1,))
        return signal_grad, *((None,) * 6)
