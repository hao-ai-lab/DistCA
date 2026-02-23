from dataclasses import dataclass
import inspect
import math
import os
from typing import Tuple

import flash_attn.flash_attn_interface
# Support multiple flash-attn API shapes across versions/builds.
# Dao-AILab source exposes _flash_attn_varlen_forward / _flash_attn_varlen_backward (with underscore).
_fa_iface = flash_attn.flash_attn_interface
_flash_attn_varlen_fwd = (
    getattr(_fa_iface, "_wrapped_flash_attn_varlen_forward", None)
    or getattr(_fa_iface, "flash_attn_varlen_forward", None)
    or getattr(_fa_iface, "_flash_attn_varlen_forward", None)
)
_flash_attn_varlen_bwd = (
    getattr(_fa_iface, "_wrapped_flash_attn_varlen_backward", None)
    or getattr(_fa_iface, "flash_attn_varlen_backward", None)
    or getattr(_fa_iface, "_flash_attn_varlen_backward", None)
    or getattr(_fa_iface, "flash_attn_varlen_bwd", None)
)
# Public API: flash_attn_varlen_func(q, k, v, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, ...) -> (out, softmax_lse)
_use_fa_varlen_func = False
if _flash_attn_varlen_fwd is None:
    _flash_attn_varlen_fwd = getattr(_fa_iface, "flash_attn_varlen_func", None) or getattr(
        __import__("flash_attn", fromlist=["flash_attn_varlen_func"]),
        "flash_attn_varlen_func",
        None,
    )
    _use_fa_varlen_func = _flash_attn_varlen_fwd is not None
if _flash_attn_varlen_bwd is None:
    _flash_attn_varlen_bwd = getattr(_fa_iface, "flash_attn_varlen_bwd", None)
if _flash_attn_varlen_fwd is None:
    raise ImportError(
        "flash_attn: need one of _wrapped_flash_attn_varlen_forward, flash_attn_varlen_forward, "
        "_flash_attn_varlen_forward, or flash_attn_varlen_func. Your build exposes none. See installation docs / flash-attn build."
    )
if _flash_attn_varlen_bwd is None:
    raise ImportError(
        "flash_attn: need one of _wrapped_flash_attn_varlen_backward, flash_attn_varlen_backward, "
        "_flash_attn_varlen_backward, or flash_attn_varlen_bwd (required for backward). Your build exposes none. See installation docs / flash-attn build."
    )
# Dao-AILab v2.5.x uses positional args and window_size tuple; no window_size_left/right kwargs, no softcap.
# Older builds may have 13-arg forward (no block_table); newer have 14.
def _fa_uses_positional_api(fn):
    try:
        return "window_size_left" not in inspect.signature(fn).parameters
    except (ValueError, TypeError):
        return False
_fa_fwd_uses_positional = _fa_uses_positional_api(_flash_attn_varlen_fwd)
_fa_bwd_uses_positional = _fa_uses_positional_api(_flash_attn_varlen_bwd)
try:
    _fa_fwd_positional_accepts_block_table = "block_table" in inspect.signature(_flash_attn_varlen_fwd).parameters
except (ValueError, TypeError):
    _fa_fwd_positional_accepts_block_table = False  # safer default: omit block_table rather than risk crash
from distca.runtime.attn_kernels.ops import DispatcherWrapper, fast_a2a
from distca.runtime.megatron.packed_seq_params import PingPangPackedSeqParams, PingPangSingleStepPackedSeqParams
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.transformer.transformer_config import TransformerConfig
import torch
from torch import Tensor

from distca.runtime.attn_kernels.dispatch import (
    # fwd send attn_out, bwd send qkv grad
    post_a2a_attn_out, pre_a2a_attn_out_grad_resend_qkv, pre_a2a_attn_out_with_lse, pre_a2a_qkv,
    # bwd recv attn_out_grad and qkv, fwd recv qkv
    post_a2a_attn_out_grad_resend_qkv, post_a2a_qkv,
)
from distca.runtime.metadata import AlltoAllMetadata

# Module-level flag to ensure warning is only printed once
_deterministic_warning_printed = False

# is_deterministic = (os.environ.get("NVTE_ALLOW_NONDETERMINISTIC_ALGO", "0") == "0")
# if is_deterministic:
#     print("🟡 Using deterministic = True as default value in FusedCommAttn! This is not recommended for production use! Forcefully set it back to False. If you need to test, come to FusedCommAttn.py to modify this logic.")
#     os.environ["NVTE_ALLOW_NONDETERMINISTIC_ALGO"] = "1"
#     is_deterministic = False

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
    deterministic: bool = None
    return_attn_probs: bool = False
    block_table=None

    def __post_init__(self):
        if self.deterministic is None:
            env_val = os.getenv("NVTE_ALLOW_NONDETERMINISTIC_ALGO", "0")
            self.deterministic = env_val != "1"
            # print(f"[FlashAttnArgs] Set deterministic to {self.deterministic} (NVTE_ALLOW_NONDETERMINISTIC_ALGO={env_val})")
        if self.deterministic:
            global _deterministic_warning_printed
            if not _deterministic_warning_printed:
                print("⚠️⚠️⚠️ Using deterministic = True! This is not recommended when profiling for performance as it will degrade backward pass performance!")
                _deterministic_warning_printed = True



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
    
    if False:
        print("🟡 _qkv_to_attn_out_fwd: q.shape = ", q.shape)
        print("🟡 _qkv_to_attn_out_fwd: k.shape = ", k.shape)
        print("🟡 _qkv_to_attn_out_fwd: v.shape = ", v.shape)
        print("🟡 _qkv_to_attn_out_fwd: cu_seqlens_q.shape = ", fa_params.cu_seqlens_q, fa_params.cu_seqlens_q.shape)
        print("🟡 _qkv_to_attn_out_fwd: cu_seqlens_kv.shape = ", fa_params.cu_seqlens_kv, fa_params.cu_seqlens_kv.shape)
        print("🟡 _qkv_to_attn_out_fwd: max_seqlen_q = ", fa_params.max_seqlen_q)
        print("🟡 _qkv_to_attn_out_fwd: max_seqlen_kv = ", fa_params.max_seqlen_kv)
        print("🟡 _qkv_to_attn_out_fwd: dropout_p = ", fa_args.dropout_p)
        print("🟡 _qkv_to_attn_out_fwd: softmax_scale = ", softmax_scale)
        print("🟡 _qkv_to_attn_out_fwd: causal = ", fa_args.causal)
        print("🟡 _qkv_to_attn_out_fwd: window_size_left = ", fa_args.window_size[0])
        print("🟡 _qkv_to_attn_out_fwd: window_size_right = ", fa_args.window_size[1])
        print("🟡 _qkv_to_attn_out_fwd: softcap = ", fa_args.softcap)
        print("🟡 _qkv_to_attn_out_fwd: alibi_slopes.shape = ", fa_args.alibi_slopes.shape if fa_args.alibi_slopes is not None else None)
        print("🟡 _qkv_to_attn_out_fwd: return_attn_probs = ", fa_args.return_attn_probs)
        print("🟡 _qkv_to_attn_out_fwd: block_table.shape = ", fa_args.block_table.shape if fa_args.block_table is not None else None)
    
    if _use_fa_varlen_func:
        # flash_attn_varlen_func(q, k, v, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, dropout_p, softmax_scale, causal)
        result = _flash_attn_varlen_fwd(
            q, k, v,
            fa_params.cu_seqlens_q,
            fa_params.cu_seqlens_kv,
            fa_params.max_seqlen_q,
            fa_params.max_seqlen_kv,
            fa_args.dropout_p,
            softmax_scale,
            causal=fa_args.causal,
        )
        out_padded, softmax_lse = result[0], result[1]
        S_dmask, rng_state = None, None
    elif _fa_fwd_uses_positional:
        # Dao-AILab v2.5.x: returns (out, q, k, v, out_padded, softmax_lse, S_dmask, rng_state). Some builds have 13 args (no block_table).
        _pos_args = (
            q, k, v,
            fa_params.cu_seqlens_q,
            fa_params.cu_seqlens_kv,
            fa_params.max_seqlen_q,
            fa_params.max_seqlen_kv,
            fa_args.dropout_p,
            softmax_scale,
            fa_args.causal,
            fa_args.window_size,
            fa_args.alibi_slopes,
            fa_args.return_attn_probs and fa_args.dropout_p > 0,
        )
        if _fa_fwd_positional_accepts_block_table:
            _pos_args = _pos_args + (fa_args.block_table,)
        _out, _q, _k, _v, out_padded, softmax_lse, S_dmask, rng_state = _flash_attn_varlen_fwd(*_pos_args)
    else:
        out_padded, softmax_lse, S_dmask, rng_state = _flash_attn_varlen_fwd(
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
    if False:
        print("🟡 _qkv_to_attn_out_bwd: dout_padded.shape = ", dout_padded.shape)
        print("🟡 _qkv_to_attn_out_bwd: q.shape = ", q.shape)
        print("🟡 _qkv_to_attn_out_bwd: k.shape = ", k.shape)
        print("🟡 _qkv_to_attn_out_bwd: v.shape = ", v.shape)
        print("🟡 _qkv_to_attn_out_bwd: out_padded.shape = ", out_padded.shape)
        print("🟡 _qkv_to_attn_out_bwd: softmax_lse.shape = ", softmax_lse.shape)
        print("🟡 _qkv_to_attn_out_bwd: cu_seqlen_q.shape = ", cu_seqlen_q, cu_seqlen_q.shape)
        print("🟡 _qkv_to_attn_out_bwd: cu_seqlen_kv.shape = ", cu_seqlen_kv, cu_seqlen_kv.shape)
        print("🟡 _qkv_to_attn_out_bwd: max_seqlen_q = ", max_seqlen_q)
        print("🟡 _qkv_to_attn_out_bwd: max_seqlen_kv = ", max_seqlen_kv)
        print("🟡 _qkv_to_attn_out_bwd: dropout_p = ", fa_args.dropout_p)
        print("🟡 _qkv_to_attn_out_bwd: softmax_scale = ", softmax_scale)
        print("🟡 _qkv_to_attn_out_bwd: causal = ", fa_args.causal)
        print("🟡 _qkv_to_attn_out_bwd: window_size_left = ", fa_args.window_size[0])
        print("🟡 _qkv_to_attn_out_bwd: window_size_right = ", fa_args.window_size[1])
        print("🟡 _qkv_to_attn_out_bwd: softcap = ", fa_args.softcap)
        print("🟡 _qkv_to_attn_out_bwd: alibi_slopes.shape = ", fa_args.alibi_slopes.shape if fa_args.alibi_slopes is not None else None)
        print("🟡 _qkv_to_attn_out_bwd: deterministic = ", fa_args.deterministic)

    if _use_fa_varlen_func:
        # Public API path: flash_attn_varlen_func handles backward internally via autograd.
        # We need to call it again with grad enabled, or use the lower-level backward if available.
        # Since we saved S_dmask=None and rng_state=None in this path, we fall through to
        # the positional/kwargs backward which should still work with _flash_attn_varlen_bwd.
        if _flash_attn_varlen_bwd is not None:
            if _fa_bwd_uses_positional:
                _flash_attn_varlen_bwd(
                    dout_padded, q, k, v, out_padded.contiguous(), softmax_lse, dq, dk, dv,
                    cu_seqlen_q, cu_seqlen_kv, max_seqlen_q, max_seqlen_kv,
                    fa_args.dropout_p, softmax_scale, fa_args.causal,
                    fa_args.window_size, fa_args.alibi_slopes, fa_args.deterministic,
                )
            else:
                _flash_attn_varlen_bwd(
                    dout_padded, q, k, v, out_padded.contiguous(), softmax_lse, dq, dk, dv,
                    cu_seqlen_q, cu_seqlen_kv, max_seqlen_q, max_seqlen_kv,
                    fa_args.dropout_p, softmax_scale, fa_args.causal,
                    fa_args.window_size[0], fa_args.window_size[1], fa_args.softcap, fa_args.alibi_slopes,
                    fa_args.deterministic,
                )
        else:
            raise RuntimeError(
                "flash_attn backward not available. The public API (flash_attn_varlen_func) was used for forward, "
                "but no low-level backward function was found. Install a flash-attn build with varlen backward support."
            )
    elif _fa_bwd_uses_positional:
        # Dao-AILab v2.5.x: _flash_attn_varlen_backward(..., window_size, alibi_slopes, deterministic) — no softcap
        _flash_attn_varlen_bwd(
            dout_padded, q, k, v, out_padded.contiguous(), softmax_lse, dq, dk, dv,
            cu_seqlen_q, cu_seqlen_kv, max_seqlen_q, max_seqlen_kv,
            fa_args.dropout_p, softmax_scale, fa_args.causal,
            fa_args.window_size, fa_args.alibi_slopes, fa_args.deterministic,
        )
    else:
        _flash_attn_varlen_bwd(
            dout_padded, q, k, v, out_padded.contiguous(), softmax_lse, dq, dk, dv,
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


def _normalize_softmax_lse(softmax_lse, total_tokens, cu_seqlens_q=None, max_seqlen_q=None):
    """Normalize softmax_lse to shape (total_tokens, num_heads) for dispatch.

    Flash-attn versions return softmax_lse in different layouts:
      - 2D (total_tokens, num_heads): already correct, pass through.
      - 2D (num_heads, total_tokens): transpose.
      - 3D (batch_size, num_heads, max_seqlen_q): C++ varlen_fwd format; unpack
        using cu_seqlens_q to strip padding per sequence.
      - 3D with trailing 1: squeeze then handle as 2D.
      - 3D with one dim == total_tokens: permute that dim to front and reshape.

    Args:
        softmax_lse: Tensor from flash-attn forward.
        total_tokens: Expected first dimension (= attn_out.shape[0]).
        cu_seqlens_q: Cumulative sequence lengths (int32 tensor), optional.
        max_seqlen_q: Maximum sequence length, optional.

    Returns:
        Tensor of shape (total_tokens, num_heads).
    """
    if softmax_lse.ndim == 3:
        if softmax_lse.shape[-1] == 1:
            softmax_lse = softmax_lse.squeeze(-1)
        else:
            batch_size = softmax_lse.shape[0]
            num_heads_fa = softmax_lse.shape[1]
            max_sq = softmax_lse.shape[2]
            # C++ varlen returns (batch_size, num_heads, max_seqlen_q); unpack to (total_tokens, num_heads).
            if (cu_seqlens_q is not None
                    and (cu_seqlens_q.shape[0] - 1) == batch_size
                    and max_sq == max_seqlen_q):
                out_lse = torch.empty(
                    total_tokens, num_heads_fa,
                    dtype=softmax_lse.dtype, device=softmax_lse.device,
                )
                for i in range(batch_size):
                    start = cu_seqlens_q[i].item()
                    end = cu_seqlens_q[i + 1].item()
                    seqlen = end - start
                    # (num_heads, max_seqlen_q) -> [:, :seqlen] -> .T -> (seqlen, num_heads)
                    out_lse[start:end].copy_(softmax_lse[i, :, :seqlen].T.contiguous())
                softmax_lse = out_lse
            else:
                for i in range(3):
                    if softmax_lse.shape[i] == total_tokens:
                        perm = [i] + [j for j in range(3) if j != i]
                        softmax_lse = softmax_lse.permute(*perm).reshape(total_tokens, -1)
                        break

    if softmax_lse.ndim == 2 and softmax_lse.shape[0] != total_tokens:
        softmax_lse = softmax_lse.T.contiguous()

    if softmax_lse.shape[0] != total_tokens:
        numel = softmax_lse.numel()
        if numel % total_tokens == 0:
            softmax_lse = softmax_lse.reshape(total_tokens, -1)
        else:
            raise RuntimeError(
                f"softmax_lse layout incompatible with packed tokens: "
                f"attn_out.shape[0] (total_tokens) = {total_tokens}, "
                f"softmax_lse.shape = {softmax_lse.shape}, softmax_lse.numel() = {numel}. "
                f"Flash-attn varlen should return (total_tokens, num_heads); "
                f"this build returns layout that cannot be reshaped."
            )

    return softmax_lse


class FusedCommAttn(torch.autograd.Function):
    """
    Fused post-recv + core attention + pre-send kernel.
    """
    @staticmethod
    def forward(
        ctx, signal: Tensor,
        fwd_qkv_metadata: AlltoAllMetadata,
        bwd_qkv_metadata: AlltoAllMetadata,
        fwd_attn_out_metadata: AlltoAllMetadata,
        bwd_attn_out_qkv_metadata: AlltoAllMetadata,
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
        post_a2a_qkv(
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
        ])
        ctx.bwd_attn_max_seqlen_q = bwd_fa_params.max_seqlen_q
        ctx.bwd_attn_max_seqlen_kv = bwd_fa_params.max_seqlen_kv
        ctx.fa_args = flash_attn_args

        # Step 3: pre-dispatch attn out
        assert attn_out.shape == recv_q.shape
        softmax_lse_dtype = softmax_lse.dtype
        total_tokens = attn_out.shape[0]
        softmax_lse = _normalize_softmax_lse(
            softmax_lse, total_tokens, fwd_fa_params.cu_seqlens_q, fwd_fa_params.max_seqlen_q,
        )
        attn_out = pre_a2a_attn_out_with_lse(
            attn_out, softmax_lse, fwd_attn_out_metadata.seq_lens[0].send_seqlens,
            fwd_attn_out_metadata.send_memcpy_metadata[0],
            dispatcher_id,
        )
        signal = torch.empty((1,), device=signal.device, dtype=signal.dtype)

        saved_tensors.extend([
            bwd_attn_out_qkv_metadata.seq_lens[0].recv_seqlens,
            bwd_attn_out_qkv_metadata.seq_lens[1].recv_seqlens,
            *bwd_attn_out_qkv_metadata.recv_memcpy_metadata,
            bwd_qkv_metadata.kv_grad_send_dedup.main_copy_mask,
            bwd_qkv_metadata.kv_grad_send_dedup.num_copies,
            bwd_qkv_metadata.kv_grad_send_dedup.copy_start_id,
        ])
        recv_seqlens_q_total = bwd_attn_out_qkv_metadata.tensor_shape[0].recv_shape[0]
        ctx.bwd_q_shape = ctx.attn_out_shape = recv_seqlens_q_total, recv_q_shape[1]
        ctx.softmax_lse_shape = recv_seqlens_q_total, softmax_lse.shape[1]
        ctx.softmax_lse_dtype = softmax_lse_dtype
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
         bwd_attn_out_qkv_recv_seqlens_q, bwd_attn_out_qkv_recv_seqlens_k,
         bwd_attn_out_qkv_recv_q_offset, bwd_attn_out_qkv_recv_k_offset,
         bwd_attn_out_qkv_recv_v_offset,
         main_copy_mask, num_copies, copy_start_id
         ) = ctx.saved_tensors

        # Step 1: post-dispatch merged_q, k, v
        recv_k_shape = ctx.bwd_k_shape
        recv_k = torch.empty(
            recv_k_shape, dtype=signal_grad.dtype, device=signal_grad.device
        )
        recv_v = torch.empty_like(recv_k)
        torch.cuda.nvtx.range_push("post_a2a_attn_out_grad_resend_qkv")
        (recv_attn_out_grad, recv_attn_out, recv_lse, recv_q, recv_k, recv_v
         ) = post_a2a_attn_out_grad_resend_qkv(
            ctx.attn_out_shape, ctx.softmax_lse_shape, ctx.bwd_q_shape,
            ctx.softmax_lse_dtype,
            recv_k, recv_v,
            None,
            bwd_attn_out_qkv_recv_seqlens_q, bwd_attn_out_qkv_recv_seqlens_k,
            bwd_attn_out_qkv_recv_q_offset, bwd_attn_out_qkv_recv_k_offset,
            bwd_attn_out_qkv_recv_v_offset,
            is_fwd=True,
            switch_buffer=switch_buffer,
            instance_id=dispatcher_id,
        )
        recv_lse = recv_lse.T.contiguous()
        torch.cuda.nvtx.range_pop()
        # Step 2: call FA bwd.
        torch.cuda.nvtx.range_push("manual_bwd_fa")
        dq, dk, dv = _qkv_to_attn_out_bwd(
            recv_q, recv_k, recv_v, recv_attn_out, recv_attn_out_grad,
            recv_lse, ctx.fa_args, bwd_attn_cu_seqlens_q, bwd_attn_cu_seqlens_kv,
            ctx.bwd_attn_max_seqlen_q, ctx.bwd_attn_max_seqlen_kv,
        )
        torch.cuda.nvtx.range_pop()
        # Step 3: pre-dispatch q_grad, k_grad, v_grad
        torch.cuda.nvtx.range_push("pre_a2a_qkv_grad_memcpy")
        dq, dk, dv = pre_a2a_qkv(
            dq, dk, dv, None, bwd_qkv_grad_send_seqlens_q, bwd_qkv_grad_send_seqlens_k,
            bwd_q_grad_send_offset, bwd_k_grad_send_offset, bwd_v_grad_send_offset,
            is_fwd=False, instance_id=dispatcher_id,
            kv_grad_copy_shard_mask=main_copy_mask,
            pre_a2a_grad_acc_args=(num_copies, copy_start_id, bwd_qkv_grad_send_seqlens_k),
        )
        torch.cuda.nvtx.range_pop()
        signal_grad = torch.empty((1,), device=recv_q.device, dtype=recv_q.dtype)
        return signal_grad, *((None,) * 8)


class post_a2a_attn_out_with_lse(torch.autograd.Function):
    """
    Post a2a attention out with lse.
    """
    @staticmethod
    def forward(ctx, signal: Tensor,
                q: Tensor, k: Tensor, v: Tensor,
                num_heads_q: int,
                metadata: AlltoAllMetadata,
                bwd_attn_out_qkv_metadata: AlltoAllMetadata,
                dispatcher_id: int,
    ):
        switch_buffer = dispatcher_id is None
        recv_shape = metadata.tensor_shape[0].recv_shape
        recv_attn_out = torch.empty(
            recv_shape, dtype=signal.dtype, device=signal.device
        ).view(torch.uint8)

        recv_attn_out = post_a2a_attn_out(
            recv_attn_out,
            metadata.seq_lens[0].recv_seqlens,
            *metadata.recv_memcpy_metadata,
            switch_buffer=switch_buffer,
            instance_id=dispatcher_id,
        )

        lse_dtype = torch.float32
        hidden_bytes = math.prod(q.shape[1:]) * signal.itemsize // torch.uint8.itemsize
        lse_bytes = num_heads_q * lse_dtype.itemsize // torch.uint8.itemsize

        attn_out = recv_attn_out[:, :hidden_bytes].view(signal.dtype).unsqueeze(1)
        softmax_lse_bytes = recv_attn_out[:, hidden_bytes:hidden_bytes + lse_bytes]

        ctx.save_for_backward(
            q, k, v, attn_out, softmax_lse_bytes,
            bwd_attn_out_qkv_metadata.kv_replica_mask,
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
        q, k, v, attn_out, softmax_lse_bytes, *send_metadata = ctx.saved_tensors

        attn_out = attn_out.reshape(attn_out.shape[0], -1)
        grad_attn_out = grad_attn_out.reshape(grad_attn_out.shape[0], -1)
        q = q.reshape(q.shape[0], -1)
        k = k.reshape(k.shape[0], -1)
        v = v.reshape(v.shape[0], -1)

        pre_a2a_attn_out_grad_resend_qkv(
            grad_attn_out, attn_out, softmax_lse_bytes, q, k, v,
            *send_metadata,
            instance_id=ctx.dispatcher_id
        )

        signal_grad = grad_attn_out.new_zeros((1,))
        return signal_grad, *((None,) * 7)


@torch.no_grad()
def dummy_backward_single_sided(
    config: TransformerConfig,
    packed_seq_params: PingPangSingleStepPackedSeqParams,
    dtype: torch.dtype,
    device: torch.device,
    attn_out_grad_all2all: bool=True,
    qkv_grad_all2all: bool=True,
):
    """
    Dummy backward for a single layer. This is exactly the backward of FusedCommAttn + All2All.
    """
    assert packed_seq_params.bwd_packed_seq_params is not None

    # create some dummy data
    num_heads = config.num_attention_heads // config.tensor_model_parallel_size
    hidden_size_tp = config.hidden_size // config.tensor_model_parallel_size

    dispatcher_id = packed_seq_params.dispatcher_id

    bwd_attn_out_qkv_metadata = packed_seq_params.attn_out_bwd_metadata
    bwd_qkv_metadata = packed_seq_params.qkv_bwd_metadata
    bwd_fa_params = packed_seq_params.bwd_packed_seq_params

    if attn_out_grad_all2all:
        # This operation does not need to wait for any prior message.
        with torch.cuda.stream(packed_seq_params.stream):
            torch.cuda.nvtx.range_push("dummy_backward_a2a_attn_grad")
            fast_a2a(
                *bwd_attn_out_qkv_metadata.fa2a_metadata,
                bwd_attn_out_qkv_metadata.my_rank_send_offset,
                bwd_attn_out_qkv_metadata.my_rank_recv_offset,
                bwd_attn_out_qkv_metadata.my_rank_send_sz,
                instance_id=dispatcher_id,
            )
            torch.cuda.nvtx.range_pop()
        if packed_seq_params.stream is not None:
            torch.cuda.current_stream().wait_stream(packed_seq_params.stream)
    else:
        assert dispatcher_id is not None

    recv_k_shape = bwd_attn_out_qkv_metadata.tensor_shape[1].recv_shape
    recv_k = torch.empty(recv_k_shape, dtype=dtype, device=device)
    recv_v = torch.empty_like(recv_k)
    recv_q_len = bwd_attn_out_qkv_metadata.tensor_shape[0].recv_shape[0]
    recv_q_shape = recv_q_len, hidden_size_tp
    softmax_lse_shape = recv_q_len, num_heads
    torch.cuda.nvtx.range_push("dummy_backward_post_a2a_attn_grad")
    (recv_attn_out_grad, recv_attn_out, recv_lse, recv_q, recv_k, recv_v
    ) = post_a2a_attn_out_grad_resend_qkv(
        recv_q_shape, softmax_lse_shape, recv_q_shape, torch.float32,
        recv_k, recv_v,
        None,
        bwd_attn_out_qkv_metadata.seq_lens[0].recv_seqlens,
        bwd_attn_out_qkv_metadata.seq_lens[1].recv_seqlens,
        *bwd_attn_out_qkv_metadata.recv_memcpy_metadata,
        is_fwd=True,
        switch_buffer=dispatcher_id is None,
        instance_id=dispatcher_id,
    )
    recv_lse = recv_lse.T.contiguous()
    torch.cuda.nvtx.range_pop()

    torch.cuda.nvtx.range_push("dummy_backward_bwd_attn")
    dq, dk, dv = _qkv_to_attn_out_bwd(
        recv_q, recv_k, recv_v, recv_attn_out, recv_attn_out_grad,
        recv_lse,
        FlashAttnArgs(
            num_heads_q=num_heads,
            num_heads_kv=config.num_query_groups // config.tensor_model_parallel_size,
            head_dim=hidden_size_tp // num_heads,
            return_attn_probs=True,
        ),
        bwd_fa_params.cu_seqlens_q, bwd_fa_params.cu_seqlens_kv,
        bwd_fa_params.max_seqlen_q, bwd_fa_params.max_seqlen_kv,
    )
    torch.cuda.nvtx.range_pop()

    torch.cuda.nvtx.range_push("dummy_backward_pre_qkv_grad_all2all")
    pre_a2a_qkv(
        dq, dk, dv, None,
        bwd_qkv_metadata.seq_lens[0].send_seqlens,
        bwd_qkv_metadata.seq_lens[1].send_seqlens,
        *bwd_qkv_metadata.send_memcpy_metadata,
        is_fwd=False,
        instance_id=dispatcher_id,
        kv_grad_copy_shard_mask=bwd_qkv_metadata.kv_grad_send_dedup.main_copy_mask,
        pre_a2a_grad_acc_args=(
            bwd_qkv_metadata.kv_grad_send_dedup.num_copies,
            bwd_qkv_metadata.kv_grad_send_dedup.copy_start_id,
            bwd_qkv_metadata.seq_lens[1].send_seqlens,
        ),
    )
    torch.cuda.nvtx.range_pop()

    if qkv_grad_all2all:
        if packed_seq_params.stream is not None:
            packed_seq_params.stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(packed_seq_params.stream):
            fast_a2a(
                *bwd_qkv_metadata.fa2a_metadata,
                bwd_qkv_metadata.my_rank_send_offset,
                bwd_qkv_metadata.my_rank_recv_offset,
                bwd_qkv_metadata.my_rank_send_sz,
                instance_id=dispatcher_id,
            )
        if dispatcher_id is None:
            DispatcherWrapper.switch_buffer()
        return None
    else:
        assert dispatcher_id is not None
        event = torch.cuda.Event()
        event.record(torch.cuda.current_stream())
        return event


def dummy_backward_send_qkv(packed_seq_params: PingPangSingleStepPackedSeqParams, event: torch.cuda.Event):
    dispatcher_id = packed_seq_params.dispatcher_id
    assert dispatcher_id is not None
    assert event is not None

    bwd_qkv_metadata = packed_seq_params.qkv_bwd_metadata
    with torch.cuda.stream(packed_seq_params.stream):
        packed_seq_params.stream.wait_event(event)
        fast_a2a(
            *bwd_qkv_metadata.fa2a_metadata,
            bwd_qkv_metadata.my_rank_send_offset,
            bwd_qkv_metadata.my_rank_recv_offset,
            bwd_qkv_metadata.my_rank_send_sz,
            instance_id=dispatcher_id,
        )
        all2all_event = torch.cuda.Event()
        all2all_event.record(packed_seq_params.stream)
    return all2all_event


@torch.no_grad()
def dummy_backward(
    config: TransformerConfig,
    packed_seq_params,
    dtype: torch.dtype,
    device: torch.device,
):
    if isinstance(packed_seq_params, PingPangSingleStepPackedSeqParams):
        dummy_backward_single_sided(config, packed_seq_params, dtype, device,)
        return
    assert isinstance(packed_seq_params, PingPangPackedSeqParams)
    # PingPong schedule:
    # Compute:              Attn_1      Attn_0
    # Comm:     out_grad_1  out_grad_0  qkv_grad_1  qkv_grad_0
    # Launch out_grad_1 and Attn_1
    with torch.cuda.nvtx.range("dummy_bwd_ping_pong_1"):
        compute_done_1 = dummy_backward_single_sided(
            config, packed_seq_params.seq_params[1], dtype, device,
            qkv_grad_all2all=False,
        )
    # Launch out_grad_0 and Attn_0
    with torch.cuda.nvtx.range("dummy_bwd_ping_pong_0"):
        compute_done_0 = dummy_backward_single_sided(
            config, packed_seq_params.seq_params[0], dtype, device,
            qkv_grad_all2all=False,
        )
    # Launch qkv_grad_1
    # At this moment, the last operation on the compute stream is the attention & memcpy
    # for ping-pong split 0 instead of split 1, so we cannot ask comm stream wait for compute
    # stream. Instead, we use a previously recorded event to synchronize.
    with torch.cuda.nvtx.range("dummy_bwd_ping_pong_qkv_all2all_1"):
        all2all_event = dummy_backward_send_qkv(packed_seq_params.seq_params[1], compute_done_1)

        # Dummy backward does not have a qkv grad to receive, so we should manually release the buffer
        torch.cuda.current_stream().wait_event(all2all_event)
        DispatcherWrapper.release(packed_seq_params.seq_params[1].dispatcher_id)

    with torch.cuda.nvtx.range("dummy_bwd_ping_pong_qkv_all2all_0"):
        all2all_event = dummy_backward_send_qkv(packed_seq_params.seq_params[0], compute_done_0)

        # Same as above.
        torch.cuda.current_stream().wait_event(all2all_event)
        DispatcherWrapper.release(packed_seq_params.seq_params[0].dispatcher_id)
