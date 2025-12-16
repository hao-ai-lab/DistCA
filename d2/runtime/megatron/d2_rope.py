import torch
import triton
import triton.language as tl
from torch import Tensor
from typing import Optional
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.models.common.embeddings.rope_utils import _apply_rotary_pos_emb_bshd


def precompute_rope_final_indices(
    cu_seqlens: Tensor,
    shard_logical_range: Tensor,
    device: str = 'cpu'
) -> Tensor:
    print("Precompute final_indices.")
    seqlens = cu_seqlens[1:] - cu_seqlens[:-1]
    offsets = shard_logical_range[:, 0] - cu_seqlens[:-1]
    token_offsets = torch.repeat_interleave(offsets, seqlens)
    total_tokens = int(token_offsets.shape[0])
    final_indices = torch.arange(total_tokens, device=device, dtype=torch.long) + token_offsets.to(device=device, dtype=torch.long)
    return final_indices


def apply_rotary_pos_emb_d2(
    t: Tensor,
    freqs: Tensor,
    config: TransformerConfig,
    cu_seqlens: Optional[Tensor] = None,
    shard_logical_range: Optional[Tensor] = None,
    final_indices: Optional[Tensor] = None,  # Precomputed indices to avoid repeated computation
    mscale: float = 1.0
) -> Tensor:
    """
    Apply rotary positional embedding with D2 support.

    Returns:
        Output tensor with RoPE applied
    """
    with torch.cuda.nvtx.range("RoPE_d2.apply_rotary_pos_emb_d2_opt"):
        print(f"[RoPE] Input tensor t.shape = {t.shape}")
        if final_indices is not None:
            print(f"[RoPE] Precomputed final_indices.shape = {final_indices.shape}")
        else:
            print(f"[RoPE] Computing final_indices from cu_seqlens and shard_logical_range")
    
        if final_indices is None:
            if cu_seqlens is None or shard_logical_range is None:
                raise ValueError("Either final_indices or both cu_seqlens and shard_logical_range must be provided")
            seqlens = cu_seqlens[1:] - cu_seqlens[:-1]
            offsets = shard_logical_range[:, 0] - cu_seqlens[:-1]
            token_offsets = torch.repeat_interleave(offsets, seqlens)
            final_indices = torch.arange(t.shape[0], device=t.device, dtype=torch.long) + token_offsets.to(device=t.device, dtype=torch.long)
        else:
            final_indices = final_indices.to(device=t.device, dtype=torch.long)
        
        curr_freqs = freqs[final_indices]
        print(f"[RoPE] freqs shape is: {freqs.shape}")
        print(f"[RoPE] curr_freqs shape is: {curr_freqs.shape}")
        with torch.cuda.nvtx.range("RoPE_d2.apply_rotary_pos_emb_bshd"):
            t_out = _apply_rotary_pos_emb_bshd(
                t.unsqueeze(1),
                curr_freqs,
                rotary_interleaved=config.rotary_interleaved,
                multi_latent_attention=config.multi_latent_attention,
                mscale=mscale
            )
        
        return t_out.squeeze(1)


@triton.jit
def _rope_kernel_split(
    t_ptr,              # [Total_Tokens, Num_Heads, Head_Dim]
    freqs_ptr,          # [Max_Seq_Len, Head_Dim]
    indices_ptr,        # [Total_Tokens]
    stride_t_n, stride_t_h, stride_t_d,
    stride_f_s, stride_f_d,
    mscale,
    ROTARY_INTERLEAVED: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid_token = tl.program_id(0)
    pid_head = tl.program_id(1)

    rot_idx = tl.load(indices_ptr + pid_token)
    t_base_offset = pid_token * stride_t_n + pid_head * stride_t_h
    f_base_offset = rot_idx * stride_f_s
    
    HALF_DIM = HEAD_DIM // 2
    
    offs_half = tl.arange(0, BLOCK_SIZE // 2)
    mask_half = offs_half < HALF_DIM

    ptr_x1 = t_ptr + t_base_offset + offs_half * stride_t_d
    ptr_x2 = t_ptr + t_base_offset + (offs_half + HALF_DIM) * stride_t_d
    
    x1 = tl.load(ptr_x1, mask=mask_half, other=0.0).to(tl.float32)
    x2 = tl.load(ptr_x2, mask=mask_half, other=0.0).to(tl.float32)

    ptr_f = freqs_ptr + f_base_offset + offs_half * stride_f_d
    theta = tl.load(ptr_f, mask=mask_half, other=0.0).to(tl.float32)

    cos_v = tl.cos(theta)
    sin_v = tl.sin(theta)

    if mscale != 1.0:
        cos_v = cos_v * mscale
        sin_v = sin_v * mscale

    out1 = x1 * cos_v - x2 * sin_v
    out2 = x2 * cos_v + x1 * sin_v
    
    tl.store(ptr_x1, out1, mask=mask_half)
    tl.store(ptr_x2, out2, mask=mask_half)


@triton.jit
def _rope_kernel_interleaved(
    t_ptr, freqs_ptr, indices_ptr,
    stride_t_n, stride_t_h, stride_t_d,
    stride_f_s, stride_f_d,
    mscale,
    HEAD_DIM: tl.constexpr,
    BLOCK_SIZE: tl.constexpr
):
    pid_token = tl.program_id(0)
    pid_head = tl.program_id(1)
    
    rot_idx = tl.load(indices_ptr + pid_token)
    
    t_offset = pid_token * stride_t_n + pid_head * stride_t_h
    f_offset = rot_idx * stride_f_s
    
    offs = tl.arange(0, BLOCK_SIZE)
    mask = offs < HEAD_DIM
    
    ptr_x = t_ptr + t_offset + offs * stride_t_d
    x = tl.load(ptr_x, mask=mask, other=0.0).to(tl.float32)
    
    ptr_f = freqs_ptr + f_offset + offs * stride_f_d
    theta = tl.load(ptr_f, mask=mask, other=0.0).to(tl.float32)
    
    cos_v = tl.cos(theta)
    sin_v = tl.sin(theta)
    
    if mscale != 1.0:
        cos_v = cos_v * mscale
        sin_v = sin_v * mscale
        
    is_odd = offs % 2
    x_swap_idx = tl.where(is_odd, offs - 1, offs + 1)
    
    ptr_x_swap = t_ptr + t_offset + x_swap_idx * stride_t_d
    x_swap = tl.load(ptr_x_swap, mask=mask, other=0.0).to(tl.float32)
    
    sign = tl.where(is_odd, 1.0, -1.0)
    
    x_rot = x_swap * sign
    out = x * cos_v + x_rot * sin_v
    
    tl.store(ptr_x, out, mask=mask)


def apply_rotary_pos_emb_d2_triton(
    t: Tensor,
    freqs: Tensor,
    config: TransformerConfig,
    cu_seqlens: Optional[Tensor] = None,
    shard_logical_range: Optional[Tensor] = None,
    final_indices: Optional[Tensor] = None,
    mscale: float = 1.0
) -> Tensor:
    """Triton implementation of RoPE."""
    
    if final_indices is None:
        if cu_seqlens is None or shard_logical_range is None:
            raise ValueError("Need cu_seqlens/shard_logical_range")
        seqlens = cu_seqlens[1:] - cu_seqlens[:-1]
        offsets = shard_logical_range[:, 0] - cu_seqlens[:-1]
        token_offsets = torch.repeat_interleave(offsets, seqlens)
        final_indices = torch.arange(t.shape[0], device=t.device, dtype=torch.long) + token_offsets
    else:
        final_indices = final_indices.to(device=t.device, dtype=torch.long)

    if freqs.ndim > 2:
        freqs = freqs.view(-1, freqs.shape[-1])
    if not freqs.is_contiguous():
        freqs = freqs.contiguous()

    if not t.is_contiguous():
        t = t.contiguous()
        
    original_shape = t.shape
    if t.ndim == 4:
        t_flat = t.view(-1, t.shape[2], t.shape[3])
    else:
        t_flat = t

    total_tokens, num_heads, head_dim = t_flat.shape
    
    if config.rotary_interleaved:
        BLOCK_SIZE = triton.next_power_of_2(head_dim)
        grid = (total_tokens, num_heads)
        _rope_kernel_interleaved[grid](
            t_flat, freqs, final_indices,
            t_flat.stride(0), t_flat.stride(1), t_flat.stride(2),
            freqs.stride(0), freqs.stride(1),
            mscale,
            HEAD_DIM=head_dim,
            BLOCK_SIZE=BLOCK_SIZE
        )
    else:
        BLOCK_SIZE = triton.next_power_of_2(head_dim)
        grid = (total_tokens, num_heads)
        _rope_kernel_split[grid](
            t_flat, freqs, final_indices,
            t_flat.stride(0), t_flat.stride(1), t_flat.stride(2),
            freqs.stride(0), freqs.stride(1),
            mscale,
            ROTARY_INTERLEAVED=False,
            HEAD_DIM=head_dim,
            BLOCK_SIZE=BLOCK_SIZE
        )

    return t_flat.view(original_shape)