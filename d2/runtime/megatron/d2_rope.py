import torch
from torch import Tensor
from typing import Optional
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.models.common.embeddings.rope_utils import _apply_rotary_pos_emb_bshd
import rich

def apply_rotary_pos_emb_d2(
    t: Tensor,
    freqs: Tensor,
    config: TransformerConfig,
    cu_seqlens: Optional[Tensor] = None,
    shard_logical_range: Optional[Tensor] = None, 
    mscale: float = 1.0
) -> Tensor:
    if cu_seqlens is None or shard_logical_range is None or freqs is None:
        rich.print("[red]Warning:[/red] Skip RoPE. Need cu_seqlens, shard_logical_range and freqs for apply_rotary_pos_emb_d2.")
        return t

    total_tokens = t.shape[0]
    
    seqlens = cu_seqlens[1:] - cu_seqlens[:-1]

    logical_lens = (shard_logical_range[:, 1] - shard_logical_range[:, 0])

    if (seqlens != logical_lens).any():
         mismatch_indices = torch.nonzero(seqlens != logical_lens).squeeze()
         raise ValueError(f"Physical sequence lengths do not match logical range lengths. Mismatch at indices: {mismatch_indices}")

    logical_starts = shard_logical_range[:, 0] 
    physical_starts = cu_seqlens[:-1]
    
    offsets = logical_starts - physical_starts
    
    token_offsets = torch.repeat_interleave(offsets, seqlens)
    
    base_indices = torch.arange(total_tokens, device=t.device, dtype=torch.long)
    
    final_indices = base_indices + token_offsets
    
    curr_freqs = freqs[final_indices]
    
    t_in = t.unsqueeze(1) 
    t_out = _apply_rotary_pos_emb_bshd(
        t_in,
        curr_freqs,
        rotary_interleaved=config.rotary_interleaved,
        multi_latent_attention=config.multi_latent_attention,
        mscale=mscale
    )
    
    return t_out.squeeze(1)