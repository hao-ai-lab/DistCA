from dataclasses import dataclass
from typing import List

import torch

from megatron.core.packed_seq_params import PackedSeqParams

from d2.runtime.inplace_metadata import Metadata

@dataclass
class PingPangSingleStepPackedSeqParams(PackedSeqParams):
    mlp_to_attn_metadata: Metadata = None
    attn_to_mlp_metadata: Metadata = None
    mlp_to_attn_kv_metadata: Metadata = None
    mlp_to_attn_kv_grad_metadata: Metadata = None
    stream: torch.cuda.Stream = None

    def to_device(self):
        return PingPangSingleStepPackedSeqParams(
            qkv_format=self.qkv_format,
            cu_seqlens_q=self.cu_seqlens_q.cuda().contiguous(),
            cu_seqlens_kv=self.cu_seqlens_kv.cuda().contiguous(),
            cu_seqlens_q_padded=self.cu_seqlens_q_padded.cuda().contiguous() if self.cu_seqlens_q_padded is not None else None,
            cu_seqlens_kv_padded=self.cu_seqlens_kv_padded.cuda().contiguous() if self.cu_seqlens_kv_padded is not None else None,
            max_seqlen_q=self.max_seqlen_q.cuda().contiguous(),
            max_seqlen_kv=self.max_seqlen_kv.cuda().contiguous(),
            mlp_to_attn_metadata=self.mlp_to_attn_metadata.normalize_dtype().cuda(),
            attn_to_mlp_metadata=self.attn_to_mlp_metadata.normalize_dtype().cuda(),
            mlp_to_attn_kv_metadata=self.mlp_to_attn_kv_metadata.normalize_dtype().cuda(),
            mlp_to_attn_kv_grad_metadata=self.mlp_to_attn_kv_grad_metadata.normalize_dtype().cuda(),
            stream=self.stream,
        )


@dataclass
class PingPangPackedSeqParams:
    seq_params: List[PingPangSingleStepPackedSeqParams]
    debug: bool = False
