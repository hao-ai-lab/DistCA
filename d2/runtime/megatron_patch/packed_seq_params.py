from dataclasses import dataclass
from typing import List, Optional

import torch

from megatron.core.packed_seq_params import PackedSeqParams

from d2.runtime.inplace_metadata import Metadata

def _to_cuda_int32(tensor: Optional[torch.Tensor]):
    if tensor is None:
        return None
    return tensor.cuda().to(torch.int32).contiguous()

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
            cu_seqlens_q=_to_cuda_int32(self.cu_seqlens_q),
            cu_seqlens_kv=_to_cuda_int32(self.cu_seqlens_kv),
            cu_seqlens_q_padded=_to_cuda_int32(self.cu_seqlens_q_padded),
            cu_seqlens_kv_padded=_to_cuda_int32(self.cu_seqlens_kv_padded),
            max_seqlen_q=_to_cuda_int32(self.max_seqlen_q),
            max_seqlen_kv=_to_cuda_int32(self.max_seqlen_kv),
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
    do_gather: bool = False

    def to_device(self):
        return PingPangPackedSeqParams(
            seq_params=[seq_param.to_device() for seq_param in self.seq_params],
            debug=self.debug,
            do_gather=self.do_gather
        )
