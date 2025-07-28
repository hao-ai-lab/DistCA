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
    # The seq params for mlp layout. This is mainly used for the RoPE.
    mlp_layout_seq_params: List[PackedSeqParams]
    debug: bool = False
    do_gather: bool = False
    # NOTE: These attributes are used for rotary seq len's max length.
    # since we do rope in the MLP layout, it should be the max length
    # at the MLP layout (i.e. the number of tokens).
    max_seqlen_q: Optional[torch.Tensor] = None
    max_seqlen_kv: Optional[torch.Tensor] = None
    qkv_format: str = "qkv"

    def to_device(self):
        # FIXME: this uses the local attention layout value, while it should
        # be the MLP layout's value. Since Attention layout may have fewer
        # tokens, this can be wrong.
        max_seqlen_q = self.max_seqlen_q
        if max_seqlen_q is None:
            max_seqlen_q = max([p.max_seqlen_q for p in self.seq_params])
        max_seqlen_kv = self.max_seqlen_kv
        if max_seqlen_kv is None:
            max_seqlen_kv = max([p.max_seqlen_kv for p in self.seq_params])
        return PingPangPackedSeqParams(
            seq_params=[seq_param.to_device() for seq_param in self.seq_params],
            mlp_layout_seq_params=[
                arg_to_cuda(seq_param) for seq_param in self.mlp_layout_seq_params
            ],
            debug=self.debug,
            do_gather=self.do_gather,
            max_seqlen_q=_to_cuda_int32(max_seqlen_q),
            max_seqlen_kv=_to_cuda_int32(max_seqlen_kv),
        )


def arg_to_cuda(v):
    if isinstance(v, torch.Tensor):
        return v.cuda()
    elif isinstance(v, PingPangPackedSeqParams):
        return v.to_device()
    elif isinstance(v, PingPangSingleStepPackedSeqParams):
        return v.to_device()
    elif isinstance(v, PackedSeqParams):
        return PackedSeqParams(
            qkv_format=v.qkv_format,
            cu_seqlens_q=_to_cuda_int32(v.cu_seqlens_q),
            cu_seqlens_kv=_to_cuda_int32(v.cu_seqlens_kv),
            max_seqlen_q=_to_cuda_int32(v.max_seqlen_q),
            max_seqlen_kv=_to_cuda_int32(v.max_seqlen_kv),
            cu_seqlens_q_padded=_to_cuda_int32(v.cu_seqlens_q_padded),
            cu_seqlens_kv_padded=_to_cuda_int32(v.cu_seqlens_kv_padded),
        )
    return v