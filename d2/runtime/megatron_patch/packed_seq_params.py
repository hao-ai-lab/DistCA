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


@dataclass
class PingPangPackedSeqParams:
    seq_params: List[PingPangSingleStepPackedSeqParams]
    debug: bool = False
