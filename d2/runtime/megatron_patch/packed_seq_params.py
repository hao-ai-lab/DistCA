from dataclasses import dataclass

import torch

from megatron.core.packed_seq_params import PackedSeqParams

from d2.runtime.inplace_metadata import Metadata

@dataclass
class PingPangPackedSeqParams(PackedSeqParams):
    debug: bool = False
    mlp_to_attn_metadata: Metadata = None
    attn_to_mlp_metadata: Metadata = None
    mlp_to_attn_kv_metadata: Metadata = None
    mlp_to_attn_kv_grad_metadata: Metadata = None
    stream: torch.cuda.Stream = None
