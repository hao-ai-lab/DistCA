from dataclasses import dataclass

import torch

from megatron.core.packed_seq_params import PackedSeqParams

@dataclass
class PingPangPackedSeqParams(PackedSeqParams):
    debug: bool = False
    query_dst_ids: torch.Tensor = None
    query_dst_offsets: torch.Tensor = None
    query_out_shape: torch.Tensor = None
    kv_dst_ids: torch.Tensor = None
    kv_dst_offsets: torch.Tensor = None
    kv_out_shape: torch.Tensor = None
    rev_query_dst_ids: torch.Tensor = None
    rev_query_dst_offsets: torch.Tensor = None
    rev_kv_dst_ids: torch.Tensor = None
    rev_kv_dst_offsets: torch.Tensor = None
    out_dst_ids: torch.Tensor = None
    out_dst_offsets: torch.Tensor = None
