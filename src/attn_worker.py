"""Worker class for attention computation using FlashAttention."""
from dataclasses import dataclass
from typing import Callable

from flash_attn import flash_attn_varlen_func
import numpy as np
import torch

# flash attn call reference:
# https://github.com/NVIDIA/TransformerEngine/blob/main/transformer_engine/pytorch/attention/dot_product_attention/backends.py#L418

# nvshmem reference:
# https://github.com/ppl-ai/pplx-kernels/blob/master/csrc/all_to_all/internode_dispatch.cu

from attn_kernels.ops import nvshmem_get_unique_id, nvshmem_init, nvshmem_finalize

@dataclass
class ScheduleMetadata:
    seq_info: np.ndarray
    # NOTE: we design the most stateless attention worker that does not consider any
    # TP/CP related communication. The Scatter is merged with sending attn_out back,
    # and Reduce operation is performed by the requester.
    # TODO: we should enable different sequences having its own number of heads.
    tp_degree: int  # we need this to know num_heads on this worker.
    # cp_degree: int
    communication_method: str = "nvshmem"   # options: dummy, signal, nvshmem, send/receive
    dummy_gen_fn: Callable = None

class AttentionWorker:
    def __init__(self, dropout_p, softmax_scale,
                 hidden_size: int, num_heads: int, num_heads_k: int,
                 dtype: torch.dtype = torch.float16):
        # attention args
        self.dropout_p = dropout_p
        self.softmax_scale = softmax_scale
        # model args
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_heads_k = num_heads_k
        self.dtype = dtype
        # communicator
        self.communicator = None

    @torch.no_grad()
    def run_attn_fwd(self, q, k, v, cu_seqlens_q, cu_seqlens_k,
                     max_seqlen_q, max_seqlen_k):
        """
        Core attention function based on FlashAttention flash_attn_varlen_func
        """
        # 1. run flash_attn_varlen_func
        out = flash_attn_varlen_func(
            q, k, v, cu_seqlens_q, cu_seqlens_k,
            max_seqlen_q, max_seqlen_k,
            dropout_p=self.dropout_p,
            softmax_scale=self.softmax_scale,
            causal=True,
        )
        return out

    def get_nvshmem_uuid(self):
        uuid = nvshmem_get_unique_id()
        return uuid

    def init_nvshmem(self, uuid, rank, world_size):
        nvshmem_init(uuid, rank, world_size)

    def receive_qkv(self, metadata: ScheduleMetadata):
        if metadata.communication_method == "dummy":
            return self.dummy_qkv(metadata)
        elif metadata.communication_method == "signal":
            # TODO: receive a signal tensor from each sender.
            return self.dummy_qkv(metadata)
        else:
            # FIXME
            pass

    def dummy_qkv(self, metadata: ScheduleMetadata):
        total_q_len = metadata.seq_info[:, 1].sum()
        total_k_len = metadata.seq_info[:, 2].sum()
        total_v_len = total_k_len

        num_heads = self.num_heads // metadata.tp_degree
        num_heads_k = self.num_heads_k // metadata.tp_degree
        hidden_size = self.hidden_size

        if metadata.dummy_gen_fn is not None:
            q, k, v = metadata.dummy_gen_fn()
            return q, k, v

        q = torch.randn(
            (total_q_len, num_heads, hidden_size), device='cuda', dtype=self.dtype)
        k = torch.randn(
            (total_k_len, num_heads_k, hidden_size), device='cuda', dtype=self.dtype)
        v = torch.randn(
            (total_v_len, num_heads_k, hidden_size), device='cuda', dtype=self.dtype)
        return q, k, v

    def send_out(self, metadata: ScheduleMetadata, attn_out):
        # FIXME: complete this function
        if metadata.communication_method == "dummy":
            return
        elif metadata.communication_method == "signal":
            # TODO: send a signal tensor to each receiver.
            return
        else:
            # FIXME
            pass

    def remote_attn(self, metadata: ScheduleMetadata, debug=False):
        """
        """
        # 0. NOTE: all metadata should be on gpu as well？
        # 1. Receive the tensor based on metadata
        assert metadata.seq_info.ndim == 2
        num_seqs, info_len = metadata.seq_info.shape
        assert info_len == 5, "each sequence has metadata of (worker_id, seq_len_q, seq_len_k, q_read_addr, kv_read_addr)"
        q, k, v = self.receive_qkv(metadata)

        # 2. Call attention.
        max_seqlen_q = metadata.seq_info[:, 1].max()
        max_seqlen_k = metadata.seq_info[:, 2].max()
        max_seqlen_q = torch.tensor(max_seqlen_q, device='cuda', dtype=torch.int32)
        max_seqlen_k = torch.tensor(max_seqlen_k, device='cuda', dtype=torch.int32)
        cu_seqlens_q = np.cumsum(metadata.seq_info[:, 1])
        cu_seqlens_k = np.cumsum(metadata.seq_info[:, 2])
        cu_seqlens_q = torch.from_numpy(cu_seqlens_q).cuda().to(torch.int32)
        cu_seqlens_k = torch.from_numpy(cu_seqlens_k).cuda().to(torch.int32)
        # prepend zero
        cu_seqlens_q, cu_seqlens_k = [
            torch.cat([torch.tensor([0], device='cuda', dtype=torch.int32), tensor])
            for tensor in [cu_seqlens_q, cu_seqlens_k]
        ]
        attn_out = self.run_attn_fwd(q, k, v, cu_seqlens_q, cu_seqlens_k,
                                     max_seqlen_q, max_seqlen_k)
        # 3. Send result back
        self.send_out(metadata, attn_out)
        # 4. for debug use, return the output to the debug controller
        return attn_out if debug else 0
