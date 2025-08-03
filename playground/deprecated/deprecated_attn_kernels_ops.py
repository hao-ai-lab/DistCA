from typing import Optional

import torch

from d2.runtime.attn_kernels.ops import _ops, nvshmem_finalize, nvshmem_init
from d2.runtime.inplace_metadata import Metadata


def create_dispatcher(
    q_stride: int,
    kv_stride: int,
    max_tokens_query: int,
    max_tokens_key_value: int,
    rank: int,
    world_size: int,
):
    return _ops.create_dispatch_helper(
        q_stride, kv_stride, max_tokens_query, max_tokens_key_value,
        rank, world_size
    )


def destroy_dispatcher(dispatcher) -> None:
    _ops.destroy_dispatch_helper(dispatcher)


class DispatcherWrapper:

    instance: Optional["DispatcherWrapper"] = None

    """
    Python wrapper for the dispatcher.
    A dispatcher is responsible for the MLP<->ATTN communication. It stores nvshmem
    buffers for the communication.
    This python wrapper will automatically update the dispatcher when its current
    buffer is not large enough.
    """
    def __init__(self,
                 q_stride: int,
                 kv_stride: int,
                 max_tokens_query: int,
                 max_tokens_key_value: int,
                 rank: int,
                 world_size: int,
                 num_sms: int = 20,
                 ):
        self.dispatcher = create_dispatcher(
            q_stride, kv_stride, max_tokens_query, max_tokens_key_value,
            rank, world_size
        )
        self.q_stride = q_stride
        self.kv_stride = kv_stride
        self.max_tokens_query = max_tokens_query
        self.max_tokens_key_value = max_tokens_key_value
        self.rank = rank
        self.world_size = world_size
        self.num_sms = num_sms
        self.set_num_sms(num_sms)

    def maybe_update(self,
                     q_stride: int,
                     kv_stride: int,
                     max_tokens_query: int,
                     max_tokens_key_value: int,
                     ):
        if (self.max_tokens_query * self.q_stride < max_tokens_query * q_stride or
            self.max_tokens_key_value * self.kv_stride < max_tokens_key_value * kv_stride):
            self.q_stride = q_stride
            self.kv_stride = kv_stride
            self.max_tokens_query = max_tokens_query
            self.max_tokens_key_value = max_tokens_key_value
            destroy_dispatcher(self.dispatcher)
            self.dispatcher = create_dispatcher(
                self.q_stride, self.kv_stride,
                self.max_tokens_query, self.max_tokens_key_value,
                self.rank, self.world_size
            )
            self.set_num_sms(self.num_sms)

    def __del__(self):
        destroy_dispatcher(self.dispatcher)
        nvshmem_finalize()

    
    @staticmethod
    def init(
        q_stride: int, kv_stride: int, max_tokens_query: int, max_tokens_key_value: int,
        rank: int, world_size: int, uid: torch.Tensor,
    ):
        nvshmem_init(uid, rank, world_size)
        DispatcherWrapper.instance = DispatcherWrapper(
            q_stride, kv_stride, max_tokens_query, max_tokens_key_value,
            rank, world_size
        )

    @staticmethod
    def get_instance():
        assert DispatcherWrapper.instance is not None, "DispatcherWrapper not initialized"
        return DispatcherWrapper.instance

    def set_num_sms(self, num_sms: int) -> None:
        self.num_sms = num_sms
        _ops.set_num_sms(self.dispatcher, num_sms)


def dispatch_qkv(
    dispatcher: DispatcherWrapper,
    tensor: torch.Tensor,
    dst_tensor: torch.Tensor,
    metadata: Metadata,
    kv_tensor: torch.Tensor,
    kv_dst_tensor: torch.Tensor,
    kv_metadata: Metadata,
):
    assert metadata.dst_rank.dtype == torch.int32
    assert metadata.dst_offset.dtype == torch.uint32
    assert metadata.num_recv_tokens.dtype == torch.uint64
    assert metadata.seq_len.dtype == torch.uint32
    assert tensor.dtype == dst_tensor.dtype

    assert kv_metadata.dst_rank.dtype == torch.int32
    assert kv_metadata.dst_offset.dtype == torch.uint32
    assert kv_metadata.num_recv_tokens.dtype == torch.uint64
    assert kv_metadata.seq_len.dtype == torch.uint32
    assert kv_tensor.dtype == kv_dst_tensor.dtype
    return _ops.dispatch_core(
        dispatcher.dispatcher,
        tensor, dst_tensor,
        metadata.dst_rank, metadata.dst_offset, metadata.num_recv_tokens, metadata.seq_len,
        kv_tensor, kv_dst_tensor,
        kv_metadata.dst_rank, kv_metadata.dst_offset, kv_metadata.num_recv_tokens,
        None, None,
    )


def dispatch_no_cp_tensor(
    dispatcher: DispatcherWrapper,
    tensor: torch.Tensor,
    dst_tensor: torch.Tensor,
    metadata: Metadata,
):
    assert metadata.dst_rank.dtype == torch.int32
    assert metadata.dst_offset.dtype == torch.uint32
    assert metadata.num_recv_tokens.dtype == torch.uint64
    assert metadata.seq_len.dtype == torch.uint32
    assert tensor.dtype == dst_tensor.dtype
    return _ops.dispatch_core(
        dispatcher.dispatcher,
        tensor, dst_tensor,
        metadata.dst_rank, metadata.dst_offset, metadata.num_recv_tokens, metadata.seq_len,
        None, None, None, None, None, None, None
    )


def dispatch_kv_backward(
    dispatcher: DispatcherWrapper,
    tensor: torch.Tensor,
    dst_tensor: torch.Tensor,
    metadata: Metadata,
):
    assert metadata.dst_rank.dtype == torch.int32
    assert metadata.dst_offset.dtype == torch.uint32
    assert metadata.num_recv_tokens.dtype == torch.uint64
    assert metadata.seq_len.dtype == torch.uint32
    assert tensor.dtype == dst_tensor.dtype
    assert metadata.seq_recv_mask.dtype == torch.uint32
    return _ops.dispatch_core(
        dispatcher.dispatcher,
        tensor, dst_tensor,
        metadata.dst_rank, metadata.dst_offset, metadata.num_recv_tokens, metadata.seq_len,
        None, None, None, None, None,
        metadata.seq_recv_mask, metadata.recv_seq_lens
    )

