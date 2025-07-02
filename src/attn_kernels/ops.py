# pyright: reportCallIssue=false

from collections.abc import Sequence
import os
from typing import Dict, Optional, Tuple

import torch

from inplace_metadata import Metadata

_lib_path = os.path.join(os.path.dirname(__file__), "libas_comm.so")
torch.ops.load_library(_lib_path)
_ops = torch.ops.dispatch_kernels

###### NVSHMEM utils ######

def nvshmem_get_unique_id() -> torch.Tensor:
    return _ops.nvshmem_get_unique_id()

def nvshmem_unique_id_size() -> int:
    return _ops.nvshmem_unique_id_size()

def nvshmem_alloc_empty_unique_id() -> torch.Tensor:
    return torch.zeros(nvshmem_unique_id_size(), dtype=torch.uint8, device="cpu")

def nvshmem_init(uid: torch.Tensor, rank: int, world_size: int) -> int:
    status = _ops.nvshmem_init(uid, rank, world_size)
    torch.cuda.synchronize()
    return status

def nvshmem_alltoall(dest: torch.Tensor, source: torch.Tensor) -> None:
    return _ops.nvshmem_alltoall(dest, source)

def nvshmem_finalize() -> None:
    torch.cuda.synchronize()
    _ops.nvshmem_finalize()

def nvshmem_my_pe() -> int:
    return _ops.nvshmem_my_pe()

def nvshmem_n_pes() -> int:
    return _ops.nvshmem_n_pes()

def nvshmem_malloc(
    shape: Sequence[int],
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    return _ops.nvshmem_malloc(shape, dtype, device)

def nvshmem_barrier_all() -> None:
    _ops.nvshmem_barrier_all()

def nvshmem_barrier_all_on_current_stream() -> None:
    _ops.nvshmem_barrier_all_on_current_stream()

###### MLP<->ATTN dispatch ######
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

    def __del__(self):
        destroy_dispatcher(self.dispatcher)


# TODO: remove this class because we only have one dispatcher.
class DispatcherStorage:

    def __init__(self):
        self._dispatcher_dict: Dict[Tuple[int, int], DispatcherWrapper] = {}
        self._current_dispatcher: Optional[DispatcherWrapper] = None
        self.rank = None
        self.world_size = None

    def get_dispatcher(
        self,
        q_stride: int,
        kv_stride: int,
        max_tokens_query: int,
        max_tokens_key_value: int,
    ):
        if self.rank is None:
            self.rank = nvshmem_my_pe()
        if self.world_size is None:
            self.world_size = nvshmem_n_pes()

        if self._current_dispatcher is None:
            # avoid allocating a buffer of size 0
            max_tokens_key_value = max(max_tokens_key_value, 1)
            kv_stride = max(kv_stride, 1)
            self._current_dispatcher = DispatcherWrapper(
                q_stride, kv_stride, max_tokens_query, max_tokens_key_value,
                self.rank, self.world_size
            )
        else:
            self._current_dispatcher.maybe_update(
                q_stride, kv_stride, max_tokens_query, max_tokens_key_value
            )
        return self._current_dispatcher


_dispatcher_storage = DispatcherStorage()


def dispatch(
    dispatcher: DispatcherWrapper,
    tensor: torch.Tensor,
    dst_tensor: torch.Tensor,
    metadata: Metadata,
    kv_tensor: Optional[torch.Tensor],
    kv_dst_tensor: Optional[torch.Tensor],
    kv_metadata: Optional[Metadata],
):
    assert metadata.dst_rank.dtype == torch.int32
    assert metadata.dst_offset.dtype == torch.uint32
    assert metadata.num_recv_tokens.dtype == torch.uint64
    assert metadata.seq_len.dtype == torch.uint32
    assert tensor.dtype == dst_tensor.dtype
    if kv_metadata is not None:
        assert kv_metadata.dst_rank.dtype == torch.int32
        assert kv_metadata.dst_offset.dtype == torch.uint32
        assert kv_metadata.num_recv_tokens.dtype == torch.uint64
        assert kv_metadata.seq_len.dtype == torch.uint32
        assert kv_tensor.dtype == kv_dst_tensor.dtype
    else:
        kv_metadata = Metadata(None, None, None, None, None)
    return _ops.dispatch(
        dispatcher.dispatcher,
        tensor, dst_tensor,
        metadata.dst_rank, metadata.dst_offset, metadata.num_recv_tokens, metadata.seq_len,
        kv_tensor, kv_dst_tensor,
        kv_metadata.dst_rank, kv_metadata.dst_offset, kv_metadata.num_recv_tokens
    )
