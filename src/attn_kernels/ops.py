# pyright: reportCallIssue=false

from collections.abc import Sequence
import os
from typing import Dict, Optional, Tuple

import torch

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
):
    return _ops.create_dispatch_helper(q_stride, kv_stride, max_tokens_query, max_tokens_key_value)


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
                 ):
        self.dispatcher = create_dispatcher(
            q_stride, kv_stride, max_tokens_query, max_tokens_key_value
        )
        self.q_stride = q_stride
        self.kv_stride = kv_stride
        self.max_tokens_query = max_tokens_query
        self.max_tokens_key_value = max_tokens_key_value

    def maybe_update(self,
                     q_stride: int,
                     kv_stride: int,
                     max_tokens_query: int,
                     max_tokens_key_value: int,
                     ):
        if (self.max_tokens_query * self.q_stride < max_tokens_query * q_stride or
            self.max_tokens_key_value * self.kv_stride < max_tokens_key_value * kv_stride or
            self.max_tokens_query < max_tokens_query or
            self.max_tokens_key_value < max_tokens_key_value):
            self.q_stride = q_stride
            self.kv_stride = kv_stride
            self.max_tokens_query = max_tokens_query
            self.max_tokens_key_value = max_tokens_key_value
            destroy_dispatcher(self.dispatcher)
            self.dispatcher = create_dispatcher(
                self.q_stride, self.kv_stride,
                self.max_tokens_query, self.max_tokens_key_value
            )

    def __del__(self):
        destroy_dispatcher(self.dispatcher)


class DispatcherStorage:

    def __init__(self):
        self._dispatcher_dict: Dict[Tuple[int, int], DispatcherWrapper] = {}
        self._current_dispatcher: Optional[DispatcherWrapper] = None

    def get_dispatcher(
        self,
        q_stride: int,
        kv_stride: int,
        max_tokens_query: int,
        max_tokens_key_value: int,
    ):
        if self._current_dispatcher is None:
            # avoid allocating a buffer of size 0
            max_tokens_key_value = max(max_tokens_key_value, 1)
            kv_stride = max(kv_stride, 1)
            self._current_dispatcher = DispatcherWrapper(
                q_stride, kv_stride, max_tokens_query, max_tokens_key_value
            )
        else:
            self._current_dispatcher.maybe_update(
                q_stride, kv_stride, max_tokens_query, max_tokens_key_value
            )
        return self._current_dispatcher


_dispatcher_storage = DispatcherStorage()


def dispatch(
    query_out: torch.Tensor,
    key_value_out: Optional[torch.Tensor],
    query_in: torch.Tensor,
    key_value_in: Optional[torch.Tensor],
    query_dst_id: torch.Tensor,
    query_dst_offset: torch.Tensor,
    key_value_dst_id: Optional[torch.Tensor],
    key_value_dst_offset: Optional[torch.Tensor],
    dispatcher = None,
):
    q_stride = query_in.stride(0) * query_in.element_size()
    kv_stride = (key_value_in.stride(0) * key_value_in.element_size()
                 if key_value_in is not None else 0)
    max_tokens_query = query_in.size(0)
    max_tokens_key_value = key_value_in.size(0) if key_value_in is not None else 0
    dispatcher = dispatcher or _dispatcher_storage.get_dispatcher(
        q_stride=q_stride,
        kv_stride=kv_stride,
        max_tokens_query=max_tokens_query,
        max_tokens_key_value=max_tokens_key_value,
    )

    num_tokens = query_in.size(0)
    num_recv_tokens_query = query_out.size(0)
    num_recv_tokens_key_value = key_value_out.size(0) if key_value_out is not None else 0

    return _ops.dispatch(
        dispatcher,
        query_out, key_value_out, query_in, key_value_in,
        query_dst_id, query_dst_offset,
        key_value_dst_id, key_value_dst_offset,
        num_tokens, num_recv_tokens_query, num_recv_tokens_key_value
    )
