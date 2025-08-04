# pyright: reportCallIssue=false

from collections.abc import Sequence
import os
from typing import Dict, Optional, Tuple

import torch

from d2.runtime.inplace_metadata import Metadata

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

def nvshmem_init(uid: torch.Tensor, rank: int, world_size: int, local_rank: int=-1) -> int:
    # NOTE: this is because we set device in python. Should move it to the cpp end.
    torch.cuda.synchronize()
    status = _ops.nvshmem_init(uid, rank, world_size, local_rank)
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


#### Fast dispatch
class FastDispatcherWrapper:
    instance: Tuple[
        "FastDispatcherWrapper", "FastDispatcherWrapper"
    ] = None
    cur_instance: int = 0

    def __init__(self, rank, local_rank, world_size, buffer_size):
        self.rank = rank
        self.local_rank = local_rank
        self.world_size = world_size
        self.buffer_size = buffer_size
        self.handle = _ops.create_fast_a2a_dispatch_helper(
            rank, local_rank, world_size, buffer_size
        )

    def __del__(self):
        _ops.destroy_fast_a2a_dispatch_helper(self.handle)

    def _update_buffer_size(self, buffer_size):
        if self.buffer_size < buffer_size:
            _ops.fast_a2a_update_buffer_size(self.handle, buffer_size)
            self.buffer_size = buffer_size

    @staticmethod
    def init(
        rank: int, local_rank: int, world_size: int, buffer_size: int,
        uid: torch.Tensor
    ):
        # NOTE: local_rank here is currently disabled, because we don't
        # consider intra-rank communication here.
        nvshmem_init(uid, rank, world_size, local_rank)
        FastDispatcherWrapper.instance = [FastDispatcherWrapper(
            rank, local_rank, world_size, buffer_size
        ) for _ in range(2)]

    @staticmethod
    def get_instance(instance_id: int=None) -> "FastDispatcherWrapper":
        assert FastDispatcherWrapper.instance is not None, "DispatcherWrapper not initialized"
        instance_id = instance_id if instance_id is not None else FastDispatcherWrapper.cur_instance
        return FastDispatcherWrapper.instance[instance_id]

    @staticmethod
    def switch_buffer():
        FastDispatcherWrapper.cur_instance = (FastDispatcherWrapper.cur_instance + 1) % 2

    @staticmethod
    def update_buffer_size(buffer_size: int):
        for instance in FastDispatcherWrapper.instance:
            instance._update_buffer_size(buffer_size)


def fast_a2a_memcpy_non_cp(
    tensor: torch.Tensor, nvshmem_offset: torch.Tensor,
    seq_tokens: torch.Tensor, to_nvshmem: bool,
    buffer: Optional[torch.Tensor]=None, instance_id: int=None
):
    if buffer is not None:
        # Debug mode, explicitly pass the "nvshmem" buffer.
        return _ops.fast_a2a_memcpy_non_cp_debug(
            tensor, nvshmem_offset, seq_tokens, to_nvshmem, buffer
        )
    return _ops.fast_a2a_memcpy_non_cp(
        FastDispatcherWrapper.get_instance(instance_id).handle,
        tensor, nvshmem_offset, seq_tokens, to_nvshmem
    )


def fast_a2a_memcpy_cp(
    tensor: torch.Tensor, do_shard: torch.Tensor,
    nvshmem_offset: torch.Tensor, seq_tokens: torch.Tensor,
    to_nvshmem: bool, buffer: Optional[torch.Tensor]=None,
    instance_id: int=None,
):
    if buffer is not None:
        # Debug mode, explicitly pass the "nvshmem" buffer.
        return _ops.fast_a2a_memcpy_cp_debug(
            tensor, do_shard, nvshmem_offset, seq_tokens, to_nvshmem, buffer
        )

    return _ops.fast_a2a_memcpy_cp(
        FastDispatcherWrapper.get_instance(instance_id).handle,
        tensor, do_shard, nvshmem_offset, seq_tokens, to_nvshmem
    )


def fast_a2a(
    sender_send_disp: torch.Tensor, sender_transfer_sz: torch.Tensor,
    sender_recv_disp: torch.Tensor, recver_transfer_sz: torch.Tensor,
    my_rank_send_offset: int, my_rank_recv_offset: int, my_rank_send_sz: int,
    instance_id: int=None,
):
    return _ops.fast_a2a(
        FastDispatcherWrapper.get_instance(instance_id).handle,
        sender_send_disp, sender_transfer_sz,
        sender_recv_disp, recver_transfer_sz,
        my_rank_send_offset, my_rank_recv_offset, my_rank_send_sz
    )


def _debug_dump_buffer(
    dump_target: str,
    buffer_dtype: torch.dtype,
    device: torch.device,
    instance_id: int=None,
):
    assert dump_target in ["send", "recv", "signal"]
    get_send = dump_target == "send"
    get_recv = dump_target == "recv"
    get_signal = dump_target == "signal"
    instance = FastDispatcherWrapper.get_instance(instance_id)
    out_tensor = torch.zeros(
        (instance.buffer_size // buffer_dtype.itemsize,),
        dtype=buffer_dtype, device=device
    )
    _ops._debug_nvshmem_buffer(
        instance.handle,
        get_send, get_recv, get_signal, out_tensor
    )
    torch.cuda.synchronize()
    return out_tensor
