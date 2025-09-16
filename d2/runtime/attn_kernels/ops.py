# pyright: reportCallIssue=false

from collections.abc import Sequence
import enum
import os
from typing import Optional, Tuple

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

import datetime
def nvshmem_init(uid: torch.Tensor, rank: int, world_size: int, local_rank: int=-1) -> int:
    # NOTE: this is because we set device in python. Should move it to the cpp end.
    print(f"Calling nvshmem_init with uid = {uid}")
    torch.cuda.synchronize()
    status = _ops.nvshmem_init(uid, rank, world_size, local_rank)
    print("nvshmem_init returns with status =", status)
    torch.cuda.synchronize()
    print("nvshmem_init synchronized. Ready to call barrier")
    import traceback
    # traceback.print_stack()
    print("nvshmem_init passed barrier.")
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
    is_acquired: list[bool] = [False, False]
    cur_instance: int = 0
    comm_stream: torch.cuda.Stream = None

    def __init__(self, rank, local_rank, world_size, buffer_size):
        self.rank = rank
        self.local_rank = local_rank
        self.world_size = world_size
        self.buffer_size = buffer_size
        self.handle = _ops.create_fast_a2a_dispatch_helper(
            rank, local_rank, world_size, buffer_size
        )
        # the buffer_released is initialized by all zeros. We should
        # manually release them here once.
        torch.cuda.synchronize()
        torch.distributed.barrier()
        _ops.release_buffer(self.handle)
        self.release_event = torch.cuda.Event()
        self.release_event.record(torch.cuda.current_stream())

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
        # sync to ensure that all operations on the buffer are done.
        torch.cuda.synchronize()
        for iid, instance in enumerate(FastDispatcherWrapper.instance):
            assert not FastDispatcherWrapper.is_acquired[iid]
            instance._update_buffer_size(buffer_size)

    # Acquire and release a dispatcher's receive buffer. Note that we do not
    # monitor the send buffer, because its availibility is guaranteed locally.
    # However, for receive buffer, we need to monitor a singal to know whether
    # the remote receive buffer is ready as well.
    def acquire(instance_id: int):
        # TODO: acquire should be on the same stream as the all2all. Try to check it.
        assert not FastDispatcherWrapper.is_acquired[instance_id]
        FastDispatcherWrapper.is_acquired[instance_id] = True
        torch.cuda.current_stream().wait_event(
            FastDispatcherWrapper.get_instance(instance_id).release_event
        )
        # TODO:(Hack) for perf test don't do anything
        _ops.wait_and_consume_buffer(FastDispatcherWrapper.get_instance(instance_id).handle)

    def release(instance_id: int):
        assert FastDispatcherWrapper.is_acquired[instance_id]
        FastDispatcherWrapper.is_acquired[instance_id] = False
        stream = FastDispatcherWrapper.comm_stream
        compute_stream = torch.cuda.current_stream()
        with torch.cuda.stream(stream):
            stream.wait_stream(compute_stream)
            _ops.release_buffer(FastDispatcherWrapper.get_instance(instance_id).handle)
        FastDispatcherWrapper.get_instance(instance_id).release_event.record(stream)


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

    if instance_id is not None:
        if to_nvshmem:
            # Ensure a strong order that "post_a2a_memcpy - release - pre_a2a_memcpy"
            assert not FastDispatcherWrapper.is_acquired[instance_id]
        else:
            assert FastDispatcherWrapper.is_acquired[instance_id]

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

    if instance_id is not None:
        if to_nvshmem:
            # Ensure a strong order that "post_a2a_memcpy - release - pre_a2a_memcpy"
            assert not FastDispatcherWrapper.is_acquired[instance_id]
        else:
            assert FastDispatcherWrapper.is_acquired[instance_id]

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
    if instance_id is not None:
        assert not FastDispatcherWrapper.is_acquired[instance_id]
        # acquiring here ensures the sync in acquire is always on the same stream as all2all
        FastDispatcherWrapper.acquire(instance_id)

    if os.environ.get("EXPERIMENT_FA2A_BARRIER", "0") == "1":
        rank = torch.distributed.get_rank()
        if rank == 0:
            print("🛑 enabled fast_a2a barrier - reached barrier")
        torch.cuda.synchronize()
        torch.distributed.barrier()
        if rank == 0:
            print("🛑 enabled fast_a2a barrier - passed")
    ret = _ops.fast_a2a(
        FastDispatcherWrapper.get_instance(instance_id).handle,
        sender_send_disp, sender_transfer_sz,
        sender_recv_disp, recver_transfer_sz,
        my_rank_send_offset, my_rank_recv_offset, my_rank_send_sz,
    )

    return ret


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
