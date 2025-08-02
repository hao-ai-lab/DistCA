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


#### Fast dispatch
class FastDispatcherWrapper:
    instance: Optional["FastDispatcherWrapper"] = None
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

    def update_buffer_size(self, buffer_size):
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
        FastDispatcherWrapper.instance = FastDispatcherWrapper(
            rank, local_rank, world_size, buffer_size
        )

    @staticmethod
    def get_instance():
        assert FastDispatcherWrapper.instance is not None, "DispatcherWrapper not initialized"
        return FastDispatcherWrapper.instance


def fast_a2a_memcpy_non_cp(
    tensor: torch.Tensor, nvshmem_offset: torch.Tensor,
    seq_tokens: torch.Tensor, to_nvshmem: bool,
    buffer: Optional[torch.Tensor]=None
):
    if buffer is not None:
        # Debug mode, explicitly pass the "nvshmem" buffer.
        return _ops.fast_a2a_memcpy_non_cp_debug(
            tensor, nvshmem_offset, seq_tokens, to_nvshmem, buffer
        )
    return _ops.fast_a2a_memcpy_non_cp(
        FastDispatcherWrapper.get_instance().handle,
        tensor, nvshmem_offset, seq_tokens, to_nvshmem
    )


def fast_a2a_memcpy_cp(
    tensor: torch.Tensor, do_shard: torch.Tensor,
    nvshmem_offset: torch.Tensor, seq_tokens: torch.Tensor,
    to_nvshmem: bool, buffer: Optional[torch.Tensor]=None
):
    if buffer is not None:
        # Debug mode, explicitly pass the "nvshmem" buffer.
        return _ops.fast_a2a_memcpy_cp_debug(
            tensor, do_shard, nvshmem_offset, seq_tokens, to_nvshmem, buffer
        )

    return _ops.fast_a2a_memcpy_cp(
        FastDispatcherWrapper.get_instance().handle,
        tensor, do_shard, nvshmem_offset, seq_tokens, to_nvshmem
    )


def fast_a2a(
    sender_send_disp: torch.Tensor, sender_transfer_sz: torch.Tensor,
    sender_recv_disp: torch.Tensor, recver_transfer_sz: torch.Tensor,
    my_rank_send_offset: int, my_rank_recv_offset: int, my_rank_send_sz: int
):
    return _ops.fast_a2a(
        FastDispatcherWrapper.get_instance().handle,
        sender_send_disp, sender_transfer_sz,
        sender_recv_disp, recver_transfer_sz,
        my_rank_send_offset, my_rank_recv_offset, my_rank_send_sz
    )


def _debug_dump_buffer(
    dump_target: str,
    buffer_dtype: torch.dtype,
    device: torch.device,
    instance: FastDispatcherWrapper=None,
):
    assert dump_target in ["send", "recv", "signal"]
    get_send = dump_target == "send"
    get_recv = dump_target == "recv"
    get_signal = dump_target == "signal"
    instance = instance or FastDispatcherWrapper.get_instance()
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
