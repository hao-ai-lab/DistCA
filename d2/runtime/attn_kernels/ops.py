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

_is_nvshmem_initialized = False
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
    global _is_nvshmem_initialized
    _is_nvshmem_initialized = True
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
    return
    if not _is_nvshmem_initialized:
        return
    _ops.nvshmem_barrier_all()

def nvshmem_barrier_all_on_current_stream() -> None:
    return
    if not _is_nvshmem_initialized:
        return
    _ops.nvshmem_barrier_all_on_current_stream()


#### Fast dispatch
class DispatcherWrapper:
    instance: Tuple[
        "DispatcherWrapper", "DispatcherWrapper"
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
        DispatcherWrapper.instance = [DispatcherWrapper(
            rank, local_rank, world_size, buffer_size
        ) for _ in range(2)]

    @staticmethod
    def get_instance(instance_id: int=None) -> "DispatcherWrapper":
        assert DispatcherWrapper.instance is not None, "DispatcherWrapper not initialized"
        instance_id = instance_id if instance_id is not None else DispatcherWrapper.cur_instance
        return DispatcherWrapper.instance[instance_id]

    @staticmethod
    def switch_buffer():
        DispatcherWrapper.cur_instance = (DispatcherWrapper.cur_instance + 1) % 2

    @staticmethod
    def update_buffer_size(buffer_size: int):
        # sync to ensure that all operations on the buffer are done.
        torch.cuda.synchronize()
        for iid, instance in enumerate(DispatcherWrapper.instance):
            assert not DispatcherWrapper.is_acquired[iid]
            instance._update_buffer_size(buffer_size)

    # Acquire and release a dispatcher's receive buffer. Note that we do not
    # monitor the send buffer, because its availibility is guaranteed locally.
    # However, for receive buffer, we need to monitor a singal to know whether
    # the remote receive buffer is ready as well.
    def acquire(instance_id: int):
        return
        # TODO: acquire should be on the same stream as the all2all. Try to check it.
        assert not DispatcherWrapper.is_acquired[instance_id]
        DispatcherWrapper.is_acquired[instance_id] = True
        torch.cuda.current_stream().wait_event(
            DispatcherWrapper.get_instance(instance_id).release_event
        )
        _ops.wait_and_consume_buffer(DispatcherWrapper.get_instance(instance_id).handle)

    def release(instance_id: int):
        return
        assert DispatcherWrapper.is_acquired[instance_id]
        DispatcherWrapper.is_acquired[instance_id] = False
        stream = DispatcherWrapper.comm_stream
        compute_stream = torch.cuda.current_stream()
        with torch.cuda.stream(stream):
            stream.wait_stream(compute_stream)
            _ops.release_buffer(DispatcherWrapper.get_instance(instance_id).handle)
        DispatcherWrapper.get_instance(instance_id).release_event.record(stream)


def a2a_memcpy_non_cp(
    tensor: torch.Tensor, nvshmem_offset: torch.Tensor,
    shard_tokens: torch.Tensor, to_nvshmem: bool,
    shard_do_copy_mask: Optional[torch.Tensor]=None,
    buffer: Optional[torch.Tensor]=None, instance_id: int=None
):
    return
    # use_pytorch_impl = os.environ.get("EXPERIMENT_USE_PYTORCH_A2A", "0") == "1"
    # print(f"a2a_memcpy_non_cp: {shard_tokens = }")
    # print(f"a2a_memcpy_non_cp: {shard_do_copy_mask = }")
    # if use_pytorch_impl:
    #     return _a2a_memcpy_non_cp_pytorch(
    #         tensor, nvshmem_offset, shard_tokens, to_nvshmem,
    #         shard_do_copy_mask, instance_id
    #     )
    
    if buffer is not None:
        # Debug mode, explicitly pass the "nvshmem" buffer.
        return _ops.fast_a2a_memcpy_non_cp_debug(
            tensor, nvshmem_offset, shard_tokens, shard_do_copy_mask,
            to_nvshmem, buffer
        )
    if shard_do_copy_mask is not None:
        assert to_nvshmem

    if instance_id is not None:
        if to_nvshmem:
            # Ensure a strong order that "post_a2a_memcpy - release - pre_a2a_memcpy"
            assert not DispatcherWrapper.is_acquired[instance_id]
        else:
            assert DispatcherWrapper.is_acquired[instance_id]

    return _ops.fast_a2a_memcpy_non_cp(
        DispatcherWrapper.get_instance(instance_id).handle,
        tensor, nvshmem_offset, shard_tokens, shard_do_copy_mask, to_nvshmem
    )


import contextlib

@contextlib.contextmanager
def sync_if_debug(msg: str):
    if os.environ.get("EXPERIMENT_SHOULD_SYNC_FAST_A2A_OPS", "0") == "1":
        import traceback
        traceback.print_stack()
        print(f"Start sync for {msg}")
        torch.cuda.synchronize()
    try:
        with torch.cuda.nvtx.range(msg):
            yield
    finally:
        if os.environ.get("EXPERIMENT_SHOULD_SYNC_FAST_A2A_OPS", "0") == "1":
            torch.cuda.synchronize()
            print(f"Finish sync for {msg}")

def _ops_fast_a2a_memcpy_cp_debug(*args):
    with sync_if_debug("_ops_fast_a2a_memcpy_cp_debug"):
        return _ops.fast_a2a_memcpy_cp_debug(*args)

def _ops_fast_a2a_memcpy_cp(*args):
    with sync_if_debug("_ops_fast_a2a_memcpy_cp"):
        return _ops.fast_a2a_memcpy_cp(*args)
        # return _ops.fast_a2a_memcpy_non_cp(*args)

def _ops_fast_a2a_memcpy_non_cp(*args):
    with sync_if_debug("_ops_fast_a2a_memcpy_non_cp"):
        return _ops.fast_a2a_memcpy_non_cp(*args)


def _ops_fast_a2a_wrapper(*args):
    with sync_if_debug("_ops_fast_a2a_wrapper"):
        return _ops.fast_a2a(*args)

def _ops_fast_a2a_grad_acc(*args):
    with sync_if_debug("_ops_fast_a2a_grad_acc"):
        return _ops.fast_a2a_grad_acc(*args)

def a2a_memcpy_cp(
    tensor: torch.Tensor, do_shard: torch.Tensor,
    nvshmem_offset: torch.Tensor, shard_tokens: torch.Tensor,
    to_nvshmem: bool, buffer: Optional[torch.Tensor]=None,
    instance_id: int=None,
):
    return 
    use_pytorch_impl = os.environ.get("EXPERIMENT_USE_PYTORCH_A2A", "0") == "1"
    if use_pytorch_impl:
        return _a2a_memcpy_cp_pytorch(
            tensor, do_shard, nvshmem_offset, shard_tokens, to_nvshmem, instance_id
        )
    
    if buffer is not None:
        # Debug mode, explicitly pass the "nvshmem" buffer.
        return _ops_fast_a2a_memcpy_cp_debug(
            tensor, do_shard, nvshmem_offset, shard_tokens, to_nvshmem, buffer
        )

    if instance_id is not None:
        if to_nvshmem:
            # Ensure a strong order that "post_a2a_memcpy - release - pre_a2a_memcpy"
            assert not DispatcherWrapper.is_acquired[instance_id]
        else:
            assert DispatcherWrapper.is_acquired[instance_id]

    return _ops_fast_a2a_memcpy_cp(
        DispatcherWrapper.get_instance(instance_id).handle,
        tensor, do_shard, nvshmem_offset, shard_tokens, to_nvshmem
    )


def pre_a2a_grad_acc(
    tensor: torch.Tensor, num_copies: torch.Tensor,
    copy_start_id: torch.Tensor, shard_tokens: torch.Tensor,
):
    """
    When kv has multiple copies, and we accumulates their gradients to the main copy,
    so that we only need to send one gradient back.
    """
    _ops_fast_a2a_grad_acc(tensor, num_copies, copy_start_id, shard_tokens)
    return tensor


def fast_a2a(
    sender_send_disp: torch.Tensor, sender_transfer_sz: torch.Tensor,
    sender_recv_disp: torch.Tensor, recver_transfer_sz: torch.Tensor,
    my_rank_send_offset: int, my_rank_recv_offset: int, my_rank_send_sz: int,
    instance_id: int=None,
):
    use_pytorch_impl = os.environ.get("EXPERIMENT_USE_PYTORCH_A2A", "0") == "1"
    should_fa2a_barrier = os.environ.get("EXPERIMENT_FA2A_BARRIER", "0") == "1"
    should_skip_fa2a_op = os.environ.get("EXPERIMENT_SKIP_FA2A_OP", "0") == "1"
    
    if use_pytorch_impl:
        r = _fast_a2a_pytorch_mock(
            sender_send_disp, sender_transfer_sz,
            sender_recv_disp, recver_transfer_sz,
            my_rank_send_offset, my_rank_recv_offset, my_rank_send_sz,
            instance_id
        )
        print(f"🔄 PyTorch mock A2A completed")
        # torch.cuda.synchronize()
        # torch.distributed.barrier()
        print(f"🔄 PyTorch mock A2A passed barrier")
        return r
    
    if instance_id is not None:
        assert not DispatcherWrapper.is_acquired[instance_id]
        # acquiring here ensures the sync in acquire is always on the same stream as all2all
        DispatcherWrapper.acquire(instance_id)

    if should_fa2a_barrier:
        rank = torch.distributed.get_rank()
        if rank == 0:
            print("🛑 enabled fast_a2a barrier - reached barrier")
        torch.cuda.synchronize()
        torch.distributed.barrier()
        if rank == 0:
            print("🛑 enabled fast_a2a barrier - passed")

    if should_skip_fa2a_op:
        # Use a module-level variable to track if we've printed the message
        if not hasattr(fast_a2a, '_printed_skip_message'):
            print("🛑 skipping fast_a2a op. This usually happens at debugging. If this is not expected, please set EXPERIMENT_SKIP_FA2A_OP to 0.")
            fast_a2a._printed_skip_message = True
        return

    ret = _ops_fast_a2a_wrapper(
        DispatcherWrapper.get_instance(instance_id).handle,
        sender_send_disp, sender_transfer_sz,
        sender_recv_disp, recver_transfer_sz,
        my_rank_send_offset, my_rank_recv_offset, my_rank_send_sz,
        True
    )
    return ret


def _fast_a2a_pytorch_mock(
    sender_send_disp: torch.Tensor, sender_transfer_sz: torch.Tensor,
    sender_recv_disp: torch.Tensor, recver_transfer_sz: torch.Tensor,
    my_rank_send_offset: int, my_rank_recv_offset: int, my_rank_send_sz: int,
    instance_id: int=None,
):
    """
    Mock PyTorch implementation of all-to-all-v using empty/zero tensors.
    
    This just performs the communication pattern without actually using NVSHMEM buffers.
    Useful for debugging the communication pattern and bypassing NVSHMEM/CUDA kernel issues.
    
    Note: The input tensors are 1D tensors of shape [as_world_size] representing 
    the per-rank view of the all-to-all operation in the attention server group.
    """
    if not torch.distributed.is_initialized():
        print("⚠️ PyTorch A2A Mock: torch.distributed not initialized, skipping")
        return
    
    # Get the attention server group for communication (NCCL, not gloo)
    from d2.runtime.megatron.create_group import get_attn_server_group, get_attn_server_rank

    group = get_attn_server_group()
    as_rank = get_attn_server_rank()
    rank = torch.distributed.get_rank()
    as_world_size = torch.distributed.get_world_size(group=group)
    
    # if as_rank == 0:
    print(f"🔄 Using PyTorch mock A2A (empty tensors, bypasses NVSHMEM) as_world_size={as_world_size}")
    
    # Move metadata to CPU for easier processing
    sender_transfer_sz_cpu = sender_transfer_sz.cpu()
    recver_transfer_sz_cpu = recver_transfer_sz.cpu()
    
    # sender_transfer_sz_cpu += 1
    # recver_transfer_sz_cpu += 1
    
    # Check the actual tensor shapes
    actual_size = sender_transfer_sz_cpu.shape[0]
    if actual_size != as_world_size:
        print(f"⚠️ Warning: tensor size {actual_size} != as_world_size {as_world_size}. Using tensor size.")
    
    # Prepare send/recv lists for torch.distributed.all_to_all
    send_list = []
    recv_list = []
    
    # Iterate over the actual tensor size
    for peer_rank in range(actual_size):
        # Create empty send tensor with the correct size (in bytes)
        send_size = sender_transfer_sz_cpu[peer_rank].item()
        send_size += 1
        if send_size > 0:
            send_tensor = torch.zeros(send_size, dtype=torch.uint8, device='cuda')
        else:
            send_tensor = torch.empty(0, dtype=torch.uint8, device='cuda')
        send_list.append(send_tensor)
        
        # Create empty recv tensor with the correct size (in bytes)
        recv_size = recver_transfer_sz_cpu[peer_rank].item()
        recv_size += 1
        if recv_size > 0:
            recv_tensor = torch.empty(recv_size, dtype=torch.uint8, device='cuda')
        else:
            recv_tensor = torch.empty(0, dtype=torch.uint8, device='cuda')
        recv_list.append(recv_tensor)
    
    # Perform the all-to-all communication with empty tensors on the attention server group
    print(f"[Rank {rank}] 🔄 Performing PyTorch mock A2A communication")
    for i, send_tensor in enumerate(send_list):
        print(f"[Rank {rank}] 🔄 send_list[{i}] shape: {send_tensor.shape}")
    for i, recv_tensor in enumerate(recv_list):
        print(f"[Rank {rank}] 🔄 recv_list[{i}] shape: {recv_tensor.shape}")
    with torch.cuda.nvtx.range("torch.distributed.all_to_all"):
        torch.distributed.all_to_all(recv_list, send_list, group=group)
    print(f"[Rank {rank}] 🔄 PyTorch mock A2A communication completed")
    # torch.cuda.synchronize()
    # torch.distributed.barrier()
    print(f"[Rank {rank}] ✅ PyTorch mock A2A completed")
    
    return None


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
    instance = DispatcherWrapper.get_instance(instance_id)
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
