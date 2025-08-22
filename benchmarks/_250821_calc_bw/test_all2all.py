"""
torchrun --nproc_per_node=1 --nnodes=8 --rdzv_backend=c10d --rdzv_endpoint=fs-mbz-gpu-172:29600 --rdzv_id=0000 --max_restarts=0 /mnt/weka/home/hao.zhang/jd/d2/benchmarks/_250821_calc_bw/test_all2all.py


torchrun --nproc_per_node=1 --nnodes=2 --rdzv_backend=c10d --rdzv_endpoint=fs-mbz-gpu-172:29600 --rdzv_id=fjfjjfjfjfj --max_restarts=0 /mnt/weka/home/hao.zhang/jd/d2/benchmarks/_250821_calc_bw/test_all2all.py
"""

from d2.runtime.attn_kernels.ops import (
    nvshmem_get_unique_id, nvshmem_alloc_empty_unique_id, FastDispatcherWrapper,
    fast_a2a, nvshmem_barrier_all,
    nvshmem_barrier_all_on_current_stream,

)
from torch import tensor
import torch
import os
import time

rank = os.environ["RANK"]
rank = int(rank)
world_size = os.environ["WORLD_SIZE"]
world_size = int(world_size)


torch.distributed.init_process_group(backend="cpu:gloo,cuda:nccl", rank=rank, world_size=world_size)
local_rank = int(os.environ.get("LOCAL_RANK"))
# local_rank = 0

device = torch.device(f"cuda:{local_rank}")


if rank == 0:
    uid = nvshmem_get_unique_id()
else:
    uid = nvshmem_alloc_empty_unique_id()
torch.distributed.broadcast(uid, src=0)

FastDispatcherWrapper.init(
    rank=rank,
    local_rank=local_rank,
    world_size=world_size,
    buffer_size=1024**3 * 4,
    # buffer_size=1024**3 * 32,
    uid=uid,
)

torch.cuda.synchronize()
torch.distributed.barrier()

nvshmem_barrier_all_on_current_stream()
print(f"rank {rank} passing the init barrier")


sender_send_disp = sender_send_offset=tensor([[        0, 214452736, 214452736, 256070144, 297687552, 297687552,
        338600960, 380215808],
    [        0,  45773312, 264801280, 310572032, 356342784, 356342784,
        402116096, 447886848],
    [        0,         0,         0, 155316736, 155316736, 155316736,
        155316736, 252575744],
    [        0,         0,         0,  61671424, 258280448, 258280448,
        258280448, 319954432],
    [        0,  45728256,  91459072, 137189888, 182920704, 403608064,
        449336320, 495067136],
    [        0,         0,         0,         0,         0,         0,
        201326592, 201326592],
    [        0,         0,         0,         0,         0,         0,
                0, 201326592],
    [        0,         0,         0,         0,         0,         0,
                0,         0]], device=device, dtype=torch.uint64)
sender_transfer_sz=tensor([[214452736,         0,  41617408,  41617408,         0,  40913408,
        41614848,  41613312],
    [ 45773312, 219027968,  45770752,  45770752,         0,  45773312,
        45770752,  45772800],
    [        0,         0, 155316736,         0,         0,         0,
        97259008, 100665856],
    [        0,         0,  61671424, 196609024,         0,         0,
        61673984,  61676544],
    [ 45728256,  45730816,  45730816,  45730816, 220687360,  45728256,
        45730816,  45725696],
    [        0,         0,         0,         0,         0, 201326592,
                0,         0],
    [        0,         0,         0,         0,         0,         0,
        201326592,         0],
    [        0,         0,         0,         0,         0,         0,
                0, 201326592]], device=device, dtype=torch.uint64)
sender_recv_disp = recver_transfer_offset = tensor([[        0,         0,         0,         0,         0,         0,
                0,         0],
    [214452736,         0,  41617408,  41617408,         0,  40913408,
        41614848,  41613312],
    [260226048, 219027968,  87388160,  87388160,         0,  86686720,
        87385600,  87386112],
    [260226048, 219027968, 242704896,  87388160,         0,  86686720,
        184644608, 188051968],
    [260226048, 219027968, 304376320, 283997184,         0,  86686720,
        246318592, 249728512],
    [305954304, 264758784, 350107136, 329728000, 220687360, 132414976,
        292049408, 295454208],
    [305954304, 264758784, 350107136, 329728000, 220687360, 333741568,
        292049408, 295454208],
    [305954304, 264758784, 350107136, 329728000, 220687360, 333741568,
        493376000, 295454208]], device=device, dtype=torch.uint64)
recver_transfer_sz=tensor([[214452736,  45773312,         0,         0,  45728256,         0,
                0,         0],
    [        0, 219027968,         0,         0,  45730816,         0,
                0,         0],
    [ 41617408,  45770752, 155316736,  61671424,  45730816,         0,
                0,         0],
    [ 41617408,  45770752,         0, 196609024,  45730816,         0,
                0,         0],
    [        0,         0,         0,         0, 220687360,         0,
                0,         0],
    [ 40913408,  45773312,         0,         0,  45728256, 201326592,
                0,         0],
    [ 41614848,  45770752,  97259008,  61673984,  45730816,         0,
        201326592,         0],
    [ 41613312,  45772800, 100665856,  61676544,  45725696,         0,
                0, 201326592]], device=device, dtype=torch.uint64)

sender_send_disp = sender_send_disp[:world_size, :world_size]
print(f"rank {rank} sender_send_disp", sender_send_disp)
sender_transfer_sz = sender_transfer_sz[:world_size, :world_size]
print(f"rank {rank} sender_transfer_sz", sender_transfer_sz)
sender_recv_disp = sender_recv_disp[:world_size, :world_size]
print(f"rank {rank} sender_recv_disp", sender_recv_disp)
recver_transfer_sz = recver_transfer_sz[:world_size, :world_size]
print(f"rank {rank} recver_transfer_sz", recver_transfer_sz)

my_rank_send_offset = sender_send_disp[rank, rank].item()
my_rank_recv_offset = sender_recv_disp[rank, rank].item()
my_rank_send_sz = sender_transfer_sz[rank, rank].item()

instance_id = 0


torch.cuda.synchronize()
torch.distributed.barrier()

print(f"rank {rank} start")
for _ in range(5):
    

    torch.cuda.synchronize()
    torch.distributed.barrier()
    print(f"rank {rank} start fast_a2a: {_}")
    with torch.cuda.nvtx.range("fast_a2a"):
        fast_a2a(
            # FastDispatcherWrapper.get_instance(instance_id).handle,
            sender_send_disp[rank], sender_transfer_sz[rank],
            sender_recv_disp[rank], recver_transfer_sz[rank],
            my_rank_send_offset, my_rank_recv_offset, my_rank_send_sz,
            instance_id=instance_id,
        )
    torch.cuda.synchronize()
    torch.distributed.barrier()
    print(f"rank {rank} after fast_a2a: {_}")
    
    nvshmem_barrier_all()
    nvshmem_barrier_all_on_current_stream()
    FastDispatcherWrapper.release(instance_id)
    time.sleep(2)
    torch.distributed.barrier()

print(f"rank {rank} done")