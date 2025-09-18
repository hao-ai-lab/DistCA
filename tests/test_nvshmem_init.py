"""
LOG_DIR="logs/$(TZ=America/Los_Angeles date +%Y%m%d_%H%M%S)"
srun -N 2 -G 8 --ntasks-per-node=1 \
    --output="${LOG_DIR}/%N.%j.out" \
    --error="${LOG_DIR}/%N.%j.out" \
    bash -lc '
        torchrun --nproc_per_node=8 --nnodes=2 --rdzv_backend=c10d --rdzv_endpoint=fs-mbz-gpu-004:29800 --rdzv_id=0000 --max_restarts=0 \
        python test_nvshmem_init.py
    '
"""
import os
import torch

rank = os.environ["RANK"]
rank = int(rank)
world_size = os.environ["WORLD_SIZE"]
world_size = int(world_size)
local_rank = os.environ["LOCAL_RANK"]
local_rank = int(local_rank)

buffer_size = 4 * 1024 ** 3

# worker.init_torch_distributed()
print(f"[Rank {rank}] init_torch_distributed: {rank = } {world_size = } {local_rank = }")
torch.distributed.init_process_group(backend="cpu:gloo,cuda:nccl", rank=rank, world_size=world_size)
device = torch.device(f"cuda:{local_rank}")
torch.cuda.set_device(device)
print(f"[Rank {rank}] init_torch_distributed done")

# worker.init_nvshmem(buffer_size, local_rank=local_rank)
from d2.runtime.attn_kernels.ops import (
    nvshmem_get_unique_id, nvshmem_alloc_empty_unique_id, FastDispatcherWrapper
)
print(f"[Rank {rank}] ====== init_nvshmem ======")
if rank == 0:
    uid = nvshmem_get_unique_id()
    # print(f"Init uid = {uid}")
else:
    uid = nvshmem_alloc_empty_unique_id()
# print(f"Broadcast uid: {uid}")
torch.distributed.broadcast(uid, src=0)
# torch.cuda.synchronize()
torch.distributed.barrier()
# print(f"[Rank {rank}] uid = {uid}")

print("FastDispatcherWrapper.init")
FastDispatcherWrapper.init(
    rank, local_rank, world_size, buffer_size, uid
)
print(f"[Rank {rank}] ====== init_nvshmem done ======")