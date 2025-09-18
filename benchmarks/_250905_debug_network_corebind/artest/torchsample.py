import os
import time
import torch
import torch.distributed as dist


print(torch.cuda.is_available())
local_rank = os.environ.get("LOCAL_RANK")
print(f"{local_rank = }")
torch.cuda.set_device(f'cuda:{local_rank}')
print(f"{torch.cuda.get_device_name() = }")

# Initialize process group
dist.init_process_group(backend='cpu:gloo,cuda:nccl')
rank = dist.get_rank()

# Create tensor on each GPU
x = torch.ones(1).cuda()

start_time = time.time()
with torch.cuda.nvtx.range("allreduce"):
    for _ in range(10000):
        print(f"Before allreduce on rank {rank}: {x}")
        dist.all_reduce(x, op=dist.ReduceOp.SUM)
        print(f"After allreduce on rank {rank}: {x}")
end_time = time.time()

print(f"{start_time = }, {end_time = }, {end_time - start_time = }")

# Cleanup
dist.destroy_process_group()