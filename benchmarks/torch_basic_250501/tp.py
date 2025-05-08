import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

def setup(rank, world_size):
    """Initialize the distributed environment."""
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '12355'
    
    # Set the device for this process
    torch.cuda.set_device(rank)
    
    # Initialize the process group with NCCL backend
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    """Clean up the distributed environment."""
    dist.destroy_process_group()

def run_demo(rank, world_size):
    """Example function to run on each process."""
    print(f"Running on rank {rank} with GPU {torch.cuda.get_device_name(rank)}")
    setup(rank, world_size)
    
    # Create a tensor on the appropriate GPU
    tensor = torch.tensor([rank], dtype=torch.float32, device=f'cuda:{rank}')
    if rank == 0:
        # Send tensor to rank 1
        dist.send(tensor=tensor, dst=1)
        print(f"Rank {rank} sent tensor: {tensor}")
    else:
        # Receive tensor from rank 0
        dist.recv(tensor=tensor, src=0)
        print(f"Rank {rank} received tensor: {tensor}")
    
    cleanup()

def main():
    # Check if CUDA is available
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. This script requires GPU support.")
    
    # Get number of available GPUs
    num_gpus = torch.cuda.device_count()
    print(f"Found {num_gpus} GPUs")
    
    world_size = num_gpus  # Use all available GPUs
    mp.spawn(
        run_demo,
        args=(world_size,),
        nprocs=world_size,
        join=True
    )

if __name__ == "__main__":
    main()
