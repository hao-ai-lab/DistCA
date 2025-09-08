import os, time, socket, torch, torch.distributed as dist
import datetime
import sys

def read_ctx_switch():
    v=nv=0
    proc_info = ""
    with open("/proc/self/status","r") as f:
        for line in f:
            proc_info += line.strip() + "\n"
            if line.startswith("voluntary_ctxt_switches"):
                v=int(line.split()[1])
            elif line.startswith("nonvoluntary_ctxt_switches"):
                nv=int(line.split()[1])
    return v, nv, proc_info.strip()


from d2.utils.logger import print_rank, Tee

import psutil, os
def main(args):


    # env from torchrun/srun
    rank = int(os.environ.get("RANK", os.environ.get("SLURM_PROCID","0")))
    world = int(os.environ.get("WORLD_SIZE", os.environ.get("SLURM_NTASKS","1")))
    local = int(os.environ.get("LOCAL_RANK", os.environ.get("SLURM_LOCALID","0")))
    master = os.environ.get("MASTER_ADDR", "127.0.0.1")
    port   = os.environ.get("MASTER_PORT", "29500")
    
    p = psutil.Process(os.getpid())
    p.cpu_affinity([local * 16, local * 16 + 1])  # pin to core based on local rank
    print("Now allowed CPUs:", p.cpu_affinity())

    # Redirect stdout and stderr to rank-specific log file
    stdout_logfile = open(os.path.join(args.output_dir, f"stdout.rank{rank}.log"), "w")
    stderr_logfile = open(os.path.join(args.output_dir, f"stderr.rank{rank}.log"), "w")
    if rank == 0:
        sys.stdout = Tee(sys.stdout, stdout_logfile)
        sys.stderr = Tee(sys.stderr, stderr_logfile)
        # also write the entire env var into the log file
    else:
        sys.stdout = Tee(stdout_logfile)
        sys.stderr = Tee(stderr_logfile)
        pass

    
    # Read CPU binding info
    print(f"{rank}: [BIND] Reading CPU binding info...")
    cpu_list = None
    with open("/proc/self/status", "r") as f:
        for line in f:
            if line.startswith("Cpus_allowed_list"):
                cpu_list = line.split()[1].strip()
                break
    print(f"{rank}: [BIND] Found CPU list: {cpu_list}")

    # Print binding info
    host = os.environ.get("SLURMD_NODENAME", socket.gethostname())
    print(f"[{host}] rank={rank} local_rank={local} device={local} cpus={cpu_list}")


    import subprocess
    print(f"{rank}: Running nvidia-smi topology check...")
    subprocess.run(["nvidia-smi", "topo", "-p2p", "w"], check=True)

    print(f"{rank}: [INIT] rank={rank} world={world} local={local} master={master} port={port}")

    print(f"{rank}: [CUDA] Setting device to cuda:{local}")
    torch.cuda.set_device(f"cuda:{local}")
    print(f"{rank}: [CUDA] Current device: {torch.cuda.current_device()}")
    print(f"{rank}: [CUDA] Synchronizing...")
    torch.cuda.synchronize()
    print(f"{rank}: [CUDA] Creating test tensor...")
    x = torch.ones(1, device="cuda", dtype=torch.float32)
    print(f"{rank}: [CUDA] Synchronizing...")
    torch.cuda.synchronize()

    print(f"{rank}: [DIST] Initializing process group...")
    os.environ["NCCL_BLOCKING_WAIT"] = "1"  # Enable timeout
    os.environ["NCCL_ASYNC_ERROR_HANDLING"] = "1"  # Enable async error handling
    
    dist.init_process_group("cpu:gloo,cuda:nccl", timeout=datetime.timedelta(seconds=30), rank=rank, world_size=world)
    
    print(f"{rank}: [DIST] Process group initialized")

    a = torch.randn(1024, 1024, device="cuda", dtype=torch.float32)
    b = torch.randn(1024, 1024, device="cuda", dtype=torch.float32)

    # small tensor to stress launch/sync overhead more than math
    N = 1 * 1024
    print(f"{rank}: [TENSOR] Creating {N} element tensor (~1MB)")
    x = torch.ones(N, device="cuda", dtype=torch.float32)
    print(f"{rank}: [TENSOR] Tensor created")

    print(f"{rank}: [WARMUP] Starting warmup phase...")
    # warmup
    for i in range(10):
        print(f"{rank}: [WARMUP] Iteration {i}/100")
        c = a @ b
    #     dist.all_reduce(x, op=dist.ReduceOp.SUM)
    print(f"{rank}: [WARMUP] Synchronizing...")
    torch.cuda.synchronize()
    print(f"{rank}: [WARMUP] Barrier...")
    dist.barrier()
    print(f"{rank}: [WARMUP] Warmup complete")

    print(f"{rank}: [BENCH] Reading initial context switches...")
    v0, nv0, proc_info = read_ctx_switch()
    print(f"{rank}: [BENCH] Initial voluntary: {v0}, non-voluntary: {nv0}")
    print(f"{rank}: [BENCH] {proc_info}")
    t0 = time.perf_counter()

    print(f"{rank}: [BENCH] Starting benchmark...")
    # Change to matmul of a and b
    iters = 5000
    for i in range(iters):
        if i % 100 == 0:
            print(f"{rank}: [BENCH] Iteration {i}/{iters}")
        c = a @ b
        pass
    
    # for i in range(iters):
    #     if i % 100 == 0:
    #         print(f"{rank}: [BENCH] Iteration {i}/{iters}")
    #     # dist.all_reduce(x, op=dist.ReduceOp.SUM)

    print(f"{rank}: [BENCH] Synchronizing...")
    torch.cuda.synchronize()
    t1 = time.perf_counter()
    print(f"{rank}: [BENCH] Reading final context switches...")
    v1, nv1, proc_info = read_ctx_switch()
    print(f"{rank}: [BENCH] Final voluntary: {v1}, non-voluntary: {nv1}")

    dt = t1 - t0
    avg_us = dt / iters * 1e6
    print(f"{rank}: [BENCH] Total time: {dt:.3f}s")
    print(f"{rank}: [BENCH] Average time per iteration: {avg_us:.1f}us")

    # Print binding info
    print(f"{rank}: [BIND] Reading CPU binding info...")
    cpu_list = "?"
    try:
        with open("/proc/self/status","r") as f:
            for line in f:
                if line.startswith("Cpus_allowed_list"):
                    cpu_list = line.split("\t",1)[1].strip()
                    print(f"{rank}: [BIND] Found CPU list: {cpu_list}")
                    break
    except Exception as e:
        print(f"{rank}: [BIND] Error reading CPU binding: {e}")
        pass

    host = socket.gethostname()
    print(f"[{host}] rank={rank} local_rank={local} device={torch.cuda.current_device()} cpus={cpu_list}")
    print(f"  iters={iters} total={dt:.3f}s avg={avg_us:.1f}us  ")
    print(f"  ctxswitch +V{v1-v0} +NV{nv1-nv0}")
    print(f"{rank}: [BIND] {proc_info}")

    dist.barrier()
    dist.destroy_process_group()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=str, default=None)
    args = parser.parse_args()
    print(args)
    main(args)