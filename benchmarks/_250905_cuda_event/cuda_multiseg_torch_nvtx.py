import os
import torch
import torch.distributed as dist

def get_local_rank():
    if "LOCAL_RANK" in os.environ:
        return int(os.environ["LOCAL_RANK"])
    rank = int(os.environ.get("RANK", 0))
    return rank % max(1, torch.cuda.device_count())

def main():
    dist.init_process_group(backend="nccl", init_method="env://")
    rank = dist.get_rank()
    world = dist.get_world_size()
    local_rank = get_local_rank()
    torch.cuda.set_device(local_rank)
    dev = torch.device(f"cuda:{local_rank}")

    # Streams
    compute_stream = torch.cuda.Stream(device=dev)
    comm_stream = torch.cuda.Stream(device=dev)

    # CUDA Events
    A_s = torch.cuda.Event(enable_timing=True); A_e = torch.cuda.Event(enable_timing=True)
    B_s = torch.cuda.Event(enable_timing=True); B_e = torch.cuda.Event(enable_timing=True)
    C_s = torch.cuda.Event(enable_timing=True); C_e = torch.cuda.Event(enable_timing=True)
    O_s = torch.cuda.Event(enable_timing=True); O_e = torch.cuda.Event(enable_timing=True)

    # Data
    M = 4096
    a = torch.randn(M, M, device=dev)
    b = torch.randn(M, M, device=dev)
    x = torch.randn(M, device=dev)
    t = torch.randn(64 * 1024 * 1024 // 4, device=dev)  # ~64MB

    # Warmup
    torch.mm(a, b); dist.all_reduce(t); torch.cuda.synchronize()

    torch.cuda.nvtx.range_push("PROFILE_MAIN")  # <— bound what nsys captures
    O_s.record()

    # --- Segment A: compute ---
    with torch.cuda.nvtx.range("Segment A: GEMM"):
        with torch.cuda.stream(compute_stream):
            A_s.record()
            c = torch.mm(a, b)
            y = c @ x
            A_e.record()

    # --- Segment B: comm (NCCL all_reduce on separate stream) ---
    with torch.cuda.nvtx.range("Segment B: NCCL all_reduce"):
        with torch.cuda.stream(comm_stream):
            B_s.record()
            req = dist.all_reduce(t, async_op=True)
            req.wait()
            B_e.record()

    # --- Segment C: post-comm compute ---
    with torch.cuda.nvtx.range("Segment C: activation"):
        with torch.cuda.stream(compute_stream):
            C_s.record()
            z = torch.relu(y) * 1.0001
            C_e.record()

    compute_stream.synchronize()
    comm_stream.synchronize()
    O_e.record()
    torch.cuda.synchronize()
    torch.cuda.nvtx.range_pop()  # end PROFILE_MAIN

    tA = A_s.elapsed_time(A_e)
    tB = B_s.elapsed_time(B_e)
    tC = C_s.elapsed_time(C_e)
    tO = O_s.elapsed_time(O_e)
    print(f"[rank {rank:02d}/{world}] A={tA:.2f}ms  B={tB:.2f}ms  C={tC:.2f}ms  overall={tO:.2f}ms")

    dist.barrier()
    if rank == 0:
        print("Done.")

if __name__ == "__main__":
    torch.backends.cuda.matmul.allow_tf32 = True
    main()