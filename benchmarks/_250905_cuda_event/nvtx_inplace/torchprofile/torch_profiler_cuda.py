import os
import torch
import torch.profiler

def heavy_stuff():
    a = torch.randn(4096, 4096, device="cuda")
    b = torch.randn(4096, 4096, device="cuda")
    return a @ b  # single large GEMM

def main():
    assert torch.cuda.is_available(), "CUDA GPU required"
    torch.cuda.init()

    # --- Work dir (PST/PDT) ---
    # If you prefer relying on the shell TZ, you can pass OUTDIR via env/CLI instead.
    from datetime import datetime, timezone, timedelta
    # NOTE: -7 matches PDT; if you want exact zone handling, pass OUTDIR from shell with TZ=America/Los_Angeles
    TS = datetime.now(timezone(timedelta(hours=-7))).strftime("%Y%m%d_%H%M%S")
    outdir = os.environ.get("OUTDIR", f"./profiles/{TS}")
    os.makedirs(outdir, exist_ok=True)

    # --- Profiler setup ---
    prof = torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.CUDA],
        record_shapes=True,
        with_stack=True,
        with_flops=True,
        on_trace_ready=torch.profiler.tensorboard_trace_handler(outdir),  # dumps per-step traces
    )

    # --- Warmup (avoid first-call jitters) ---
    heavy_stuff(); torch.cuda.synchronize()

    # --- Profile a few steps ---
    with prof:
        for step in range(3):
            _ = heavy_stuff()
            torch.cuda.synchronize()
            prof.step()  # mark iteration for the trace viewer

    # --- Human-readable summary ---
    print("\n=== Top ops by CUDA time ===")
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=20))

    # --- Chrome trace for quick inspection (in addition to tensorboard files) ---
    prof.export_chrome_trace(os.path.join(outdir, "trace.json"))
    print(f"\nTraces written to: {outdir}")

if __name__ == "__main__":
    torch.backends.cuda.matmul.allow_tf32 = True
    main()