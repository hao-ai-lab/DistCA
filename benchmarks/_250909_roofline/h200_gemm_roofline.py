#!/usr/bin/env python3
# h200_gemm_roofline.py
# Roofline-style GEMM sweep for NVIDIA GPUs (H200-friendly).
# Outputs CSV: M,N,K,dtype,ms,TFLOPs,AI(FLOP/byte),GBps,notes

import argparse, math, time, os, sys
import torch

def bytes_mnk(M,N,K, dtype, beta0=True):
    """Estimated DRAM traffic for C = A @ B (+ beta*C if beta0==False).
       Counts: read A (M*K), read B (K*N), write C (M*N), read C if beta!=0.
       Does not model caches; good for roofline-style AI."""
    e = torch.tensor([], dtype=dtype).element_size()
    reads = M*K + K*N + (0 if beta0 else M*N)
    writes = M*N
    return (reads + writes) * e

def flops_mnk(M,N,K):
    # GEMM does 2*M*N*K FLOPs (multiply + add)
    return 2.0 * M * N * K

@torch.inference_mode()
def bench_gemm_once(M, N, K, dtype, transA=False, transB=False, beta0=True, iters=50, warmup=10):
    dev = torch.device("cuda:0")
    # Pick shapes for A (M x K or K x M) and B (K x N or N x K) depending on transpose flags
    Ashape = (K, M) if transA else (M, K)
    Bshape = (N, K) if transB else (K, N)

    A = torch.randn(*Ashape, device=dev, dtype=dtype)
    B = torch.randn(*Bshape, device=dev, dtype=dtype)
    C = torch.empty(M, N, device=dev, dtype=dtype)

    # Multiply by T if requested (by swapping .t() views)
    Aop = A.t() if transA else A
    Bop = B.t() if transB else B

    # We’ll use beta=0 path by default to avoid reading C from memory
    beta0_flag = beta0

    # Ensure Tensor Cores: use matmul/torch.mm on contiguous inputs
    # For bf16/fp16, multiples of 8/16 help; we leave to user sweep.

    # Warmup
    torch.cuda.synchronize()
    for _ in range(warmup):
        if beta0_flag:
            C = Aop @ Bop
        else:
            C.mul_(0.0)
            C = C + (Aop @ Bop)
    torch.cuda.synchronize()

    # Timed runs
    start = torch.cuda.Event(enable_timing=True)
    end   = torch.cuda.Event(enable_timing=True)

    start.record()
    for _ in range(iters):
        if beta0_flag:
            C = Aop @ Bop
        else:
            C.mul_(0.0)
            C = C + (Aop @ Bop)
    end.record()
    torch.cuda.synchronize()

    ms = start.elapsed_time(end) / iters  # per-iter ms
    return float(ms)

def measure_d2d_bandwidth(bytes_target=4<<30, repeat=8):
    """Empirical device-to-device memcpy bandwidth (GB/s)."""
    # Use a big chunk to approach HBM limit
    nbytes = bytes_target
    dtype = torch.uint8
    x = torch.empty(nbytes, device="cuda", dtype=dtype)
    y = torch.empty_like(x)

    torch.cuda.synchronize()
    # warmup copies
    for _ in range(3):
        y.copy_(x)
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end   = torch.cuda.Event(enable_timing=True)

    start.record()
    for _ in range(repeat):
        y.copy_(x)
    end.record()
    torch.cuda.synchronize()

    ms = start.elapsed_time(end) / repeat
    gbps = (nbytes / 1e9) / (ms / 1e3)
    return gbps

def find_empirical_peak_tflops(dtype, sizes=(4096, 8192, 12288, 16384), iters=30, warmup=5):
    best = (0.0, None, None, None, None)  # tflops, M,N,K,ms
    for M in sizes:
        for N in sizes:
            for K in sizes:
                ms = bench_gemm_once(M,N,K,dtype, iters=iters, warmup=warmup)
                tflops = flops_mnk(M,N,K) / (ms/1e3) / 1e12
                if tflops > best[0]:
                    best = (tflops, M, N, K, ms)
    return best  # TFLOPs, M,N,K,ms

def as_dtype(s):
    s = s.lower()
    if s in ("fp16","half","float16"):  return torch.float16
    if s in ("bf16","bfloat16"):        return torch.bfloat16
    if s in ("fp32","float","float32"): return torch.float32
    raise ValueError(f"Unsupported dtype: {s}")

# --- Roofline Plot Helper -----------------------------------------------
# Usage:
#   python h200_gemm_roofline.py --dtype bf16 ... --csv_file results.csv
#   python h200_gemm_roofline.py --plot results.csv --peak_tflops 831.6 --peak_bw 2137.1
#
# If --plot is provided, script will plot roofline from CSV instead of running benchmarks.

import sys
import numpy as np

def plot_roofline(csv_file, peak_tflops, peak_bandwidth_gbps, title="H200 GEMM Roofline"):
    import pandas as pd
    import matplotlib.pyplot as plt

    df = pd.read_csv(csv_file)
    if "AI_FLOP_per_byte" in df.columns:
        df.rename(columns={"AI_FLOP_per_byte":"AI"}, inplace=True)

    ridge_ai = (peak_tflops * 1e3) / peak_bandwidth_gbps

    plt.figure(figsize=(7,5))
    x = df["AI"]
    y = df["TFLOPs"]
    plt.scatter(x, y, c=df["M"], cmap="viridis", s=40, label="Measurements")

    xs = np.linspace(min(x)*0.95, max(x)*1.05, 200)
    bw_tf = (peak_bandwidth_gbps * xs) / 1000.0
    plt.plot(xs, bw_tf, label=f"BW ceiling ~{peak_bandwidth_gbps:.0f} GB/s")
    plt.axhline(peak_tflops, linestyle="--", color="r",
                label=f"Compute ceiling {peak_tflops:.1f} TFLOP/s")
    plt.axvline(ridge_ai, linestyle=":", color="k", label=f"Ridge AI {ridge_ai:.1f}")

    plt.xlabel("Arithmetic Intensity (FLOP/byte)")
    plt.ylabel("Throughput (TFLOP/s)")
    plt.title(title)
    plt.legend()
    plt.grid(True, which="both")
    out_png = csv_file.replace(".csv", ".roofline.png")
    plt.savefig(out_png, dpi=160, bbox_inches="tight")
    print(f"Saved roofline plot to {out_png}")

def main():
    parser = argparse.ArgumentParser(description="H200 GEMM roofline sweep (empirical).")
    parser.add_argument("--dtype", type=str, default="bf16", help="bf16|fp16|fp32")
    parser.add_argument("--sizes", type=str, default="1024,2048,4096,8192,12288,16384",
                        help="Comma list of base sizes to sweep; we try (M,N,K) over these.")
    parser.add_argument("--fixed", type=str, default="none",
                        help="Optionally fix one dimension: none|M=4096|N=4096|K=4096 etc.")
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--beta0", action="store_true", help="Use beta=0 model (skip reading C).")
    parser.add_argument("--no-tc", action="store_true", help="Disable TensorCore-friendly flags.")
    parser.add_argument("--empirical-ceilings", action="store_true",
                        help="First measure D2D GB/s and peak GEMM TFLOP/s and print ridge point.")
    parser.add_argument("--transpose", type=str, default="NN", help="GEMM op: N/T for A and B, e.g., NN, NT, TN, TT")
    parser.add_argument("--csv_file", type=str, default="", help="If set, also write CSV here.")
    args = parser.parse_args()

    # Device info
    torch.cuda.init()
    torch.cuda.set_device("cuda:0")
    props = torch.cuda.get_device_properties("cuda:0")
    name = props.name
    sm   = props.major * 10 + props.minor
    print(f"# Device: {name} (SM {sm}), {props.multi_processor_count} SMs, {props.total_memory/1e9:.2f} GB HBM")
    print(f"# CUDA:   {torch.version.cuda}, PyTorch: {torch.__version__}")

    # Matmul precision knobs
    if args.no_tc:
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.set_float32_matmul_precision("highest")
    else:
        # Let PyTorch/ cuBLASLt use Tensor Cores where possible (TF32/F16/BF16)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.set_float32_matmul_precision("high")

    dtype = as_dtype(args.dtype)
    sizes = [int(x) for x in args.sizes.split(",")]
    trans = args.transpose.upper()
    assert trans in ("NN","NT","TN","TT")
    tA = (trans[0] == "T"); tB = (trans[1] == "T")

    csv_out = None
    if args.csv_file:
        csv_out = open(args.csv_file, "w", buffering=1)
    def emit(line):
        print(line)
        if csv_out:
            csv_out.write(line + "\n")

    # Optional empirical ceilings
    if args.empirical_ceilings:
        print("# Measuring empirical device-to-device bandwidth...")
        gbps = measure_d2d_bandwidth()
        print(f"# Empirical bandwidth: {gbps:.1f} GB/s")

        print(f"# Searching empirical peak compute for {args.dtype}...")
        peak_tflops, Mm, Nn, Kk, ms = find_empirical_peak_tflops(dtype)
        print(f"# Empirical peak: {peak_tflops:.1f} TFLOP/s @ ({Mm},{Nn},{Kk}) ~ {ms:.2f} ms/iter")

        ridge_ai = (peak_tflops * 1e12) / (gbps * 1e9)  # FLOP/byte where compute=line meets bandwidth=line
        print(f"# Ridge point (AI*): {ridge_ai:.2f} FLOP/byte  (*from empirical ceilings)\n")

    # CSV header
    emit("M,N,K,dtype,ms,TFLOPs,AI_FLOP_per_byte,GBps,op,notes")

    # Prepare sweep list
    fixed = args.fixed.strip().lower()
    def fixed_dim(dim):
        if not fixed.startswith(dim+"="):
            return None
        try:
            return int(fixed.split("=")[1])
        except:
            return None

    fixM = fixed_dim("m")
    fixN = fixed_dim("n")
    fixK = fixed_dim("k")

    Ms = [fixM] if fixM else sizes
    Ns = [fixN] if fixN else sizes
    Ks = [fixK] if fixK else sizes

    # Run sweep
    for M in Ms:
        for N in Ns:
            for K in Ks:
                # Favor multiples-of-16 for F16/BF16 Tensor Cores
                note = ""
                if dtype in (torch.float16, torch.bfloat16):
                    if (M % 16) or (N % 16) or (K % 16):
                        note = "non-multiple-of-16; TC under-util"

                ms = bench_gemm_once(M,N,K,dtype, transA=tA, transB=tB,
                                     beta0=args.beta0, iters=args.iters, warmup=args.warmup)
                flops = flops_mnk(M,N,K)
                tflops = flops / (ms/1e3) / 1e12
                traffic = bytes_mnk(M,N,K,dtype, beta0=args.beta0)
                ai = flops / traffic
                gbps = (traffic / 1e9) / (ms / 1e3)

                emit(f"{M},{N},{K},{args.dtype},{ms:.3f},{tflops:.3f},{ai:.3f},{gbps:.1f},{trans},{note}")


    import io
    import pandas as pd
    import matplotlib.pyplot as plt
    import numpy as np

    with open(args.csv_file, "r") as f:
        csv_out = f.read()
    df = pd.read_csv(io.StringIO(csv_out))

    # Empirical ceilings from user metadata
    peak_tflops = 989
    peak_bandwidth_gbps = 2137.1
    ridge_ai = (peak_tflops * 1e3) / peak_bandwidth_gbps

    # Plot roofline
    plt.figure(figsize=(7,5))
    x = df["AI"]
    y = df["TFLOPs"]
    plt.scatter(x, y, c=df["M"], cmap="viridis", s=40, label="Measurements")

    # Ceilings
    xs = np.linspace(min(x)*0.98, max(x)*1.05, 200)
    bw_tf = (peak_bandwidth_gbps * xs) / 1000.0
    plt.plot(xs, bw_tf, label=f"BW ceiling ~{peak_bandwidth_gbps:.0f} GB/s")
    plt.axhline(peak_tflops, linestyle="--", color="r", label=f"Compute ceiling {peak_tflops:.1f} TFLOP/s")
    plt.axvline(ridge_ai, linestyle=":", color="k", label=f"Ridge AI {ridge_ai:.1f}")

    plt.xlabel("Arithmetic Intensity (FLOP/byte)")
    plt.ylabel("Throughput (TFLOP/s)")
    plt.title("H200 GEMM Roofline (bf16, N=128)")
    plt.legend()
    plt.grid(True, which="both")
    roofline_png2 = "h200_gemm_roofline_bf16_N128_full.png"
    plt.savefig(roofline_png2, dpi=160, bbox_inches="tight")
    plt.close()

    # TFLOPs vs K grouped by M
    plt.figure(figsize=(7,5))
    for M, g in df.groupby("M"):
        g_sorted = g.sort_values("K")
        plt.plot(g_sorted["K"], g_sorted["TFLOPs"], marker="o", label=f"M={M}")
    plt.xscale("log")
    plt.xlabel("K (log scale)")
    plt.ylabel("Throughput (TFLOP/s)")
    plt.title("TFLOPs vs K (bf16, N=128)")
    plt.legend()
    plt.grid(True, which="both")
    tflops_vs_k_png2 = "h200_tflops_vs_k_bf16_N128_full.png"
    plt.savefig(tflops_vs_k_png2, dpi=160, bbox_inches="tight")
    plt.close()

    if csv_out:
        csv_out.close()

    if "--plot" in sys.argv:
        idx = sys.argv.index("--plot")
        try:
            csv_file = sys.argv[idx+1]
        except IndexError:
            print("Usage: --plot <csv_file> --peak_tflops <val> --peak_bw <GB/s>")
            sys.exit(1)

        # Defaults; override with command-line flags
        peak_tflops = 800.0
        peak_bw = 2000.0
        if "--peak_tflops" in sys.argv:
            peak_tflops = float(sys.argv[sys.argv.index("--peak_tflops")+1])
        if "--peak_bw" in sys.argv:
            peak_bw = float(sys.argv[sys.argv.index("--peak_bw")+1])

        plot_roofline(csv_file, peak_tflops, peak_bw)
        sys.exit(0)

if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("CUDA device not available.", file=sys.stderr)
        sys.exit(1)
    main()