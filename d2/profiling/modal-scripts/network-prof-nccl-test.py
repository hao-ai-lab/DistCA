# nccl_modal.py
import modal

NCCL_IMAGE = (
    modal.Image.from_registry(
        "nvidia/cuda:12.4.0-devel-ubuntu22.04",
        add_python="3.12",
    )
    .run_commands(
        "apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends git build-essential libnuma-dev",
        "git clone --depth 1 https://github.com/NVIDIA/nccl-tests /opt/nccl-tests && make -C /opt/nccl-tests -j$(nproc) CUDA_HOME=/usr/local/cuda MPI=0"
    )
)

app = modal.App("nccl-bench")


def run_command(cmd):
    import subprocess
    print()
    print()
    print(f"$ {cmd}")
    print()
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    print(result.stdout)
    print()
    print()
    return result.stdout

# @app.function(
#     image=NCCL_IMAGE,
#     gpu="H100:8",
#     timeout=60,
# )
@app.function(
    image=NCCL_IMAGE,
    gpu="H100:4",
    timeout=60,
)
def run_nccl(
    ngpu: int,
    op_name: str, # all_reduce, all_gather
    dtype: str, # half, float
    max_bytes: str = "8G",
):
    import subprocess, os
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, range(ngpu)))
    cmd = f"/opt/nccl-tests/build/{op_name}_perf -t 1 -b 8M -e {max_bytes} -f 2 -d {dtype} -g {ngpu}"
    s = run_command(cmd)
    return f"$ {cmd}\n{s}"


@app.function(
    image=NCCL_IMAGE,
    gpu="H100:8",
    timeout=60,
)
def warmup():
    import subprocess
    cmd = "nvidia-smi"
    subprocess.run(cmd, shell=True)
    run_command("/opt/nccl-tests/build/all_reduce_perf --help")
    run_command("/opt/nccl-tests/build/all_gather_perf --help")
    return



# @app.local_entrypoint()
def main():
    for op_name in ["all_reduce", "all_gather", "reduce_scatter"]:
        for ngpu in [8, 4, 2]:
            for dtype in ["half"]:
                s = run_nccl.remote(ngpu, op_name, dtype)
                with open(f"{op_name}-{ngpu}-{dtype}.perf.txt", "w") as f:
                    f.write(s)

@app.local_entrypoint()
def main2():
    for op_name in ["all_reduce", "all_gather"]:
        for ngpu in [4]:
            for dtype in ["half"]:
                s = run_nccl.remote(ngpu, op_name, dtype)
                with open(f"{op_name}-{ngpu}-{dtype}.perf.txt", "w") as f:
                    f.write(s)

"""
USAGE: all_reduce_perf 
        [-t,--nthreads <num threads>] 
        [-g,--ngpus <gpus per thread>] 
        [-b,--minbytes <min size in bytes>] 
        [-e,--maxbytes <max size in bytes>] 
        [-i,--stepbytes <increment size>] 
        [-f,--stepfactor <increment factor>] 
        [-n,--iters <iteration count>] 
        [-m,--agg_iters <aggregated iteration count>] 
        [-w,--warmup_iters <warmup iteration count>] 
        [-N,--run_cycles <cycle count> run & print each cycle (default: 1; 0=infinite)] 
        [-p,--parallel_init <0/1>] 
        [-c,--check <check iteration count>] 
        [-o,--op <sum/prod/min/max/avg/mulsum/all>] 
        [-d,--datatype <nccltype/all>] 
        [-r,--root <root>] 
        [-z,--blocking <0/1>] 
        [-y,--stream_null <0/1>] 
        [-T,--timeout <time in seconds>] 
        [-G,--cudagraph <num graph launches>] 
        [-C,--report_cputime <0/1>] 
        [-a,--average <0/1/2/3> report average iteration time <0=RANK0/1=AVG/2=MIN/3=MAX>] 
        [-R,--local_register <0/1/2> enable local (1) or symmetric (2) buffer registration on send/recv buffers (default: disable (0))]
        [-h,--help]

$ /opt/nccl-tests/build/all_gather_perf --help

USAGE: all_gather_perf 
        [-t,--nthreads <num threads>] 
        [-g,--ngpus <gpus per thread>] 
        [-b,--minbytes <min size in bytes>] 
        [-e,--maxbytes <max size in bytes>] 
        [-i,--stepbytes <increment size>] 
        [-f,--stepfactor <increment factor>] 
        [-n,--iters <iteration count>] 
        [-m,--agg_iters <aggregated iteration count>] 
        [-w,--warmup_iters <warmup iteration count>] 
        [-N,--run_cycles <cycle count> run & print each cycle (default: 1; 0=infinite)] 
        [-p,--parallel_init <0/1>] 
        [-c,--check <check iteration count>] 
        [-o,--op <sum/prod/min/max/avg/mulsum/all>] 
        [-d,--datatype <nccltype/all>] 
        [-r,--root <root>] 
        [-z,--blocking <0/1>] 
        [-y,--stream_null <0/1>] 
        [-T,--timeout <time in seconds>] 
        [-G,--cudagraph <num graph launches>] 
        [-C,--report_cputime <0/1>] 
        [-a,--average <0/1/2/3> report average iteration time <0=RANK0/1=AVG/2=MIN/3=MAX>] 
        [-R,--local_register <0/1/2> enable local (1) or symmetric (2) buffer registration on send/recv buffers (default: disable (0))]
        [-h,--help]
"""