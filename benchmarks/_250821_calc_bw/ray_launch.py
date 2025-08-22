# ray_launcher.py
import ray
import argparse
import subprocess
import time
import os
from datetime import datetime

# Define a remote function that runs the command on one node with all 8 GPUs
@ray.remote(num_gpus=8)
def run_cmd(cmd: str, duplicate_log: bool = False, work_dir: str = None):
    """Run a shell command and stream output in real-time."""
    if work_dir:
        os.chdir(work_dir)

    hostname = os.popen("hostname").read().strip()
    if "{hostname}" in cmd:
        cmd = cmd.replace("{hostname}", hostname)
    print(f"Running on {hostname}: {cmd}")
    
    os.environ["RAY_DEDUP_LOGS"] = "1" if duplicate_log else "0"
    try:
        process = subprocess.Popen(
            cmd, shell=True,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1, universal_newlines=True
        )
        
        stdout_lines = []
        # Stream output line by line
        for line in iter(process.stdout.readline, ''):
            line = line.rstrip()
            if line:
                print(f"[Remote] {line}", flush=True)
                stdout_lines.append(line)
        
        # Wait for process to complete
        process.wait()
        
        return {
            "hostname": hostname,
            "command": cmd,
            "returncode": process.returncode,
            "stdout": "\n".join(stdout_lines),
            "stderr": "",  # stderr is merged with stdout
        }
    except Exception as e:
        return {"hostname": hostname, "command": cmd, "error": str(e)}
import fire

def main(
    cmd: str, 
    num_tasks: int = 1,
    address: str = "auto",
    torchrun: bool = False,
    nsys: bool = False,
    nsys_output_file: str = "{now}.{hostname}.{comment}.rep",
    comment: str = "",
    no_duplicate_log: bool = False,
    master_endpoint: str = None
):
    """General Ray cluster command launcher.
    
    Args:
        cmd: Command line to run
        num_tasks: Number of parallel tasks (default=1)
        address: Ray cluster address (default=auto)
        torchrun: Use torchrun for distributed execution
        nsys: Use nsys for profiling
        nsys_output_file: Nsys output file (default={now}.{hostname}.{comment}.rep)
        comment: Comment for the output file
        duplicate_log: Duplicate log to stdout
    """
    os.environ["RAY_DEDUP_LOGS"] = "0" if no_duplicate_log else "1"

    pwd = os.getcwd()
    print(f"Working directory: {pwd}")
    if nsys:
        nsys_output_file = os.path.join(pwd, nsys_output_file)

    # Connect to Ray cluster
    ray.init(address=address)

    # Launch the tasks - each task will use one node with all 8 GPUs
    nnodes = num_tasks
    if master_endpoint is None:
        get_master_ip = os.popen("hostname").read().strip()
        master_endpoint = f"{get_master_ip}:29600"
        print(f"Master endpoint: {master_endpoint}")
    rdvz_id = f"ray_launch_{int(time.time())}_{os.getpid()}"
    
    if torchrun:
        cmd = f"torchrun --rdzv_backend=c10d --rdzv_endpoint={master_endpoint} --rdzv_id={rdvz_id} --max_restarts=0 --nnodes={nnodes} {cmd}"
    
    if nsys:
        if "{now}" in nsys_output_file:
            nsys_output_file = nsys_output_file.replace("{now}", datetime.now().strftime("%Y%m%d_%H%M%S"))
        if "{comment}" in nsys_output_file:
            nsys_output_file = nsys_output_file.replace("{comment}", comment)
        cmd = f"nsys profile --force-overwrite=true -o {nsys_output_file} -t cuda,nvtx {cmd}"
    print(f"Running command: {cmd}")
    tasks = [run_cmd.remote(cmd, duplicate_log=duplicate_log, work_dir=pwd) for _ in range(num_tasks)]
    results = ray.get(tasks)

    # Print results
    # for i, res in enumerate(results):
    #     print(f"\n=== Task {i} ===")
    #     for k, v in res.items():
    #         print(f"{k}: {v}")

if __name__ == "__main__":
    fire.Fire(main)