import os
import sys
import contextlib
from rich import print as rich_print
from runsim_func import run_simulation
from contextlib import redirect_stdout

from d2.simulator.optimizers.samples import (
    batch_documents, 
    sample_wlbllm_docs_altered,
)

os.makedirs("log", exist_ok=True)

def dual_print(s, f):
    print(s, flush=True)
    f.write(s + "\n")
    f.flush()

K = 1024

# batches = [
#     [ctx_len] * (128 * K // ctx_len)
#     for ctx_len in [1*K, 2*K, 4*K, 8*K, 16*K, 32*K, 64*K, 128*K]
# ]
batches = [
    # '[64 * K]',
    # '[32 * K] * 2',
    '[60 * K, 32 * K, 45 * K, 1 * K]',
    # '[40 * K] + [1 * K] * 3',
    # '[40 * K] + [8 * K] * 3',
    # '[40 * K] + [1 * K] * 8',
    # '[40 * K] + [1 * K] * 15',
    # '[40 * K] + [1 * K] * 30',
    # '[8 * K] * 5',
]

# wlbdocs = batch_documents(
#     sample_wlbllm_docs_altered(size=10000),
#     max_ctx_length=128*K,
# )
# batches = list(wlbdocs)



f = open("wlb_vs_attnserver_diff.psv", "w+")

rich_print("* [green] green [/] means attnserver is faster")
dual_print("idx|batch|num_total_devices|num_workers|diff(ms)|did_time_out", f)

should_redirect = True
should_redirect = False
def redirect_ctx(log_file):
    if should_redirect:
        return redirect_stdout(log_file)
    else:
        return contextlib.nullcontext()


idx = 0
for batch_ in batches:
    # for num_workers in [1, 2, 4, 8]:
    for num_workers in [4]:
        for num_total_devices in [32]:
            if isinstance(batch_, str):
                batch = eval(batch_)
            else:
                batch = batch_
            
            log_file_path = f"log/{idx}.log"
            with open(log_file_path, "w") as log_file:
                with redirect_ctx(log_file):
                    diff, did_time_out = run_simulation(
                        batch=batch, 
                        num_total_devices=num_total_devices,
                        num_workers=num_workers,
                    )

            did_time_out = f"[bold red on white]{did_time_out}[/bold red on white]" if did_time_out else str(did_time_out)
            diff = 0.0 if abs(diff) < 1e-3 else diff
            diff_color = "[green]" if diff > 0 else ("[red]" if diff < 0 else "[white]")
            rich_print(f"{diff_color}{idx}|{batch_}|{num_total_devices}|{num_workers}|{diff:.3f}|{did_time_out}[/]")
            # print(f"{idx}|{batch_}|{num_total_devices}|{num_workers}|{diff:.3f}", flush=True)
            f.write(f"{idx}|{batch_}|{num_total_devices}|{num_workers}|{diff:.3f}|{did_time_out}\n")
            f.flush()
            idx += 1
