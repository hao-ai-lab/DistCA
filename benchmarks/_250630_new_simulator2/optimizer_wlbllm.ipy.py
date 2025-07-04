from ortools.sat.python import cp_model
import numpy as np
from dataclasses import dataclass
import rich
import timemodule as tm
from typing import Dict, List
INF = tm.INF

verbose = True
K = 1024
# batch = [128 * K, 64 * K, 32 * K, 32 * K, 16 * K]
# batch = [64 * K] * 8 + [32 * K] * 8
# batch = [64 * K] * 8
batch = [64 * K] * 4 + [4 * K] * 4
max_length = sum(batch) # memory constraint


model = cp_model.CpModel()
num_docs     = len(batch)

should_include_all_gather = True
# should_include_all_gather = False

num_total_devices = 64
tp = 4
cp = 8
dp = num_workers = num_total_devices // (tp * cp)

# ================================
# Setup the Attention time (us)
# - Assume a pre-defined MLP (tp, cp, dp) combination.
# ================================

costs = {}
costs_info = {}
for d in range(num_docs):
    doc_len = batch[d]
    this_attn_time = tm.get_attn_time(tp, cp, doc_len) * 1000 # ms -> us
    this_mlp_time, _ = tm.get_mlp_time(tp, cp, doc_len)
    this_mlp_time *= 1000 # ms -> us
    this_allgather_time = tm.get_allgather_time_with_config(cp, doc_len, max(tm.num_kv_head // tp, 1) * tm.head_dim * 2) * 1000 # ms -> us
    this_allreduce_time = tm.get_allreduce_time_with_config(tp, doc_len, tm.hidden_size) * 1000 # ms -> us
    
    lat = (
        this_mlp_time + this_attn_time + this_allreduce_time
    )
    if should_include_all_gather:
        lat += this_allgather_time
    lat = int(lat)
    costs[d] = lat
    costs_info[d] = {
        "doc_len": doc_len,
        "attn_time": this_attn_time,
        "mlp_time": this_mlp_time,
        "allgather_time": this_allgather_time,
        "allreduce_time": this_allreduce_time,
    }

from rich.table import Table
from rich.console import Console

console = Console()
table = Table(title="Document Latency Breakdown")

# Add columns for document ID and document length
table.add_column("Doc ID", justify="center", style="cyan", no_wrap=True)
table.add_column("Doc Len", justify="center", style="magenta")

# Add columns for each latency component
table.add_column("Attention Time (us)", justify="center", style="green")
table.add_column("MLP Time (us)", justify="center", style="green")
table.add_column("Allgather Time (us)", justify="center", style="green")
table.add_column("Allreduce Time (us)", justify="center", style="green")
table.add_column("Total Latency (us)", justify="center", style="red")

# Populate the table with data
for d in range(num_docs):
    doc_len = costs_info[d]["doc_len"]
    attn_time = costs_info[d]["attn_time"]
    mlp_time = costs_info[d]["mlp_time"]
    allgather_time = costs_info[d]["allgather_time"]
    allreduce_time = costs_info[d]["allreduce_time"]
    total_latency = costs[d]
    
    table.add_row(
        str(d),
        str(doc_len),
        f"{attn_time:.2f}",
        f"{mlp_time:.2f}",
        f"{allgather_time:.2f}",
        f"{allreduce_time:.2f}",
        f"{total_latency:.2f}"
    )

table.title = f"[WLBLLM] Document Latency Breakdown (tp={tp}, cp={cp}, dp={dp})"
console.print(table)


# ================================
# Calculate the Total worker time
# ================================
model = cp_model.CpModel()

# Decision: x[d,w] == 1  ⇔  doc d served by worker w
x = {
    (d, w): model.NewBoolVar(f"x_{d}_{w}")
    for d in range(num_docs)
    for w in range(num_workers)
}

# 1. Each doc goes to exactly one worker
for d in range(num_docs):
    model.Add(sum(x[d, w] for w in range(num_workers)) == 1)

# 2. Per-worker length budget  Σ len_d * x[d,w] ≤ L_max
for w in range(num_workers):
    model.Add(
        sum(batch[d] * x[d, w] for d in range(num_docs)) <= max_length
    )

# 3. Latency per worker  lat_w = Σ cost_d * x[d,w]
lat_worker = [
    model.NewIntVar(0, INF, f"lat_{w}") for w in range(num_workers)
]
for w in range(num_workers):
    # TODO: The cost of MLP is subject to the parallel plan to split the MLP.
    model.Add(
        lat_worker[w]
        == sum(costs[d] * x[d, w] for d in range(num_docs))
    )

# 4. Objective  —  minimise the maximum worker latency
lat_max = model.NewIntVar(0, INF, "lat_max")
for w in range(num_workers):
    model.Add(lat_worker[w] <= lat_max)
model.Minimize(lat_max)

# ——— Solve ———————————————————————————————————————————————
solver = cp_model.CpSolver()
solver.parameters.max_time_in_seconds = 1
solver.parameters.num_search_workers = 0  # use all cores
status = solver.Solve(model)


# ——— Extract assignment ————————————————————————————
doc2worker: Dict[int, int] = {}
batches: List[List[int]] = [[] for _ in range(num_workers)]
for d in range(num_docs):
    for w in range(num_workers):
        if solver.Value(x[d, w]):
            doc2worker[d] = w
            batches[w].append(batch[d])
            break

lat_max_value = solver.Value(lat_max)
lat_worker_values = [solver.Value(lw) for lw in lat_worker]

from rich.console import Console
from rich.table import Table

console = Console()

# Create a table for worker assignments
worker_table = Table(title=f"Worker Assignment and Latency (tp={tp}, cp={cp}, dp={dp})")

worker_table.add_column("Worker ID", justify="right", style="cyan", no_wrap=True)
worker_table.add_column("DocIDs", style="magenta")
worker_table.add_column("Total Length", justify="right", style="white")
worker_table.add_column("Latency (us)", justify="right", style="bold white")
worker_table.add_column("MLP (us)", justify="right", style="green")
worker_table.add_column("Attention (us)", justify="right", style="green")
worker_table.add_column("Allreduce (us)", justify="right", style="green")
worker_table.add_column("Allgather (us)" if should_include_all_gather else "Allgather (us)\n(disabled)", justify="right", style="green" if should_include_all_gather else "bright_black")

# Populate the table with data
for w in range(num_workers):
    assigned_docs = [doc for doc, worker in doc2worker.items() if worker == w]
    total_batch_length = sum(batch[doc] for doc in assigned_docs)
    mlp_cost = sum(costs_info[doc]['mlp_time'] for doc in assigned_docs)
    attn_cost = sum(costs_info[doc]['attn_time'] for doc in assigned_docs)
    allreduce_cost = sum(costs_info[doc]['allreduce_time'] for doc in assigned_docs)
    allgather_cost = sum(costs_info[doc]['allgather_time'] for doc in assigned_docs)
    assigned_docs_str = ", ".join(str(doc) for doc in assigned_docs)
    worker_table.add_row(
        str(w),
        assigned_docs_str,
        str(total_batch_length),
        f"{lat_worker_values[w]:.2f}",
        f"{mlp_cost:.2f}",
        f"{attn_cost:.2f}",
        f"{allreduce_cost:.2f}",
        f"{allgather_cost:.2f}"
    )

console.print(worker_table)

# Print the maximum latency in a panel
from rich.panel import Panel

panel_content = (
    f"[bold white]WLBLLM Maximum Worker Latency: {lat_max_value:.2f} us[/bold white]\n"
    f"[bold white]Config: tp={tp}, cp={cp}, dp={dp}[/bold white]"
)

console.print(Panel(panel_content, title="WLBLLM Latency Information", border_style="white"))
