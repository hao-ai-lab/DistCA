from ortools.sat.python import cp_model
import numpy as np
from dataclasses import dataclass
import rich
import timemodule as tm
INF = tm.INF

verbose = True
K = 1024

# [128K * 1-2], [30-60K * 5-8]

# 1LnS -> 128K + [1*K (????)] * N

# batch = [128 * K, 64 * K, 32 * K, 32 * K, 16 * K]
batch = [64 * K] * 8 + [32 * K] * 8
# batch = [64 * K] * 8
# batch = [64 * K] * 4 + [4 * K] * 4

parallel_plan = [(tp, cp) for tp in (1,2,4,8) for cp in (1,2,4,8)] + [(0,0)]
resource = [tp * cp for tp, cp in parallel_plan]


num_workers  = 8
num_total_devices = 64
mlp_tp, mlp_cp = 4, 8
mlp_dp = num_total_devices // (mlp_tp * mlp_cp)



model = cp_model.CpModel()
num_plans    = len(parallel_plan)
num_docs     = len(batch)

# should_include_all_gather = True
should_include_all_gather = False


# ================================
# Setup the Attention time (us)
# - Assume a pre-defined MLP (tp, cp, dp) combination.
# ================================

# latency[(j,k)] = latency of parallelism plan j for document k
latency = {}
for j in range(num_plans):
    for k in range(num_docs):
        tp, cp = parallel_plan[j]
        doc_len = batch[k]
        if tp == 0 and cp == 0:
            latency[(j,k)] = tm.INF
            continue
        
        lat = (tm.get_attn_time(tp, cp, doc_len))
        if should_include_all_gather:
            lat += tm.get_allgather_time_with_config(num_workers, doc_len, tm.num_kv_head * tm.head_dim * 2)
        latency[(j,k)] = int(lat * 1000) # ms -> us

from rich.table import Table
from rich.console import Console

console = Console()
table = Table(title="Document Latency Table")

# Add columns for doc id and doc len
table.add_column("Doc ID", justify="center", style="cyan", no_wrap=True)
table.add_column("Doc Len\ntp\ncp", justify="center", style="magenta")

# Add columns for each (tp, cp) combination
for tp in (1, 2, 4, 8):
    for cp in (1, 2, 4, 8):
        table.add_column(f"{tp}\n{cp}", justify="center", style="green")

# Populate the table with data
for k in range(num_docs):
    row = [str(k), f"{batch[k]}"]
    for tp in (1, 2, 4, 8):
        for cp in (1, 2, 4, 8):
            j = parallel_plan.index((tp, cp))
            row.append(f"{latency[(j, k)]:>8}")
    table.add_row(*row)
table.title = f"[AttnServer] Attention latency table"
console.print(table)


# ================================
# Calculate the Attention time
# - Assume a pre-defined MLP (tp, cp, dp) combination.
# ================================

x = {(i,j): model.NewBoolVar(f"x_{i}_{j}") for i in range(num_workers) for j in range(num_plans)}
y = {(k,i): model.NewBoolVar(f"y_{k}_{i}") for k in range(num_docs) for i in range(num_workers)}

lat_worker = [model.NewIntVar(0, INF, f"lat_{i}") for i in range(num_workers)]
lat_max    =  model.NewIntVar(0, INF, "lat_max")

# 1. Each worker picks one plan
for i in range(num_workers):
    model.Add(sum(x[i,j] for j in range(num_plans)) == 1)

# 2. Each document assigned to one worker
for k in range(num_docs):
    model.Add(sum(y[k,i] for i in range(num_workers)) == 1)

# TODO: Cut the i-dimension of `z`
# 3. Linearise product: z_{i,j,k} = x_{i,j} * y_{k,i}
z = {}
for i in range(num_workers):
    for j in range(num_plans):
        for k in range(num_docs):
            z[(i,j,k)] = model.NewBoolVar(f"z_{i}_{j}_{k}")
            model.AddBoolAnd([x[i,j], y[k,i]]).OnlyEnforceIf(z[(i,j,k)])
            model.AddBoolOr([x[i,j].Not(), y[k,i].Not(), z[(i,j,k)]])

    # latency per worker
    model.Add(
        lat_worker[i] == sum(latency[(j,k)] * z[(i,j,k)] for j in range(num_plans) for k in range(num_docs))
    )
    model.Add(lat_worker[i] <= lat_max)


# 4. Resource budget
total_devices = sum(
    resource[j] * x[i,j]
    for i in range(num_workers) 
    for j in range(num_plans)
)
model.Add(total_devices <= num_total_devices)

# 5. Objective
model.Minimize(lat_max)

# Solve – parallel by default
solver = cp_model.CpSolver()
solver.parameters.num_search_workers = 0
# add timeout
solver.parameters.max_time_in_seconds = 10
import time

start_time = time.time()
status = solver.Solve(model)
end_time = time.time()

solve_time = end_time - start_time
from rich.panel import Panel

status_name = solver.StatusName()
if status_name != "OPTIMAL":
    rich.print(Panel(f"⚠️ ILP solved in {solve_time:.2f} seconds. Solved Status: {status_name}. This solution is not optimal!", style="yellow"))
else:
    rich.print(Panel(f"✅ ILP solved in {solve_time:.2f} seconds. Solved Status: {status_name}", style="bold white"))

# Extract the solution
# Extract the solutions
from rich.table import Table
from rich.console import Console

console = Console()
table = Table(title="Worker Plan Assignment")

table.add_column("Worker", justify="right", style="cyan", no_wrap=True)
table.add_column("Plan (tp,cp)", style="green")
table.add_column("Latency (us)", justify="right", style="red")


minimum_worker_latency = INF
xs = dict()  # worker -> plan
for i in range(num_workers):
    for j in range(num_plans):
        if solver.Value(x[i, j]) == 1:
            xs[i] = parallel_plan[j]
            
            if xs[i] != (0, 0):
                worker_lat = solver.Value(lat_worker[i])
                if worker_lat < minimum_worker_latency:
                    minimum_worker_latency = worker_lat
                table.add_row(str(i), str(parallel_plan[j]), f"{worker_lat:.2f}")

table.title = f"[AttnServer] Worker Plan Assignment\nMinimum worker latency: {minimum_worker_latency:.2f} us"
console.print(table)

ys = dict()  # doc -> worker

# Create a table for document assignments
doc_table = Table(title="Document to Worker Assignment")

doc_table.add_column("Document ID", justify="right", style="cyan", no_wrap=True)
doc_table.add_column("Length", justify="right", style="magenta")
doc_table.add_column("Assigned Worker", justify="right", style="green")
doc_table.add_column("Attention Time (us)", justify="right", style="red")

for k in range(num_docs):
    for i in range(num_workers):
        if solver.Value(y[k, i]) == 1:
            ys[k] = i
            tp, cp = xs[i]
            # print(tp, cp, batch[k], tm.get_attn_time(tp, cp, batch[k]))
            attn_time = tm.get_attn_time(tp, cp, batch[k]) * 1000 # ms -> us
            doc_table.add_row(str(k), str(batch[k]), str(i), f"{attn_time:.2f} us")

console.print(doc_table)

batch_attn_time = minimum_worker_latency 

# ================================
# Setup the MLP times (us)
# - Enumerate all possible (tp, cp, dp) combinations.
# ================================

token_per_dp_shard = sum(batch) // mlp_dp
batch_mlp_time, _ = tm.get_mlp_time(mlp_tp, mlp_cp, token_per_dp_shard)
batch_mlp_time *= 1000 # ms -> us

batch_allreduce_time = tm.get_allreduce_time_with_config(mlp_tp, token_per_dp_shard, tm.hidden_size) * 2 * 1000 # ms -> us


# ================================
# Calculate the total time
# ================================
batch_total_time = minimum_worker_latency + batch_mlp_time + batch_allreduce_time

from rich.panel import Panel

panel_content = (
    f"[bold white]Batch Linear time:    {batch_mlp_time:>10.2f} us[/bold white]\n"
    f"[bold white]Batch attention time: {batch_attn_time:>10.2f} us[/bold white]\n"
    f"[bold white]Batch allreduce time: {batch_allreduce_time:>10.2f} us[/bold white]\n"
    f"[bold white]--------------------------------------[/bold white]\n"
    f"[bold white]Batch total time:     {batch_total_time:>10.2f} us[/bold white]"
)

rich.print(Panel(panel_content, title="Batch Timing Information", border_style="white"))