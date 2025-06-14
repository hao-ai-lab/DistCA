# %%

import d2.simulator.optimizers.wlbllm as wlbllm
import d2.simulator.optimizers.attnserver as attnserver

from d2.simulator.optimizers.attnserver import AttnServerSolver
from d2.simulator.optimizers.wlbllm import WlbLlmSolver


# %%
K = 1024
batch = [64, 32]
batch =  [i * K for i in batch]

num_workers = 4
num_total_devices = 32

print("Qwen3-235B")


# %%
import timemodule as tm

# %%
# tm.get_attn_time(64 * 1024, 8, 2)
print("Check attention time: ")
for ctx_len in [1, 2, 4, 32, 64, 96]:
    # a = tm.get_attn_time(ctx_len * 1024, 8, 2)
    a = tm.get_attn_time(ctx_len * 1024, 8, 4)
    print(f"ctx_len: {ctx_len}, attn: {a}ms")

# %%
best_solution = 1e15
from rich.console import Console
from rich.table import Table

console = Console()
table = Table(title="WLB-LLM Solution Latency")

# Add columns for cp values
table.add_column("tp/cp", justify="right", style="cyan", no_wrap=True)
for cp in [8, 4, 2, 1]:
    table.add_column(str(cp), justify="right")

# Prepare a matrix to store results
results = {tp: {cp: float('inf') for cp in [8, 4, 2, 1]} for tp in [8, 4, 2, 1]}

print("WLB-LLM Solution:")
for tp in [8, 4, 2, 1]:
    for cp in [8, 4, 2, 1]:
        if tp * cp > num_total_devices:
            continue
        parallel_plan = (tp, cp)
        num_workers = num_total_devices // (tp * cp)
        assert num_workers * tp * cp == num_total_devices, "num_workers * tp * cp != num_total_devices"
        
        solver = WlbLlmSolver()
        solution = solver.solve(
            batch, 
            max_length=sum(batch),
            num_workers=num_workers,
            parallel_plan=parallel_plan,
        )
        lat_max = solution.lat_max
        results[tp][cp] = lat_max
        best_solution = min(best_solution, lat_max)

# Populate the table
for tp in [8, 4, 2, 1]:
    row = [str(tp)]
    for cp in [8, 4, 2, 1]:
        value = results[tp][cp]
        if value == best_solution:
            row.append(f"[bold spring_green2]{value}[/bold spring_green2]")
        else:
            row.append(str(value) if value != float('inf') else 'inf')
    table.add_row(*row)

table.caption = f"Best Solution: {best_solution} ms\nPlan = {parallel_plan}"
console.print(table)
print(f"Best solution: {best_solution}")

# %%

solver = AttnServerSolver()
solution = solver.solve(
    batch, 
    num_workers=num_total_devices, 
    num_total_devices=num_total_devices,
)
lat_max = solution.lat_max
solution.print_solution()


# %%
