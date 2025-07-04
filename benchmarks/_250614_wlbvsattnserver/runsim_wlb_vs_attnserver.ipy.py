# %%
import d2.timemodule as tm
from d2.simulator.optimizers.attnserver import AttnServerSolver
from d2.simulator.optimizers.wlbllm import WlbLlmSolver


# %%
K = 1024
# batch = [64 * K, 64 * K]
batch = [2030, 22, 5521, 2260, 4800, 5912, 4524, 4160, 2253, 2958, 3119, 3473, 1408, 579, 2887, 1793, 4614, 1369, 4687, 707, 5225, 816, 419]
# batch = [64] * 64
# batch =  [i * K for i in batch]

num_workers = 4
num_total_devices = 16

print("Qwen3-235B")

# %%
best_latency = 1e15
best_solution = None
best_plan = None
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
        if lat_max < best_latency:
            best_latency = lat_max
            best_solution = solution
            best_plan = parallel_plan

# Populate the table
for tp in [8, 4, 2, 1]:
    row = [str(tp)]
    for cp in [8, 4, 2, 1]:
        value = results[tp][cp]
        if value == best_latency:
            row.append(f"[bold spring_green2]{value}[/bold spring_green2]")
        else:
            row.append(f"{value:.2f}" if value < tm.INF else '-')
    table.add_row(*row)

table.caption = f"Best Latency: {best_latency} ms\nPlan = {best_plan}"
console.print(table)
print(f"Best latency: {best_latency}")
best_solution.print_solution()

# %%

solver = AttnServerSolver()
solution = solver.solve(
    batch, 
    num_workers=num_total_devices, 
    num_total_devices=num_total_devices,
    timeout=30,
)
lat_max = solution.lat_max
solution.print_solution()

# 806186
# 804746

# %%
len(batch)
# %%
