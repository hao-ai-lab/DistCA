from d2.simulator.optimizers.attnserver import AttnServerSolver
from d2.simulator.optimizers.wlbllm import WlbLlmSolver

import numpy as np
from rich.console import Console
from rich.table import Table

K = 1024
M = 1024 ** 2

def run_simulation(
    batch, 
    num_workers: int = 1,
    num_total_devices: int = 8,
):

    best_latency = 1e15
    best_solution = None
    best_plan = None

    console = Console()
    table = Table(title="WLB-LLM Solution Latency")

    # Add columns for cp values
    table.add_column("tp/cp", justify="right", style="cyan", no_wrap=True)
    for cp in [8, 4, 2, 1]:
        table.add_column(str(cp), justify="right")

    # Prepare a matrix to store results
    results = {tp: {cp: float('inf') for cp in [8, 4, 2, 1]} for tp in [8, 4, 2, 1]}

    print("WLB-LLM Solution:")
    for tp in [1, 2, 4, 8]:
        for cp in [1, 2, 4, 8]:
            if tp * cp > num_total_devices:
                continue
            parallel_plan = (tp, cp)
            num_workers = num_total_devices // (tp * cp)
            assert num_workers * tp * cp == num_total_devices, f"num_workers * tp * cp != num_total_devices: {num_workers} * {tp} * {cp} != {num_total_devices}"
            
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
                row.append(f"[bold spring_green2]{value * 1000:.2f}[/bold spring_green2]")
            else:
                row.append(f"{value * 1000:.2f}" if value != float('inf') else 'inf')
        table.add_row(*row)

    best_num_workers = num_total_devices // (best_plan[0] * best_plan[1])
    table.caption = f"Best Latency: {best_latency * 1000:.2f} us\nPlan = {best_plan} x DP = {best_num_workers}"
    console.print(table)
    print(f"Best latency: {best_latency * 1000:.2f} us")
    best_solution.print_solution()
    wlb_llm_solution = best_solution

    print("------------------------------")
    solver = AttnServerSolver()
    attn_server_solution = solver.solve(
        batch, 
        num_workers=num_workers, 
        num_total_devices=num_total_devices,
        timeout=30,
    )
    lat_max = attn_server_solution.lat_max
    attn_server_solution.print_solution()
    did_time_out = attn_server_solution.did_time_out

    print("------------------------------")

    print("Sanity Checking:")
    print(f"- Batch: {batch}")
    print(f"- Num Workers: {num_workers}")
    print(f"- Num Total Devices: {num_total_devices}")

    print(f"- WLB-LLM Parallel Plan:    {wlb_llm_solution.parallel_plan}")
    print(f"- AttnServer Parallel Plan: {attn_server_solution.get_parallel_plan()}")
    print(f"- WLB-LLM Batch Assignment:    {wlb_llm_solution.get_batch_assignment()}")
    print(f"- AttnServer Batch Assignment: {attn_server_solution.get_batch_assignment()}")

    diff = wlb_llm_solution.lat_max - attn_server_solution.lat_max
    print(f"- WLB-LLM:    {wlb_llm_solution.lat_max:>8.2f} ms")
    print(f"- AttnServer: {attn_server_solution.lat_max:>8.2f} ms")
    print(f"- Diff:       {diff:>8.2f} ms")

    return diff, did_time_out