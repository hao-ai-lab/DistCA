#!/usr/bin/env python3
# attn_solver_fast.py  –  compact CP-SAT version
from ortools.sat.python import cp_model
import numpy as np
import rich, time
from rich.table import Table
from rich.console import Console
from rich.panel import Panel

import timemodule as tm   # ← your own timing helpers
INF = tm.INF
verbose = True
K = 1024

def main(
    batch="[64*K]*8+[32*K]*8",
    num_total_devices=64,
    mlp_tp=4,
    mlp_cp=8,
    max_time_in_seconds=360,
    sample_name="test",
    sample_id="0",
    num_workers=16,
    sweep_all_mlp_plans=False,
):
    """
    Main function for attention server optimization.
    
    Args:
        batch: Either a string to be evaluated (e.g., "[64*K]*8+[32*K]*8") or a list of integers
        num_total_devices: Total number of devices available
        mlp_tp: MLP tensor parallelism
        mlp_cp: MLP context parallelism
        max_time_in_seconds: Maximum solving time
        sample_name: Name for the sample
        sample_id: ID for the sample
        num_workers: Number of workers
        sweep_all_mlp_plans: Whether to sweep all MLP plan combinations
    """
    # Handle batch parameter - can be string or list
    if isinstance(batch, str):
        batch = eval(batch)
    elif not isinstance(batch, list):
        raise ValueError("batch must be either a string or a list")

    # ------------------------------------------------------------------
    # Parallel-plan catalogue & resources
    parallel_plan = [(tp, cp) for tp in (1, 2, 4, 8) for cp in (1, 2, 4, 8)] + [(0, 0)]
    resource      = [tp * cp for tp, cp in parallel_plan]
    # ------------------------------------------------------------------
    # 1)  Pre-compute per-plan / per-doc latency (µs) and show table
    # ------------------------------------------------------------------
    num_plans, num_docs = len(parallel_plan), len(batch)
    latency = {}              # (j,k) -> µs
    for j, (tp, cp) in enumerate(parallel_plan):
        for k, doc_len in enumerate(batch):
            if (tp, cp) == (0, 0):
                latency[j, k] = INF
                continue
            lat = tm.get_attn_time(tp, cp, doc_len)
            latency[j, k] = int(lat * 1000)        # ms -> µs

    if verbose:
        console = Console()
        tbl = Table(title="[AttnServer] Attention latency table")
        tbl.add_column("Doc ID", style="cyan", justify="center")
        tbl.add_column("Len", style="magenta", justify="center")
        for tp in (1, 2, 4, 8):
            for cp in (1, 2, 4, 8):
                tbl.add_column(f"{tp}/{cp}", style="green", justify="center")
        for k in range(num_docs):
            row = [str(k), str(batch[k])]
            for tp in (1, 2, 4, 8):
                for cp in (1, 2, 4, 8):
                    j = parallel_plan.index((tp, cp))
                    row.append(f"{latency[j,k]:>8}")
            tbl.add_row(*row)
        console.print(tbl)

    # ------------------------------------------------------------------
    # 2)  Optimisation model  (compact CP-SAT)
    # ------------------------------------------------------------------
    def model_size(mdl):
        """Return (#vars, #cons) on any OR-Tools version."""
        try:
            return mdl.NumVariables(), mdl.NumConstraints()
        except AttributeError:
            p = mdl.Proto()     # older API
            return len(p.variables), len(p.constraints)

    model = cp_model.CpModel()
    P, D = num_plans, num_docs
    MAX_LAT = max(latency.values())
    MAX_DEV = max(resource)

    # 2-a) integer plan index for each worker
    plan = [model.NewIntVar(0, P-1, f"plan_{i}") for i in range(num_workers)]

    # 2-b) assignment y[k,i]  (doc k -> worker i)
    y = {(k,i): model.NewBoolVar(f"y_{k}_{i}") for k in range(D) for i in range(num_workers)}
    for k in range(D):
        model.Add(sum(y[k,i] for i in range(num_workers)) == 1)

    # 2-c) device budget  via Element(plan, resource)
    dev_i = [model.NewIntVar(0, MAX_DEV, f"dev_{i}") for i in range(num_workers)]
    for i in range(num_workers):
        model.AddElement(plan[i], resource, dev_i[i])
    model.Add(sum(dev_i) <= num_total_devices)

    # 2-d) latency lookup with Element, optional via y
    lat_vec = [[latency[j,k] for j in range(P)] for k in range(D)]
    lat_used = {}
    for i in range(num_workers):
        for k in range(D):
            lsel = model.NewIntVar(0, MAX_LAT, f"lSel_{i}_{k}")
            model.AddElement(plan[i], lat_vec[k], lsel)
            luse = model.NewIntVar(0, MAX_LAT, f"lUse_{i}_{k}")
            model.Add(luse == lsel).OnlyEnforceIf(y[k,i])
            model.Add(luse == 0   ).OnlyEnforceIf(y[k,i].Not())
            lat_used[i,k] = luse

    # 2-e) per-worker & batch latency
    lat_worker = [model.NewIntVar(0, MAX_LAT*D, f"latW_{i}") for i in range(num_workers)]
    lat_max    =  model.NewIntVar(0, MAX_LAT*D, "lat_max")
    for i in range(num_workers):
        model.Add(lat_worker[i] == sum(lat_used[i,k] for k in range(D)))
        model.Add(lat_worker[i] <= lat_max)

    model.Minimize(lat_max)

    # ------------------------------------------------------------------
    # 3)  Solve
    # ------------------------------------------------------------------
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = max_time_in_seconds
    t0 = time.time(); status = solver.Solve(model); t1 = time.time()

    n_vars, n_cons = model_size(model)
    rich.print(f"[green]Solved with {n_vars} vars / {n_cons} cons in {t1-t0:.2f}s "
            f"(status: {solver.StatusName(status)})[/]")

    # ------------------------------------------------------------------
    # 4)  Pretty-print solution (worker plans, doc assignment) + Extract structured data
    # ------------------------------------------------------------------
    console = Console()

    # Worker -> plan table
    w_tbl = Table(title="[AttnServer] Worker Plan Assignment")
    w_tbl.add_column("Worker", style="cyan", justify="right")
    w_tbl.add_column("Plan (tp,cp)", style="green")
    w_tbl.add_column("Latency (µs)", style="red", justify="right")

    xs = {}                      # plan chosen per worker
    min_worker_lat = INF
    worker_plan_data = []        # For structured data

    for i in range(num_workers):
        pidx = solver.Value(plan[i])
        tp, cp = parallel_plan[pidx]
        xs[i] = (tp, cp)
        w_lat = solver.Value(lat_worker[i])
        if (tp, cp) != (0, 0):
            min_worker_lat = min(min_worker_lat, w_lat)
            w_tbl.add_row(str(i), f"{tp},{cp}", f"{w_lat:.0f}")
            
            # Store structured data
            worker_plan_data.append({
                "worker_id": i,
                "tp": tp,
                "cp": cp,
                "latency_us": w_lat
            })

    console.print(w_tbl)

    # Doc -> worker table
    d_tbl = Table(title="Document to Worker Assignment")
    d_tbl.add_column("Doc", style="cyan", justify="right")
    d_tbl.add_column("Len", style="magenta", justify="right")
    d_tbl.add_column("Worker", style="green", justify="right")
    d_tbl.add_column("Attn time (µs)", style="red", justify="right")

    doc_assignment_data = []     # For structured data

    for k in range(D):
        for i in range(num_workers):
            if solver.Value(y[k,i]):
                tp, cp = xs[i]
                d_lat = tm.get_attn_time(tp, cp, batch[k]) * 1000
                d_tbl.add_row(str(k), str(batch[k]), str(i), f"{d_lat:.0f}")
                
                # Store structured data
                doc_assignment_data.append({
                    "doc_id": k,
                    "doc_len": batch[k],
                    "worker_id": i,
                    "attn_time_us": d_lat
                })

    console.print(d_tbl)

    batch_attn_time = min_worker_lat

    # ------------------------------------------------------------------
    # 5)  MLP + all-reduce timing  (unchanged)
    # ------------------------------------------------------------------
    if sweep_all_mlp_plans:
        sweep_results = {}
        for mlp_tp_sweep in (1, 2, 4, 8):
            for mlp_cp_sweep in (1, 2, 4, 8):
                mlp_dp = num_total_devices // (mlp_tp_sweep * mlp_cp_sweep)
                token_per_dp_shard = sum(batch) // mlp_dp
                batch_mlp_time, _   = tm.get_mlp_time(mlp_tp_sweep, mlp_cp_sweep, token_per_dp_shard)
                batch_mlp_time     *= 1000          # ms → µs
                batch_allreduce_time = (tm.get_allreduce_time_with_config(
                    mlp_tp_sweep, token_per_dp_shard, tm.hidden_size) * 2 * 1000)
                batch_total_time = batch_attn_time + batch_mlp_time + batch_allreduce_time
                sweep_results[(mlp_tp_sweep, mlp_cp_sweep)] = dict(
                    batch_total_time=batch_total_time,
                    batch_attn_time=batch_attn_time,
                    batch_mlp_time=batch_mlp_time,
                    batch_allreduce_time=batch_allreduce_time,
                    solver_status=solver.StatusName(status),
                    solved_time_s=t1-t0,
                    num_vars=n_vars,
                    num_cons=n_cons,
                    worker_plans=worker_plan_data,
                    doc_assignments=doc_assignment_data,
                    mlp_tp=mlp_tp_sweep,
                    mlp_cp=mlp_cp_sweep,
                )
    
    # Always compute the main result with the specified mlp_tp/mlp_cp
    mlp_dp = num_total_devices // (mlp_tp * mlp_cp)
    token_per_dp_shard = sum(batch) // mlp_dp
    batch_mlp_time, _   = tm.get_mlp_time(mlp_tp, mlp_cp, token_per_dp_shard)
    batch_mlp_time     *= 1000          # ms → µs
    batch_allreduce_time = (tm.get_allreduce_time_with_config(
        mlp_tp, token_per_dp_shard, tm.hidden_size) * 2 * 1000)

    batch_total_time = batch_attn_time + batch_mlp_time + batch_allreduce_time

    panel = Panel(
        f"[bold white]Batch Linear time:    {batch_mlp_time:>10.0f} µs\n"
        f"Batch attention time: {batch_attn_time:>10.0f} µs\n"
        f"Batch allreduce time: {batch_allreduce_time:>10.0f} µs\n"
        f"--------------------------------------\n"
        f"Batch total time:     {batch_total_time:>10.0f} µs[/bold white]",
        title="Batch Timing Information",
        border_style="white"
    )
    rich.print(panel)

    # Create timing breakdown
    timing_breakdown = {
        "batch_linear_time_us": batch_mlp_time,
        "batch_attention_time_us": batch_attn_time,
        "batch_allreduce_time_us": batch_allreduce_time,
        "batch_total_time_us": batch_total_time,
        "solver_status": solver.StatusName(status),
        "solver_status_code": (status),
        "solved_time_s": t1-t0,
        "num_vars": n_vars,
        "num_cons": n_cons,
    }

    if sweep_all_mlp_plans:
        return {
            'sweep_results': sweep_results,
            'main_result': {
                'batch_total_time': batch_total_time,
                'batch_attn_time': batch_attn_time,
                'batch_mlp_time': batch_mlp_time,
                'batch_allreduce_time': batch_allreduce_time,
                'worker_plans': worker_plan_data,
                'doc_assignments': doc_assignment_data,
                'timing_breakdown': timing_breakdown,
            },
            "solver_status": solver.StatusName(status),
            "solver_status_code": (status),
            "solved_time_s": t1-t0,
            "num_vars": n_vars,
            "num_cons": n_cons,
        }

    return {
        'batch_total_time': batch_total_time,
        'batch_attn_time': batch_attn_time,
        'batch_mlp_time': batch_mlp_time,
        'batch_allreduce_time': batch_allreduce_time,
        'worker_plans': worker_plan_data,
        'doc_assignments': doc_assignment_data,
        'timing_breakdown': timing_breakdown,
        "solver_status": solver.StatusName(status),
        "solved_time_s": t1-t0,
        "num_vars": n_vars,
        "num_cons": n_cons,
    }


def main_with_args():
    """Entry point that uses command line arguments (for backward compatibility)"""
    import argparse
    parser = argparse.ArgumentParser()
    # batch
    parser.add_argument("--batch", type=str, default="[64*K]*8+[32*K]*8")
    # num_total_devices
    parser.add_argument("--num_total_devices", type=int, default=64)
    # mlp_tp
    parser.add_argument("--mlp_tp", type=int, default=4)
    # mlp_cp
    parser.add_argument("--mlp_cp", type=int, default=8)
    # max_time_in_seconds
    parser.add_argument("--max_time_in_seconds", type=int, default=10)
    # sample_name
    parser.add_argument("--sample_name", type=str, default="test")
    # sample_id
    parser.add_argument("--sample_id", type=str, default="0")
    # num_workers
    parser.add_argument("--num_workers", type=int, default=16)
    # sweep_all_mlp_plans
    parser.add_argument("--sweep_all_mlp_plans", action="store_true", default=False)
    args = parser.parse_args()
    print(args)

    return main(
        batch=args.batch,
        num_total_devices=args.num_total_devices,
        mlp_tp=args.mlp_tp,
        mlp_cp=args.mlp_cp,
        max_time_in_seconds=args.max_time_in_seconds,
        sample_name=args.sample_name,
        sample_id=args.sample_id,
        num_workers=args.num_workers,
        sweep_all_mlp_plans=args.sweep_all_mlp_plans,
    )


if __name__ == "__main__":
    main_with_args()