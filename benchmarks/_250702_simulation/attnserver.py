#!/usr/bin/env python3
from ortools.sat.python import cp_model
import numpy as np
import rich, time
from rich.table import Table
from rich.console import Console
from rich.panel import Panel

import timemodule as tm   # ← your own timing helpers
INF = tm.INF

K = 1024

def attnserver_solver(
    batch,
    num_workers,
    num_total_devices,
    mlp_tp,
    mlp_cp,
    verbose = True,
    max_time_in_seconds = 360,
):

    # ------------------------------------------------------------------
    # Example batch   (keep whatever batches you like)
    # batch = [64 * K] * 8 + [32 * K] * 8
    # ------------------------------------------------------------------

    # Parallel-plan catalogue & resources
    parallel_plan = [(tp, cp) for tp in (1, 2, 4, 8) for cp in (1, 2, 4, 8)] + [(0, 0)]
    resource      = [tp * cp for tp, cp in parallel_plan]

    # Hardware / MLP parameters
    mlp_dp            = num_total_devices // (mlp_tp * mlp_cp)

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
    # 4)  Pretty-print solution (worker plans, doc assignment)
    # ------------------------------------------------------------------
    console = Console()

    # Worker -> plan table
    w_tbl = Table(title="[AttnServer] Worker Plan Assignment")
    w_tbl.add_column("Worker", style="cyan", justify="right")
    w_tbl.add_column("Plan (tp,cp)", style="green")
    w_tbl.add_column("Latency (µs)", style="red", justify="right")

    xs = {}                      # plan chosen per worker
    min_worker_lat = INF
    for i in range(num_workers):
        pidx = solver.Value(plan[i])
        tp, cp = parallel_plan[pidx]
        xs[i] = (tp, cp)
        w_lat = solver.Value(lat_worker[i])
        if (tp, cp) != (0, 0):
            min_worker_lat = min(min_worker_lat, w_lat)
            w_tbl.add_row(str(i), f"{tp},{cp}", f"{w_lat:.0f}")

    console.print(w_tbl)

    # Doc -> worker table
    d_tbl = Table(title="Document to Worker Assignment")
    d_tbl.add_column("Doc", style="cyan", justify="right")
    d_tbl.add_column("Len", style="magenta", justify="right")
    d_tbl.add_column("Worker", style="green", justify="right")
    d_tbl.add_column("Attn time (µs)", style="red", justify="right")

    for k in range(D):
        for i in range(num_workers):
            if solver.Value(y[k,i]):
                tp, cp = xs[i]
                d_lat = tm.get_attn_time(tp, cp, batch[k]) * 1000
                d_tbl.add_row(str(k), str(batch[k]), str(i), f"{d_lat:.0f}")
    console.print(d_tbl)

    batch_attn_time = min_worker_lat

    # ------------------------------------------------------------------
    # 5)  MLP + all-reduce timing  (unchanged)
    # ------------------------------------------------------------------
    token_per_dp_shard = sum(batch) // mlp_dp
    batch_mlp_time     = tm.get_mlp_time(mlp_tp, mlp_cp, token_per_dp_shard)
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

    result = dict(
        batch_total_time=batch_total_time,
        batch_attn_time=batch_attn_time,
        batch_mlp_time=batch_mlp_time,
        batch_allreduce_time=batch_allreduce_time,
        
    )
    return result

if __name__ == "__main__":
    batch = [64 * K] * 4 + [32 * K] * 8
    num_workers = 8
    num_total_devices = 64
    mlp_tp = 4
    mlp_cp = 8
    result = attnserver_solver(
        batch, num_workers, num_total_devices, mlp_tp, mlp_cp,
        max_time_in_seconds=360,
    )
    rich.print(result)