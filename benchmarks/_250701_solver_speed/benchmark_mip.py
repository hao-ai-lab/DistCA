#!/usr/bin/env python3
# benchmark_mip.py
import time, argparse, itertools, collections
from dataclasses import dataclass
from typing import List, Tuple
from ortools.sat.python import cp_model
from ortools.linear_solver import pywraplp
from rich import print, box
from rich.table import Table

# ---------- Problem-specific constants ----------
K = 1024
parallel_plan = [(tp, cp)
                 for tp in (1, 2, 4, 8)
                 for cp in (1, 2, 4, 8)] + [(0, 0)]
resource = [tp * cp for tp, cp in parallel_plan]
num_workers, num_total_devices = 8, 64

# dummy timing functions -------------------------
import math, random
def attn_time(tp, cp, doc_len):            # use your real tm.get_attn_time
    return (doc_len / (tp+1)) * math.log(cp+1) + 5
def mlp_time(tokens):                       # stand-in for tm.get_mlp_time
    return 2.3 * tokens
def allreduce_time(tokens):
    return 0.8 * tokens
# ------------------------------------------------

def model_size(mdl):
    """Return (n_vars, n_cons) regardless of OR-Tools version."""
    try:
        return mdl.NumVariables(), mdl.NumConstraints()
    except AttributeError:
        proto = mdl.Proto()          # older API
        return len(proto.variables), len(proto.constraints)


# ---------- Model builders ----------
def build_cp_sat(batch: List[int]):
    """Return (model, solver, lat_max Var) built with CP-SAT."""
    mdl = cp_model.CpModel()
    n_plans, n_docs = len(parallel_plan), len(batch)

    x = {(i,j): mdl.NewBoolVar(f"x_{i}_{j}")      # worker i uses plan j
         for i in range(num_workers) for j in range(n_plans)}
    y = {(k,i): mdl.NewBoolVar(f"y_{k}_{i}")      # doc k → worker i
         for k in range(n_docs) for i in range(num_workers)}
    z = {(i,j,k): mdl.NewBoolVar(f"z_{i}_{j}_{k}")   # x·y linearisation
         for i in range(num_workers) for j in range(n_plans) for k in range(n_docs)}

    # 1. each worker one plan
    for i in range(num_workers):
        mdl.Add(sum(x[i,j] for j in range(n_plans)) == 1)

    # 2. each document one worker
    for k in range(n_docs):
        mdl.Add(sum(y[k,i] for i in range(num_workers)) == 1)

    # 3. z linearisation
    for (i,j,k), zvar in z.items():
        mdl.AddBoolAnd([x[i,j], y[k,i]]).OnlyEnforceIf(zvar)
        mdl.AddBoolOr([x[i,j].Not(), y[k,i].Not(), zvar])

    # Latencies
    lat = {}
    for j,(tp,cp) in enumerate(parallel_plan):
        for k,doc in enumerate(batch):
            lat[j,k] = int(attn_time(tp,cp,doc)*1000) if (tp,cp)!=(0,0) else 10**9

    lat_worker = [mdl.NewIntVar(0, 10**12, f"lat_{i}") for i in range(num_workers)]
    lat_max    =  mdl.NewIntVar(0, 10**12, "lat_max")
    for i in range(num_workers):
        mdl.Add(lat_worker[i] ==
                sum(lat[j,k]*z[i,j,k] for j in range(n_plans) for k in range(n_docs)))
        mdl.Add(lat_worker[i] <= lat_max)

    # 4. device budget
    mdl.Add(sum(resource[j]*x[i,j] for i in range(num_workers) for j in range(n_plans))
            <= num_total_devices)

    mdl.Minimize(lat_max)
    solver = cp_model.CpSolver()
    n_vars, n_cons = model_size(mdl)
    return mdl, solver, lat_max, n_vars, n_cons



def randint(min, max):
    """Return a random integer in the range [min, max]."""
    return random.randint(min, max)

def run_trial(N: int, solver_kind: str):
    batch = []
    for i in range(N):
        doc_len = randint(1, 64) * K
        batch.append(doc_len)
    if solver_kind.lower() == "cp-sat":
        mdl, solver, lat_max, n_vars, n_cons = build_cp_sat(batch)
        solver.parameters.max_time_in_seconds = 360
        t0 = time.perf_counter()
        status = solver.Solve(mdl)
        dur = time.perf_counter() - t0
        return status, solver.ObjectiveValue(), dur, n_vars, n_cons
    else:
        raise NotImplementedError(f"Solver '{solver_kind}' is not implemented in this benchmark.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sizes", type=int, nargs="+", default=[8,16,24,32,36,40,64],
                        help="Batch sizes N to test")
    parser.add_argument("--solvers", nargs="+",
                        default=["cp-sat","CBC_MIXED_INTEGER_PROGRAMMING"],
                        help="Solver IDs (cp-sat, CBC_MIXED_INTEGER_PROGRAMMING, SCIP, GUROBI, ...)")
    args = parser.parse_args()

    table = Table(title="Solver benchmark", box=box.SIMPLE_HEAVY)
    table.add_column("Solver")
    table.add_column("N (docs)")
    table.add_column("Status")
    table.add_column("Opt value (us)")
    table.add_column("Time (s)")
    table.add_column("Num Vars")
    table.add_column("Num Cons")

    for sol in args.solvers:
        for N in args.sizes:
            print(f"Running {sol} for N={N} ...")
            stat,val,tm, n_vars, n_cons = run_trial(N, sol)
            print(f"  Status: {stat}, Opt value: {val:.0f}, Time: {tm:.2f}s, Num Vars: {n_vars}, Num Cons: {n_cons}")
            table.add_row(sol, str(N), str(stat), f"{val:.0f}", f"{tm:.2f}", 
                          f"{n_vars}", f"{n_cons}")
    print(table)


if __name__ == "__main__":
    main()