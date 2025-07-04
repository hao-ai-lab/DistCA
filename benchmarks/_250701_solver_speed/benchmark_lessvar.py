#!/usr/bin/env python3
# cp_sat_element.py  —  fewer variables, no z(i,j,k)
import math, time, random, argparse
from typing import List
from ortools.sat.python import cp_model
from rich import print, box
from rich.table import Table

# ---------------- constants & helpers ----------------
K = 1024
tps, cps = (1, 2, 4, 8), (1, 2, 4, 8)
plans    = [(tp, cp) for tp in tps for cp in cps] + [(0, 0)]   # index 0..31
P        = len(plans)
resource = [tp * cp for tp, cp in plans]

def attn_time(tp, cp, doc_len):
    return (doc_len / (tp + 1)) * math.log(cp + 1) + 5          # dummy (ms)

MAX_LAT  = int(attn_time(1, 8, 64*K) * 1000) + 1               # µs upper-bound
MAX_DEV  = max(resource)

# ---------------- model builder ----------------
def solve(batch: List[int], W: int, dev_budget: int, time_limit=360):
    D = len(batch)

    model = cp_model.CpModel()

    # 1. worker plan (integer 0..P-1)
    plan = [model.NewIntVar(0, P-1, f"plan_{i}") for i in range(W)]

    # 2. doc → worker assignment
    y = {(k,i): model.NewBoolVar(f"y_{k}_{i}") for k in range(D) for i in range(W)}
    for k in range(D):
        model.Add(sum(y[k,i] for i in range(W)) == 1)

    # 3. device budget  (use AddElement)
    dev_i = [model.NewIntVar(0, MAX_DEV, f"dev_{i}") for i in range(W)]
    for i in range(W):
        model.AddElement(plan[i], resource, dev_i[i])
    model.Add(sum(dev_i) <= dev_budget)

    # 4. latency lookup   lat_sel[i,k] = latency(plan[i], doc_k)
    lat_vec = [[int(attn_time(*plans[j], dlen) * 1000)    # µs
                for j in range(P)] for dlen in batch]

    lat_sel  = {}   # raw lookup value (no y)
    lat_used = {}   # == lat_sel if y==1 else 0
    for i in range(W):
        for k in range(D):
            lat_sel[i,k]  = model.NewIntVar(0, MAX_LAT, f"lSel_{i}_{k}")
            model.AddElement(plan[i], lat_vec[k], lat_sel[i,k])

            lat_used[i,k] = model.NewIntVar(0, MAX_LAT, f"lUse_{i}_{k}")
            # if y==1  -> lat_used == lat_sel
            model.Add(lat_used[i,k] == lat_sel[i,k]).OnlyEnforceIf(y[k,i])
            # if y==0  -> lat_used == 0
            model.Add(lat_used[i,k] == 0).OnlyEnforceIf(y[k,i].Not())

    # 5. worker & batch latency
    lat_w   = [model.NewIntVar(0, MAX_LAT * D, f"latW_{i}") for i in range(W)]
    lat_max = model.NewIntVar(0, MAX_LAT * D, "lat_max")
    for i in range(W):
        model.Add(lat_w[i] == sum(lat_used[i,k] for k in range(D)))
        model.Add(lat_w[i] <= lat_max)

    model.Minimize(lat_max)

    # ----- solve
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = time_limit
    status = solver.Solve(model)
    # n_vars = model.NumVariables()
    # n_cons = model.NumConstraints()
    return status, solver.ObjectiveValue(), solver.WallTime(), "-", "-"

# ---------------- CLI wrapper ----------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", type=int, nargs="+", default=[16, 32, 64, 128, 256])
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--devices", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = random.Random(args.seed)

    tbl = Table(title="CP-SAT (element) benchmark", box=box.SIMPLE_HEAD)
    tbl.add_column("Docs"); tbl.add_column("#vars"); tbl.add_column("#cons")
    tbl.add_column("Status"); tbl.add_column("Opt µs"); tbl.add_column("Time s")

    for N in args.sizes:
        batch = [rng.randint(512, 64 * K) for _ in range(N)]
        st, val, tm, nv, nc = solve(batch, args.workers, args.devices)
        print(f"Docs: {N}, Workers: {args.workers}, Devices: {args.devices}, "
                f"Opt µs: {val:.0f}, Time s: {tm:.2f}")
        tbl.add_row(str(N), str(nv), str(nc),
                    str(st), f"{val:.0f}", f"{tm:.2f}")
    print(tbl)

if __name__ == "__main__":
    main()