# %% 
# GPT-o3 Optimized solver

from ortools.sat.python import cp_model
import numpy as np
from models import Llama8B

raise Exception("This is buggy!! Likely something wrong with the latency calculation. Need to fix.")

# %% 
K = 1024

# ----- constants you already have -----
batch             = np.array([32] + [1] * 32) * K          # token counts
num_total_devices = 32
parallel_plan     = [
    (tp, cp)
    for tp in (1, 2, 4, 8)
    for cp in (1, 2, 4, 8)
    if tp * cp <= num_total_devices and tp <= 8
]
P                 = len(parallel_plan)
resource_per_pln  = [tp*cp for tp, cp in parallel_plan]
BIG_M             = 10**9                               # instead of np.inf

latency = np.zeros((len(batch), P), dtype=int)
for p, (tp, cp) in enumerate(parallel_plan):
    this = Llama8B(tp=tp, cp=cp)
    for q, tok in enumerate(batch):
        latency[q, p] = int(this._attn(tok))
# latency = latency # us 


# %% 
# ---------- CP-SAT model ----------
model = cp_model.CpModel()

W              = min(len(batch), num_total_devices)     # upper bound
assign         = [[model.NewBoolVar(f"a_{k}_{i}") for i in range(W)]
                  for k in range(len(batch))]
plan           = [model.NewIntVar(0, P-1, f"plan_{i}")  for i in range(W)]
worker_active  = [model.NewBoolVar(f"act_{i}")          for i in range(W)]
lat_worker     = []

# %% 
# each datum goes to exactly one worker
for k in range(len(batch)):
    model.Add(sum(assign[k][i] for i in range(W)) == 1)

# %% 
# if any datum is assigned, the worker is active
for i in range(W):
    for k in range(len(batch)):
        # assign[k][i] ⇒ worker_active[i]
        model.AddImplication(assign[k][i], worker_active[i])
    # no datum ⇒ inactive
    model.Add(sum(assign[k][i] for k in range(len(batch))) >= 1).\
         OnlyEnforceIf(worker_active[i])
    model.Add(sum(assign[k][i] for k in range(len(batch))) == 0).\
         OnlyEnforceIf(worker_active[i].Not())

# %% 
# resource budget
res_i = []
for i in range(W):
    r = model.NewIntVar(0, max(resource_per_pln), f"res_{i}")
    model.AddElement(plan[i], resource_per_pln, r)
    res_i.append(r)
model.Add(sum(res_i) <= num_total_devices)

# %%
# ----------
# latency per worker and objective
# ----------
max_lat = model.NewIntVar(0, BIG_M, "max_latency")
lat_worker = []

n_data = len(batch)          # just to shorten the lines

for i in range(W):
    lat_k_i = []             # latency contribution of each datum k on worker i

    for k in range(n_data):
        # 1️⃣  Pick the latency that corresponds to this worker’s plan
        #     temp = latency[k][plan[i]]
        temp = model.NewIntVar(0, BIG_M, f"tmp_{k}_{i}")
        model.AddElement(plan[i], latency[k].tolist(), temp)

        # 2️⃣  Copy it into lk only if datum k is assigned to worker i;
        #     otherwise set lk to 0
        lk = model.NewIntVar(0, BIG_M, f"l_{k}_{i}")
        model.Add(lk == temp).OnlyEnforceIf(assign[k][i])
        model.Add(lk == 0).OnlyEnforceIf(assign[k][i].Not())
        lat_k_i.append(lk)

    # 3️⃣  Total latency for worker i
    lat_i = model.NewIntVar(0, BIG_M, f"lat_{i}")
    model.Add(lat_i == sum(lat_k_i))
    model.Add(max_lat >= lat_i)          # keep track of the maximum
    lat_worker.append(lat_i)

# %% 
model.Minimize(max_lat)

# %% 
# ---------- solve ----------
solver = cp_model.CpSolver()
solver.parameters.max_time_in_seconds = 30            # optional
import time
start_time = time.time()
status = solver.Solve(model)
end_time = time.time()
print(f"Solver time: {end_time - start_time:.2f}s")


# %% 
assert status == cp_model.OPTIMAL, solver.StatusName(status)

# %% 
# ---------- extract ----------
results = []
for i in range(W):
    if solver.Value(worker_active[i]):
        pln_idx       = solver.Value(plan[i])
        tp, cp        = parallel_plan[pln_idx]
        assigned_idxs = [k for k in range(len(batch))
                         if solver.Value(assign[k][i])]
        results.append((batch[assigned_idxs], tp, cp))

# %% 
print("balanced plan:", results)
print("max latency :", solver.Value(max_lat), "us")
# %%
llm = Llama8B(tp=8, cp=2)
mlp_time = sum([llm._mlp(batch[i]) for i in range(len(batch))]) / 1000
mlp_time
# %%
# %%

