# %%
# ------------------------------------------------------------
# 2. CP-SAT  (replaces the old PuLP MILP section)
# ------------------------------------------------------------
import numpy as np
from ortools.sat.python import cp_model
from models import Llama8B


# %%

K = 1024
batch = np.array([32] + [1] * 32) * K
num_total_devices = 32
num_worker_max = 32
num_data = len(batch)

_infinity = 100000000 # 1e9

# %%
parallel_plan = [
    (tp, cp)
    for tp in (1, 2, 4, 8)
    for cp in (1, 2, 4, 8)
    if tp * cp <= num_total_devices and tp <= 8
]

resource = [tp*cp for tp, cp in parallel_plan]
num_sols = len(parallel_plan)

# %%

latency = np.zeros((len(batch), num_sols), dtype=int)
for p, (tp, cp) in enumerate(parallel_plan):
    this = Llama8B(tp=tp, cp=cp)
    for q, tok in enumerate(batch):
        latency[q, p] = int(this._attn(tok))
# latency = latency # us 


# %%
# --- helper: convert latency to integers (µs) ----------------
SCALE = 1          # ms → µs so we keep <2^31 values
latency_int = (latency * SCALE).astype(int)   # (num_sols, num_data)

model = cp_model.CpModel()

# ---------- decision variables --------------------------------
# %%
x = {}  # x[i,j]  worker i chooses parallel_plan j
for i in range(num_worker_max):
    for j in range(num_sols):
        x[i, j] = model.NewBoolVar(f"x_{i}_{j}")

# %%
y = {}  # y[k,i]  data item k assigned to worker i
for k in range(num_data):
    for i in range(num_worker_max):
        y[k, i] = model.NewBoolVar(f"y_{k}_{i}")

# %%
# auxiliary AND-vars:  z[i,j,k] = x[i,j] ∧ y[k,i]
z = {}
for i in range(num_worker_max):
    for j in range(num_sols):
        for k in range(num_data):
            z[i, j, k] = model.NewBoolVar(f"z_{i}_{j}_{k}")
            # z <= x  ,  z <= y  ,  z >= x + y – 1
            model.Add(z[i, j, k] <= x[i, j])
            model.Add(z[i, j, k] <= y[k, i])
            model.Add(z[i, j, k] >= x[i, j] + y[k, i] - 1)

# %%
# latency per worker and global max
lat_worker = []
for i in range(num_worker_max):
    lw = model.NewIntVar(0, int(_infinity * SCALE), f"lat_{i}")
    lat_worker.append(lw)
lat_max = model.NewIntVar(0, int(_infinity * SCALE), "lat_max")

# ---------- constraints ---------------------------------------
# %%
# 1) each worker picks exactly one (tp,cp) solution
for i in range(num_worker_max):
    model.Add(sum(x[i, j] for j in range(num_sols)) == 1)

# %%
# 2) each data item goes to exactly one worker
for k in range(num_data):
    model.Add(sum(y[k, i] for i in range(num_worker_max)) == 1)

# %%
# 3) define latency of each worker   lat_i = Σ_j Σ_k  latency[j,k] * z[i,j,k]
for i in range(num_worker_max):
    model.Add(lat_worker[i] ==
              sum(latency_int[j, k] * z[i, j, k]
                  for j in range(num_sols)
                  for k in range(num_data)))
    model.Add(lat_worker[i] <= lat_max)

# %%
# 4) resource budget:  Σ_i Σ_j  resource[j] * x[i,j]  ≤  num_total_devices
model.Add(
    sum(resource[j] * x[i, j]
        for i in range(num_worker_max)
        for j in range(num_sols))
    <= num_total_devices
)
# %%
# ---------- objective -----------------------------------------
model.Minimize(lat_max)

# %%
# ---------- solve ---------------------------------------------
solver = cp_model.CpSolver()
solver.parameters.max_time_in_seconds = 3600        # 1-hour limit
solver.parameters.num_search_workers  = 128         # <-- use many cores!
solver.parameters.log_search_progress = True

status = solver.Solve(model)
assert status == cp_model.OPTIMAL, "CP-SAT did not find an optimal solution"

# ---------- extract solution ----------------------------------
results = []
for i in range(num_worker_max):
    # which parallel plan j did this worker pick?
    chosen_j = next(j for j in range(num_sols) if solver.BooleanValue(x[i, j]))
    tp, cp = parallel_plan[chosen_j]

    assigned = [k for k in range(num_data) if solver.BooleanValue(y[k, i])]
    if assigned:
        results.append((batch[assigned], tp, cp))

print("assignment =", results)
print("max-latency =",
      solver.Value(lat_max) / SCALE, "ms")