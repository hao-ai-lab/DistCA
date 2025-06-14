# %%
from ortools.sat.python import cp_model
import numpy as np

# %%
parallel_plan = [(tp, cp) for tp in [1, 2, 4, 8] for cp in [1, 2, 4, 8]]
parallel_plan.append((0, 0))
resource = [tp * cp for tp, cp in parallel_plan]
# %%
# batch = [32] + [8] * 2 + [4] * 2 + [1] * 8
batch = [1, 2, 3, 4] # doc_lengths

# %%
latency = np.zeros((len(parallel_plan), len(batch)))
for j, (tp, cp) in enumerate(parallel_plan):
    for k in range(len(batch)):
        doc_length = batch[k]
        # TODO: Put the proper modeling here.
        latency[j, k] = doc_length / (1 + tp * (cp ** 0.5))
latency *= 1000
latency = latency.astype(int)
print(latency)
# %%
model = cp_model.CpModel()
num_workers  = 4
num_plans    = len(parallel_plan)
num_batches  = len(batch)   
num_total_devices = 16

# %%
# Constants 
# - i ∈ {0,…,W-1}: worker index (W = num_workers_max, 16)
# - j ∈ {0,…,P-1}: plan index (P = len(parallel_plan))
# - k ∈ {0,…,B-1}: document index (B = len(batch))
# - resource[j]: devices consumed by plan j (tp × cp)
# - latency[j,k]: latency (ms) if plan j processes document k

# Decision vars
# - x[i,j]: {0,1} worker i chooses plan j
# - y[k,i]: {0,1} document k is processed by worker i
# - z[i,j,k]: {0,1} helper var = x[i,j] ∧ y[k,i] (“worker i with plan j serves document k”)
# - lat_worker[i]: ℤ ≥ 0 total latency accumulated on worker i
# - lat_max: ℤ ≥ 0 slowest worker’s latency (objective)


x = {(i,j): model.NewBoolVar(f"x_{i}_{j}") for i in range(num_workers) for j in range(num_plans)}
y = {(k,i): model.NewBoolVar(f"y_{k}_{i}") for k in range(num_batches) for i in range(num_workers)}

lat_worker = [model.NewIntVar(0, int(1e9), f"lat_{i}") for i in range(num_workers)]
lat_max    = model.NewIntVar(0, int(1e9), "lat_max")

# %%
# 1. Each worker picks one plan
for i in range(num_workers):
    model.Add(sum(x[i,j] for j in range(num_plans)) == 1)

# %%
# 2. Each document assigned to one worker
for k in range(num_batches):
    model.Add(sum(y[k,i] for i in range(num_workers)) == 1)

# %%
# 3. Linearise product: z_{i,j,k} = x_{i,j} * y_{k,i}
z = {}
for i in range(num_workers):
    for j in range(num_plans):
        for k in range(num_batches):
            z[(i,j,k)] = model.NewBoolVar(f"z_{i}_{j}_{k}")
            model.AddBoolAnd([x[i,j], y[k,i]]).OnlyEnforceIf(z[(i,j,k)])
            model.AddBoolOr([x[i,j].Not(), y[k,i].Not(), z[(i,j,k)]])

    # latency per worker
    model.Add(lat_worker[i] ==
              sum(int(latency[j,k]) * z[(i,j,k)]
                  for j in range(num_plans) for k in range(num_batches)))
    model.Add(lat_worker[i] <= lat_max)

# %%
# 4. Resource budget
total_devices = sum(resource[j] * x[i,j]
                    for i in range(num_workers) for j in range(num_plans))
model.Add(total_devices <= num_total_devices)

# %%
# 5. Objective
model.Minimize(lat_max)

# %%
# Solve – parallel by default
solver = cp_model.CpSolver()
solver.parameters.num_search_workers = 0   # 0 → use all logical cores
solver.parameters.max_time_in_seconds = 60 # optional time-limit
status = solver.Solve(model)

# %%
print(f"Status: {solver.StatusName(status)}")
print(f"Objective: {solver.ObjectiveValue()}")
print(f"lat_max: {solver.Value(lat_max)}")
# %%
# show each worker's plan
worker_plans = []
for i in range(num_workers):
    for j in range(num_plans):
        if solver.Value(x[i,j]) == 1:
            print(f"Worker {i} chooses plan (tp, cp) = {parallel_plan[j]}")
            worker_plans.append(j)
# %%
# show each documents' processed time
for k in range(num_batches):
    for i in range(num_workers):
        if solver.Value(y[k,i]) == 1:
            j = worker_plans[i]
            print(f"Doc {k} by worker {i} at {solver.Value(lat_worker[i])}ms (itself needs {latency[j,k]}ms)")

# %%
