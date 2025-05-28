# %%
import time_module
from time_module import compute, network
from models import Llama8B
import numpy as np
import pulp
import typing
from typing import List, Tuple

K = 1024
tp = 8
cp = 1
num_worker_max = 4
assert tp * cp * num_worker_max == 16, "Total number of workers must be 16"
model = Llama8B(tp=tp, cp=cp)

# %%

batch = [64] + [2] * 32
batch = [i * K for i in batch]
batch = np.array(batch)
num_data = batch.shape[0]

latency_table = np.zeros((batch.shape[0]))
for i in range(batch.shape[0]):
    latency_table[i] = model._attn(batch[i]) + model._mlp(batch[i])
latency_table = latency_table / 1000 # us -> ms

# %%
prob = pulp.LpProblem("WLB", pulp.LpMinimize)

x = [[pulp.LpVariable(f"x_{k}_{i}", cat="Binary") for i in range(num_worker_max)] for k in range(num_data)]
# Latency for each worker
lat_worker = [pulp.LpVariable(f"lat_{i}") for i in range(num_worker_max)]
lat_max = pulp.LpVariable("lat_max")
# %%
# Objective: minimize the maximum latency across all workers
prob += lat_max

# %%
# Constraint: each data item is assigned to exactly one worker
for k in range(num_data):
    prob += pulp.lpSum(x[k]) == 1

# %%
# Compute latency per worker
for i in range(num_worker_max):
    # latency of worker i = sum_k (x[k][i] * latency_table[k])
    lat_expr = pulp.lpSum(x[k][i] * latency_table[k] for k in range(num_data))
    prob += lat_worker[i] == lat_expr
    prob += lat_worker[i] <= lat_max


# %%
# Solve
status = prob.solve(pulp.PULP_CBC_CMD(msg=False))
assert pulp.LpStatus[status] == "Optimal", "ILP did not find optimal solution"

# %%
# 3. Extract solution
results = []
for i in range(num_worker_max):
    assigned_data = [k for k in range(num_data) if pulp.value(x[k][i]) > 0.5]
    if assigned_data:
        results.append(batch[assigned_data])

# %%
# %%
print(f"WLB LLM[TP={tp}, CP={cp}, DP={num_worker_max}]:")
results, pulp.value(lat_max)

# %%
