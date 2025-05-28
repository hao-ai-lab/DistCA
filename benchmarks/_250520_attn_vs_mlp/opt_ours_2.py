# %%
import time_module
from time_module import compute, network
from models import Llama8B
import numpy as np
import pulp
import typing
from typing import List, Tuple

# %%

"""
Ours: balance attn by using different tp_degree and cp_degree for each worker.
    NOTE: ideal case can be solved by an ILP:
    for worker i, use parallelisation solution j
    sol[i][j]: worker i uses sol j or not
    data[k][i]: data k is assigned to worker i
    sum_j sol[i][j] = 1
    sum_i data[k][i] = 1
    lat_i = sum_j_k sol[i][j] * lat_table[j][k]
    resource_i = sum_j_k sol[i][j] * resource_table[j]
    minimize max_i lat_i, under sum_i resource_i <= num_total_devices
"""


# %%
K = 1024
# batch = [32] + [8] * 2 + [4] * 2 + [1] * 8
# batch = [32] + [16] + [8] * 2
batch = [32]  + [1] 
batch = [i * K for i in batch]
batch = np.array(batch)
num_total_devices = 16
num_worker_max = 16


# 1. Prepare constants
parallel_plan = []
for tp in [1, 2, 4, 8]:
    for cp in [1, 2, 4, 8]:
        # TODO: 8 is for intra-node, but CP can go beyond a node?
        if tp * cp <= num_total_devices and tp <= 8:
            parallel_plan.append((tp, cp))

resource = [tp * cp for tp, cp in parallel_plan]

latency = np.zeros((len(parallel_plan), batch.shape[0]))
for j, (tp, cp) in enumerate(parallel_plan):
    this_model = Llama8B(tp=tp, cp=cp)
    for k in range(batch.shape[0]):
        latency[j, k] = this_model._attn(batch[k])
    pass
latency = latency / 1000 # us -> ms, otherwise the infinity is not large enough.

# NOTE: we may not really have num_worker_max workers. 
# Hence, we need to add a special column
# where all latencies are inf, and resource is 0.
parallel_plan.append((0, 0))
resource.append(0)
resource = np.array(resource)
_infinity = 1e15
latency = np.concatenate((latency, np.full((1, batch.shape[0]), _infinity)), axis=0)

# %%

# 2. ILP
num_sols = len(parallel_plan)
num_data = batch.shape[0]
# Create the problem
prob = pulp.LpProblem("AttnBalancing", pulp.LpMinimize)

# Decision variables
x = [[pulp.LpVariable(f"x_{i}_{j}", cat="Binary") for j in range(num_sols)] for i in range(num_worker_max)]
y = [[pulp.LpVariable(f"y_{k}_{i}", cat="Binary") for i in range(num_worker_max)] for k in range(num_data)]

# Latency for each worker
lat_worker = [pulp.LpVariable(f"lat_{i}") for i in range(num_worker_max)]
lat_max = pulp.LpVariable("lat_max")

# Objective: minimize the maximum latency across all workers
prob += lat_max

# %%

# Constraint: each worker chooses exactly one solution
for i in range(num_worker_max):
    prob += pulp.lpSum(x[i]) == 1

# Constraint: each data item is assigned to exactly one worker
for k in range(num_data):
    prob += pulp.lpSum(y[k]) == 1

# %%
# TODO: Quadratic constraint is not solved using the solver pulp solver. 
# Compute latency per worker
for i in range(num_worker_max):
    # latency of worker i = sum_k sum_j (x[i][j] * y[k][i] * latency[j][k])
    lat_expr = pulp.lpSum(
        x[i][j] * y[k][i] * latency[j][k]
        for j in range(num_sols)
        for k in range(num_data)
    )
    prob += lat_worker[i] == lat_expr
    prob += lat_worker[i] <= lat_max

# # %%
# # Compute latency per worker
# for i in range(num_worker_max):
#     # latency of worker i = sum_k sum_j (x[i][j] * y[k][i] * latency[j][k])

#     lat_expr = None
#     for j in range(num_sols):
#         for k in range(num_data):
#             z = pulp.LpVariable(f"z_{i}_{j}_{k}", cat="Binary")
#             prob += z <= x[i][j]
#             prob += z <= y[k][i]
#             prob += z >= x[i][j] + y[k][i] - 1

#             if lat_expr is None:
#                 lat_expr = z * latency[j][k]
#             else:
#                 lat_expr += z * latency[j][k]
    
#     prob += (lat_worker[i] == lat_expr)
#     prob += (lat_worker[i] <= lat_max)

# %%
# Resource constraint
prob += pulp.lpSum(x[i][j] * resource[j] for i in range(num_worker_max) for j in range(num_sols)) <= num_total_devices

# %%
prob
# %%
# Solve
# status = prob.solve(pulp.PULP_CBC_CMD(msg=False))
status = prob.solve(pulp.PULP_CBC_CMD(msg=True))
assert pulp.LpStatus[status] == "Optimal", "ILP did not find optimal solution"
# %%

# 3. Extract solution
results = []
for i in range(num_worker_max):
    for j in range(num_sols):
        if pulp.value(x[i][j]) > 0.5:
            tp, cp = parallel_plan[j]
            assigned_data = [k for k in range(num_data) if pulp.value(y[k][i]) > 0.5]
            if assigned_data:
                results.append((batch[assigned_data], tp, cp))
            break

results
# %% 
results, pulp.value(lat_max)

# %%
sum([model._mlp(item) / 1000 for item in batch])

# %%
results
# %%
