# %%
from ortools.sat.python import cp_model
import numpy as np


parallel_plan = [(tp, cp) for tp in [1, 2, 4, 8] for cp in [1, 2, 4, 8]]
parallel_plan.append((0, 0))
resource = [tp * cp for tp, cp in parallel_plan]

def solve_attnserver(batch: list[int], num_workers: int, num_total_devices: int):
    """
    Solve the attnserver problem.

    Args:
        batch (list[int]): The batch of documents.
        num_workers (int): The number of workers.
        num_total_devices (int): The total number of devices.

    Returns:
        list[int]: The plan index for each worker.
    """

    latency = np.zeros((len(parallel_plan), len(batch)))
    for j, (tp, cp) in enumerate(parallel_plan):
        for k in range(len(batch)):
            doc_length = batch[k]
            # TODO: Put the proper modeling here.
            latency[j, k] = doc_length / (1 + tp * (cp ** 0.5))
        
    # latency = (latency * 1000)
    # print(f"latency: {latency}")
    latency = latency.astype(int)

    model = cp_model.CpModel()
    num_plans    = len(parallel_plan)
    num_docs     = len(batch)


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
    y = {(k,i): model.NewBoolVar(f"y_{k}_{i}") for k in range(num_docs) for i in range(num_workers)}

    INF = 1e9
    lat_worker = [model.NewIntVar(0, int(INF), f"lat_{i}") for i in range(num_workers)]
    lat_max    = model.NewIntVar(0, int(INF), "lat_max")


    # 1. Each worker picks one plan
    for i in range(num_workers):
        model.Add(sum(x[i,j] for j in range(num_plans)) == 1)


    # 2. Each document assigned to one worker
    for k in range(num_docs):
        model.Add(sum(y[k,i] for i in range(num_workers)) == 1)


    # 3. Linearise product: z_{i,j,k} = x_{i,j} * y_{k,i}
    z = {}
    for i in range(num_workers):
        for j in range(num_plans):
            for k in range(num_docs):
                z[(i,j,k)] = model.NewBoolVar(f"z_{i}_{j}_{k}")
                model.AddBoolAnd([x[i,j], y[k,i]]).OnlyEnforceIf(z[(i,j,k)])
                model.AddBoolOr([x[i,j].Not(), y[k,i].Not(), z[(i,j,k)]])

        # latency per worker
        model.Add(lat_worker[i] ==
                sum(int(latency[j,k]) * z[(i,j,k)]
                    for j in range(num_plans) for k in range(num_docs)))
        model.Add(lat_worker[i] <= lat_max)

    # 4. Resource budget
    total_devices = sum(resource[j] * x[i,j]
                        for i in range(num_workers) for j in range(num_plans))
    model.Add(total_devices <= num_total_devices)


    # 5. Objective
    model.Minimize(lat_max)


    # Solve – parallel by default
    solver = cp_model.CpSolver()
    solver.parameters.num_search_workers = 0   # 0 → use all logical cores
    solver.parameters.max_time_in_seconds = 60 # optional time-limit
    status = solver.Solve(model)

    # Extract the solutions
    xs = dict() # worker -> plan
    for i in range(num_workers):
        for j in range(num_plans):
            if solver.Value(x[i,j]) == 1:
                xs[i] = parallel_plan[j]
    
    ys = dict() # doc -> worker
    for k in range(num_docs):
        for i in range(num_workers):
            if solver.Value(y[k,i]) == 1:
                ys[k] = i

    worker2plan = xs
    doc2worker  = ys

    return (model, solver), worker2plan, doc2worker, solver.Value(lat_max)


def test_attnserver():
    batch = [1, 2, 3, 4]
    num_workers = 4
    num_total_devices = 16
    (model, solver), worker2plan, doc2worker, lat_max = solve_attnserver(batch, num_workers, num_total_devices)
    
    for i in range(num_workers):
        print(f"Worker {i} chooses plan {worker2plan[i]}")
    
    for k in range(len(batch)):
        print(f"Doc {k} is processed by worker {doc2worker[k]}")

    print(f"lat_max: {lat_max}")

    # print(f"Status: {solver.StatusName(status)}")
    # print(f"Objective: {solver.ObjectiveValue()}")
    # print(f"lat_max: {(lat_max)}")
# %%
def tests():
    test_attnserver()


def main():
    pass
# %%
if __name__ == "__main__":
    main()

# %%
