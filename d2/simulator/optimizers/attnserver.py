from ortools.sat.python import cp_model
import numpy as np
from dataclasses import dataclass

import d2.timemodule as tm
INF = tm.INF

def print_latency_table(batch: list[int], parallel_plan: list[tuple[int, int]], latency: np.ndarray):
    from rich.console import Console
    from rich.table import Table    
    console = Console()

    seen_length = set()
    for k in range(len(batch)):
        doc_length = batch[k]
        if doc_length in seen_length:
            continue
        seen_length.add(doc_length)
        table = Table(title=f"Doc[{k}] = {doc_length} Latency")

        # Add columns for cp values
        table.add_column("tp/cp", justify="right", style="cyan", no_wrap=True)
        for cp in [8, 4, 2, 1]:
            table.add_column(str(cp), justify="right")
        # Populate the table
        for tp in [8, 4, 2, 1]:
            row = [str(tp)]
            for cp in [8, 4, 2, 1]:
                plan_idx = parallel_plan.index((tp, cp))
                value = latency[plan_idx, k]
                row.append(f"{value:.2f}" if value != float('inf') else 'inf')
            table.add_row(*row)

        console.print(table)


@dataclass
class AttnServerSolution:
    # worker -> plan
    worker2plan: dict[int, tuple[int, int]]
    # doc -> worker
    doc2worker: dict[int, int]
    # batches = worker -> batch
    batches: list[int]
    # parallel plan
    parallel_plan: list[tuple[int, int]]
    # max latency
    lat_max: int
    # latency per worker
    lat_worker: list[int]
    # Latency per document
    lat_doc_table: np.ndarray

    model: cp_model.CpModel
    solver: cp_model.CpSolver
    variables: dict[str, 'cp_model.BoolVarT | cp_model.IntVar']

    mlp_time: float

    def get_parallel_plan(self):
        return [plan for plan in self.worker2plan.values() if plan != (0, 0)]

    def get_batch_assignment(self):
        return self.batches

    def dump_object(self):
        return dict(
            batches=self.batches,
            lat_max=self.lat_max,
            lat_worker=self.lat_worker,
            worker2plan=self.worker2plan,
            doc2worker=self.doc2worker,
        )

    def print_solution(self):
        print("AttnServer Solution:")
        for i, (worker, plan) in enumerate(self.worker2plan.items()):
            if len(self.batches[i]) == 0:
                continue
            print(f"- Worker {i}: {plan} docs {self.batches[i]} - latency: {self.lat_worker[i] / 1000} ms")
        print(f"- MLP Time: {self.mlp_time} ms")
        print(f"- Maximum Latency: {self.lat_max}")



class AttnServerSolver:
    def __init__(self, parallel_plan: list[tuple[int, int]] = None):
        if not parallel_plan:
            parallel_plan = [(tp, cp) for tp in [1, 2, 4, 8] for cp in [1, 2, 4, 8]]
            parallel_plan.append((0, 0))
        self.parallel_plan = parallel_plan
        self.resource = [tp * cp for tp, cp in parallel_plan]
        self.num_plans = len(parallel_plan)

    def get_latency_table(self, batch: list[int], parallel_plan: list[tuple[int, int]], num_total_devices: int):
        hqo = 64
        hkv = 4
        d = 128

        latency = np.zeros((len(parallel_plan), len(batch)))
        for j, (tp, cp) in enumerate(parallel_plan):
            for k in range(len(batch)):
                doc_length = batch[k]
                if tp * cp == 0:
                    latency[j, k] = tm.INF
                else:
                    lat = 0
                    attn_time = tm.get_attn_time(doc_length, tp, cp)
                    allreduce_elem = doc_length * hqo * d // tp
                    allgather_elem = doc_length * d * max(1, hkv // cp)
                    allreduce_time = tm.get_allreduce_time(allreduce_elem, tp)
                    allgather_time = tm.get_allgather_time(allgather_elem, cp)

                    lat = attn_time + allreduce_time + allgather_time
                    latency[j, k] = lat
                    print(f"[AttnServer] [{tp=}, {cp=}] d: {doc_length}, latency: {lat:.2f}, attn_time: {attn_time:.2f}, allreduce_time: {allreduce_time:.2f}, allgather_time: {allgather_time:.2f}")


        mcp = max(num_total_devices // 8, 1)
        mtp = min(num_total_devices, 8)
        total_length = sum(batch)

        # TODO: The MLP time is not exactly correct...
        # latency += tm.get_mlp_time(total_length, mtp, mcp)
        
        mlp_time = tm.get_mlp_time(sum(batch), mtp, mcp)
        print(f"[AttnServer] MLP(tp={mtp},cp={mcp}): {mlp_time:.2f} ms")
        # mlp_time = tm.get_mlp_time(batch[0], mtp, mcp)
        # mlp_time = tm.get_mlp_time(batch[0], 2, 2)
        return latency, mlp_time
        
    def solve(
            self, batch: list[int], num_workers: int, num_total_devices: int,
            timeout: float = 15,
    ) -> AttnServerSolution:
        """
        Solve the attnserver problem.

        Args:
            batch (list[int]): The batch of documents.
            num_workers (int): The number of workers.
            num_total_devices (int): The total number of devices.

        Returns:
            AttnServerSolution: The solution to the attnserver problem.
        """
        parallel_plan = self.parallel_plan
        resource = self.resource
        
        latency_ms, mlp_time = self.get_latency_table(batch, parallel_plan, num_total_devices)
        latency_us = latency_ms * 1000
        
        latency = latency_us.astype(int)

        print_latency_table(batch, parallel_plan, latency_ms)

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


        lat_worker = [model.NewIntVar(0, tm.INF, f"lat_{i}") for i in range(num_workers)]
        lat_max    = model.NewIntVar(0, tm.INF, "lat_max")


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
        solver.parameters.max_time_in_seconds = timeout # optional time-limit
        status = solver.Solve(model)

        # Extract the solutions
        xs = dict() # worker -> plan
        for i in range(num_workers):
            for j in range(num_plans):
                if solver.Value(x[i,j]) == 1:
                    xs[i] = parallel_plan[j]
                    
                    if xs[i] != (0, 0):
                        worker_lat = solver.Value(lat_worker[i])
                        print(f"Worker {i} plan {j} = {parallel_plan[j]} latency {(worker_lat / 1000):.2f} ms")
        
        ys = dict() # doc -> worker
        for k in range(num_docs):
            for i in range(num_workers):
                if solver.Value(y[k,i]) == 1:
                    ys[k] = i

        worker2plan = xs
        doc2worker  = ys

        batches = []
        for i in range(num_workers):
            batches.append([])
        
        for k in range(num_docs):
            i = doc2worker[k]
            batches[i].append(batch[k])


        lat_max_value = solver.Value(lat_max) / 1000

        return AttnServerSolution(
            worker2plan=worker2plan,
            doc2worker=doc2worker,
            batches=batches,
            parallel_plan=parallel_plan,
            lat_max=lat_max_value + mlp_time,
            lat_worker=[solver.Value(lat_worker[i]) for i in range(num_workers)],
            lat_doc_table=latency,
            model=model,
            solver=solver,
            variables=dict(
                x=x, y=y, z=z,
                lat_worker=lat_worker,
                lat_max=lat_max,
            ),
            mlp_time=mlp_time,
        )
    
def test_attnserver():
    batch = [1, 2, 3, 4]
    num_workers = 4
    num_total_devices = 16
    
    solver = AttnServerSolver()
    solution = solver.solve(batch, num_workers, num_total_devices)
    solution.print_solution()
    return solution


if __name__ == "__main__":
    test_attnserver()