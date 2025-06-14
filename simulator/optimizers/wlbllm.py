# wlb_llm_solver.py
# Re-implementation of the PuLP demo in the same style as the AttnServer example
from ortools.sat.python import cp_model
import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Tuple, Any

import timemodule as tm

INF = int(1e15)
# ————————————————————————————————————————————————————————————
#  Dataclass to hold the results
# ————————————————————————————————————————————————————————————
@dataclass
class WlbLlmSolution:
    # document → worker assignment
    doc2worker: Dict[int, int]
    # worker → docs actually served
    batches: List[List[int]]
    # latency per worker and objective
    lat_worker: List[int]
    lat_max: int

    # raw artefacts (optional – handy for debugging / tweaking)
    model: cp_model.CpModel
    solver: cp_model.CpSolver
    variables: Dict[str, object]

    def print_solution(self) -> None:  # noqa: D401 (simple “Print …”)
        print("WLB-LLM ILP Solution")
        for w, docs in enumerate(self.batches):
            print(f"- Worker {w:<2d}: docs {docs}  —  latency {self.lat_worker[w]} ms")
        print(f"- Maximum latency: {self.lat_max}\n")

    def dump_object(self) -> Dict[str, Any]:
        return dict(
            batches=self.batches,
            lat_max=self.lat_max,
            lat_worker=self.lat_worker,
            doc2worker=self.doc2worker,
        )


# ————————————————————————————————————————————————————————————
#  Main solver class
# ————————————————————————————————————————————————————————————
class WlbLlmSolver:
    """Minimise the slowest worker's latency subject to length and assignment constraints."""

    def get_attn_time(self, x: int, tp: int, cp: int) -> float:

        hqo = 64
        hkv = 4
        d = 128

        allreduce_perdevice_nelem = x * hqo * d // tp
        allgather_perdevice_nelem = x * d * max(1, hkv // cp)

        
        attn = tm.get_attn_time(x, tp, cp)
        allreduce_time = tm.get_allreduce_time(allreduce_perdevice_nelem, tp)
        allgather_time = tm.get_allgather_time(allgather_perdevice_nelem, cp)

        return attn + allreduce_time + allgather_time

    
    def get_mlp_time(self, x: int, tp: int, cp: int) -> float:
        import timemodule as tm
        return tm.get_mlp_time(x, tp, cp)

    def solve(
        self,
        doc_lengths: List[int],
        max_length: int,
        num_workers: int,
        parallel_plan: tuple[int, int],
        *,
        time_limit_s: int | None = 30,
    ) -> WlbLlmSolution:
        tp, cp = parallel_plan

        n_docs = len(doc_lengths)
        attn_time = self.get_attn_time
        mlp_time = self.get_mlp_time

        costs = [int(attn_time(d, tp = tp, cp = cp) + mlp_time(d, tp = tp, cp = cp)) for d in doc_lengths]  # ms, cast to int
        print(costs)

        # ——— CP-SAT model ——————————————————————————————————————————
        model = cp_model.CpModel()

        # Decision: x[d,w] == 1  ⇔  doc d served by worker w
        x = {
            (d, w): model.NewBoolVar(f"x_{d}_{w}")
            for d in range(n_docs)
            for w in range(num_workers)
        }

        # 1. Each doc goes to exactly one worker
        for d in range(n_docs):
            model.Add(sum(x[d, w] for w in range(num_workers)) == 1)

        # 2. Per-worker length budget  Σ len_d * x[d,w] ≤ L_max
        for w in range(num_workers):
            model.Add(
                sum(doc_lengths[d] * x[d, w] for d in range(n_docs)) <= max_length
            )

        # 3. Latency per worker  lat_w = Σ cost_d * x[d,w]
        lat_worker = [
            model.NewIntVar(0, INF, f"lat_{w}") for w in range(num_workers)
        ]
        for w in range(num_workers):
            # TODO: The cost of MLP is subject to the parallel plan to split the MLP.
            model.Add(
                lat_worker[w]
                == sum(costs[d] * x[d, w] for d in range(n_docs))
            )

        # 4. Objective  —  minimise the maximum worker latency
        lat_max = model.NewIntVar(0, INF, "lat_max")
        for w in range(num_workers):
            model.Add(lat_worker[w] <= lat_max)
        model.Minimize(lat_max)

        # ——— Solve ———————————————————————————————————————————————
        solver = cp_model.CpSolver()
        if time_limit_s:
            solver.parameters.max_time_in_seconds = time_limit_s
        solver.parameters.num_search_workers = 0  # use all cores
        status = solver.Solve(model)

        if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            raise RuntimeError("No feasible solution found")

        # ——— Extract assignment ————————————————————————————
        doc2worker: Dict[int, int] = {}
        batches: List[List[int]] = [[] for _ in range(num_workers)]
        for d in range(n_docs):
            for w in range(num_workers):
                if solver.Value(x[d, w]):
                    doc2worker[d] = w
                    batches[w].append(doc_lengths[d])
                    break

        return WlbLlmSolution(
            doc2worker=doc2worker,
            batches=batches,
            lat_worker=[solver.Value(lw) for lw in lat_worker],
            lat_max=solver.Value(lat_max),
            model=model,
            solver=solver,
            variables=dict(x=x, lat_worker=lat_worker, lat_max=lat_max),
        )


# ————————————————————————————————————————————————————————————
#  tiny smoke-test
# ————————————————————————————————————————————————————————————

def test_solver():
    solver = WlbLlmSolver()
    doc_lengths = [1, 2, 3, 4]
    sol = solver.solve(doc_lengths, max_length=16, num_workers=4)
    sol.print_solution()

if __name__ == "__main__":
    test_solver()

