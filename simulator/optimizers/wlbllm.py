# wlb_llm_solver.py
# Re-implementation of the PuLP demo in the same style as the AttnServer example
from ortools.sat.python import cp_model
import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Tuple


# ————————————————————————————————————————————————————————————
#  Problem-specific helpers
# ————————————————————————————————————————————————————————————
def attn_time(x: int) -> float:
    """O(seq²) quadratic attention cost (arbitrary demo model)."""
    return x ** 2


def mlp_time(x: int) -> float:
    """O(seq) MLP cost (arbitrary demo model)."""
    return x


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


# ————————————————————————————————————————————————————————————
#  Main solver class
# ————————————————————————————————————————————————————————————
class WlbLlmSolver:
    """Minimise the slowest worker’s latency subject to length and assignment constraints."""

    def solve(
        self,
        doc_lengths: List[int],
        max_length: int,
        num_workers: int,
        *,
        time_limit_s: int | None = 30,
    ) -> WlbLlmSolution:
        n_docs = len(doc_lengths)
        costs = [int(attn_time(d) + mlp_time(d)) for d in doc_lengths]  # ms, cast to int

        # ——— CP-SAT model ——————————————————————————————————————————
        model = cp_model.CpModel()
        INF = 10**9

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