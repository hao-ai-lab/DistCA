import argparse
from typing import List, Tuple

import numpy as np
import pulp

from datasets import load_dataset
from transformers import AutoTokenizer
import time_module.compute as compute

class Simulator:
    def __init__(
        self, hqo: int = 32, hkv: int = 8, d: int = 128, 
        d1: int = 4096, d2: int = None, 
        gpu: str = "A100-SXM-80GB",
        dtype: str = "half"
    ):
        """
        Initialize the simulator with model configuration parameters
        Args:
            hqo: Number of query/output heads
            hkv: Number of key/value heads
            d: Head dimension
            d1: Hidden dimension
            d2: MLP intermediate dimension (defaults to 3.5 * d1)
            gpu: GPU name
            dtype: data type
        """
        self.hqo = hqo
        self.hkv = hkv
        self.d = d
        self.d1 = d1
        self.d2 = int(d1 * 3.5) if d2 is None else d2
        self.gpu = gpu
        self.dtype = dtype

    def get_batch_attn_time(self, num_tokens: np.ndarray, tp_degree: int,
                            cp_degree: int, do_sum: bool=True) -> float:
        """
        Get the attention time for a batch of tokens using direct computation
        Input:
            num_tokens: 1-dim array of shape (num_data,). Each datapoint is the number of tokens
            of a data sample in the batch.
        """
        if not isinstance(num_tokens, np.ndarray):
            raise TypeError("num_tokens must be a numpy array.")
        if num_tokens.ndim != 1:
            raise ValueError("num_tokens must be a 1-dimensional array.")
        if np.any(num_tokens <= 0):
            raise ValueError("All values in num_tokens must be positive.")
        if tp_degree <= 0 or cp_degree <= 0:
            raise ValueError("tp_degree and cp_degree must be positive.")
        if (tp_degree & (tp_degree - 1)) != 0 or (cp_degree & (cp_degree - 1)) != 0:
            raise ValueError("tp_degree and cp_degree must be powers of 2.")

        assert self.hqo % tp_degree == 0 and tp_degree <= self.hqo
        assert self.hkv % tp_degree == 0 and tp_degree <= self.hkv
        hqo = self.hqo // tp_degree
        hkv = self.hkv // tp_degree

        result = 0
        for tokens in num_tokens:
            # TODO: This is too slow. 
            attn_time = compute.attn_time(
                gpu=self.gpu,
                cp=cp_degree,
                head_dim=self.d,
                nhead=hqo,
                tokens=tokens,
                dtype=self.dtype,
                is_fwd=True,
            )
            result += attn_time

        return result

    def get_batch_mlp_time(self, num_tokens: np.ndarray, tp_degree: int,
                           cp_degree: int, do_sum: bool=True) -> float:
        """
        Get the MLP time for a batch of tokens using direct computation
        Input:
            num_tokens: 1-dim array of shape (num_data,). Each datapoint is the number of tokens
            of a data sample in the batch.
        """
        # Scale dimensions by tp_degree
        d2 = self.d2 / tp_degree
        hqo = self.hqo / tp_degree
        hkv = self.hkv / tp_degree

        # Scale tokens by cp_degree
        if do_sum:
            T = np.sum(num_tokens)

            def M(m, k, n):
                return compute.gemm_time(
                    gpu=self.gpu,
                    m=m, k=k, n=n,
                    dtype=self.dtype,
                )

            # Compute total MLP time
            mlp_time = (
                M(T, self.d1, d2) + M(T, d2, self.d1) + 
                M(T, self.d1, hqo * self.d) + 2 * M(T, hkv * self.d, self.d1)
            )

            return mlp_time
        else:
            return np.array([
                (
                    M(T, self.d1, d2) + M(T, d2, self.d1) + 
                    M(T, self.d1, hqo * self.d) + 2 * M(T, hkv * self.d, self.d1)
                ) 
                for T in num_tokens
            ])


    def get_batch_time(self, num_tokens: np.ndarray, tp_degree: int, cp_degree: int) -> float:
        attn_time = self.get_batch_attn_time(num_tokens, tp_degree, cp_degree)
        mlp_time = self.get_batch_mlp_time(num_tokens, tp_degree, cp_degree)
        return attn_time + mlp_time
    
    def get_dp_batch_time(self, dp_batches: List[np.ndarray], tp_degree: int, cp_degree: int) -> float:
        dp_time = [
            self.get_batch_time(batch, tp_degree, cp_degree) for batch in dp_batches
        ]
        return np.max(dp_time)

    # Planner
    def wlb_balance_ilp(self, batch: np.ndarray, tp_degree: int, cp_degree: int, dp_degree: int, num_worker_max: int) -> List[np.ndarray]:
        """Baseline method that fixes the tp_degree and cp_degree for all workers, but allows different MLP for each worker."""
        # NOTE: ideal case can be solved by an ILP:
        # for worker i:
        # data[k][i]: data k is assigned to worker i
        # sum_i data[k][i] = 1
        # lat_i = sum_j_k sol[i][j] * lat_table[j][k] // unlike balance_attn, this time the latency is summed by mlp as well.
        # minimize max_i lat_i, under sum_i resource_i <= num_total_devices
        # 1. Prepare constants
        num_total_devices = tp_degree * cp_degree * dp_degree
        num_data = batch.shape[0]
        latency_table = (
            self.get_batch_attn_time(batch, tp_degree, cp_degree, do_sum=False) 
            + self.get_batch_mlp_time(batch, tp_degree, cp_degree, do_sum=False)
        )

        # 2. ILP
        # Create the problem
        prob = pulp.LpProblem("WLB", pulp.LpMinimize)
        # Decision variables
        x = [[pulp.LpVariable(f"x_{k}_{i}", cat="Binary") for i in range(num_worker_max)] for k in range(num_data)]
        # Latency for each worker
        lat_worker = [pulp.LpVariable(f"lat_{i}") for i in range(num_worker_max)]
        lat_max = pulp.LpVariable("lat_max")

        # Objective: minimize the maximum latency across all workers
        prob += lat_max

        # Constraint: each data item is assigned to exactly one worker
        for k in range(num_data):
            prob += pulp.lpSum(x[k]) == 1

        # Compute latency per worker
        for i in range(num_worker_max):
            # latency of worker i = sum_k (x[k][i] * latency_table[k])
            lat_expr = pulp.lpSum(x[k][i] * latency_table[k] for k in range(num_data))
            prob += lat_worker[i] == lat_expr
            prob += lat_worker[i] <= lat_max

        # Solve
        status = prob.solve(pulp.PULP_CBC_CMD(msg=False))
        assert pulp.LpStatus[status] == "Optimal", "ILP did not find optimal solution"
        # 3. Extract solution
        results = []
        for i in range(num_worker_max):
            assigned_data = [k for k in range(num_data) if pulp.value(x[k][i]) > 0.5]
            if assigned_data:
                results.append(batch[assigned_data])
        return results, pulp.value(lat_max)

    def _balance_attn(self, batch: np.ndarray, num_total_devices: int, num_worker_max: int) -> List[Tuple[np.ndarray, int, int]]:
        """Ours: balance attn by using different tp_degree and cp_degree for each worker."""
        # NOTE: ideal case can be solved by an ILP:
        # for worker i, use parallelisation solution j
        # sol[i][j]: worker i uses sol j or not
        # data[k][i]: data k is assigned to worker i
        # sum_j sol[i][j] = 1
        # sum_i data[k][i] = 1
        # lat_i = sum_j_k sol[i][j] * lat_table[j][k]
        # resource_i = sum_j_k sol[i][j] * resource_table[j]
        # minimize max_i lat_i, under sum_i resource_i <= num_total_devices

        # 1. Prepare constants
        parallel_plan = []
        for tp in [1, 2, 4, 8]:
            for cp in [1, 2, 4, 8]:
                # TODO: 8 is for intra-node, but CP can go beyond a node?
                if tp * cp <= min(8, num_total_devices):
                    parallel_plan.append((tp, cp))
        resource = [tp * cp for tp, cp in parallel_plan]
        latency = np.zeros((len(parallel_plan), batch.shape[0]))
        for i, (tp, cp) in enumerate(parallel_plan):
            latency[i] = self.get_batch_attn_time(batch, tp, cp, do_sum=False)
        # NOTE: we may not really have num_worker_max workers. Hence, we need to add a special column
        # where all latencies are inf, and resource is 0.
        parallel_plan.append((0, 0))
        resource.append(0)
        resource = np.array(resource)
        latency = np.concatenate((latency, np.full((1, batch.shape[0]), np.inf)), axis=0)

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

        # Constraint: each worker chooses exactly one solution
        for i in range(num_worker_max):
            prob += pulp.lpSum(x[i]) == 1

        # Constraint: each data item is assigned to exactly one worker
        for k in range(num_data):
            prob += pulp.lpSum(y[k]) == 1

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

        # Resource constraint
        prob += pulp.lpSum(x[i][j] * resource[j] for i in range(num_worker_max) for j in range(num_sols)) <= num_total_devices

        # Solve
        status = prob.solve(pulp.PULP_CBC_CMD(msg=False))
        assert pulp.LpStatus[status] == "Optimal", "ILP did not find optimal solution"

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

        return results

    def get_dp_attn_balanced_time(self, dp_batches: List[np.ndarray], tp_degree: int, cp_degree: int) -> float:
        mlp_time = [
            self.get_batch_mlp_time(batch, tp_degree, cp_degree) for batch in dp_batches
        ]
        mlp_time = np.max(mlp_time)
        num_total_devices = tp_degree * cp_degree * len(dp_batches)

        total_batch = np.concatenate(dp_batches)
        attn_configs = self._balance_attn(total_batch, num_total_devices)

        attn_time = [
            self.get_batch_attn_time(*attn_config) for attn_config in attn_configs
        ]
        attn_time = np.max(attn_time)
        # NOTE: add communication time?
        return attn_time + mlp_time

def test(data_path, tokenizer, batch_size,
         tp_degree, cp_degree, dp_degree,
         num_tokens_per_data,
         hqo=32, hkv=8, d=128, d1=4096, d2=None,
         batch_samples=100):
    doc_dataset = load_dataset(data_path, streaming=True)
    assert batch_size % dp_degree == 0

    sim = Simulator(hqo=hqo, hkv=hkv, d=d, d1=d1, d2=d2)

    def get_data():
        data_budget = num_tokens_per_data
        doc_lens = []
        for data in doc_dataset:
            token_count = len(tokenizer(data["text"]).input_ids)
            while data_budget < token_count:
                doc_lens.append(data_budget)
                yield doc_lens
                doc_lens = []
                data_budget = num_tokens_per_data
                token_count -= data_budget

            if data_budget >= token_count:
                doc_lens.append(token_count)
                data_budget -= token_count

    # collect time from all samples
    ours = []
    baseline = []
    wlb = []
    for sample in range(batch_samples):
        datas = [
            get_data() for _ in range(batch_size)
        ]
        # based on dp_degree, split the batch
        dp_batches = [
            np.concatenate(datas[i:i + batch_size // dp_degree])
            for i in range(0, batch_size, batch_size // dp_degree)
        ]
        batch = np.concatenate(datas)
        # 1. ours
        ours_time = sim.get_dp_attn_balanced_time(dp_batches, tp_degree, cp_degree)
        # 2. baseline
        baseline_time = sim.get_dp_batch_time(dp_batches, tp_degree, cp_degree)
        # 3. wlb balance
        balanced_batch, wlb_time = sim.wlb_balance_ilp(batch, tp_degree, cp_degree, dp_degree, batch_size)
        ours.append(ours_time)
        baseline.append(baseline_time)
        wlb.append(wlb_time)
        print(f"Sample {sample + 1}/{batch_samples}: Ours: {ours_time:.2f}, Baseline: {baseline_time:.2f}, WLB: {wlb_time:.2f}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="./data/fake.json")
    parser.add_argument("--tokenizer", type=str, default="gpt2")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--tp_degree", type=int, default=1)
    parser.add_argument("--cp_degree", type=int, default=1)
    parser.add_argument("--dp_degree", type=int, default=1)
    parser.add_argument("--num_tokens_per_data", type=int, default=512)
    parser.add_argument("--hqo", type=int, default=32)
    parser.add_argument("--hkv", type=int, default=8)
    parser.add_argument("--d", type=int, default=128)
    parser.add_argument("--d1", type=int, default=4096)
    parser.add_argument("--d2", type=int, default=None)
    parser.add_argument("--batch_samples", type=int, default=100)
    args = parser.parse_args()
    # Load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    # Run the test
    test(args.data_path, tokenizer, args.batch_size,
         args.tp_degree, args.cp_degree, args.dp_degree,
         args.num_tokens_per_data,
         hqo=args.hqo, hkv=args.hkv, d=args.d, d1=args.d1, d2=args.d2,
         batch_samples=args.batch_samples)

if __name__ == "__main__":
    main()
