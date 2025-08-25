# %%
K = 1024
megatron_batch_size = 2 # because we do pingpong.
base_seq_len = K * 64
attn_base_time = 12.5020
# linear_base_time = (13.5 + 8.5) # mlp + qkvo
# linear_base_time = (13.5 + 8.5)  # mlp + qkvo
mlp_base_time = 13.5  # assume expert parallel
qkvo_base_time = 8.5 
linear_base_time = (mlp_base_time + qkvo_base_time)  # mlp + qkvo
# linear_base_time = 0
# linear_base_time = 0



# %%
# total_seq_len = K * 128
# total_seq_len = K * 256
total_seq_len = K * 512

batch_size = 8

wlb_dp = 4
wlb_cp = 2
total_ranks = 8


# %%

def get_attn_time(batch):
    total_time = 0
    for l in batch:
        ratio = l / base_seq_len
        total_time += attn_base_time * (ratio ** 2)
    return total_time

def get_mlp_time(batch):
    total_time = 0
    for l in batch:
        ratio = l / base_seq_len
        total_time += linear_base_time * (ratio)
    return total_time

def get_network_time(token_per_batch, cp_degree):
    base_token_per_batch = 512 * 1024
    if cp_degree == 1:
        return 0
    if cp_degree == 2:
        base_time = 8
    elif cp_degree == 4:
        base_time = 20
    elif cp_degree == 8:
        base_time = 46
    else:
        raise ValueError(f"Invalid cp_degree: {cp_degree}")

    total_time = base_time * (token_per_batch / base_token_per_batch)
    return total_time



def flatten(batch):
    return [item for sublist in batch for item in sublist]


# %%
from d2.simulator.optimizers.samples import sample_wlbllm_docs_upsample, batch_documents

N = 10
# up_sample_factor = 8
up_sample_factor = 16
elongate_factor = 2
# filter_threshold = 2 * K
# filter_threshold = 4 * K
filter_threshold = 64 * K
filter_ratio = 0.10
# filter_ratio = 0.50

# %%

GLOBAL_BATCH = batch_documents(
    sample_wlbllm_docs_upsample(
        size=10000,
        upsample_long_factor=up_sample_factor,
        filter_threshold=filter_threshold,
        filter_ratio=filter_ratio,
        elongate_factor=elongate_factor,
    ), max_ctx_length=total_seq_len
)

# %%
def flatten(batch):
    return [item for sublist in batch for item in sublist]

# %%


all_speedups = []
for i in range(N):
    batches = []
    for _ in range(batch_size * 2):
        batches.append(next(GLOBAL_BATCH))

    # 
    token_per_rank = total_seq_len * batch_size * 2 // total_ranks
    
    
    # Calculate wlbllm with balance
    # Lmax = total_seq_len * 2 * batch_size // wlb_dp
    Lmax = token_per_rank * wlb_cp
    all_docs = flatten(batches)
    all_docs.sort(reverse=True)
    new_batch = []
    for r in range(wlb_dp):
        new_batch.append([])
    
    def get_workload(micro_batch: list[int]) -> int:
        # TODO: Fix this get_workload function to calculate the `breakpoint` of a model.
        a = [ i / (64 * K) for i in micro_batch]
        return sum(i ** 2 + i for i in a)

    def get_length(micro_batch: list[int]) -> int:
        return sum(micro_batch)

    # Step 1: Pack the docs into the new batch.
    remained_docs = []
    for doc in all_docs:
        workloads = [get_workload(batch) for batch in new_batch]
        lengths = [get_length(batch) for batch in new_batch]
        min_workload_idx = workloads.index(min(workloads))
        min_length_idx = lengths.index(min(lengths))
        
        if lengths[min_workload_idx] + doc <= Lmax:
            new_batch[min_workload_idx].append(doc)
        else:
            if lengths[min_length_idx] + doc <= Lmax:
                new_batch[min_length_idx].append(doc)
            else:
                remained_docs.append(doc)
        pass

    # Step 2: Pack the remained docs, by workload.
    for doc in remained_docs:
        workloads = [get_workload(batch) for batch in new_batch]
        lengths = [get_length(batch) for batch in new_batch]
        min_workload_idx = workloads.index(min(workloads))
        new_batch[min_workload_idx].append(doc)

    # Step 3: Calculate the baseline attention time
    network_time = get_network_time(token_per_rank, wlb_cp)
    baseline_times = [
        (get_attn_time(batch) + get_mlp_time(batch)) / wlb_cp + network_time
        for batch in new_batch
    ]
    
    max_baseline_time = max(baseline_times)
    baseline_time = max_baseline_time
    
    # Calculate d2
    d2_attn_time = get_attn_time(all_docs) / total_ranks 
    mlp_time = linear_base_time * (token_per_rank / base_seq_len)
    d2_time = d2_attn_time + mlp_time

    # Calculate speedup
    speedup = - (d2_time - baseline_time) / baseline_time
    all_speedups.append(speedup)
    print(f"Sample {i}: Speedup: {speedup:.2%}; Baseline: {baseline_time:.2f} ms; D2: {d2_time:.2f} ms.")
    # print(f"Sample {i}: Batches: {batches}")

import numpy as np
import rich
rich.print(f"""
Experiment Config:
- dp_size: {dp_size}
- total_seq_len: {total_seq_len}
- up_sample_factor: {up_sample_factor}
- filter_threshold: {filter_threshold}
- filter_ratio: {filter_ratio}

Speedup: d2 vs baseline
- Average: {np.mean(all_speedups):.2%}
- Max: {max(all_speedups):.2%}
- Min: {min(all_speedups):.2%}
- Median: {np.median(all_speedups):.2%}
- Std: {np.std(all_speedups):.2%}
""")
import matplotlib.pyplot as plt
plt.hist(all_speedups, bins=20)
plt.show()

# %%

# %%

# %%
