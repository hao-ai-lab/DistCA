# %%
# Implementing the varlen packing heuristic in WLBLLM paper
# Algorithm 1
# 
# Their algorithm is essentially do a "pre-reordering" of batch.

from collections import deque
from d2.simulator.optimizers.samples import sample_wlbllm_docs_upsample, batch_documents
import rich

K = 1024
N = 4 # microbatch per iteration
Lmax = 256 * K
outlier_threshold = 64 * K

GLOBAL_BATCH = batch_documents(
    sample_wlbllm_docs_upsample(
        size=10000,
        upsample_long_factor = 4,
        filter_threshold = 64 * K,
        filter_ratio = 0.50,
        elongate_factor = 1,
    ), max_ctx_length=(K * 128),
)

# Compute imbalance
Lmax = 256 * K
outlier_threshold = 64 * K
GLOBAL_BATCH = [
    [256 * K],
    [16 * K] * 16,
    [16 * K] * 16,
    [16 * K] * 16,

    [256 * K],
    [16 * K] * 16,
    [16 * K] * 16,
    [16 * K] * 16,

    [64 * K] + [16 * K] * ((256 - 64) // 16),
    [16 * K] * 16,
    [16 * K] * 16,
    [16 * K] * 16,

    [64 * K] + [16 * K] * ((256 - 64) // 16),
    [16 * K] * 16,
    [16 * K] * 16,
    [16 * K] * 16,
] 
# GLOBAL_BATCH = [
#     [256 * K],
#     [128 * K] * 2,
#     [64 * K] * 4,
#     [32 * K] * 8,    
# ] * 4

# # # Memory imbalance
# Lmax = 256 * K
# outlier_threshold = 30 * K
# GLOBAL_BATCH = [
#     [256 * K],
#     [32 * K] + [1 * K] * (256 - 32),
#     [32 * K] + [1 * K] * (256 - 32),
#     [1* K] * 256,
# ] + [
#     [32 * K] + [1 * K] * (256 - 32),
#     [1* K] * 256,
#     [1* K] * 256,
#     [1* K] * 256,
# ] * 16

GLOBAL_BATCH = iter(GLOBAL_BATCH)

def is_outlier(doc: int) -> bool:
    return doc >= outlier_threshold

def get_workload(micro_batch: list[int]) -> int:
    a = [ i / (64 * K) for i in micro_batch]
    # print(a)
    # attention + linear time
    return sum(i ** 2 + i for i in a)

def get_length(micro_batch: list[int]) -> int:
    return sum(micro_batch)


Q = deque()
B = []
remained_docs = []
# for batch in GLOBAL_BATCH:
for round_idx in range(7):
    rich.print(f"============= Round {round_idx} =============")
    batch = []
    try:
        for i in range(N):
            _batch = next(GLOBAL_BATCH)
            batch.extend(_batch)
    except StopIteration:
        pass
    pass
    doc_set = remained_docs
    remained_docs = []

    if len(batch) == 0 and len(doc_set) == 0:
        print("No more docs")
        break

    # Delay execution of outlier docs
    for doc in batch:
        if is_outlier(doc):
            Q.append(doc)
        else:
            doc_set.append(doc)
    pass
    print("batch", batch)
    print("Q", Q)

    
    # Pop outliers for the current batch.
    if len(Q) >= N:
        for i in range(N):
            doc_set.append(Q.popleft())
        pass
    
    
    # Sort the doc_set in descending order and do greedy packing.
    doc_set.sort(reverse=True) 
    print("doc_set", doc_set)

    # Start packing
    new_batch = []
    for _ in range(N):
        new_batch.append([])

    for doc in doc_set:
        # Get the microbatch with the least workload and length
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
    
    B.append(new_batch)
    print("new_batch", new_batch)
    workloads = [get_workload(batch) for batch in new_batch]
    lengths = [get_length(batch) for batch in new_batch]
    print("workloads", workloads)
    print("lengths", lengths)
    print("remained_docs", remained_docs)
    print("workload diff", max(workloads) - min(workloads))
    print("length diff", max(lengths) - min(lengths))

    pass
# %%
B
# %%
memory_diffs = []
compute_diffs = []
for i, batch in enumerate(B):
    memory_load = [get_length(microbatch) for microbatch in batch]
    compute_load = [get_workload([i / 1024 for i in microbatch]) for microbatch in batch]
    memory_diff = max(memory_load) - min(memory_load)
    memory_diffs.append(memory_diff)
    compute_diff = max(compute_load) - min(compute_load)
    compute_diffs.append(compute_diff)
    # print(f"Batch {i}: Memory load: {(memory_load)}, Compute load: {(compute_load)}, batch: {batch}")
    pass
print(f"Average memory diff: {sum(memory_diffs) / len(memory_diffs)}")
print(f"Max memory diff: {max(memory_diffs)}, Min memory diff: {min(memory_diffs)}")
print(f"Average compute diff: {sum(compute_diffs) / len(compute_diffs)}")
print(f"Max compute diff: {max(compute_diffs)}, Min compute diff: {min(compute_diffs)}")
# %%


