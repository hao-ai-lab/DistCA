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
total_seq_len = K * 128
# total_seq_len = K * 256
# total_seq_len = K * 512

def get_attn_time(batch):
    total_time = 0
    for l in batch:
        ratio = l / base_seq_len
        total_time += attn_base_time * (ratio ** 2)
    return total_time

def flatten(batch):
    return [item for sublist in batch for item in sublist]


from d2.simulator.optimizers.samples import sample_wlbllm_docs_upsample, batch_documents

N = 100
# up_sample_factor = 8
up_sample_factor = 16
elongate_factor = 1
# filter_threshold = 2 * K
# filter_threshold = 4 * K
filter_threshold = 64 * K
filter_ratio = 0.10
# filter_ratio = 0.50
dp_size = 4


GLOBAL_BATCH = batch_documents(
    sample_wlbllm_docs_upsample(
        size=20000,
        upsample_long_factor=up_sample_factor,
        filter_threshold=filter_threshold,
        filter_ratio=filter_ratio,
        elongate_factor=elongate_factor,
    ), max_ctx_length=total_seq_len
)

# def create_global_batch():
#     GLOBAL_BATCH_1 = batch_documents(
#         sample_wlbllm_docs_upsample(
#             size=20000,
#             # upsample_long_factor=up_sample_factor,
#             upsample_long_factor=1,
#             # filter_threshold=10000,
#             filter_threshold=8 * K,
#             # filter_threshold=filter_threshold,
#             # filter_ratio=filter_ratio,
#             filter_ratio=0.5,
#         ), max_ctx_length=total_seq_len
#     )
#     GLOBAL_BATCH_1 = list(GLOBAL_BATCH_1)
#     GLOBAL_BATCH_1 = GLOBAL_BATCH_1[: -2] # remove the last few samples

#     GLOBAL_BATCH_2 = batch_documents(
#         [total_seq_len] * 100000,
#         max_ctx_length=total_seq_len,
#     )

#     while True:
#         # Yield a long sequence
#         yield next(GLOBAL_BATCH_1)
#         yield next(GLOBAL_BATCH_1)
#         # Then yield a long sequence
#         yield next(GLOBAL_BATCH_2)
#         yield next(GLOBAL_BATCH_2)

# GLOBAL_BATCH = create_global_batch()


all_speedups = []
for i in range(N):
    batches = []
    for _ in range(dp_size * megatron_batch_size):
        batches.append(next(GLOBAL_BATCH))

    # 
    mlp_time = linear_base_time * (total_seq_len / base_seq_len)
    
    # Calculate Baseline
    dp_batches = [
        flatten(batches[rid] + batches[rid + dp_size])
        for rid in range(dp_size)
    ]
    baseline_attn_time = [
        get_attn_time(batch)
        for batch in dp_batches
    ]
    max_baseline_attn_time = max(baseline_attn_time)
    baseline_time = max_baseline_attn_time + mlp_time
    
    # Calculate d2
    d2_attn_time = get_attn_time(flatten(batches)) / dp_size 
    d2_time = d2_attn_time + mlp_time

    # Calculate speedup
    speedup = - (d2_time - baseline_time) / baseline_time
    all_speedups.append(speedup)
    print(f"Sample {i}: Speedup: {speedup:.2%}; Baseline: {baseline_time:.2f} ms; D2: {d2_time:.2f} ms. Attn ratio: {(max_baseline_attn_time / baseline_time):.2%}. Baseline attention time: {max_baseline_attn_time:.2f} ms, D2 attention time: {d2_attn_time:.2f} ms")
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
