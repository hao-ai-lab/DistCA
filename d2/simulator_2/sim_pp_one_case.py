# %%
import sim_pp_d2
import sim_pp_wlb
import matplotlib.pyplot as plt

import importlib

def reload_packages():
    """Reload the simulator packages."""
    importlib.reload(sim_pp_d2)
    importlib.reload(sim_pp_wlb)

reload_packages()

# %%[markdown]
# # Simulate the speedups of D2 vs WLBLLM. PP.

# %%[markdown]
# ## Prepare data
# %%
K = 1024

from d2.simulator.optimizers.samples import (
    sample_wlbllm_docs_upsample, 
    batch_documents,
)

GLOBAL_BATCH = batch_documents(
    sample_wlbllm_docs_upsample(
        size=10000,
        filter_threshold=64 * K,
        filter_ratio=0.90,
        upsample_long_factor=2,
        elongate_factor=4,
    ), max_ctx_length=K * 256
)

def flatten(lst):
    """Recursively flatten nested lists into a single list."""
    result = []
    for item in lst:
        if isinstance(item, list):
            result.extend(flatten(item))
        else:
            result.append(item)
    return result

def pair(lst):
    """Pair up elements in a list."""
    return [
        [lst[i], lst[i + 1]]
        for i in range(0, len(lst), 2)
    ]


num_batches = 16
batches = [next(GLOBAL_BATCH) for _ in range(num_batches)]

# %%
batches
print(f"{len(batches) = }")
# %%
# ---- WLBLLM & D2 Config ----
total_layers = 32
pp_size = num_stages = 4
nlayers = 32 // pp_size
d2_dpcp_size = 1

# %%[markdown]
# ## WLBLLM
# %%
# Run WLBLLM

reports = {}

# for wlb_dp_size in [1,2,4,8]:
#     for wlb_cp_size in [1,2,4,8]:
#         for pp_size in [1,2,4,8]:


_batches = [
    [64 * K] * 2,
] * (8 - 1)

disable_wlb_reorder = True


for idx in range(8):
    batches = _batches.copy()
    batches.insert(idx, [128 * K])
    # print(f"Insert [128 * K] at idx = {idx}: {batches}")

    for (wlb_dp_size, wlb_cp_size, pp_size) in [
        (1, 1, 8),
        # (1, 1, 4),
        # (1, 1, 2),
    ]:
        # if wlb_dp_size * wlb_cp_size * pp_size != 8:
        #     continue

        DP = wlb_dp_size
        CP = wlb_cp_size
        PP = pp_size
        factor = 4

        max_time = 0
        if disable_wlb_reorder:
            new_batches = batches
        else:
            num_buckets = min(DP * PP * factor, len(batches)) # to make PP enter steady state
            new_batches = sim_pp_wlb.get_workload_balancing_batches_no_defer(
                batches, num_buckets=num_buckets,
            )

        # print(f"{DP = } {CP = } {PP = }: {len(new_batches) = }, {sum(sum(b) for b in new_batches)}")
        for dp_rank in range(DP):
            dp_batch = new_batches[dp_rank * (PP * factor): (dp_rank + 1) * (PP * factor)]
            # print(f"DP Rank = {dp_rank}: {len(dp_batch) = }, dp_batch = {dp_batch}")
            # print(dp_batch)
            wlbllm_events = sim_pp_wlb.run_iteration(
                dp_batch, 
                PP, 
                nlayers=nlayers, 
                wlb_cp=wlb_cp_size,
            )

            _ = sim_pp_wlb.plot_timeline(
                wlbllm_events,
                title_suffix=(
                    f" | NumBatches = {num_batches}, DP = {dp_rank} / {wlb_dp_size}, "
                    f"PP = {pp_size}, CP = {wlb_cp_size}"
                ),
                granularity=10000, 
                save_path="wlbllm_timeline.png"
            )
            # plt.show()
            wlb_end_time = max([e[-1] for e in wlbllm_events])
            max_time = max(max_time, wlb_end_time)
            # print(f"WLBLLM End Time DP Rank = {dp_rank}: {wlb_end_time}ms")
        # print(f"WLBLLM {DP = } {CP = } {PP = }: {max_time}ms")
        reports[(idx, DP, CP, PP)] = max_time


for (idx, DP, CP, PP), max_time in reports.items():
    print(f"{idx = } {DP = } {CP = } {PP = }: {max_time:.2f} ms")


# %%[markdown]
# ## D2
# %%
batches = [
    [64 * K] * 2,
] * 7
batches.insert(0, [128 * K])

d2_batches = pair(batches)
d2_events = sim_pp_d2.simulate_d2_pipeline(
    d2_batches, pp_size=pp_size, 
    dpcp_size=d2_dpcp_size,
    nlayers=nlayers,
    verbose=False
)
sim_pp_d2.plot_d2_timeline(d2_events, save_path="d2_timeline.png")
plt.show()

d2_end_time = max([e[-1] for e in d2_events])
print("D2 End Time: ", d2_end_time)

# %%
speedup =  wlb_end_time/d2_end_time
print(f"Speedup: {speedup:.2f}x")
# %%
