# %%
import sim_pp_d2
import sim_pp_wlb
import matplotlib.pyplot as plt
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

# %%
# ---- WLBLLM & D2 Config ----
total_layers = 32
pp_size = num_stages = 4
nlayers = 32 // pp_size
d2_dpcp_size = 1
wlb_dp_size = 1
wlb_cp_size = 1

num_batches = 16

batches = [next(GLOBAL_BATCH) for _ in range(num_batches)]
print("batches", batches)


# %%[markdown]
# ## WLBLLM

# %%
# Run WLBLLM

new_batches = sim_pp_wlb.get_workload_balancing_batches_no_defer(batches)
wlbllm_events = sim_pp_wlb.run_iteration(new_batches, num_stages, nlayers=nlayers, wlb_cp=wlb_cp_size)
_ = sim_pp_wlb.plot_timeline(wlbllm_events, title_suffix=f" | NumBatches = {num_batches}, Stages = {num_stages}", granularity=1000)
plt.show()  # Display the figure

wlb_end_time = max([e[-1] for e in wlbllm_events])
print("WLBLLM End Time: ", wlb_end_time)



# %%
d2_batches = pair(batches)
d2_events = sim_pp_d2.simulate_d2_pipeline(
    d2_batches, pp_size=pp_size, 
    dpcp_size=d2_dpcp_size,
    nlayers=nlayers,
    verbose=False
)
sim_pp_d2.plot_d2_timeline(d2_events)

d2_end_time = max([e[-1] for e in d2_events])
print("D2 End Time: ", d2_end_time)

# %%
speedup =  wlb_end_time/d2_end_time
print(f"Speedup: {speedup:.2f}x")
# %%
