# %%
import sim_pp_d2
import sim_pp_wlb
import matplotlib.pyplot as plt

import importlib

def reload_packages():
    """Reload the simulator packages."""
    importlib.reload(sim_pp_d2)
    importlib.reload(sim_pp_wlb)

from d2.simulator.optimizers.samples import (
    sample_wlbllm_docs_upsample, 
    batch_documents,
)

K = 1024

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

set_of_batches = []
# for i in range(128):
#     set_of_batches.append(next(GLOBAL_BATCH))
for i in range(16):
    set_of_batches.append(next(GLOBAL_BATCH))
set_of_batches
# %%

def chunk(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]
# %%
# Run D2
for dpcp in [1,2,4,8]:
    for pp in [1,2,4,8]:
        if dpcp * pp != 8: continue

        config = dict(mode="d2", dpcp=dpcp, pp=pp)
        num_batches = dpcp * pp * 2
        nlayers = 32 // pp
        d2_dpcp_size = dpcp
        num_stages = pp

        current_time = 0
        for sid, batches in enumerate(chunk(set_of_batches, num_batches)):
            new_batches = pair(batches)
            d2_events = sim_pp_d2.simulate_d2_pipeline(
                new_batches, pp_size=pp, 
                dpcp_size=d2_dpcp_size,
                nlayers=nlayers,
                verbose=False
            )
            sim_pp_d2.plot_d2_timeline(d2_events, save_path=f"d2.dpcp{dpcp}.pp{pp}.sid{sid}.png")
            d2_end_time = max([e[-1] for e in d2_events])
            current_time += d2_end_time

        print(f"D2 dpcp{dpcp}.pp{pp}: duration = {current_time:.2f} ms")


# %%

# Run WLBLLM
for dp in [1,2,4,8]:
    for pp in [1,2,4,8]:
        for cp in [1,2,4,8]:
            if dp * pp * cp != 8: continue

            config = dict(mode="wlbllm", dp=dp, pp=pp, cp=cp)
            num_batches = dp * pp
            nlayers = 32 // pp
            wlb_cp_size = cp
            num_stages = pp

            current_time = 0
            for sid, batches in enumerate(chunk(set_of_batches, num_batches)):

                new_batches = sim_pp_wlb.get_workload_balancing_batches_no_defer(
                    batches, num_buckets=dp * pp
                )
                
                wlbllm_one_batch_end_times = []
                for dp_rank in range(dp):
                    dp_batches = new_batches[dp_rank::dp]
                    wlbllm_events = sim_pp_wlb.run_iteration(
                        new_batches,
                        num_stages=num_stages, 
                        nlayers=nlayers, 
                        wlb_cp=wlb_cp_size
                    )
                    wlbllm_one_batch_end_time = max([e[-1] for e in wlbllm_events])
                    wlbllm_one_batch_end_times.append(wlbllm_one_batch_end_time)

                    _ = sim_pp_wlb.plot_timeline(
                        wlbllm_events, 
                        title_suffix=f" | NumBatches = {num_batches}, Stages = {num_stages}", 
                        granularity=10000, 
                        save_path=f"wlbllm.dp{dp}.pp{pp}.cp{cp}.sid{sid}.dp_rank{dp_rank}.png"
                    )
                current_time += max(wlbllm_one_batch_end_times)

            print(f"WLBLLM dp{dp}.pp{pp}.cp{cp}: duration = {current_time:.2f} ms")
            


# %%
