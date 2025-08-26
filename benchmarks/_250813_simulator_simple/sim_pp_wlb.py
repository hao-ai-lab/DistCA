# %%[markdown]
# # WLBLLM PP Simulator
# ## Few problems to solve
# - [ ] Calculating the linear time properly (divide cp or not?)
# - [ ] Add defer logic.
# - [ ] Make sure the stage's threshold is correct.

# %%
K = 1024

# %%
# ---- Timeline plot with per-microbatch colors & labels ----
import matplotlib.pyplot as plt

def plot_timeline(execution_log, title_suffix="", granularity=100):
    def _darken(rgb, factor=0.6) -> tuple:
        """Darken an RGB color by a factor."""
        r, g, b = rgb
        return (r * factor, g * factor, b * factor)

    if not execution_log:
        print("No log.")
        return

    # Blue shades for forward, green shades for backward
    blue_colors = [
        (0.300, 0.650, 0.900),  # lightest blue
        (0.200, 0.550, 0.800),  # lighter blue
    ]

    green_colors = [
        (0.350, 0.800, 0.350),  # lightest green
        (0.250, 0.700, 0.250),  # lighter green
    ]

    end_time = max(t1 for _, _, _, _, t1 in execution_log)
    busy = defaultdict(float)
    
    # Get the number of stages from the execution_log
    num_stages = max(s for _, s, _, _, _ in execution_log) + 1

    _, ax = plt.subplots(figsize=(11, 0.8 * num_stages + 2))
    yheight, ygap = 10, 6
    yticks, ylabels = [], []

    # group events by stage for easy drawing
    per_stage = defaultdict(list)
    for op, s, m, t0, t1 in execution_log:
        per_stage[s].append((op, m, t0, t1))
        busy[s] += (t1 - t0)

    # draw per stage
    for s in range(num_stages):
        y = s * (yheight + ygap)
        yticks.append(y + yheight / 2)
        ylabels.append(f"S{s}")

        for op, m, t0, t1 in sorted(per_stage[s], key=lambda x: x[2]):
            start_ms = t0
            dur_ms = (t1 - t0)

            if op == "F":
                color = blue_colors[m % 2]  # forward uses blue shades
            else:  # op == "B"
                color = green_colors[m % 2]  # backward uses green shades

            # one rectangle per (op, microbatch) segment so each can have its own color
            ax.broken_barh([(start_ms, dur_ms)], (y, yheight), facecolors=color, edgecolors="black", linewidth=0.4)

            # label microbatch id at bar center
            ax.text(start_ms + dur_ms / 2, y + yheight / 2, f"{m}",
                    ha="center", va="center", fontsize=8, color="white")

    total_ms = end_time
    utils = [100.0 * (busy[s] / end_time) if end_time > 0 else 0.0 for s in range(num_stages)]
    util_str = " • ".join([f"S{s}:{u:4.1f}%" for s, u in enumerate(utils)])

    # cosmetics
    ax.set_xlabel("Time (ms)")
    ax.set_yticks(yticks, ylabels)
    ax.set_title(f"WLBLLM PP: 1F1B Timeline {title_suffix}\n"
                 f"Total={total_ms:.1f} ms; Util {util_str}")
    ax.grid(True, axis="x", linestyle="--", alpha=0.35)

    # Set x-axis ticks to specified granularity
    import numpy as np
    max_time_ms = total_ms
    max_time_ms = (max_time_ms + granularity - 1) // granularity * granularity
    x_ticks = np.arange(0, max_time_ms + granularity, granularity)
    ax.set_xticks(x_ticks)
    
    # Add a vertical line to mark the final time
    ax.axvline(x=total_ms, color='red', linestyle='--', linewidth=1.5, alpha=0.7, label=f'Final Time: {total_ms:.1f} ms')

    # custom legend (2 blue shades for F, 2 green shades for B, plus final time marker)
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    
    legend_patches = []
    # Add microbatch color patches
    for i, c in enumerate(blue_colors):
        legend_patches.append(Patch(facecolor=c, edgecolor="black", label=f"mb%2={i} (F)"))
    for i, c in enumerate(green_colors):
        legend_patches.append(Patch(facecolor=c, edgecolor="black", label=f"mb%2={i} (B)"))
    
    # Add final time line to legend
    legend_patches.append(Line2D([0], [0], color='red', linestyle='--', linewidth=1.5, 
                                label=f'Final: {total_ms:.1f} ms'))
    
    ax.legend(handles=legend_patches, ncols=5, loc="upper right", fontsize=8, framealpha=0.9)

    plt.tight_layout()
    # Return the figure to allow further customization if needed
    return plt.gcf()


# %%

base_seq_len = K * 64
attn_base_time = 12.5020
# linear_base_time = (13.5 + 8.5) # mlp + qkvo
# linear_base_time = (13.5 + 8.5)  # mlp + qkvo
mlp_base_time = 13.5  # assume expert parallel
qkvo_base_time = 8.5
linear_base_time = (mlp_base_time + qkvo_base_time)  # mlp + qkvo
# linear_base_time = 0
# linear_base_time = 0

wlb_dp = 4
wlb_cp = 2
total_ranks = 8


def get_attn_time(batch) -> float:
    total_time = 0
    for l in batch:
        ratio = l / base_seq_len
        total_time += attn_base_time * (ratio ** 2)
    return total_time


def get_mlp_time(batch) -> float:
    total_time = 0
    for l in batch:
        ratio = l / base_seq_len
        total_time += linear_base_time * (ratio)
    return total_time


def get_network_time(token_per_batch, cp_degree) -> float:
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


def get_batch_time(batch: list[int], is_backward: bool = False, wlb_cp: int = 2, nlayers: int = 1) -> float:
    token_per_batch = sum(batch)
    network_time = get_network_time(token_per_batch, wlb_cp)
    attn_time = get_attn_time(batch)
    mlp_time = get_mlp_time(batch)
    if is_backward:
        attn_time *= 2.5
        mlp_time *= 2
    compute_time = (attn_time + mlp_time) / wlb_cp
    total_time = compute_time + network_time
    total_time *= nlayers
    return total_time


# %%
# 4-stage pipeline parallel (1F1B) SimPy model with a Matplotlib timeline.
# Forward = 130 ms, Backward = 2.5x (325 ms)
# One PriorityStore per stage: grad (prio=0) > act (prio=1)
# ---- Sim model (tiny + readable) ----


import simpy
from collections import defaultdict

K = 1024

def run_iteration(batches, num_stages=4, nlayers=1, threshold=None):
    """
    Run a pipeline parallel simulation with the given batches and number of stages.
    
    Args:
        batches: List of batches, each batch is a list of sequence lengths
        num_stages: Number of pipeline stages/devices
    
    Returns:
        execution_log: Execution log for timeline visualization
    """
    env = simpy.Environment()
    inboxes = [simpy.PriorityStore(env) for _ in range(num_stages)]
    done_counter = [0] * num_stages
    execution_log = []
    
    # Number of microbatches is the length of batches list
    num_microbatches = len(batches)
    # threshold = threshold or num_microbatches // 2

    # Create completion events for each stage
    completion_events = [env.event() for _ in range(num_stages)]

    # Function to check if a stage is complete and trigger its event
    def check_stage_completion(stage_idx):
        if done_counter[stage_idx] >= num_microbatches and not completion_events[stage_idx].triggered:
            completion_events[stage_idx].succeed()

    # Modify the stage function to signal completion
    def stage_with_signal(env, idx, inbox, next_inbox, prev_inbox, num_microbatches, done_counter, log_data, nlayers=1, ):
        """Main stage function to perform pipeline parallelism."""
        
        threshold = num_stages - idx
        active_batch_count = 0
        while done_counter[idx] < num_microbatches:
            # Get all batch from the inbox, 
            # take the batch that conform the 1F1B scheduling
            item = yield inbox.get()
            _, kind, m, batch = item


            # if active_batch_count <= 2:
            #     pass
            # else:
            #     # wait for a grad batch to arrive.
            #     active_batch_count -= 1
            #     pass

            if kind == "act":
                active_batch_count += 1
                pass
            
            
            if active_batch_count > threshold:
                print(f"[stage {idx}] active_batch_count > 4: {active_batch_count}")
                # need to wait for at least one backward batch to arrive.
                buffer = [item]
                while True:
                    item = yield inbox.get()
                    print(f"[stage {idx}] take item: {item}")
                    _, kind, m, batch = item
                    if kind == "act":
                        buffer.append(item)
                        continue
                    
                    assert kind == "grad"
                    print(f"[stage {idx}] take grad item: {item}")
                    active_batch_count -= 1
                    for t in buffer:
                        yield inbox.put(t)
                    print(f"[stage {idx}] put all items back to buffer: {buffer}")
                    break
                    
                assert item[1] == "grad"
                pass

            is_backward = (kind == "grad")

            if is_backward:
                active_batch_count -= 1
                t0 = env.now
                time_spent = get_batch_time(batch, is_backward=is_backward, nlayers=nlayers)
                yield env.timeout(time_spent)
                t1 = env.now
                log_data.append(("B", idx, m, t0, t1))
                print(f"[stage {idx}] B {m} {t0} {t1}")
                done_counter[idx] += 1
                # Check if stage is complete
                check_stage_completion(idx)
                if prev_inbox is not None:
                    yield prev_inbox.put((1, "grad", m, batch))

            else:  # "act" -> forward
                t0 = env.now
                time_spent = get_batch_time(batch, is_backward=is_backward, nlayers=nlayers)
                yield env.timeout(time_spent)
                t1 = env.now
                log_data.append(("F", idx, m, t0, t1))
                print(f"[stage {idx}] F {m} {t0} {t1}")
                if next_inbox is not None:
                    yield next_inbox.put((0, "act", m, batch))
                else:
                    active_batch_count -= 1
                    # last stage: immediately do backward for this microbatch
                    t0b = env.now
                    time_spent = get_batch_time(batch, is_backward=True, nlayers=nlayers)
                    yield env.timeout(time_spent)
                    t1b = env.now
                    log_data.append(("B", idx, m, t0b, t1b))
                    print(f"[stage {idx}] B {m} {t0b} {t1b}")
                    done_counter[idx] += 1
                    # Check if stage is complete
                    check_stage_completion(idx)
                    if prev_inbox is not None:
                        yield prev_inbox.put((1, "grad", m, batch))

    # Start stage processes
    for i in range(num_stages):
        next_inbox = inboxes[i + 1] if i < num_stages - 1 else None
        prev_inbox = inboxes[i - 1] if i > 0 else None
        env.process(stage_with_signal(env, i, inboxes[i], next_inbox, prev_inbox, num_microbatches, done_counter, execution_log, nlayers=nlayers))

    # Feed microbatches to stage 0 as activations
    def feeder():
        for m, batch in enumerate(batches):
            # (prio, kind, m=batch_id, batch)
            yield inboxes[0].put((0, "act", m, batch))

    env.process(feeder())

    # Wait for all completion events
    all_complete = simpy.AllOf(env, completion_events)
    env.run(until=all_complete)

    return execution_log



# %%
# ---- Quick demo ----
# Create 4 batches with the same sequence length
# batches = [[64 * K] for _ in range(num_batches)]
batches = [
    [128 * K] * 4,
    [256 * K] * 2,
    [512 * K] * 1,
    [256 * K] * 2,
    [128 * K] * 4,
    [256 * K] * 2,
    [512 * K] * 1,
    [256 * K] * 2,

]
num_batches = len(batches)
num_stages = 4
# threshold = num_batches // 2
# threshold = num_stages
execution_log = run_iteration(batches, num_stages)
_ = plot_timeline(execution_log, title_suffix=f" | M={num_batches}, S={num_stages}", granularity=1000)
plt.show()  # Display the figure
# %%
# ---- Actually using a distribution to try out ----
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
    ), max_ctx_length=K * 512
)
num_batches = 10
batches = [next(GLOBAL_BATCH) for _ in range(num_batches)]
flops = []
for batch in batches:
    flops.append(get_batch_time(batch, is_backward=False, nlayers=1))
import rich
rich.print(flops)

execution_log = run_iteration(batches, num_stages, nlayers=1)
_ = plot_timeline(execution_log, title_suffix=f" | NumBatches = {num_batches}, Stages = {num_stages}", granularity=1000)
plt.show()  # Display the figure

# %%
# ---- Adding workload balancing across the batches ----
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
    ), max_ctx_length=K * 512
)
num_batches = 16
num_stages = 4  
# threshold = num_stages
batches = [next(GLOBAL_BATCH) for _ in range(num_batches)]

def get_workload_balancing_batches_no_defer(batches: list[list[int]]) -> list[list[int]]:
    
    def get_workload(micro_batch: list[int]) -> int:
        # TODO: Fix this get_workload function to calculate the `breakpoint` of a model.
        a = [ i / (64 * K) for i in micro_batch]
        return sum(i ** 2 + i for i in a)
    
    def get_length(micro_batch: list[int]) -> int:
        return sum(micro_batch)
    
    
    Lmax = max(
        get_length(batch) for batch in batches
    )

    new_batch = []
    for r in range(len(batches)):
        new_batch.append([])

    # Step 1: Pack the docs into the new batch.
    all_docs = [doc for batch in batches for doc in batch]
    all_docs.sort(reverse=True)

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
    
    # Step 2: Pack the remained docs, directly by workload, no defer to next stage.
    for doc in remained_docs:
        workloads = [get_workload(batch) for batch in new_batch]
        lengths = [get_length(batch) for batch in new_batch]
        min_workload_idx = workloads.index(min(workloads))
        new_batch[min_workload_idx].append(doc)
    
    return new_batch

new_batches = get_workload_balancing_batches_no_defer(batches)
execution_log = run_iteration(new_batches, num_stages, nlayers=1)
_ = plot_timeline(execution_log, title_suffix=f" | NumBatches = {num_batches}, Stages = {num_stages}", granularity=1000)
plt.show()  # Display the figure

# %%
