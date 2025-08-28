# %%[markdown]
# # WLBLLM PP Simulator
# ## Few problems to solve
# - [ ] Calculating the linear time properly (divide cp or not?)
# - [ ] Add defer logic.
# - [ ] Make sure the stage's threshold is correct.

# %%
import simpy
from collections import defaultdict
from queue import deque
K = 1024

# %%
# ---- Timeline plot with per-microbatch colors & labels ----
import matplotlib.pyplot as plt

def plot_timeline(
    execution_log, title_suffix="", granularity=100,
    show_microbatch_duration = True,
    save_path: str = None,
):
    """Generate a timeline plot from the WLBLLM execution log."""
    
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

            if show_microbatch_duration:
                
                # Add duration text rotated 90 degrees in the center
                ax.text(start_ms + dur_ms / 2, y + yheight / 2, f"{dur_ms:.1f}",
                        ha="center", va="center", fontsize=8, color="white", rotation=90)
                # Add microbatch id below the box
                ax.text(start_ms + dur_ms / 2, y - 1, f"{m}",
                        ha="center", va="top", fontsize=8, color="black")
                
            else:
                ax.text(start_ms + dur_ms / 2, y + yheight / 2, f"{m}",
                        ha="center", va="center", fontsize=8, color="white")
                pass

    total_ms = end_time
    utils = [100.0 * (busy[s] / end_time) if end_time > 0 else 0.0 for s in range(num_stages)]
    util_str = " • ".join([f"S{s}:{u:4.1f}%" for s, u in enumerate(utils)])

    # cosmetics
    ax.set_xlabel("Time (ms)")
    ax.set_yticks(yticks, ylabels)
    ax.set_title(f"WLBLLM PP: 1F1B Timeline {title_suffix}\n"
                 f"Total={total_ms:.1f} ms; "
                #  f"Util {util_str}"
                 )
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
    if save_path is not None:
        plt.savefig(save_path, dpi=300)
    
    return plt.gcf()


# %%

from timemodule import get_wlbllm_batch_time
# %%
# 4-stage pipeline parallel (1F1B) SimPy model with a Matplotlib timeline.
# Forward = 130 ms, Backward = 2.5x (325 ms)
# One PriorityStore per stage: grad (prio=0) > act (prio=1)
# ---- Sim model (tiny + readable) ----



def run_iteration(batches, num_stages=4, nlayers=1, threshold=None, wlb_cp=2, verbose=False):
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
    def stage_with_signal(env, idx, inbox, next_inbox, prev_inbox, num_microbatches, done_counter, log_data, nlayers=1, verbose=False):
        """Main stage function to perform pipeline parallelism."""
        
        def log(msg):
            if verbose:
                print(msg)
        
        # Construct 1f1b constraint
        FORWARD_PASS, BACKWARD_PASS = False, True
        def get_1f1b_constraint(pp_size: int, rank: int = idx, num_batches: int = num_microbatches):
            forwarded_batches = 0
            backwarded_batches = 0
            action: list[bool] = []
            log: list[str] = []
            # Phase 1: Forward
            for i in range(pp_size - rank):
                forwarded_batches += 1
                action.append(FORWARD_PASS)
                log.append(f'forward[{forwarded_batches}]')
                pass

            # Phase 2: Backward - Forward
            while forwarded_batches < num_batches:
                # backward
                backwarded_batches += 1
                action.append(BACKWARD_PASS)
                log.append(f'backward[{backwarded_batches}]')
            
                # forward
                forwarded_batches += 1
                action.append(FORWARD_PASS)
                log.append(f'forward[{forwarded_batches}]')
            
            # Phase 3: Backward only
            while backwarded_batches < num_batches:
                backwarded_batches += 1
                action.append(BACKWARD_PASS)
                log.append(f'backward[{backwarded_batches}]')
            
            # import rich
            # rich.print(f"Rank{rank}")
            # rich.print(action)
            # return action
            
            return deque(action)

        _1f1b_constraint: deque = get_1f1b_constraint(
            pp_size=num_stages, rank=idx, 
            num_batches=num_microbatches
        )
        
        while done_counter[idx] < num_microbatches:

            if len(_1f1b_constraint) == 0:
                log(f"[stage {idx}] 1f1b constraint is empty. we actually should not be here.")
                break

            action = _1f1b_constraint.popleft()
            log(f"[stage {idx}] expect next action: {"act" if action == FORWARD_PASS else "grad"}")
            expected_action = "act" if action == FORWARD_PASS else "grad"


            # ------------------------------
            # Get the item from the inbox
            # ------------------------------
            buffer = []
            while True:
                # Loop until we get the expected action
                log(f"[stage {idx}] waiting for {expected_action} item")
                item = yield inbox.get()
                _, kind, m, batch = item
                if kind != expected_action:
                    log(f"[stage {idx}] get item: {item}, but expected {expected_action}")
                    buffer.append(item)
                    continue
                assert kind == expected_action
                log(f"[stage {idx}] take {expected_action} item: {item}")
                
                # Put all other items back to buffer
                for t in buffer:
                    yield inbox.put(t)
                    log(f"[stage {idx}] put item back to inbox: {t}")
                break
            # assert action == expected_action, f"action: {action}, expected_action: {expected_action}"
            assert item[1] == expected_action, f"item: {item}, expected_action: {expected_action}"
            # successfully get the item.


            is_backward = (action == BACKWARD_PASS)

            if is_backward:
                t0 = env.now
                time_spent = get_wlbllm_batch_time(batch, is_backward=is_backward, nlayers=nlayers, wlb_cp=wlb_cp)
                yield env.timeout(time_spent)
                t1 = env.now
                log_data.append(("B", idx, m, t0, t1))
                log(f"[stage {idx}] B {m} {t0:.2f} {t1:.2f}")
                done_counter[idx] += 1
                # Check if stage is complete
                check_stage_completion(idx)
                if prev_inbox is not None:
                    yield prev_inbox.put((1, "grad", m, batch))
                    log(f"[stage {idx}] put grad item to prev inbox")

            else:  # "act" -> forward
                t0 = env.now
                time_spent = get_wlbllm_batch_time(batch, is_backward=is_backward, nlayers=nlayers, wlb_cp=wlb_cp)
                yield env.timeout(time_spent)
                t1 = env.now
                log_data.append(("F", idx, m, t0, t1))
                log(f"[stage {idx}] F {m} {t0:.2f} {t1:.2f}")
                if next_inbox is not None:
                    yield next_inbox.put((0, "act", m, batch))
                    log(f"[stage {idx}] put act item to next inbox")
                else:
                    assert _1f1b_constraint.popleft() == BACKWARD_PASS, f"action: {_1f1b_constraint.popleft()}, expected_action: {BACKWARD_PASS}"
                    # last stage: immediately do backward for this microbatch
                    t0b = env.now
                    time_spent = get_wlbllm_batch_time(batch, is_backward=True, nlayers=nlayers, wlb_cp=wlb_cp)
                    yield env.timeout(time_spent)
                    t1b = env.now
                    log_data.append(("B", idx, m, t0b, t1b))
                    log(f"[stage {idx}] B {m} {t0b:.2f} {t1b:.2f}")
                    done_counter[idx] += 1
                    # Check if stage is complete
                    check_stage_completion(idx)
                    if prev_inbox is not None:
                        yield prev_inbox.put((1, "grad", m, batch))
                        log(f"[stage {idx}] put grad item to prev inbox")

    # Start stage processes
    for i in range(num_stages):
        next_inbox = inboxes[i + 1] if i < num_stages - 1 else None
        prev_inbox = inboxes[i - 1] if i > 0 else None
        env.process(stage_with_signal(env, i, inboxes[i], next_inbox, prev_inbox, num_microbatches, done_counter, execution_log, nlayers=nlayers, verbose=verbose))

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
def get_workload_balancing_batches_no_defer(
    batches: list[list[int]],
    num_buckets: int = 8,
    Lmax: int = None,
) -> list[list[int]]:
    """
    Get the workload balancing batches without deferring to the next stage.
    """
    
    def get_workload(micro_batch: list[int]) -> int:
        # TODO: Fix this get_workload function to calculate the `breakpoint` of a model.
        a = [ i / (64 * K) for i in micro_batch]
        return sum(i ** 2 + i for i in a)
    
    def get_length(micro_batch: list[int]) -> int:
        return sum(micro_batch)
    
    
    Lmax = Lmax or max(
        get_length(batch) for batch in batches
    )

    new_batch = []
    for r in range(num_buckets):
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

# %%
# ---- Quick demo ----
# Create 4 batches with the same sequence length
# batches = [[64 * K] for _ in range(num_batches)]
def quick_demo():
    # batches = [
    #     [128 * K] * 4,
    #     [256 * K] * 2,
    #     [512 * K] * 1,
    #     [256 * K] * 2,
    #     [128 * K] * 4,
    #     [256 * K] * 2,
    #     [512 * K] * 1,
    #     [256 * K] * 2,

    # ]
    batches = [
        [64*K] * 2,
        [64*K] * 2,
        [128*K] * 1,
        [64*K] * 2,
        [64*K] * 2,
        [64*K] * 2,
        [64*K] * 2,
        [64*K] * 2,

    ]
    num_batches = len(batches)
    num_stages = 4
    # threshold = num_batches // 2
    # threshold = num_stages
    execution_log = run_iteration(batches, num_stages, verbose=False)
    _ = plot_timeline(execution_log, title_suffix=f" | M={num_batches}, S={num_stages}", granularity=1000)
    plt.show()  # Display the figure
    return


# quick_demo()
# %%

# %%
def actual_demo_with_distribution():
    # ---- Actually using a distribution to try out ----
    from d2.simulator.optimizers.samples import (
        sample_wlbllm_docs_upsample, 
        batch_documents,
    )

    num_stages = 4

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
        flops.append(get_wlbllm_batch_time(batch, is_backward=False, nlayers=1))
    import rich
    rich.print(flops)

    execution_log = run_iteration(batches, num_stages, nlayers=1)
    _ = plot_timeline(execution_log, title_suffix=f" | NumBatches = {num_batches}, Stages = {num_stages}", granularity=1000)
    plt.show()  # Display the figure



# %%
# ---- Adding workload balancing across the batches ----
def actual_demo_with_workload_balancing():
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

    new_batches = get_workload_balancing_batches_no_defer(batches)
    execution_log = run_iteration(new_batches, num_stages, nlayers=1)
    _ = plot_timeline(execution_log, title_suffix=f" | NumBatches = {num_batches}, Stages = {num_stages}", granularity=1000, show_microbatch_duration=True)
    plt.show()  # Display the figure

# actual_demo_with_workload_balancing()

# %%
