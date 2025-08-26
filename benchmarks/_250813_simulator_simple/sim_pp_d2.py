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
    ax.set_title(f"D2 PP: 1F1B Timeline {title_suffix}\n"
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
mlp_base_time = 13.5  # assume expert parallel
qkvo_base_time = 8.5
linear_base_time = (mlp_base_time + qkvo_base_time)  # mlp + qkvo

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

def get_qkv_time(batch) -> float:
    total_time = 0
    for l in batch:
        ratio = l / base_seq_len
        total_time += qkvo_base_time * (ratio)
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

def make_dynamic_alltoall(env, comm_time_fn=None):
    """
    Dynamic reusable all-to-all with value passing.

    API:
      register(worker_id)                 # call once when a stage starts
      deregister(worker_id)               # call once when a stage is done
      ev = launch_comm(worker_id, payload)  # non-blocking; 'data = yield ev'

    Round semantics:
      - The *first arrival* of a round snapshots the set of currently-active workers.
      - Only that set is required to arrive for this round.
      - If a required worker deregisters before arriving, it is dropped.
      - Event succeeds with a dict of payloads from the required subset.
    """
    active = set()            # currently active workers
    gen = 0

    # per-generation state
    arrivals = {}             # wid -> payload
    required = None           # set of wids required this gen (snapshot at first arrival)
    release_event = env.event()
    releasing = False

    def _reset_generation():
        nonlocal arrivals, required, release_event, releasing, gen
        gen += 1
        arrivals = {}
        required = None
        release_event = env.event()
        releasing = False

    def register(worker_id: int):
        active.add(worker_id)

    def deregister(worker_id: int):
        nonlocal required, releasing
        active.discard(worker_id)
        # If this worker was required but hasn't arrived yet, drop it.
        if required is not None and worker_id in required and not release_event.triggered:
            required.discard(worker_id)
            if not releasing and required.issubset(arrivals.keys()):
                _schedule_release()

    def _schedule_release():
        nonlocal releasing
        if releasing or release_event.triggered:
            return
        releasing = True
        snap = {wid: arrivals[wid] for wid in required if wid in arrivals}

        def _release_proc():
            delay = float(comm_time_fn(snap)) if comm_time_fn else 0.0
            if delay > 0:
                yield env.timeout(delay)
            release_event.succeed(snap)
        env.process(_release_proc())

    def launch_comm(worker_id: int, payload):
        nonlocal required
        # First arrival sets required set to currently active snapshot
        if required is None:
            required = set(active) if active else {worker_id}  # at least self

        # Record arrival
        if worker_id not in arrivals:
            arrivals[worker_id] = payload

        # If everyone required has arrived, release
        if required.issubset(arrivals.keys()):
            _schedule_release()

        return release_event

    return register, deregister, launch_comm

def run_iteration(batches, num_stages=4, nlayers=2):
    """
    Same structure as your version, but:
      - dynamic barrier (register/deregister)
      - each stage increments its own done counter when it finishes a grad send
        for a microbatch (so later stages can exit and deregister cleanly)
    """
    env = simpy.Environment()
    inboxes = [simpy.PriorityStore(env) for _ in range(num_stages)]
    done_counter = [0] * num_stages
    completion_events = [env.event() for _ in range(num_stages)]
    num_microbatches = len(batches)

    # Build per-env dynamic barrier
    # (Put your delay model in comm_time_fn if you want)
    register, deregister, launch_comm = make_dynamic_alltoall(env, comm_time_fn=None)

    def check_stage_completion(stage_idx):
        if done_counter[stage_idx] >= num_microbatches and not completion_events[stage_idx].triggered:
            completion_events[stage_idx].succeed()

    def flatten_payloads(payloads_dict):
        out = []
        for v in payloads_dict.values():
            out.extend(v)
        return out

    def stage(env, idx, inbox, next_inbox, prev_inbox, num_microbatches, nlayers):
        # Enter pipeline (become eligible for comm rounds)
        register(idx)
        try:
            while done_counter[idx] < num_microbatches:
                prio, kind, m, batch = yield inbox.get()
                ping, pong = batch

                is_backward = (kind == "grad")
                is_forward  = not is_backward
                is_last     = (idx == num_stages - 1)

                # ---------------- FORWARD (two microbatches) ----------------
                if is_forward:
                    # L1: qkv[0] -> comm -> qkv[1] -> comm -> attn[0] -> comm -> attn[1] -> comm
                    # qkv[0]
                    yield env.timeout(get_qkv_time(ping))
                    ev_m2a0 = launch_comm(idx, ping)

                    # qkv[1]
                    yield env.timeout(get_qkv_time(pong))
                    ev_m2a1 = launch_comm(idx, pong)

                    # attn[0] -> a2m[0]
                    payloads = yield ev_m2a0
                    attn_0_batch = flatten_payloads(payloads)
                    yield env.timeout(get_attn_time(attn_0_batch) / total_ranks)
                    ev_a2m0 = launch_comm(idx, ping)

                    # attn[1] -> a2m[1]
                    payloads = yield ev_m2a1
                    attn_1_batch = flatten_payloads(payloads)
                    yield env.timeout(get_attn_time(attn_1_batch) / total_ranks)
                    ev_a2m1 = launch_comm(idx, pong)

                    # Middle layers 1..L-2
                    for l in range(1, nlayers - 1):
                        # mlp+qkv[0] -> comm
                        yield env.timeout(get_mlp_time(ping))
                        ev_m2a0 = launch_comm(idx, ping)

                        # mlp+qkv[1] -> comm
                        yield env.timeout(get_mlp_time(pong))
                        ev_m2a1 = launch_comm(idx, pong)

                        # attn[0] -> comm
                        payloads = yield ev_m2a0
                        yield env.timeout(get_attn_time(flatten_payloads(payloads)) / total_ranks)
                        ev_a2m0 = launch_comm(idx, ping)

                        # attn[1] -> comm
                        payloads = yield ev_m2a1
                        yield env.timeout(get_attn_time(flatten_payloads(payloads)) / total_ranks)
                        ev_a2m1 = launch_comm(idx, pong)

                    # Last layer mlp only
                    yield env.timeout(get_mlp_time(ping))
                    yield env.timeout(get_mlp_time(pong))

                # Forward → next stage
                if is_forward and next_inbox is not None:
                    yield next_inbox.put((1, "act", m, batch))

                # ---------------- BACKWARD trigger ----------------
                if is_backward or (is_forward and next_inbox is None):
                    # Simple mirrored backward (kept from your version)
                    # mlp_o[1] -> comm; mlp_o[0] -> comm
                    yield env.timeout(get_mlp_time(pong) * 2)
                    ev_m2a1 = launch_comm(idx, ping)

                    yield env.timeout(get_mlp_time(ping) * 2)
                    ev_m2a0 = launch_comm(idx, pong)

                    # attn_b[1] -> comm; attn_b[0] -> comm
                    payloads = yield ev_m2a1
                    yield env.timeout(get_attn_time(flatten_payloads(payloads)) * 2.5 / total_ranks)
                    ev_a2m1 = launch_comm(idx, pong)

                    payloads = yield ev_m2a0
                    yield env.timeout(get_attn_time(flatten_payloads(payloads)) * 2.5 / total_ranks)
                    ev_a2m0 = launch_comm(idx, ping)

                    # middle layers
                    for l in range(1, nlayers - 1):
                        yield env.timeout(get_mlp_time(pong) * 2)
                        ev_m2a1 = launch_comm(idx, ping)

                        yield env.timeout(get_mlp_time(ping) * 2)
                        ev_m2a0 = launch_comm(idx, pong)

                        payloads = yield ev_m2a1
                        yield env.timeout(get_attn_time(flatten_payloads(payloads)) * 2.5 / total_ranks)
                        ev_a2m1 = launch_comm(idx, pong)

                        payloads = yield ev_m2a0
                        yield env.timeout(get_attn_time(flatten_payloads(payloads)) * 2.5 / total_ranks)
                        ev_a2m0 = launch_comm(idx, ping)

                    # first layer qkv_b
                    yield env.timeout(get_qkv_time(pong) * 1.7)
                    yield env.timeout(get_qkv_time(ping) * 1.7)

                # Backward → previous stage (or mark done if first stage)
                if (is_backward or (is_forward and next_inbox is None)) and (inboxes and idx > 0):
                    yield inboxes[idx - 1].put((0, "grad", m, batch))

                    # mark microbatch done for this stage once we sent its grad upstream
                    done_counter[idx] += 1
                    if done_counter[idx] >= num_microbatches:
                        check_stage_completion(idx)

                elif (is_backward or (is_forward and next_inbox is None)) and idx == 0:
                    # Stage 0 finishes the mb completely
                    done_counter[idx] += 1
                    if done_counter[idx] >= num_microbatches:
                        check_stage_completion(idx)

        finally:
            # Leave pipeline: future comm rounds no longer wait for this stage
            deregister(idx)

    # Spin up stages
    for i in range(num_stages):
        next_inbox = inboxes[i + 1] if i < num_stages - 1 else None
        prev_inbox = inboxes[i - 1] if i > 0 else None
        env.process(stage(env, i, inboxes[i], next_inbox, prev_inbox, num_microbatches, nlayers))

    # Feed microbatches (two per item, as in your code)
    def feeder():
        for m, batch in enumerate(batches):
            yield inboxes[0].put((1, "act", m, batch))
    env.process(feeder())

    all_complete = simpy.events.AllOf(env, completion_events)
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
]
num_batches = len(batches)
num_stages = 4
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
