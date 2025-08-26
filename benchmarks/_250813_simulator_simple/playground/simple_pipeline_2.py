# %%
import simpy
from collections import defaultdict
import matplotlib.pyplot as plt

# %%


# %%[markdown]
# # D2 PP Simulator
# ## Few problems to solve
# - [ ] Calculating the linear time properly (divide cp or not?)
# - [ ] Add defer logic.
# - [ ] Correct the time functions
# - [ ] Add logics for the last layer / first layer.
# - [ ] Implement the pipeline "init" logic where the other ranks are not activated yet.

# %%
K = 1024

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

# Global env for logging
env = None

def get_attn_time(batch) -> float:
    total_time = 0
    for l in batch:
        ratio = l / base_seq_len
        total_time += attn_base_time * (ratio ** 2)
    log(f"Attention time for batch {batch}: {total_time:.2f}")
    return total_time


def get_qkv_time(batch) -> float:
    total_time = 0
    for l in batch:
        ratio = l / base_seq_len
        total_time += qkvo_base_time * (ratio)
    log(f"QKV time for batch {batch}: {total_time:.2f}")
    return total_time

def get_mlp_time(batch) -> float:
    total_time = 0
    for l in batch:
        ratio = l / base_seq_len
        total_time += mlp_base_time * (ratio)
    log(f"MLP time for batch {batch}: {total_time:.2f}")
    return total_time


def get_network_time(token_per_batch, cp_degree) -> float:
    base_token_per_batch = 512 * 1024
    if cp_degree == 1:
        log(f"No network time needed for cp_degree=1")
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
    log(f"Network time for {token_per_batch} tokens with cp_degree {cp_degree}: {total_time:.2f}")
    return total_time


def get_batch_time(batch: list[int], is_backward: bool = False, wlb_cp: int = 2, nlayers: int = 1) -> float:
    token_per_batch = sum(batch)
    log(f"Calculating time for batch with {token_per_batch} total tokens")
    
    network_time = get_network_time(token_per_batch, wlb_cp)
    attn_time = get_attn_time(batch)
    mlp_time = get_mlp_time(batch)
    
    if is_backward:
        log("Backward pass - scaling attention and MLP times")
        attn_time *= 2.5
        mlp_time *= 2
        
    compute_time = (attn_time + mlp_time) / wlb_cp
    total_time = compute_time + network_time
    total_time *= nlayers
    
    log(f"Total batch time: {total_time:.2f} (compute: {compute_time:.2f}, network: {network_time:.2f}, layers: {nlayers})")
    return total_time

# %%
# ---------- Logging (light & colored) ----------
LOGGING_ENABLED = True
COLORED_OUTPUT = True

def log(message, time=None):
    if not LOGGING_ENABLED:
        return
    colors = {-1:'\033[0m', 0:'\033[92m', 1:'\033[94m', 2:'\033[93m', 3:'\033[91m'}
    wid = -1
    if "Worker " in message:
        try:
            wid = int(message.split("Worker ")[1].split()[0])
        except ValueError:
            pass
    cs = colors.get(wid, '\033[0m') if COLORED_OUTPUT else ''
    ce = '\033[0m' if COLORED_OUTPUT else ''
    
    # Use provided time or get current simulation time if env exists
    current_time = time if time is not None else (env.now if env else None)
    if current_time is not None:
        print(f"{cs}[Time {current_time:.2f}] {message}{ce}")
    else:
        print(f"{cs}[LOG] {message}{ce}")

# %%
# --- Dynamic all-to-all barrier: register/deregister/launch_comm ---
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
        log(f"Resetting generation {gen}")
        arrivals = {}
        required = None
        release_event = env.event()
        releasing = False

    def register(worker_id: int):
        active.add(worker_id)
        log(f"Worker {worker_id} registered. Active workers: {active}")

    def deregister(worker_id: int):
        nonlocal required, releasing
        active.discard(worker_id)
        log(f"Worker {worker_id} deregistered. Active workers: {active}")
        # If this worker was required but hasn't arrived yet, drop it.
        if required is not None and worker_id in required and not release_event.triggered:
            required.discard(worker_id)
            log(f"Dropped required worker {worker_id}")
            if not releasing and required.issubset(arrivals.keys()):
                _schedule_release()

    def _schedule_release():
        nonlocal releasing
        if releasing or release_event.triggered:
            return
        releasing = True
        snap = {wid: arrivals[wid] for wid in required if wid in arrivals}
        log(f"Scheduling release with payloads from workers: {list(snap.keys())}")

        def _release_proc():
            delay = float(comm_time_fn(snap)) if comm_time_fn else 0.0
            if delay > 0:
                log(f"Communication delay: {delay:.2f}")
                yield env.timeout(delay)
            release_event.succeed(snap)
            log("Release event succeeded")
        env.process(_release_proc())

    def launch_comm(worker_id: int, payload):
        nonlocal required
        # First arrival sets required set to currently active snapshot
        if required is None:
            required = set(active) if active else {worker_id}  # at least self
            log(f"First arrival from worker {worker_id}. Required set: {required}")

        # Record arrival
        if worker_id not in arrivals:
            arrivals[worker_id] = payload
            log(f"Worker {worker_id} arrived with payload size {len(payload)}")

        # If everyone required has arrived, release
        if required.issubset(arrivals.keys()):
            log("All required workers have arrived")
            _schedule_release()

        return release_event

    return register, deregister, launch_comm


# --- Use the dynamic barrier inside your pipeline run ---
def run_iteration(batches, num_stages=4, nlayers=2):
    """
    Same structure as your version, but:
      - dynamic barrier (register/deregister)
      - each stage increments its own done counter when it finishes a grad send
        for a microbatch (so later stages can exit and deregister cleanly)
    """
    global env
    env = simpy.Environment()
    inboxes = [simpy.PriorityStore(env) for _ in range(num_stages)]
    done_counter = [0] * num_stages
    completion_events = [env.event() for _ in range(num_stages)]
    num_microbatches = len(batches)

    log(f"Starting simulation with {num_stages} stages and {num_microbatches} microbatches")

    # Build per-env dynamic barrier
    # (Put your delay model in comm_time_fn if you want)
    register, deregister, launch_comm = make_dynamic_alltoall(env, comm_time_fn=None)

    def check_stage_completion(stage_idx):
        if done_counter[stage_idx] >= num_microbatches and not completion_events[stage_idx].triggered:
            completion_events[stage_idx].succeed()
            log(f"Stage {stage_idx} completed all {num_microbatches} microbatches")

    def flatten_payloads(payloads_dict):
        out = []
        for v in payloads_dict.values():
            out.extend(v)
        return out

    def stage(env, idx, inbox, next_inbox, prev_inbox, num_microbatches, nlayers):
        # Enter pipeline (become eligible for comm rounds)
        has_registered = False
        try:
            while done_counter[idx] < num_microbatches:
                
                prio, kind, m, batch = yield inbox.get()

                if not has_registered:
                    register(idx)
                    has_registered = True

                ping, pong = batch

                is_backward = (kind == "grad")
                is_forward  = not is_backward
                is_last     = (idx == num_stages - 1)

                log(f"[Stage {idx}] processing {'backward' if is_backward else 'forward'} pass for microbatch {m}")

                # ---------------- FORWARD (two microbatches) ----------------
                if is_forward:
                    # L1: qkv[0] -> comm -> qkv[1] -> comm -> attn[0] -> comm -> attn[1] -> comm
                    # qkv[0]
                    layer_id = 0
                    log(f"[Stage {idx}] forward {layer_id} - qkv[0] start")
                    yield env.timeout(get_qkv_time(ping))
                    log(f"[Stage {idx}] forward {layer_id} - qkv[0] done")

                    log(f"[Stage {idx}] forward {layer_id} - m2a[0] comm launch")
                    ev_m2a0 = launch_comm(idx, ping)

                    # qkv[1]
                    layer_id = 1
                    log(f"[Stage {idx}] forward {layer_id} - qkv[1] start")
                    yield env.timeout(get_qkv_time(pong))
                    log(f"[Stage {idx}] forward {layer_id} - qkv[1] done")

                    log(f"[Stage {idx}] forward {layer_id} - m2a[1] comm start")
                    ev_m2a1 = launch_comm(idx, pong)

                    # attn[0] -> a2m[0]
                    layer_id = 0
                    log(f"[Stage {idx}] forward {layer_id} - m2a[0] comm wait")
                    payloads = yield ev_m2a0
                    attn_0_batch = flatten_payloads(payloads)
                    log(f"[Stage {idx}] forward {layer_id} - m2a[0] comm done: {attn_0_batch}")
                    log(f"[Stage {idx}] forward {layer_id} - attn[0] start")
                    yield env.timeout(get_attn_time(attn_0_batch) / total_ranks)
                    log(f"[Stage {idx}] forward {layer_id} - attn[0] done")

                    log(f"[Stage {idx}] forward {layer_id} - a2m[0] comm launch")
                    ev_a2m0 = launch_comm(idx, ping)


                    # attn[1] -> a2m[1]
                    layer_id = 1
                    log(f"[Stage {idx}] forward {layer_id} - m2a[1] comm wait")
                    payloads = yield ev_m2a1
                    attn_1_batch = flatten_payloads(payloads)
                    log(f"[Stage {idx}] forward {layer_id} - m2a[1] comm done: {attn_1_batch}")
                    yield env.timeout(get_attn_time(attn_1_batch) / total_ranks)
                    log(f"[Stage {idx}] forward {layer_id} - attn[1] done")

                    log(f"[Stage {idx}] forward {layer_id} - a2m[1] comm launch")
                    ev_a2m1 = launch_comm(idx, pong)

                    # Middle layers 1..L-2
                    for l in range(1, nlayers - 1):
                        # mlp+qkv[0] -> comm
                        log(f"[Stage {idx}] forward {layer_id} - a2m[0] comm wait")
                        yield ev_a2m0
                        log(f"[Stage {idx}] forward {layer_id} - a2m[0] comm done")

                        log(f"[Stage {idx}] forward {layer_id} - mlp+qkv[0] start")
                        yield env.timeout(get_mlp_time(ping))
                        log(f"[Stage {idx}] forward {layer_id} - mlp+qkv[0] done")

                        log(f"[Stage {idx}] forward {layer_id} - m2a[0] comm launch")
                        ev_m2a0 = launch_comm(idx, ping)

                        # mlp+qkv[1] -> comm
                        log(f"[Stage {idx}] forward {layer_id} - a2m[1] comm wait")
                        yield ev_a2m1
                        log(f"[Stage {idx}] forward {layer_id} - a2m[1] comm done")

                        log(f"[Stage {idx}] forward {layer_id} - mlp+qkv[1] start")
                        yield env.timeout(get_mlp_time(pong))
                        log(f"[Stage {idx}] forward {layer_id} - mlp+qkv[1] done")

                        log(f"[Stage {idx}] forward {layer_id} - m2a[1] comm launch")
                        ev_m2a1 = launch_comm(idx, pong)

                        # attn[0] -> comm
                        log(f"[Stage {idx}] forward {layer_id} - m2a[0] comm wait")
                        payloads = yield ev_m2a0
                        log(f"[Stage {idx}] forward {layer_id} - m2a[0] comm done")
                        yield env.timeout(get_attn_time(flatten_payloads(payloads)) / total_ranks)
                        log(f"[Stage {idx}] forward {layer_id} - attn[0] done")

                        log(f"[Stage {idx}] forward {layer_id} - a2m[0] comm launch")
                        ev_a2m0 = launch_comm(idx, ping)

                        # attn[1] -> comm
                        log(f"[Stage {idx}] forward {layer_id} - m2a[1] comm wait")
                        payloads = yield ev_m2a1
                        log(f"[Stage {idx}] forward {layer_id} - m2a[1] comm done")
                        yield env.timeout(get_attn_time(flatten_payloads(payloads)) / total_ranks)
                        log(f"[Stage {idx}] forward {layer_id} - attn[1] done")

                        log(f"[Stage {idx}] forward {layer_id} - a2m[1] comm launch")
                        ev_a2m1 = launch_comm(idx, ping)

                    # Last layer mlp only
                    layer_id = 0
                    log(f"[Stage {idx}] forward {layer_id} - a2m[0] comm wait")
                    yield ev_a2m0
                    log(f"[Stage {idx}] forward {layer_id} - a2m[0] comm done")
                    yield env.timeout(get_mlp_time(ping))
                    log(f"[Stage {idx}] forward {layer_id} - mlp[0] done")

                    log(f"[Stage {idx}] forward {layer_id} - a2m[1] comm wait")
                    yield ev_a2m1
                    log(f"[Stage {idx}] forward {layer_id} - a2m[1] comm done")
                    yield env.timeout(get_mlp_time(pong))
                    log(f"[Stage {idx}] forward {layer_id} - mlp[1] done")

                # Forward → next stage
                if is_forward and next_inbox is not None:
                    log(f"[Stage {idx}] forwarding activation to next stage: {idx+1}")
                    yield next_inbox.put((1, "act", m, batch))

                # ---------------- BACKWARD trigger ----------------
                if is_backward or (is_forward and next_inbox is None):
                    log(f"[Stage {idx}] starting backward pass")
                    layer_id = 0
                    # Simple mirrored backward (kept from your version)
                    # mlp_o[1] -> comm; mlp_o[0] -> comm
                    log(f"[Stage {idx}] backward {layer_id} - mlp_o[1] start")
                    yield env.timeout(get_mlp_time(pong) * 2)
                    log(f"[Stage {idx}] backward {layer_id} - mlp_o[1] done")

                    log(f"[Stage {idx}] backward {layer_id} - m2a[1] comm launch")
                    ev_m2a1 = launch_comm(idx, ping)

                    log(f"[Stage {idx}] backward {layer_id} - mlp_o[0] start")
                    yield env.timeout(get_mlp_time(ping) * 2)
                    log(f"[Stage {idx}] backward {layer_id} - mlp_o[0] done")

                    log(f"[Stage {idx}] backward {layer_id} - m2a[0] comm launch")
                    ev_m2a0 = launch_comm(idx, pong)

                    log(f"[Stage {idx}] backward {layer_id} - attn_b[1] start")
                    # attn_b[1] -> comm; attn_b[0] -> comm
                    log(f"[Stage {idx}] backward {layer_id} - m2a[1] comm wait")
                    payloads = yield ev_m2a1
                    log(f"[Stage {idx}] backward {layer_id} - m2a[1] comm done")
                    yield env.timeout(get_attn_time(flatten_payloads(payloads)) * 2.5 / total_ranks)
                    log(f"[Stage {idx}] backward {layer_id} - attn_b[1] done")

                    log(f"[Stage {idx}] backward {layer_id} - a2m[1] comm launch")
                    ev_a2m1 = launch_comm(idx, pong)

                    log(f"[Stage {idx}] backward {layer_id} - attn_b[0] start")
                    payloads = yield ev_m2a0
                    log(f"[Stage {idx}] backward {layer_id} - m2a[0] comm done")
                    yield env.timeout(get_attn_time(flatten_payloads(payloads)) * 2.5 / total_ranks)
                    log(f"[Stage {idx}] backward {layer_id} - attn_b[0] done")

                    log(f"[Stage {idx}] backward {layer_id} - a2m[0] comm launch")
                    ev_a2m0 = launch_comm(idx, ping)

                    # middle layers
                    for l in range(1, nlayers - 1):
                        layer_id = l
                        log(f"[Stage {idx}] backward {layer_id} - a2m[1] comm wait")
                        yield ev_a2m1
                        log(f"[Stage {idx}] backward {layer_id} - a2m[1] comm done")
                        yield env.timeout(get_mlp_time(pong) * 2)
                        log(f"[Stage {idx}] backward {layer_id} - mlp_o[1] done")

                        log(f"[Stage {idx}] backward {layer_id} - m2a[1] comm launch")
                        ev_m2a1 = launch_comm(idx, ping)

                        log(f"[Stage {idx}] backward {layer_id} - a2m[0] comm wait")
                        yield ev_a2m0
                        log(f"[Stage {idx}] backward {layer_id} - a2m[0] comm done")
                        yield env.timeout(get_mlp_time(ping) * 2)
                        log(f"[Stage {idx}] backward {layer_id} - mlp_o[0] done")

                        log(f"[Stage {idx}] backward {layer_id} - m2a[0] comm launch")
                        ev_m2a0 = launch_comm(idx, pong)

                        log(f"[Stage {idx}] backward {layer_id} - m2a[1] comm wait")
                        payloads = yield ev_m2a1
                        log(f"[Stage {idx}] backward {layer_id} - m2a[1] comm done")
                        yield env.timeout(get_attn_time(flatten_payloads(payloads)) * 2.5 / total_ranks)
                        log(f"[Stage {idx}] backward {layer_id} - attn_b[1] done")

                        log(f"[Stage {idx}] backward {layer_id} - a2m[1] comm launch")
                        ev_a2m1 = launch_comm(idx, pong)

                        log(f"[Stage {idx}] backward {layer_id} - m2a[0] comm wait")
                        payloads = yield ev_m2a0
                        log(f"[Stage {idx}] backward {layer_id} - m2a[0] comm done")
                        yield env.timeout(get_attn_time(flatten_payloads(payloads)) * 2.5 / total_ranks)
                        log(f"[Stage {idx}] backward {layer_id} - attn_b[0] done")

                        log(f"[Stage {idx}] backward {layer_id} - a2m[0] comm launch")
                        ev_a2m0 = launch_comm(idx, ping)

                    # first layer qkv_b
                    layer_id = 0
                    log(f"[Stage {idx}] backward {layer_id} - a2m[0] comm wait")
                    yield ev_a2m0
                    log(f"[Stage {idx}] backward {layer_id} - a2m[0] comm done")
                    yield env.timeout(get_qkv_time(pong) * 1.7)
                    log(f"[Stage {idx}] backward {layer_id} - qkv_b[1] done")

                    log(f"[Stage {idx}] backward {layer_id} - a2m[1] comm wait")
                    yield ev_a2m1
                    log(f"[Stage {idx}] backward {layer_id} - a2m[1] comm done")
                    yield env.timeout(get_qkv_time(ping) * 1.7)
                    log(f"[Stage {idx}] backward {layer_id} - qkv_b[0] done")

                # Backward → previous stage (or mark done if first stage)
                # TODO(FIXME): isn't this `inboxes[idx - 1]` just prev_inbox?
                if (is_backward or (is_forward and next_inbox is None)) and (inboxes and idx > 0):
                    log(f"Stage {idx} sending gradients to previous stage")
                    yield inboxes[idx - 1].put((0, "grad", m, batch))

                    # mark microbatch done for this stage once we sent its grad upstream
                    done_counter[idx] += 1
                    if done_counter[idx] >= num_microbatches:
                        check_stage_completion(idx)

                elif (is_backward or (is_forward and next_inbox is None)) and idx == 0:
                    # Stage 0 finishes the mb completely
                    log(f"Stage 0 completed microbatch {m}")
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
            log(f"Feeding microbatch {m} to stage 0")
            yield inboxes[0].put((1, "act", m, batch))
    env.process(feeder())

    all_complete = simpy.events.AllOf(env, completion_events)
    env.run(until=all_complete)
    log("Simulation completed")

# %%

# %%
# ---- Quick demo ----
# Create 4 batches with the same sequence length
# batches = [[64 * K] for _ in range(num_batches)]
batches = [
    [
        [128 * K] * 4, [256 * K] * 2,
    ],
    # [
    #     [512 * K] * 1, [256 * K] * 2,
    # ]
]
num_batches = len(batches)
# num_stages = 4
num_stages = 2
log(f"Starting demo with {num_batches} batches and {num_stages} stages")
execution_log = run_iteration(batches, num_stages)

# %%
