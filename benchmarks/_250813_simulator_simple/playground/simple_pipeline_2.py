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


def get_attn_time(batch) -> float:
    total_time = 0
    for l in batch:
        ratio = l / base_seq_len
        total_time += attn_base_time * (ratio ** 2)
    return total_time


def get_qkv_time(batch) -> float:
    total_time = 0
    for l in batch:
        ratio = l / base_seq_len
        total_time += qkvo_base_time * (ratio)
    return total_time

def get_mlp_time(batch) -> float:
    total_time = 0
    for l in batch:
        ratio = l / base_seq_len
        total_time += mlp_base_time * (ratio)
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
    if time is not None:
        print(f"{cs}[Time {time:.2f}] {message}{ce}")
    else:
        print(f"{cs}[LOG] {message}{ce}")

# %%
# ---------- Tagged All-to-All Comm (functional, per-tag) ----------
def make_alltoall_comm(env, num_workers, comm_time_fn=None):
    """
    Factory that returns a function `launch_comm(worker_id, payload)`:
      - Marks arrival + stores payload for this generation
      - Returns an event that will succeed with a snapshot dict {wid: payload}
        once *all* workers have arrived and an optional comm delay elapses.
    No classes; just a closure with tiny shared state per generation.
    """

    # --- shared state for the current generation ---
    gen = 0
    arrivals = [env.event() for _ in range(num_workers)]  # one arrival event per worker
    release_event = env.event()                           # what callers will yield on
    payloads = {}                                         # {wid: payload} for this gen
    releasing = False                                     # ensure single releaser

    def reset_generation():
        nonlocal gen, arrivals, release_event, payloads, releasing
        gen += 1
        arrivals = [env.event() for _ in range(num_workers)]
        release_event = env.event()
        payloads = {}
        releasing = False

    def launch_comm(worker_id, payload):
        nonlocal releasing, payloads, arrivals, release_event, gen

        my_gen = gen
        my_release = release_event
        my_arrivals = arrivals  # keep a handle to this generation's events

        # Record payload and mark arrival (idempotent against double calls)
        if not my_arrivals[worker_id].triggered:
            payloads[worker_id] = payload
            my_arrivals[worker_id].succeed()
            log(f"Worker {worker_id} arrived at comm barrier gen={my_gen} (count={len(payloads)}/{num_workers})", env.now)

        # If all arrived and we haven't scheduled a release yet, do it once
        if (not releasing) and all(e.triggered for e in my_arrivals):
            releasing = True

            def _release_process():
                # Wait (trivial since all triggered, but ensures consistent ordering)
                yield simpy.events.AllOf(env, my_arrivals)
                # Optional comm delay based on payloads
                delay = float(comm_time_fn(payloads)) if comm_time_fn else 0.0
                if delay > 0:
                    log(f"Barrier gen={my_gen} simulating all-to-all delay={delay}", env.now)
                    yield env.timeout(delay)
                # Snapshot & release
                snapshot = dict(payloads)
                log(f"Barrier gen={my_gen} releasing ({len(snapshot)} payloads)", env.now)
                my_release.succeed(snapshot)
                # Prepare next generation
                reset_generation()

            env.process(_release_process())

        return my_release

    return launch_comm

# %%

env = simpy.Environment()
num_workers = 4

launch_comm = make_alltoall_comm(env, num_workers, comm_time_fn=None)


# %%
# 4-stage pipeline parallel (1F1B) SimPy model with a Matplotlib timeline.
# Forward = 130 ms, Backward = 2.5x (325 ms)
# One PriorityStore per stage: grad (prio=0) > act (prio=1)
# ---- Sim model (tiny + readable) ----


def run_iteration(batches, num_stages=4, nlayers=1):
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

    # Create completion events for each stage
    completion_events = [env.event() for _ in range(num_stages)]

    # Function to check if a stage is complete and trigger its event
    def check_stage_completion(stage_idx):
        if done_counter[stage_idx] >= num_microbatches and not completion_events[stage_idx].triggered:
            completion_events[stage_idx].succeed()

    # Modify the stage function to signal completion
    def stage_with_signal(env, idx, inbox, next_inbox, prev_inbox, num_microbatches, done_counter, log_data, nlayers=1):
        """Main stage function to perform pipeline parallelism."""
        while done_counter[idx] < num_microbatches:
            _, kind, m, batch = yield inbox.get()
            ping, pong = batch

            is_backward = (kind == "grad")
            is_forward = not is_backward

            is_first_stage = (idx == 0)
            is_last_stage = (idx == num_stages - 1)


            # forward pass, regardless of the stage, will need to go through the forward pass logic.
            if is_forward:
                # "act" -> forward
                
                if idx == 0:
                    # TODO: Add embedding time logic here.
                    pass

                # for first layer, 
                # qkv[0], m2a[0], qkv[1], m2a[1], attn[0], a2m[0], attn[1], a2m[1]

                layer_id = 0

                # qkv[0] + m2a[0]
                yield env.timeout(get_qkv_time(ping))
                m2a0_ev = launch_comm(idx, ping)

                # qkv[1] + m2a[1]
                yield env.timeout(get_qkv_time(pong))
                m2a1_ev = launch_comm(idx, pong)

                # wait for m2a[0], then attn[0] + a2m[0]
                attn_0_batch = yield m2a0_ev
                yield env.timeout(get_attn_time(attn_0_batch) // total_ranks)
                a2m0_ev = launch_comm(idx, ping)

                # wait for m2a[1], then attn[1] + a2m[1]
                attn_1_batch = yield m2a1_ev
                yield env.timeout(get_attn_time(attn_1_batch) // total_ranks)
                a2m1_ev = launch_comm(idx, pong)

                for layer_id in range(1, nlayers - 1):
                    # layer_id = 1 ... nlayers - 2

                    # for all until last layer
                    # o_mlp_qkv[0], m2a[0], o_mlp_qkv[1], m2a[1], attn[0], a2m[0], attn[1], a2m[1]

                    # o_mlp_qkv[0], m2a[0]
                    yield env.timeout(get_mlp_time(ping))
                    m2a0_ev = launch_comm(idx, ping)

                    # o_mlp_qkv[1], m2a[1]
                    yield env.timeout(get_mlp_time(pong))
                    m2a1_ev = launch_comm(idx, pong)

                    # attn[0], a2m[0]
                    attn_0_batch = yield m2a0_ev
                    yield env.timeout(get_attn_time(attn_0_batch) // total_ranks)
                    a2m0_ev = launch_comm(idx, ping)

                    # attn[1], a2m[1]
                    attn_1_batch = yield m2a1_ev
                    yield env.timeout(get_attn_time(attn_1_batch) // total_ranks)
                    a2m1_ev = launch_comm(idx, pong)

                    pass

                # Handle last layer

                # o_mlp[0]
                yield env.timeout(get_mlp_time(ping))
                
                # o_mlp[1]
                yield env.timeout(get_mlp_time(pong))


                if is_last_stage:
                    # TODO: Add output mlp logic here.
                    pass

            
            # forward, and is not the last stage.
            if is_forward and next_inbox is not None:
                yield next_inbox.put((1, "act", m, batch))
            
            # backward; or forward, encounter the last stage, and need to issue a backward immediately.
            if is_backward or (is_forward and next_inbox is None):
                # actuall do a round of backward

                # backward logic

                if is_last_stage:
                    # TODO: add loss to output mlp logic here.
                    pass

                # Reverse of: qkv[0], m2a[0], qkv[1], m2a[1], attn[0], a2m[0], attn[1], a2m[1], 
                # o_mlp_qkv[0], m2a[0], o_mlp_qkv[1], m2a[1], attn[0], a2m[0], attn[1], a2m[1],

                layer_id = 0
                
                # TODO: Fix the time function...
                # mlp_o[1], m2a[1]
                yield env.timeout(get_mlp_time(pong))
                m2a1_ev = launch_comm(idx, ping)
                
                # mlp_o[0], m2a[0]
                yield env.timeout(get_mlp_time(ping))
                m2a0_ev = launch_comm(idx, pong)


                # attn[1], a2m[1]
                attn_1_batch = yield m2a1_ev
                yield env.timeout(get_attn_time(attn_1_batch) // total_ranks)
                a2m1_ev = launch_comm(idx, pong)

                # attn[0], a2m[0]
                attn_0_batch = yield m2a0_ev
                yield env.timeout(get_attn_time(attn_0_batch) // total_ranks)
                a2m0_ev = launch_comm(idx, ping)

                for layer_id in range(1, nlayers - 1):

                    # qkv_mlp_o[1], m2a[1]
                    yield env.timeout(get_mlp_time(pong))
                    m2a1_ev = launch_comm(idx, ping)

                    # qkv_mlp_o[0], m2a[0]
                    yield env.timeout(get_mlp_time(ping))
                    m2a0_ev = launch_comm(idx, pong)

                    # attn[1], a2m[1]
                    attn_1_batch = yield m2a1_ev
                    yield env.timeout(get_attn_time(attn_1_batch) // total_ranks)
                    a2m1_ev = launch_comm(idx, pong)

                    # attn[0], a2m[0]
                    attn_0_batch = yield m2a0_ev
                    yield env.timeout(get_attn_time(attn_0_batch) // total_ranks)
                    a2m0_ev = launch_comm(idx, ping)
                    pass

                # handle first layer

                # qkv[1]
                yield env.timeout(get_qkv_time(pong))

                # qkv[0]
                yield env.timeout(get_qkv_time(ping))

                pass

                

    # Start stage processes
    for i in range(num_stages):
        next_inbox = inboxes[i + 1] if i < num_stages - 1 else None
        prev_inbox = inboxes[i - 1] if i > 0 else None
        env.process(stage_with_signal(env, i, inboxes[i], next_inbox, prev_inbox, num_microbatches, done_counter, execution_log, nlayers=nlayers))

    # Feed microbatches to stage 0 as activations
    def feeder():
        for m, batch in enumerate(batches):
            # (prio, kind, m=batch_id, batch)
            yield inboxes[0].put((1, "act", m, batch))

    env.process(feeder())

    # Wait for all completion events
    all_complete = simpy.AllOf(env, completion_events)
    env.run(until=all_complete)

    return execution_log



# %%

# %%
# ---- Quick demo ----
# Create 4 batches with the same sequence length
# batches = [[64 * K] for _ in range(num_batches)]
batches = [
    [[128 * K] * 4, [256 * K] * 2,]
    # [[512 * K] * 1, [256 * K] * 2,]
]
num_batches = len(batches)
num_stages = 4
execution_log = run_iteration(batches, num_stages)

# %%
