# %%
import simpy

# %%
# Logging (kept from your version)
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
# Example usage matching your flow (overlap compute between launch and wait)
env = simpy.Environment()
num_workers = 4

def example_comm_time_fn(payloads):
    # Just a toy model: time proportional to total tokens / 80_000
    total = 0
    for v in payloads.values():
        total += sum(v) if isinstance(v, list) else int(v)
    return total / 80000.0

launch_comm = make_alltoall_comm(env, num_workers, comm_time_fn=None)

def worker_process(sim_env, worker_id):
    log(f"Worker {worker_id} starting", sim_env.now)
    batch = [(64 // (worker_id + 1)) * 1024] * (worker_id + 1)
    log(f"Worker {worker_id} created batch: len={len(batch)}", sim_env.now)

    # First compute
    t1 = worker_id + 1
    log(f"Worker {worker_id} computing for {t1}", sim_env.now)
    yield sim_env.timeout(t1)
    log(f"Worker {worker_id} finished first compute", sim_env.now)

    # Launch comm (non-blocking) and keep computing to overlap
    ev = launch_comm(worker_id, batch)
    log(f"Worker {worker_id} launched communication", sim_env.now)

    # Second compute (overlaps comm delay)
    yield sim_env.timeout(1.0)
    log(f"Worker {worker_id} finished second compute; now waiting for comm", sim_env.now)

    # Sync and get all payloads
    all_payloads = yield ev
    log(f"Worker {worker_id} comm done; received from {sorted(all_payloads.keys())}", sim_env.now)

    # Show a peek
    for k, v in sorted(all_payloads.items()):
        log(f"Worker {worker_id} <- worker {k}: len={len(v)}", sim_env.now)

# %%
log(f"Starting simulation with {num_workers} workers")
for w in range(num_workers):
    env.process(worker_process(env, w))

env.run()
log(f"Simulation completed at time {env.now}")
# %%
