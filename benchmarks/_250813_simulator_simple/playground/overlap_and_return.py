# %%
import simpy

# %%
# Logging configuration
LOGGING_ENABLED = True
COLORED_OUTPUT = True

def log(message, time=None):
    """
    Logging function that can be enabled or disabled with colored output.
    
    Args:
        message: The message to log
        time: Current simulation time (if available)
    """
    if not LOGGING_ENABLED:
        return
        
    # Define colors for different worker IDs
    colors = {
        -1: '\033[0m',      # Default - white
        0: '\033[92m',      # Worker 0 - green
        1: '\033[94m',      # Worker 1 - blue
        2: '\033[93m',      # Worker 2 - yellow
        3: '\033[91m',      # Worker 3 - red
    }
    
    # Extract worker ID from message if possible
    msg_worker_id = -1
    if "Worker " in message:
        try:
            msg_worker_id = int(message.split("Worker ")[1].split()[0])
        except ValueError:
            pass
    
    # Apply color based on worker ID
    color_start = colors.get(msg_worker_id, '\033[0m') if COLORED_OUTPUT else ''
    color_end = '\033[0m' if COLORED_OUTPUT else ''
    
    if time is not None:
        print(f"{color_start}[Time {time:.2f}] {message}{color_end}")
    else:
        print(f"{color_start}[LOG] {message}{color_end}")

# %%
env = simpy.Environment()

# %%

num_workers = 4
comm_values = {}
comm_events = {}

def init_comm_events(sim_env, worker_count):
    a = {}
    for worker_id in range(worker_count):
        a[worker_id] = sim_env.event()
    return a

comm_events = init_comm_events(env, num_workers)


def communication(
    sim_env, worker_id, success_event, 
    input_value, return_values,
):
    global comm_values, comm_events

    log(f"Worker {worker_id} arrived at communication barrier with batch size {len(input_value)}", sim_env.now)
    
    # Mark myself as arrived at the barrier.
    comm_values[worker_id] = input_value
    comm_events[worker_id].succeed()
    log(f"Worker {worker_id} marked as arrived", sim_env.now)

    # Wait until all workers have arrived at this point.
    log(f"Worker {worker_id} waiting for all workers to arrive", sim_env.now)
    for other_id in range(num_workers):
        log(f"Worker {worker_id} waiting for worker {other_id}", sim_env.now)
        yield comm_events[other_id]
        log(f"Worker {worker_id} done waiting for worker {other_id}", sim_env.now)

    log(f"Worker {worker_id} all workers have arrived at barrier", sim_env.now)
    
        # Calculate the actual communication time
    # In a real implementation, we would compute the time needed for all-to-all communication
    # that exceeds what can be overlapped with compute.
    log(f"Worker {worker_id} simulating communication time", sim_env.now)
    yield sim_env.timeout(0)
    
    # Finish communication, copy the values before signaling success
    log(f"Worker {worker_id} copying communication values", sim_env.now)
    for key in comm_values:
        return_values[key] = comm_values[key]
        
    # Make sure all workers have a chance to copy values before worker 0 clears them
    yield sim_env.timeout(0)
    
    # Signal success to the waiting worker process
    log(f"Worker {worker_id} communication completed, signaling success", sim_env.now)
    success_event.succeed()
    
    # Only worker 0 resets the communication state after all workers have copied values
    if worker_id == 0:
        # Small delay to ensure all workers have copied the values
        yield sim_env.timeout(0.01)
        log(f"Worker {worker_id} resetting communication state", sim_env.now)
        # Reset the communication events
        temp_events = init_comm_events(sim_env, num_workers)
        for k in range(num_workers):
            comm_events[k] = temp_events[k]
        # Clear values after all workers have copied them
        comm_values.clear()
        log(f"Worker {worker_id} communication events reset", sim_env.now)
    
    return return_values

# %%


def worker_process(sim_env, worker_id):
    log(f"Worker {worker_id} starting", sim_env.now)
    batch = [(64 // (worker_id + 1)) * 1024] * (worker_id + 1)
    log(f"Worker {worker_id} created batch: {batch}", sim_env.now)
    
    # compute
    compute_time = worker_id + 1
    log(f"Worker {worker_id} computing for {compute_time} time units", sim_env.now)
    yield sim_env.timeout(compute_time)
    log(f"Worker {worker_id} finished first compute phase", sim_env.now)

    # comm
    log(f"Worker {worker_id} initiating communication", sim_env.now)
    comm_event = sim_env.event()
    attn_batches = {}
    sim_env.process(communication(sim_env, worker_id, comm_event, input_value=batch, return_values=attn_batches))
    log(f"Worker {worker_id} started communication process", sim_env.now)

    # compute
    log(f"Worker {worker_id} starting second compute phase", sim_env.now)
    yield sim_env.timeout(1)
    log(f"Worker {worker_id} finished second compute phase", sim_env.now)

    # wait for comm to finish
    log(f"Worker {worker_id} waiting for communication to complete", sim_env.now)
    yield comm_event
    log(f"Worker {worker_id} communication completed", sim_env.now)

    # inspect the attn_batches values
    log(f"Worker {worker_id} received data: {len(attn_batches)} entries", sim_env.now)
    for key, value in attn_batches.items():
        log(f"Worker {worker_id} - received from worker {key}: {len(value)} elements {value}", sim_env.now)

    return attn_batches


log("Starting simulation with {} workers".format(num_workers))
workers = []
for w_id in range(num_workers):
    worker = worker_process(env, w_id)
    workers.append(worker)
    env.process(worker)
    log(f"Registered worker {w_id} process")

# %%

log("Running simulation")
env.run()
log("Simulation completed at time {}".format(env.now))
# %%
