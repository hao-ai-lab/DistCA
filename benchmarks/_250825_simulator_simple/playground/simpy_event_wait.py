# %%
import simpy

# %%
# Logging configuration
LOGGING_ENABLED = True

def log(message, time=None):
    """
    Logging function that can be enabled or disabled.
    
    Args:
        message: The message to log
        time: Current simulation time (if available)
    """
    if LOGGING_ENABLED:
        if time is not None:
            print(f"[Time {time:.2f}] {message}")
        else:
            print(f"[LOG] {message}")

# %%
env = simpy.Environment()
# %%
def communication(environment, comm_event):
    log("Starting communication process", environment.now)
    
    # Simulate communication taking 3 time units
    yield environment.timeout(3)
    
    # Signal that communication is complete
    log("Communication completed", environment.now)
    comm_event.succeed()

def worker(environment):
    log("Worker process starting", environment.now)
    
    # First compute phase
    log("Starting first computation phase", environment.now)
    yield environment.timeout(1)
    log("Completed first computation phase", environment.now)
    
    # Issue a comm that overlaps with the compute
    log("Initiating communication", environment.now)
    comm_event = environment.event()
    environment.process(communication(environment, comm_event))
    
    # Continue with compute while comm is happening
    log("Starting second computation phase (overlapping with communication)", environment.now)
    yield environment.timeout(1)
    log("Completed second computation phase", environment.now)
    
    # Wait until the comm is finished before proceeding
    log("Waiting for communication to complete", environment.now)
    yield comm_event
    log("Communication completed, proceeding with execution", environment.now)
    
    # Compute after comm is complete
    log("Starting final computation phase", environment.now)
    yield environment.timeout(1)
    log("Completed final computation phase", environment.now)



log("Starting simulation")
env.process(worker(env))
# %%
log("Running simulation")
env.run()
log(f"Simulation completed at time {env.now}")
# %%