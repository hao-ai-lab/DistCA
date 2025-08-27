# %%

from rich import print

K = 1024

base_seq_len = K * 64
attn_base_time = 12.5020
# linear_base_time = (13.5 + 8.5) # mlp + qkvo
# linear_base_time = (13.5 + 8.5)  # mlp + qkvo
mlp_base_time = 13.5  # assume expert parallel
qkvo_base_time = 8.5
linear_base_time = (mlp_base_time + qkvo_base_time)  # mlp + qkvo
# linear_base_time = 0
# linear_base_time = 0
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

def flatten(batch: list[list[int]]) -> list[int]:
    return [item for sublist in batch for item in sublist]


def window_slice(idx, size, max_idx):    
    lo = max(0, idx - size + 1)
    hi = min(max_idx, idx + 1)
    return slice(lo, hi)
# %%

def simulate_d2_pipeline(
    batches: list[list[list[int]]], 
    pp_size: int = 4, 
    dpcp_size: int = 1,
    verbose: bool = False,
):
    """
    Simulate the D2 with pipeline parallelism.
    """

    def window(idx):
        return window_slice(idx, pp_size, len(batches))

    def log(*args, **kwargs):
        if verbose:
            print(*args, **kwargs)

    forward_idx = 0
    backward_idx = 0

    events = []
    mlp_time = None
    current_time = 0

    # Phase 1: forward only
    while forward_idx < pp_size:
        fwd_window = window(forward_idx)
        log(f"Phase 1: forward only, forward_idx: {forward_idx} ({fwd_window})")

        ping, pong = batches[0]
        mlp_batch = ping + pong
        log(f"mlp_batch: {mlp_batch}")
        if mlp_time is None:
            mlp_time = get_mlp_time(mlp_batch) / dpcp_size
        log(f"mlp_time: {mlp_time}")

        attn_batch = batches[fwd_window]
        attn_batch = flatten(flatten(attn_batch))
        attn_time = get_attn_time(attn_batch) / pp_size / dpcp_size

        this_time = mlp_time + attn_time
        future_time = current_time + this_time
        events.append((f"forward", forward_idx, fwd_window, current_time, future_time))
        current_time = future_time
        
        forward_idx += 1
        pass

    # Phase 2: forward and backward alternate
    while forward_idx < len(batches) + pp_size - 1:
        
        # first backward
        bwd_window = window(backward_idx)
        log(f"Phase 2: forward and backward alternate, backward_idx: {backward_idx} ({bwd_window})")
        
        backward_batch = batches[bwd_window]
        attn_batch = flatten(flatten(backward_batch))
        attn_time = get_attn_time(attn_batch) / pp_size / dpcp_size
        log(f"attn_time: {attn_time}")

        # TODO: Fix the time.
        this_time = attn_time * 2.5 + mlp_time * 2
        future_time = current_time + this_time
        events.append((f"backward", backward_idx, bwd_window, current_time, future_time))
        current_time = future_time
        backward_idx += 1
        log(f"backward_batch: {...} (len={len(backward_batch)})")
        
        # then forward
        fwd_window = window(forward_idx)
        log(f"Phase 2: forward and backward alternate, forward_idx: {forward_idx} ({fwd_window})")
        forward_batch = batches[fwd_window]
        attn_batch = flatten(flatten(forward_batch))
        attn_time = get_attn_time(attn_batch) / pp_size / dpcp_size
        log(f"attn_time: {attn_time}")
        this_time = attn_time * 2.5 + mlp_time * 2
        future_time = current_time + this_time
        events.append((f"forward", forward_idx, fwd_window, current_time, future_time))
        current_time = future_time
        log(f"forward_batch: {...} (len={len(forward_batch)})")
        forward_idx += 1
        pass

    # Phase 3: backward only
    while backward_idx < pp_size + len(batches) - 1:
        
        bwd_window = window(backward_idx)
        log(f"Phase 3: backward only, backward_idx: {backward_idx} ({bwd_window})")
        backward_batch = batches[bwd_window]
        attn_batch = flatten(flatten(backward_batch))
        attn_time = get_attn_time(attn_batch) / pp_size / dpcp_size
        this_time = attn_time * 2.5 + mlp_time * 2
        future_time = current_time + this_time
        events.append((f"backward", backward_idx, bwd_window, current_time, future_time))
        current_time = future_time
        backward_idx += 1
        pass

    return events
# %%
import matplotlib.pyplot as plt

def plot_d2_timeline(events):
    # Create figure and axis
    fig, ax = plt.subplots(figsize=(12, 6))

    for event_name, idx, window, start_time, end_time in events:
        duration = end_time - start_time
        if "forward" in event_name:
            color = 'lightblue'
            label = 'Forward Pass' if 'Forward Pass' not in ax.get_legend_handles_labels()[1] else None
        else:
            color = 'lightgreen'
            label = 'Backward Pass' if 'Backward Pass' not in ax.get_legend_handles_labels()[1] else None
        
        # Draw filled bar
        ax.barh(y=event_name, width=duration, left=start_time, color=color, label=label)
        
        # Draw black border around bar
        ax.barh(y=event_name, width=duration, left=start_time, color='none', edgecolor='black')
        
        # Add text labels in the middle of the bar
        pos = 0.5 - (1 if "forward" in event_name else -1) * 0.05
        
        ax.text(start_time + duration/2, pos, f'{idx}', 
                horizontalalignment='center', verticalalignment='top', rotation=90)
        ax.text(start_time + duration/2, event_name, f'{duration:.1f}', 
                horizontalalignment='center', verticalalignment='center', rotation=90)

    # Add vertical line at end time (and the time of the last event)
    max_time = max(end_time for _, _, _, _, end_time in events)
    ax.axvline(x=max_time, color='red', linestyle='--', label='End Time', linewidth=2)
    ax.text(max_time, ax.get_ylim()[0], f'{max_time:.1f}', 
            horizontalalignment='right', verticalalignment='bottom', color='red')

    # Customize the plot
    ax.set_xlabel('Time')
    ax.set_ylabel('Events')
    ax.set_title('D2 PP Timeline (backward assume 2.5x attn, 2x mlp)')
    ax.grid(True, axis='x', linestyle='--', alpha=0.7)

    # Add legend
    ax.legend()

    # Adjust layout to prevent label cutoff
    plt.tight_layout()

    # Show the plot
    plt.show()

# %%
# ---- Quick demo ----
# Create 4 batches with the same sequence length
# batches = [[64 * K] for _ in range(num_batches)]
def quick_demo():
    batches = [
        [[128 * K] * 4, [256 * K] * 2],
        [[512 * K] * 1, [128 * K] * 4],
        [[64 * K] * 8, [128 * K] * 4],
        [[64 * K] * 8, [128 * K] * 4],
        [[64 * K] * 8, [128 * K] * 4],
        [[64 * K] * 8, [128 * K] * 4],
        [[64 * K] * 8, [128 * K] * 4],
        [[64 * K] * 8, [128 * K] * 4],
        
    ]
    pp_size = 4
    events = simulate_d2_pipeline(batches, pp_size=pp_size, verbose=True)
    plot_d2_timeline(events)

    forward_pass_times = [e[3] for e in events if "forward" in e[0]]
    backward_pass_times = [e[3] for e in events if "backward" in e[0]]
    assert len(forward_pass_times) == len(batches) + pp_size - 1, f"Forward pass times: {forward_pass_times}"
    assert len(backward_pass_times) == len(batches) + pp_size - 1, f"Backward pass times: {backward_pass_times}"

# %%
# ---- Actually using a distribution to try out ----
def actual_demo_with_distribution():
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
    batches = [
        [next(GLOBAL_BATCH) , next(GLOBAL_BATCH)]
        for _ in range(num_batches)
    ]
    sim_events = simulate_d2_pipeline(batches, pp_size=pp_size, verbose=True)
    plot_d2_timeline(sim_events)

    end_time = max([
        e[-1]
        for e in sim_events
    ])
    print("End Time: ", end_time)



if __name__ == "__main__":
    quick_demo()
    actual_demo_with_distribution()

