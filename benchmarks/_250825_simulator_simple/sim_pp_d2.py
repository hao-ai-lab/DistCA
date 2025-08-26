# %%

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

# %%
from rich import print
# %%


K = 1024

tick = 0
batches = [
    [
        [128 * K] * 4, [256 * K] * 2,
    ],
    [
        [512 * K] * 1, [128 * K] * 4,
    ],
    [
        [64 * K] * 8, [128 * K] * 4,
    ],
    [
        [64 * K] * 8, [128 * K] * 4,
    ],
    [
        [64 * K] * 8, [128 * K] * 4,
    ],
    [
        [64 * K] * 8, [128 * K] * 4,
    ],
    [
        [64 * K] * 8, [128 * K] * 4,
    ],
    [
        [64 * K] * 8, [128 * K] * 4,
    ],
    
    
]


active_forward = 0
active_backward = 0
forward_idx = 0
backward_idx = 0

phases = [
    "forward_only",
    "forward_and_backward",
    "backward_only",
]
phase = "forward"
threshold = 4
pp_size = 4



events = []

mlp_time = None

current_time = 0


def window(idx, size=None, max_idx=None):
    size = size or pp_size
    max_idx = max_idx or len(batches)
    
    lo = max(0, idx - size + 1)
    hi = min(max_idx, idx + 1)
    return slice(lo, hi)

# Phase 1: forward only
while forward_idx < pp_size:
    fwd_window = window(forward_idx)
    print(f"Phase 1: forward only, forward_idx: {forward_idx} ({fwd_window})")

    ping, pong = batches[0]
    mlp_batch = ping + pong
    print(f"mlp_batch: {mlp_batch}")
    if mlp_time is None:
        mlp_time = get_mlp_time(mlp_batch)
    print(f"mlp_time: {mlp_time}")

    attn_batch = batches[fwd_window]
    attn_batch = flatten(flatten(attn_batch))
    attn_time = get_attn_time(attn_batch) / pp_size

    this_time = mlp_time + attn_time
    future_time = current_time + this_time
    events.append((f"forward", forward_idx, fwd_window, current_time, future_time))
    current_time = future_time
    
    forward_idx += 1
    pass

# Phase 2: forward and backward alternate
while forward_idx < len(batches) + pp_size - 2: # TODO: Inspect this condition
    
    # first backward
    bwd_window = window(backward_idx)
    print(f"Phase 2: forward and backward alternate, backward_idx: {backward_idx} ({bwd_window})")
    
    backward_batch = batches[bwd_window]
    attn_batch = flatten(flatten(backward_batch))
    attn_time = get_attn_time(attn_batch) / pp_size
    print(f"attn_time: {attn_time}")

    # TODO: Fix the time.
    this_time = attn_time * 2.5 + mlp_time * 2
    future_time = current_time + this_time
    events.append((f"backward", backward_idx, bwd_window, current_time, future_time))
    current_time = future_time
    backward_idx += 1
    print(f"backward_batch: {...} (len={len(backward_batch)})")
    
    # then forward
    fwd_window = window(forward_idx)
    print(f"Phase 2: forward and backward alternate, forward_idx: {forward_idx} ({fwd_window})")
    forward_batch = batches[fwd_window]
    attn_batch = flatten(flatten(forward_batch))
    attn_time = get_attn_time(attn_batch) / pp_size
    print(f"attn_time: {attn_time}")
    this_time = attn_time * 2.5 + mlp_time * 2
    future_time = current_time + this_time
    events.append((f"forward", forward_idx, fwd_window, current_time, future_time))
    current_time = future_time
    print(f"forward_batch: {...} (len={len(forward_batch)})")
    forward_idx += 1
    pass

# Phase 3: backward only
while backward_idx < pp_size + len(batches) - 1:
    
    bwd_window = window(backward_idx)
    print(f"Phase 3: backward only, backward_idx: {backward_idx} ({bwd_window})")
    backward_batch = batches[bwd_window]
    attn_batch = flatten(flatten(backward_batch))
    attn_time = get_attn_time(attn_batch) / pp_size
    this_time = attn_time * 2.5 + mlp_time * 2
    future_time = current_time + this_time
    events.append((f"backward", backward_idx, bwd_window, current_time, future_time))
    current_time = future_time
    backward_idx += 1
    pass
# %%
import matplotlib.pyplot as plt

# Create figure and axis
fig, ax = plt.subplots(figsize=(12, 6))

# Plot each event as a horizontal bar
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
    
    # Add text label with index in the middle of the bar
    ax.text(start_time + duration/2, event_name, f'{idx}', 
            horizontalalignment='center', verticalalignment='center')

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
events
# %%
