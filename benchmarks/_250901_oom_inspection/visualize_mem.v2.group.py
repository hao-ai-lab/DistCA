# %%
import numpy as np
import json
import os
import pandas as pd
from IPython.display import display, Markdown
import matplotlib.pyplot as plt

# %%
! pwd

# %%
names = [
    "wlbllm_cp8_0",
    "wlbllm_cp4_0", 
    "wlbllm_cp2_0",
    "wlbllm_cp1_0",
    "d2_b8_0",
    "d2_b4_0",
    "d2_b1_0"
]
config = "mem.n8.n131072.b32.l4"
log_dir_base = f"./logs/20250902104205_PST"
# %%
# Create a figure with larger size
plt.figure(figsize=(15, 10))

# Create a color map for each name
colors = plt.cm.rainbow(np.linspace(0, 1, len(names)))
name_to_color = dict(zip(names, colors))

# Plot for each name and rank
for name in names:
    log_dir = f"{log_dir_base}/{config}/{name}/mem"
    color = name_to_color[name]
    
    # Plot for each rank
    for rank in range(64):
        if rank % 8 != 0:
            continue
        file_name = f"mem.rank{rank}.jsonl"
        file_path = os.path.join(log_dir, file_name)
        
        if not os.path.exists(file_path):
            continue
            
        with open(file_path, "r") as f:
            memory_usage = [json.loads(line) for line in f]

        # Convert to DataFrame
        df = pd.DataFrame(memory_usage)
        df['allocated_cur'] = (df['allocated_cur'].astype(float) / 1024).round(2)
        
        # Plot line for this rank using the same color for the name
        plt.plot(df.index, df['allocated_cur'], label=f'{name} Rank {rank}', 
                color=color, alpha=0.5)

plt.title('Memory Usage Across Ranks and Configurations')
plt.xlabel('Step')
plt.ylabel('Memory (GB)')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True)
plt.tight_layout()
plt.show()


# Use plotly to plot the figure above
import plotly.graph_objects as go
from plotly.subplots import make_subplots

rank0_only = True  # Flag to only plot rank 0
# rank0_only = False


# Create figure
fig = go.Figure()

# Plot for each name and rank
for name_idx, name in enumerate(names):
    log_dir = f"{log_dir_base}/{config}/{name}/mem"
    
    # Plot for each rank
    for rank in range(64):
        if rank0_only and rank != 0:
            continue
        if not rank0_only and rank % 8 != 0:
            continue
            
        file_name = f"mem.rank{rank}.jsonl"
        file_path = os.path.join(log_dir, file_name)
        
        if not os.path.exists(file_path):
            continue
        with open(file_path, "r") as f:
            memory_usage = [json.loads(line) for line in f]

        # Convert to DataFrame
        df = pd.DataFrame(memory_usage)
        df['allocated_cur'] = (df['allocated_cur'].astype(float) / 1024).round(2)
        
        # Add trace for this rank with hover text
        fig.add_trace(go.Scatter(
            x=df.index,
            y=df['allocated_cur'],
            name=f'{name} Rank {rank}',
            text=df.apply(lambda row: f"Config: {name}<br>Step: {row.name}<br>Memory: {row['allocated_cur']} GB<br>" + 
                                    "<br>".join([f"{k}: {v}" for k,v in row.items()]), axis=1),
            hoverinfo='text',
            visible=True,
            legendgroup=name  # Group traces by name
        ))

# Create buttons for each name
buttons = []
for i, name in enumerate(names):
    # Create visibility list - True for current name's traces, unchanged for others
    visible = [True if trace.legendgroup == name else None for trace in fig.data]
    buttons.append(dict(
        label=name,
        method="restyle",
        args=[{"visible": visible}]
    ))

# Update layout with buttons
fig.update_layout(
    title='Memory Usage Across Ranks and Configurations',
    xaxis_title='Step',
    yaxis_title='Memory (GB)',
    showlegend=True,
    updatemenus=[dict(
        type="buttons",
        direction="right",
        x=0.7,
        y=1.2,
        showactive=True,
        # buttons=buttons
    )]
)

fig.show()

# %%
# Create a new figure for the diff plot
fig_diff = go.Figure()

# Plot diff for each configuration and rank
for name in names:
    for rank in range(8):  # Assuming 8 ranks based on context
        file_path = os.path.join(log_dir_base, config, name, f"mem.rank{rank}.jsonl")
        if not os.path.exists(file_path):
            continue
            
        with open(file_path, "r") as f:
            memory_usage = [json.loads(line) for line in f]

        # Convert to DataFrame
        df = pd.DataFrame(memory_usage)
        df['allocated_cur'] = (df['allocated_cur'].astype(float) / 1024).round(2)
        
        # Calculate diff with previous step
        df['memory_diff'] = df['allocated_cur'].diff()
        
        # Add trace for this rank with hover text
        fig_diff.add_trace(go.Scatter(
            x=df.index,
            y=df['memory_diff'],
            name=f'{name} Rank {rank}',
            text=df.apply(lambda row: f"Config: {name}<br>Step: {row.name}<br>Memory Diff: {row['memory_diff']:.2f} GB<br>" + 
                                    "<br>".join([f"{k}: {v}" for k,v in row.items()]), axis=1),
            hoverinfo='text',
            visible=True,
            legendgroup=name  # Group traces by name
        ))

# Create buttons for each name
diff_buttons = []
for i, name in enumerate(names):
    # Create visibility list - True for current name's traces, unchanged for others
    visible = [True if trace.legendgroup == name else None for trace in fig_diff.data]
    diff_buttons.append(dict(
        label=name,
        method="restyle",
        args=[{"visible": visible}]
    ))

# Update layout with buttons
fig_diff.update_layout(
    title='Memory Usage Difference Across Ranks and Configurations',
    xaxis_title='Step',
    yaxis_title='Memory Difference (GB)',
    showlegend=True,
    updatemenus=[dict(
        type="buttons",
        direction="right",
        x=0.7,
        y=1.2,
        showactive=True,
        # buttons=diff_buttons
    )]
)

fig_diff.show()

# %%
