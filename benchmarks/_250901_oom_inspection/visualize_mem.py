# %%

import json
import os
import pandas as pd
from IPython.display import display, Markdown
import matplotlib.pyplot as plt

# %%
! pwd

# %%
name = "wlbllm_cp8_0"
# name = "d2_b8_0"
config = "mem.n8.n65536.b1.l32"
log_dir = f"./logs/20250902091626_PST/{config}/{name}/mem"


# %%

# Read the rank 0's mem.rank0.jsonl and print a markdown table
with open(os.path.join(log_dir, "mem.rank0.jsonl"), "r") as f:
    memory_usage = [json.loads(line) for line in f]
    df = pd.DataFrame(memory_usage)
    display(Markdown(df.to_markdown()))


# %%
# Create a figure with larger size
plt.figure(figsize=(15, 10))

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
    
    # Plot line for this rank
    plt.plot(df.index, df['allocated_cur'], label=f'Rank {rank}', alpha=0.5)

plt.title('Memory Usage Across Ranks')
plt.xlabel('Step')
plt.ylabel('Memory (KB)')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True)
plt.tight_layout()
plt.show()


# %%
# Use plotly to plot the figure above
import plotly.graph_objects as go

# Create figure
fig = go.Figure()

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
    
    # Add trace for this rank with hover text
    fig.add_trace(go.Scatter(
        x=df.index,
        y=df['allocated_cur'],
        name=f'Rank {rank}',
        text=df.apply(lambda row: f"Step: {row.name}<br>Memory: {row['allocated_cur']} KB<br>" + 
                                 "<br>".join([f"{k}: {v}" for k,v in row.items()]), axis=1),
        hoverinfo='text'
    ))

fig.update_layout(
    title='Memory Usage Across Ranks',
    xaxis_title='Step',
    yaxis_title='Memory (KB)',
    showlegend=True
)

fig.show()

# %%