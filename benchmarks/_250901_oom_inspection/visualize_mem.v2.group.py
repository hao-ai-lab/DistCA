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
    "wlbllm_cp8_1",
    "wlbllm_cp4_1", 
    "wlbllm_cp2_1",
    "wlbllm_cp1_1",
    "d2_b8_1",
    "d2_b4_1",
    "d2_b1_1"
]
# config = "mem.n4.n65536.b4.l4"
# config = "mem.n4.n131072.b4.l4"
# config = "mem.n4.n65536.b8.l4"
# log_dir_base = f"/mnt/weka/home/yonghao.zhuang/jd/d2/benchmarks/_250901_oom_inspection/logs/20250902155443_PST"
log_dir_base = f"/mnt/weka/home/yonghao.zhuang/jd/d2/benchmarks/_250901_oom_inspection/logs/20250902182812_PST"
config = "mem.n8.n65536.b32.l4"

# %%

# Use plotly to plot the figure above
import plotly.graph_objects as go
from plotly.subplots import make_subplots

rank0_only = True  # Flag to only plot rank 0
# rank0_only = False

# Create figure
fig = go.Figure()


peak_memory_usage = {}
# Plot for each name and rank
for name_idx, name in enumerate(names):
    log_dir = f"{log_dir_base}/{config}/{name}/mem"
    
    # Plot for each rank
    for rank in range(64):
        if rank0_only and rank != 0:
            continue
        if not rank0_only and rank % 8 != 0:
            continue
        
        # Generate the memory usage dataframe
        file_name = f"mem.rank{rank}.jsonl"
        file_path = os.path.join(log_dir, file_name)
        
        if not os.path.exists(file_path):
            continue
        with open(file_path, "r") as f:
            memory_usage = [json.loads(line) for line in f]

        # Convert to DataFrame
        df = pd.DataFrame(memory_usage)
        if df.empty:
            continue

        df_output_dir = f"{log_dir_base}/{config}/{name}/mem-tsv"
        os.makedirs(df_output_dir, exist_ok=True)
        df_output_file = f"{df_output_dir}/mem.rank{rank}.tsv"
        df_copy = df.copy()
        df_copy['name'] = f'{config}/{name}'
        df_copy.to_csv(df_output_file, index=False, sep='\t')

        df['allocated_cur'] = (df['allocated_cur'].astype(float) / 1024).round(2)
        df['allocated_peak'] = (df['allocated_peak'].astype(float) / 1024).round(2)
        peak_memory_usage[name] = max(
            peak_memory_usage.get(name, 0), 
            df['allocated_peak'].max().item()
        )
        
        # Choose color based on name prefix
        if name.startswith('d2'):
            if 'b1' in name:
                color = '#8B0000'  # darkred
            elif 'b2' in name:
                color = '#B22222'  # firebrick
            elif 'b4' in name:
                color = '#DC143C'  # crimson
            elif 'b8' in name:
                color = '#FF4444'  # lighter red
            else:
                color = '#FF6666'  # lightest red
        else:  # wlbllm
            if 'cp1' in name:
                color = 'royalblue'
            elif 'cp2' in name:
                color = '#4169E1'  # royal blue
            elif 'cp4' in name:
                color = '#6495ED'  # cornflower blue
            else:
                color = '#87CEEB'  # sky blue
            
        # Add trace for this rank with hover text
        fig.add_trace(go.Scatter(
            x=df.index,
            y=df['allocated_cur'],
            name=f'{name} Rank {rank}',
            text=df.apply(lambda row: f"Config: {name}<br>Step: {row.name}<br>Memory: {row['allocated_cur']} GB<br>" + 
                                    "<br>".join([f"{k}: {v}" for k,v in row.items()]), axis=1),
            hoverinfo='text',
            visible=True,
            legendgroup=name,  # Group traces by name
            line=dict(color=color)  # Set the color
        ))

    # Search for the log and see if oom happened, and also search for the buffer size actually set.
    # glob the log files
    import glob
    log_files = glob.glob(
        f"{log_dir_base}/{config}/{name}/*.log"
    )
    for log_file in log_files:
        with open(log_file, "r") as f:
            log = f.read()
        if "OutOfMemoryError" in log:
            # Use regex to find OOM error line
            import re
            oom_match = re.search(r'torch\.OutOfMemoryError:.*?(?=\n|$)', log, re.MULTILINE)
            if oom_match:
                print(oom_match.group(0))
            print(f"OOM happened in {log_file}")
        # rich.print(f"🟡 [Rank {rank}] Overflow check passed for fa2a_metadata_0 and fa2a_metadata_1 with tolerance_factor {tolerance_factor} and buffer_size {buffer_size / 1024**3} GB")
        if "Overflow check passed" in log:
            # Use regex to find buffer size
            import re
            buffer_size_matches = re.findall(r'buffer_size (\d+\.?\d*) GB', log)
            if buffer_size_matches:
                buffer_size = float(buffer_size_matches[-1])
                print(f"Buffer size {buffer_size} GB found in {log_file}")
        pass


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
    title=f'Memory (config = {config})',
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

# save to html
fig.write_html(f"{log_dir_base}/{config}/mem.html")
# %%
peak_memory_usage

# %%


# %%
# Create a new figure for the diff plot
fig_diff = go.Figure()
ranks_to_plot = [0]
# Plot diff for each configuration and rank
for name in names:
    log_dir = f"{log_dir_base}/{config}/{name}/mem"
    for rank in ranks_to_plot:
        file_path = os.path.join(log_dir, f"mem.rank{rank}.jsonl")
        print(file_path)
        if not os.path.exists(file_path):
            continue
            
        with open(file_path, "r") as f:
            memory_usage = [json.loads(line) for line in f]

        # Convert to DataFrame
        df = pd.DataFrame(memory_usage)
        if df.empty:
            continue
        df['allocated_cur'] = (df['allocated_cur'].astype(float) / 1024).round(2)
        
        # Calculate diff with previous step
        df['memory_diff'] = df['allocated_cur'].diff()
        
        # Choose color based on name prefix
        if name.startswith('d2'):
            if 'b1' in name:
                color = '#8B0000'  # darkred
            elif 'b2' in name:
                color = '#B22222'  # firebrick
            elif 'b4' in name:
                color = '#DC143C'  # crimson
            elif 'b8' in name:
                color = '#FF4444'  # lighter red
            else:
                color = '#FF6666'  # lightest red
        else:  # wlbllm
            if 'cp1' in name:
                color = 'royalblue'
            elif 'cp2' in name:
                color = '#4169E1'  # royal blue
            elif 'cp4' in name:
                color = '#6495ED'  # cornflower blue
            else:
                color = '#87CEEB'  # sky blue
        
        # Add trace for this rank with hover text
        fig_diff.add_trace(go.Scatter(
            x=df.index,
            y=df['memory_diff'],
            name=f'{name} Rank {rank}',
            text=df.apply(lambda row: f"Config: {name}<br>Step: {row.name}<br>Memory Diff: {row['memory_diff']:.2f} GB<br>" + 
                                    "<br>".join([f"{k}: {v}" for k,v in row.items()]), axis=1),
            hoverinfo='text',
            visible=True,
            legendgroup=name,  # Group traces by name
            mode='lines+markers',  # Add points to the plot
            line=dict(color=color)  # Set the color
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
    title=f'Memory Difference (config = {config})',
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
# save to html
fig_diff.write_html(f"{log_dir_base}/{config}/mem_diff.html")
# %%
