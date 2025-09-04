
# %%
import json
import math
import os
import re

# folder = "/mnt/weka/home/yonghao.zhuang/jd/d2/benchmarks/_250903_large_scale_v4/logs/20250904_015232_PST_bs1_nt1048576_ef16/mem-log"
# folder = "/mnt/weka/home/yonghao.zhuang/jd/d2/benchmarks/_250903_large_scale_v4/logs/20250904_021034_PST_bs1_nt524288_ef8/mem-log"
# folder = "/mnt/weka/home/yonghao.zhuang/jd/d2/benchmarks/_250903_large_scale_v4/logs/20250904_022632_PST_bs1_nt16384_ef8/mem-log"
# folder = "/mnt/weka/home/yonghao.zhuang/jd/d2/benchmarks/_250903_large_scale_v4/logs/20250904_023702_PST_bs1_nt8192_ef1/mem-log"
# folder = "/mnt/weka/home/yonghao.zhuang/jd/d2/benchmarks/_250903_large_scale_v4/logs/20250904_075850_PST_bs8_nt131072_ef2/mem-log"
# folder = "/mnt/weka/home/yonghao.zhuang/jd/d2/benchmarks/_250903_large_scale_v4/logs/20250904_082337_PST_bs4_nt262144_ef4/mem-log"
# folder = "/mnt/weka/home/yonghao.zhuang/jd/d2/benchmarks/_250903_large_scale_v4/logs/20250904_091012_PST_bs4_nt262144_ef4/mem-log"
name = "20250904_093005_PST_bs4_nt262144_ef4"
folder = f"/mnt/weka/home/yonghao.zhuang/jd/d2/benchmarks/_250903_large_scale_v4/logs/{name}/mem-log"

mem_data = {}
for file in os.listdir(folder):
    print(file)
    rank = int(re.search(r'\.rank(\d+)\.', file).group(1))
    print(rank)
    with open(os.path.join(folder, file), "r") as f:
        data = [] 
        for line in f:
            data.append(json.loads(line))
    if not data:
        continue
    mem_data[rank] = data
    # break
# %%
# Sort mem_data by rank and create new dict
mem_data = dict(sorted(mem_data.items()))

# %%
# %%
def get_events(data):
    results = []
    for item in data:
        message = item['message']
        # message should replace '/mnt/weka/home/yonghao.zhuang/jd/d2/' to ''
        message = message.replace('/mnt/weka/home/yonghao.zhuang/jd/d2/', '')
        results.append(message)
    return results
    
    pass
def get_metrics(data, key='allocated_cur'):

    return [
        (item[key] / 1024) 
        for item in data
    ]

events = get_events(mem_data[0])

plot_data = {}
for rank in mem_data:
    # if rank % 8 != 0:
    #     continue
    plot_data[rank] = get_metrics(mem_data[rank])

# # %%
# # Plot a line chart for each rank
# import matplotlib.pyplot as plt

# for rank in plot_data:
#     plt.plot(plot_data[rank], label=f'Rank {rank}')
# plt.legend()
# plt.show()

# %%
# Plot a plotly figure
import plotly.graph_objects as go

fig = go.Figure()
for rank in plot_data:
    fig.add_trace(go.Scatter(
        x=list(range(len(plot_data[rank]))), 
        y=plot_data[rank],
        name=f'Rank {rank}',
        text=events[:len(plot_data[rank])],  # Add event text as tooltips
        hovertemplate='Event: %{text}<br>Memory: %{y:.2f} GB<extra></extra>'
    ))

fig.show()
# Save the plotly figure as HTML
fig.write_html("memory_usage.html")

# %%

mem_data[0]
# %%
# For each rank, plot the difference between the current and previous step
diff_data = {}
# ranks_to_plot = [8, 184]  # List of ranks to plot
ranks_to_plot = [
    i for i in range(0, 128, 8)
]

for rank in mem_data:
    if rank in ranks_to_plot:
        diff_data[rank] = [mem_data[rank][i]['allocated_cur'] - mem_data[rank][i-1]['allocated_cur'] for i in range(1, len(mem_data[rank]))]

# # Plot the difference for each rank
# for rank in diff_data:
#     plt.plot(diff_data[rank], label=f'Rank {rank}')
# plt.legend()
# plt.show()

# Plot the difference data using plotly
fig = go.Figure()
# Only plot specified ranks
for rank in diff_data:
    fig.add_trace(go.Scatter(
        x=list(range(len(diff_data[rank]))),
        y=diff_data[rank],
        name=f'Rank {rank}',
        text=events[1:len(diff_data[rank])+1],  # Offset events by 1 since diff starts from second element
        hovertemplate='Event: %{text}<br>Memory Diff: %{y:.2f} MB<extra></extra>',
        mode='lines+markers',  # Show both lines and dots
        marker=dict(size=8)  # Set dot size
    ))

fig.update_layout(
    title=f'Memory Usage Difference Between Steps (Ranks {ranks_to_plot})',
    xaxis_title='Step', 
    yaxis_title='Memory Allocated (MB)'
)

fig.show()

# Save the plotly figure as HTML
fig.write_html("memory_diff.html")
# %%

max_step = 100
# Find shortest trace length
min_length = min(len(trace) for trace in diff_data.values())

# Get min and max values at each step across all ranks
min_vals = []
max_vals = []
diff_vals = []
for step in range(min_length):
    step_vals = [diff_data[rank][step] for rank in diff_data]
    min_val = min(step_vals)
    max_val = max(step_vals)
    min_vals.append(min_val)
    max_vals.append(max_val)
    diff_vals.append(max_val - min_val)


min_vals = min_vals[:max_step]
max_vals = max_vals[:max_step]
diff_vals = diff_vals[:max_step]
# Create min/max plot
fig = go.Figure()

fig.add_trace(go.Scatter(
    x=list(range(min_length)),
    y=min_vals,
    name='Min across ranks',
    text=events[1:min_length+1],
    hovertemplate='Event: %{text}<br>Min Memory Diff: %{y:.2f} MB<extra></extra>',
    mode='lines+markers',
    marker=dict(size=8)
))

fig.add_trace(go.Scatter(
    x=list(range(min_length)), 
    y=max_vals,
    name='Max across ranks',
    text=events[1:min_length+1],
    hovertemplate='Event: %{text}<br>Max Memory Diff: %{y:.2f} MB<extra></extra>',
    mode='lines+markers',
    marker=dict(size=8)
))

fig.add_trace(go.Scatter(
    x=list(range(min_length)),
    y=diff_vals,
    name='Max-Min Difference',
    text=events[1:min_length+1], 
    hovertemplate='Event: %{text}<br>Max-Min Diff: %{y:.2f} MB<extra></extra>',
    mode='lines+markers',
    marker=dict(size=8)
))

fig.update_layout(
    title='Min/Max Memory Usage Difference Across Ranks',
    xaxis_title='Step',
    yaxis_title='Memory Allocated (MB)'
)

fig.show()

# Save the plotly figure as HTML
fig.write_html("memory_diff_minmax.html")

# %%
# Create a table with x, ymin, ymax, ydiff and message
import pandas as pd

data = {
    'Step': list(range(max_step)),
    'Min Memory (MB)': min_vals,
    'Max Memory (MB)': max_vals, 
    'Max-Min Diff (MB)': diff_vals,
    'Event': events[1:max_step+1]
}

df = pd.DataFrame(data)
df = df[df['Max-Min Diff (MB)'] > 1]
# Display as markdown
print("\nMemory Usage Statistics Table:")
print(df.to_markdown(index=False))

# Display the table
# print("\nMemory Usage Statistics Table:")
# print(df.to_string(index=False))

# # Save to CSV
# df.to_csv('memory_stats.csv', index=False)
# print("\nSaved memory statistics to memory_stats.csv")

# %%
