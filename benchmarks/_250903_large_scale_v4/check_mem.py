
# %%
import json
import math
import os
import re

# folder = "/mnt/weka/home/yonghao.zhuang/jd/d2/benchmarks/_250903_large_scale_v4/logs/20250904_015232_PST_bs1_nt1048576_ef16/mem-log"
folder = "/mnt/weka/home/yonghao.zhuang/jd/d2/benchmarks/_250903_large_scale_v4/logs/20250904_021034_PST_bs1_nt524288_ef8/mem-log"

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
mem_data
# %%
def get_events(data):
    return [
        item['message']
        for item in data
    ]
    pass
def get_metrics(data, key='allocated_cur'):

    return [
        (item[key] / 1024) 
        for item in data
    ]

events = get_events(mem_data[0])

plot_data = {}
for rank in mem_data:
    if rank % 8 != 0:
        continue
    plot_data[rank] = get_metrics(mem_data[rank])

# %%
# Plot a line chart for each rank
import matplotlib.pyplot as plt

for rank in plot_data:
    plt.plot(plot_data[rank], label=f'Rank {rank}')
plt.legend()
plt.show()

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

# %%
fig.show()
# %%
