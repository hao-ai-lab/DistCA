
# %%
import os
import json
root_path = "/mnt/weka/home/yonghao.zhuang/jd/d2/benchmarks/_250918_large_scale_v11_dpcp/logs.v1-pretrain-8b"
a = os.listdir(root_path)

success_configs = {}
rows = []
for folder in a:
    if not os.path.isdir(os.path.join(root_path, folder)):
        continue
    
    # Read the config file to check the configurations
    readme_config = os.path.join(root_path, folder, "README.md")
    if not os.path.exists(readme_config):
        continue
    with open(readme_config, "r") as f:
        readme_config_data = f.read()
    readme_config_data = readme_config_data.split("\n")
    readme_config = {}
    for x in readme_config_data:
        if not x.startswith("-"):
            continue
        key, value = x.split(":")
        key = key.strip().lower().replace("-", "").strip()
        value = value.strip()
        readme_config[key] = value
    name = folder[27:]
    mode = readme_config["mode"].strip()
    cp_size = int(readme_config["cp_size"].strip())
    nodes = int(readme_config["nnodes"].strip())
    batch_size = int(readme_config["batch_size"].strip())
    num_tokens = int(readme_config["num_tokens"].strip())


    # Read the benchmark file
    file = os.path.join(root_path, folder, "benchmark.raw.jsonl")
    if not os.path.exists(file):
        continue


    with open(file, "r") as f:
        data = [json.loads(line) for line in f]
    # print(data)
    durations = [x['duration_ms'] for x in data]
    # print(durations)
    average_duration = sum(durations) / len(durations)
    # groups = name.split("-")
    # mode, cp_size, nodes, batch_size, num_tokens = groups
    print(f"{name}: {average_duration:.2f}ms")
    config = dict(
        mode=mode,
        cp_size=cp_size,
        nodes=nodes,
        batch_size=batch_size,
        num_tokens=num_tokens,
    )
    row = dict(
        name=name,
        dataset="pretrain",
        group_id=f"{nodes:02d}_{num_tokens}_{batch_size}",
        **config,
        average_duration=average_duration,
    )
    for i, d in enumerate(durations):
        row[f"sample_{i}"] = d
    success_configs[name] = config
    rows.append(row)

# %%
success_configs

# %%


# %%
import pandas as pd
df = pd.DataFrame(rows)
df = df.sort_values(by=[
    "nodes", "batch_size", "num_tokens", "mode", "average_duration", "cp_size",
], ascending=False)
df

# %%
wlb_groups_best = df[
    df['mode'] == 'wlbllm'
].groupby([ "nodes", "batch_size", "num_tokens", "dataset"])['average_duration'].min().reset_index()
wlb_groups_best
# %%
d2_groups_best = df[
    df['mode'] == 'd2'
].groupby([ "nodes", "batch_size", "num_tokens", "dataset"])['average_duration'].min().reset_index()
d2_groups_best
# %%
# merge wlb_groups_best and d2_groups_best - 
merged_wlb_vs_d2 = pd.merge(wlb_groups_best, d2_groups_best, on=['nodes', 'batch_size', 'num_tokens', 'dataset'], how='left', suffixes=('_wlb', '_d2'))
merged_wlb_vs_d2['speedup'] = merged_wlb_vs_d2['average_duration_wlb'] / merged_wlb_vs_d2['average_duration_d2']
merged_wlb_vs_d2['linear_speedup'] = merged_wlb_vs_d2['num_tokens'] / merged_wlb_vs_d2['batch_size']
# merged_wlb_vs_d2['line_id'] = merged_wlb_vs_d2['num_tokens'] / merged_wlb_vs_d2['batch_size'] * merged_wlb_vs_d2['nodes']
# merged_wlb_vs_d2['line_id'] = merged_wlb_vs_d2['line_id'].apply(
#     lambda x: '128k' if x == 262144.0 else (
#         '256k' if x == 1048576.0 else (
#             '512k' if x == 4194304.0 else None
#         )
#     )
# )
merged_wlb_vs_d2['line_id'] = merged_wlb_vs_d2['num_tokens'].apply(
    lambda x: (x // 1024)
)
merged_wlb_vs_d2.sort_values(by=['num_tokens', 'nodes'], ascending=True)
# %%
# Plot speedup vs nodes for each num_tokens/batch_size combination
import matplotlib.pyplot as plt

# Create figure and axis
fig, ax = plt.subplots(figsize=(5, 4))

# Get unique line_ids
line_ids = merged_wlb_vs_d2['line_id'].unique()

# Plot a line for each line_id
for line_id in sorted(line_ids):
    data = merged_wlb_vs_d2[merged_wlb_vs_d2['line_id'] == line_id]
    
    # Convert nodes to strings for categorical x-axis
    nodes_str = data['nodes'].astype(str)
    
    ax.plot(nodes_str, data['speedup'], 
            marker='o', markersize=8,
            label=f'NTokens {line_id:.0f}',
            linewidth=2)

# Set labels and title
ax.set_xlabel('Number of Nodes')
ax.set_ylabel('Speedup (WLBLLM/D2)')
ax.set_title('Speedup vs Number of Nodes')

# Add legend
ax.legend()

# Add grid
ax.grid(True, which="both", ls="-", alpha=0.2)

# Show plot
plt.tight_layout()
plt.show()

# %%
