# %%
import os
import json
root_path = "/mnt/weka/home/yonghao.zhuang/jd/d2/benchmarks/_250920_test_pp/logs.v13-large-scale-pp"
a = os.listdir(root_path)
a = sorted(a)

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
    pp_size = int(readme_config["pp_size"].strip())
    nodes = int(readme_config["nnodes"].strip())
    batch_size = float(readme_config["batch_size"].strip())
    num_microbatch = int(readme_config["num_microbatch"].strip())
    num_tokens = int(readme_config["num_tokens"].strip())
    model_path = readme_config["model_path"].strip()
    model_size = None
    if "34b" in model_path.lower():
        model_size = "34b"
    elif "8b" in model_path.lower():
        model_size = "8b"
    # dataset = readme_config["sample_name"].strip()

    # Read the environment file
    environment_file = os.path.join(root_path, folder, "slurm.env")
    if not os.path.exists(environment_file):
        continue
    with open(environment_file, "r") as f:
        environment_data = f.read()
        environment_data = environment_data.split("\n")
        environment_data = {x.split("=")[0]: x.split("=")[1] for x in environment_data if "=" in x}
    dataset = environment_data["SAMPLE_NAME"].strip()


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

    total_batch_size = batch_size * num_microbatch
    ratio = 8 / (nodes / total_batch_size)
    config = dict(
        mode=mode,
        cp_size=cp_size,
        pp_size=pp_size,
        nodes=nodes,
        ratio=ratio,
        batch_size=batch_size,
        num_tokens=num_tokens,
        num_microbatch=num_microbatch,
        total_batch_size=total_batch_size,
        model_path=model_path,
        model_size=model_size,
    )
    row = dict(
        name=name,
        dataset=dataset,
        group_id=f"{nodes:02d}_{num_tokens}_{batch_size}",
        **config,
        average_duration=average_duration,
    )
    for i, d in enumerate(durations):
        row[f"sample_{i}"] = d
    success_configs[name] = config
    rows.append(row)

# %%
dataset

# %%
import pandas as pd
df = pd.DataFrame(rows)
df = df.sort_values(by=[
    "model_size", "num_tokens", "ratio", "nodes", "mode",
    "average_duration", "total_batch_size",  "cp_size",
    'num_microbatch', 'batch_size',
], ascending=False)
df_to_save = df.copy()
front_columns = ['model_size', 'num_tokens', 'ratio', 'nodes', 'total_batch_size', 'num_microbatch', 'batch_size', 'mode', 'cp_size',]
back_columns = [x for x in df_to_save.columns if x not in front_columns]
df_to_save = df_to_save[front_columns + back_columns]
df_to_save = df_to_save.drop(columns=['name', 'group_id', 'model_path'])
df_to_save.rename(columns={
    'num_microbatch': 'mb',
    'batch_size': 'bs',
    'total_batch_size': 'tbs',
}, inplace=True)
df_to_save
df_to_save.to_csv("table.tsv", index=True, sep="\t")

# %%
wlb_groups_best = df[
    df['mode'] == 'wlbllm'
].groupby(["model_size",  "num_tokens", "ratio", "nodes", "total_batch_size", "dataset"])['average_duration'].min().reset_index()
wlb_groups_best
# %%
d2_groups_best = df[
    df['mode'] == 'd2'
].groupby(["model_size",  "num_tokens", "ratio", "nodes", "total_batch_size", "dataset"])['average_duration'].min().reset_index()
d2_groups_best
# %%
# merge wlb_groups_best and d2_groups_best - 
merged_wlb_vs_d2 = pd.merge(wlb_groups_best, d2_groups_best, on=['model_size', 'nodes', 'num_tokens', 'ratio', 'total_batch_size', 'dataset'], how='left', suffixes=('_wlb', '_d2'))
merged_wlb_vs_d2['speedup'] = merged_wlb_vs_d2['average_duration_wlb'] / merged_wlb_vs_d2['average_duration_d2']
merged_wlb_vs_d2['linear_speedup'] = merged_wlb_vs_d2['num_tokens'] / merged_wlb_vs_d2['total_batch_size']
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
merged_wlb_vs_d2_display = merged_wlb_vs_d2.sort_values(by=['model_size', 'num_tokens', 'nodes'], ascending=True)
front_columns = ['model_size', 'nodes', 'num_tokens', 'ratio', 'total_batch_size', 
'speedup', 'dataset',
'average_duration_wlb', 'average_duration_d2', 
'linear_speedup', 'line_id']
back_columns = [x for x in merged_wlb_vs_d2_display.columns if x not in front_columns]
merged_wlb_vs_d2_display = merged_wlb_vs_d2_display[front_columns + back_columns]
merged_wlb_vs_d2_display__wlbllm = merged_wlb_vs_d2_display[merged_wlb_vs_d2_display['dataset'] == 'wlbllm']
merged_wlb_vs_d2_display__prolong = merged_wlb_vs_d2_display[merged_wlb_vs_d2_display['dataset'] == 'prolong']

# %%
print("wlbllm")
display(merged_wlb_vs_d2_display__wlbllm)
print(
    merged_wlb_vs_d2_display__wlbllm.to_csv("merged_wlb_vs_d2_display__wlbllm.tsv", index=False, sep="\t")
)

print("prolong")
display(merged_wlb_vs_d2_display__prolong)
print(
    merged_wlb_vs_d2_display__prolong.to_csv("merged_wlb_vs_d2_display__prolong.tsv", index=False, sep="\t")
)

# %%
# Plot speedup vs nodes for each num_tokens/batch_size combination
import matplotlib.pyplot as plt

# Get unique model sizes and datasets
model_sizes = sorted(merged_wlb_vs_d2['model_size'].unique())
datasets = sorted(merged_wlb_vs_d2['dataset'].unique())

# Create subplot grid - one row per dataset, one column per model size
fig, axes = plt.subplots(len(datasets), len(model_sizes), 
                        figsize=(5*len(model_sizes), 4*len(datasets)))

# Handle case of single row or column
if len(datasets) == 1:
    axes = axes.reshape(1, -1)
if len(model_sizes) == 1:
    axes = axes.reshape(-1, 1)

# Plot data for each dataset and model size combination
for row_idx, dataset in enumerate(datasets):
    for col_idx, model_size in enumerate(model_sizes):
        ax = axes[row_idx, col_idx]
        
        # Get data for this dataset and model size
        data = merged_wlb_vs_d2[
            (merged_wlb_vs_d2['dataset'] == dataset) & 
            (merged_wlb_vs_d2['model_size'] == model_size)
        ]
        line_ids = sorted(data['line_id'].unique())
        
        # Plot a line for each line_id
        for line_id in line_ids:
            line_data = data[data['line_id'] == line_id]
            nodes_str = line_data['nodes'].astype(str)
            
            ax.plot(nodes_str, line_data['speedup'],
                    marker='o', markersize=8,
                    label=f'NTokens {line_id:.0f}k',
                    linewidth=2)
        
        # Set labels and title
        ax.set_xlabel('Number of Nodes')
        if col_idx == 0:  # Only leftmost plots need y-label
            ax.set_ylabel('Speedup (WLBLLM/D2)')
        
        # Set title only for top row
        if row_idx == 0:
            ax.set_title(f'Model Size: {model_size}')
            
        # Add dataset label to rightmost plots
        if col_idx == len(model_sizes)-1:
            ax.text(1.05, 0.5, f'Dataset: {dataset}', 
                   rotation=-90, transform=ax.transAxes,
                   verticalalignment='center')
        
        # Add legend and grid
        ax.legend()
        ax.grid(True, which="both", ls="-", alpha=0.2)

# Adjust layout
plt.tight_layout()
plt.show()

# %%
