# %%
!ls logs.v1-ablation/
# %%
import os
import json

root_paths = [
    # Good line 8B 16 Node
    # "./logs.v1-ablation/0918_194000_PST/",
    # "./logs.v1-ablation/0918_184545_PST",
    # "./logs.v1-ablation/0919_183239_PST",
    # "./logs.v1-ablation/0919_014011_PST",
    # "./logs.v1-ablation/0919_181816_PST"

    # "./logs.v1-ablation/0919_183239_PST",
    # "./logs.v1-ablation/0919_193940_PST",

    # "./logs.v1-ablation/0918_184545_PST",

    # "./logs.v1-ablation/0918_191358_PST",
    # "./logs.v1-ablation/0918_194000_PST",
    # "./logs.v1-ablation/0918_210117_PST",
    # "./logs.v1-ablation/0918_211408_PST",
    # "./logs.v1-ablation/0918_211502_PST",
    # "./logs.v1-ablation/0919_013952_PST",
    # "./logs.v1-ablation/0919_014011_PST",
    # "./logs.v1-ablation/0919_123001_PST",
    # "./logs.v1-ablation/0919_181816_PST",
    # "./logs.v1-ablation/0919_183239_PST",
    # "./logs.v1-ablation/0919_192312_PST",
    # "./logs.v1-ablation/0919_192318_PST",
    # "./logs.v1-ablation/0919_193940_PST",
    # "./logs.v1-ablation/0919_195402_PST",
    # "./logs.v1-ablation/0919_195430_PST"
    "./logs.v1-ablation/0919_205936_PST"
]
success_configs = {}
rows = []
for root_path in root_paths:
    a = os.listdir(root_path)
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
        
        tolerance_factor = 0.05
        if "tol" in name:
            tolerance_factor = float(
                name.split("tol")[-1].strip()
            )
        is_signal = ("signal" in name)
        is_single_stream = ("single-stream" in name)

        # Read the benchmark file
        file = os.path.join(root_path, folder, "benchmark.raw.jsonl")
        if not os.path.exists(file):
            continue


        with open(file, "r") as f:
            data = [json.loads(line) for line in f]
        # print(data)
        # print(durations)
        # groups = name.split("-")
        # mode, cp_size, nodes, batch_size, num_tokens = groups
        print(f"{name}: {average_duration:.2f}ms")
        config = dict(
            mode=mode,
            cp_size=cp_size,
            nodes=nodes,
            batch_size=batch_size,
            num_tokens=num_tokens,
            tolerance_factor=tolerance_factor,
            is_signal=is_signal,
            is_single_stream=is_single_stream,
        )
        
        row = dict(
            name=name,
            dataset="pretrain",
            group_id=f"{nodes:02d}_{num_tokens}_{batch_size}",
            **config,
        )

        # should_sample_ids = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}
        should_sample_ids = None
        if should_sample_ids:
            durations = [x['duration_ms'] for idx, x in enumerate(data) if idx in should_sample_ids]
        else:
            durations = [x['duration_ms'] for idx, x in enumerate(data)]
        average_duration = sum(durations) / len(durations) if len(durations) > 0 else 0
        
        row['average_duration'] = average_duration
        for i, d in enumerate(durations):
            if should_sample_ids is not None and i in should_sample_ids:
                row[f"sample_{i}"] = d
        success_configs[name] = config
        rows.append(row)

# %%
rows


# %%
import pandas as pd
df = pd.DataFrame(rows)
df = df.sort_values(by=[
    "nodes", "batch_size", "num_tokens", "mode", "average_duration", "cp_size",
], ascending=True)
df

# %%

df_tol_only = df[
    (df['is_signal'] == False) &
    (df['is_single_stream'] == False)
]

df_tol_only = df_tol_only.sort_values(
    by=[
        'nodes',
        'tolerance_factor',
    ]
)
df_tol_only



# %%
# Only plot the 
# x-axis: tolerance factor
# y-axis: average duration

df1 = df[
    (df['is_signal'] == False) &
    (df['is_single_stream'] == False)
]

df1_n8 = df1[df1['nodes'] == 8].sort_values(by='tolerance_factor')
df1_n16 = df1[df1['nodes'] == 16].sort_values(by='tolerance_factor')

import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))

# Plot for 8 nodes
plt.plot(df1_n8['tolerance_factor'], df1_n8['average_duration'], 
         marker='o', label='8 nodes', linewidth=2)

# Plot for 16 nodes  
plt.plot(df1_n16['tolerance_factor'], df1_n16['average_duration'],
         marker='o', label='16 nodes', linewidth=2)

plt.xlabel('Tolerance Factor')
plt.ylabel('Average Latency (ms)')
plt.title('Ablation - Tolerance Factor (8B)')
plt.grid(True)
plt.legend()

plt.savefig("figure.ablation.tolerance_factor.8b.png")
plt.show()


# %%
df2 = df[
    (df['tolerance_factor'] == 0.05)
]
# %%

# Get unique node counts
nodes = sorted(df['nodes'].unique())

# Set up data for each mode and node count
signal_data = []
single_stream_data = []
normal_data = []

for n in nodes:
    signal_data.append(df[df['is_signal'] == True][df['nodes'] == n]['average_duration'].mean())
    single_stream_data.append(df[(df['is_signal'] == False) & 
                                (df['is_single_stream'] == True) & 
                                (df['nodes'] == n)]['average_duration'].mean())
    normal_data.append(df[(df['is_signal'] == False) & 
                         (df['is_single_stream'] == False) & 
                         (df['tolerance_factor'] == 0.05) &
                         (df['nodes'] == n)]['average_duration'].mean())

# Set up the plot
plt.figure(figsize=(10, 6))

# Set width of bars and positions of the bars
barWidth = 0.25
r1 = range(len(nodes))
r3 = [x + barWidth for x in r1]
r2 = [x + barWidth for x in r3]

# Create bars
plt.bar(r1, signal_data, width=barWidth, color='skyblue', label='Signal')
plt.bar(r2, single_stream_data, width=barWidth, color='lightgreen', label='Single Stream')
plt.bar(r3, normal_data, width=barWidth, color='salmon', label='Normal (TF=0.05)')

# Add value labels on top of each bar
for i, v in enumerate(signal_data):
    plt.text(r1[i], v, f'{v:.2f}', ha='center', va='bottom')
for i, v in enumerate(single_stream_data):
    plt.text(r2[i], v, f'{v:.2f}', ha='center', va='bottom')
for i, v in enumerate(normal_data):
    plt.text(r3[i], v, f'{v:.2f}', ha='center', va='bottom')

# Customize the plot
plt.xlabel('Number of Nodes')
plt.ylabel('Average Latency (ms)')
plt.title('Ablation - Signal vs Single Stream vs Normal (8B)')
plt.xticks([r + barWidth for r in range(len(nodes))], nodes)
plt.grid(True, axis='y')
plt.legend()

plt.savefig("figure.ablation.signal-vs-single_stream-vs-normal.8b.png")
plt.show()


# %%
