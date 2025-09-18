# %%

import os
os.chdir("/mnt/weka/home/yonghao.zhuang/jd/d2/benchmarks/_250907_large_scale_v6")
# %%
import wandb
import json
import os
import pandas as pd
import time
import datetime

benchmark_data = []
folders = os.listdir("logs.v2-large_scale")
for folder in folders:
    print(folder)
    name = folder.split(".")[-1]
    
    benchmark_file = os.path.join("logs.v2-large_scale", folder, "benchmark.raw.jsonl")
    if not os.path.exists(benchmark_file):
        continue

    # d2-cp1-n32-b1-t131072-normal
    parts = name.split("-")
    mode = parts[0]
    cp_size = parts[1]
    nodes = parts[2]
    batch_size = parts[3]
    num_tokens = parts[4]
    
    this_benchmark_data = []
    with open(benchmark_file, "r") as f:
        for line in f:
            data = json.loads(line)
            benchmark_data.append((
                name,
                data['sample_id'],
                data['duration_ms'],
                mode,
                cp_size,
                nodes,
                batch_size,
                num_tokens,
            ))

    print(f"🟡 {name} has {len(this_benchmark_data)} samples")

df = pd.DataFrame(benchmark_data, columns=['name', 'sample_id', 'duration_ms', 'mode', 'cp_size', 'nodes', 'batch_size', 'num_tokens'])
df = df.drop_duplicates(subset=['sample_id', 'name'], keep='last')


# Pivot the dataframe to have sample_id as rows and name as columns
pivot_df = df.pivot(index='sample_id', columns='name', values='duration_ms').round(2)
pivot_df.to_csv(f"pivot_df.csv")
print(pivot_df)

# Speedup dataframe
speedup_df = df[[
    'nodes', 'batch_size', 'num_tokens', 'mode', 'cp_size', 'sample_id', 'duration_ms',
]].sort_values(by=['nodes', 'batch_size', 'num_tokens', 'mode', 'cp_size'], ascending=True)
speedup_df.to_csv(f"speedup_df.tsv", sep="\t", index=False)

# Pivot speedup table such that fore each sample_id we have a column, and each row is ('nodes', 'batch_size', 'num_tokens', 'mode', 'cp_size')
pivot_speedup_df = speedup_df.pivot(index=['nodes', 'batch_size', 'num_tokens', 'mode', 'cp_size'], columns='sample_id', values='duration_ms').round(2)
pivot_speedup_df.to_csv(f"pivot_speedup_df.tsv", sep="\t", index=True)
print(pivot_speedup_df)
# %%
speedups = []
avg_times = []
for row in pivot_speedup_df.iterrows():
    node, bs, t, mode, cp = row[0]
    print(row[0])
    if mode == 'wlbllm':
        d2_avg_time = speedup_df[
            (speedup_df['nodes'] == node) &
            (speedup_df['batch_size'] == bs) &
            (speedup_df['num_tokens'] == t) &
            (speedup_df['mode'] == 'd2')
        ]['duration_ms'].mean()
        wlbllm_avg_time = row[1].mean()
        avg_times.append(wlbllm_avg_time)
        speedup = wlbllm_avg_time / d2_avg_time
        print(speedup)
        speedups.append(speedup)
    else:
        avg_times.append(row[1].mean())
        speedups.append(1)

pivot_speedup_df['avg_times'] = avg_times
pivot_speedup_df['speedup'] = speedups
pivot_speedup_df.to_csv(f"pivot_speedup_df_with_speedup.tsv", sep="\t", index=True)



# %%
benchmark_data = []
folders = os.listdir("logs.v2-large_scale")
for folder in folders:
    # print(folder)
    if not os.path.isdir(os.path.join("logs.v2-large_scale", folder)):
        continue
    name = folder.split(".")[-1]
    parts = name.split("-")
    mode = parts[0]
    cp_size = parts[1]
    nodes = parts[2]
    batch_size = parts[3]
    num_tokens = parts[4]

    log_files = os.listdir(os.path.join("logs.v2-large_scale", folder, "logs"))
    for log_file in log_files:
        log_file_path = os.path.join("logs.v2-large_scale", folder, "logs", log_file)
        if not log_file_path.endswith(".log"):
            continue
        with open(log_file_path, "r") as f:
            data = f.read()
        if "OutOfMemory" in data:
            # nodes	batch_size	num_tokens	mode	cp_size
            print(f"{nodes}\t{batch_size}\t{num_tokens}\t{mode}\t{cp_size}\t- OOM\t{name}")
            break

# %%
