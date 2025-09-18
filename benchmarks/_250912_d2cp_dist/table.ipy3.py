# %%
!ls /mnt/weka/home/yonghao.zhuang/jd/d2/benchmarks/_250912_d2cp_dist/logs.v1-sweep-memory-70b

# %%
import json
import os
root = "/mnt/weka/home/yonghao.zhuang/jd/d2/benchmarks/_250912_d2cp_dist/logs.v1-sweep-memory-70b"
root2 = "/mnt/weka/home/yonghao.zhuang/jd/d2/benchmarks/_250912_d2cp_dist/logs.v1-sweep-memory-70b"


rows = []

a = os.listdir(root2)
for folder in a:
    if not os.path.isdir(os.path.join(root2, folder)):
        continue
    file = os.path.join(root2, folder, "benchmark.raw.jsonl")
    if not os.path.exists(file):
        continue
    # print(file)
    with open(file, "r") as f:
        data = [json.loads(line) for line in f]
    # print(data)
    durations = [x['duration_ms'] for x in data]
    # print(durations)
    average_duration = sum(durations) / len(durations)
    name = folder[27:]
    groups = name.split("-")
    mode, cp_size, nodes, batch_size, num_tokens = groups
    print(f"{name}: {average_duration:.2f}ms")
    row = dict(
        name=name,
        mode=mode,
        cp_size=cp_size,
        nodes=nodes,
        batch_size=batch_size,
        num_tokens=num_tokens,
        average_duration=average_duration,
    )
    for i, d in enumerate(durations):
        row[f"sample_{i}"] = d
    rows.append(row)

a = os.listdir(root)
for folder in a:
    if not os.path.isdir(os.path.join(root, folder)):
        continue
    file = os.path.join(root, folder, "benchmark.raw.jsonl")
    if not os.path.exists(file):
        continue
    # print(file)
    with open(file, "r") as f:
        data = [json.loads(line) for line in f]
    # print(data)
    durations = [x['duration_ms'] for x in data]
    # print(durations)
    average_duration = sum(durations) / len(durations)
    name = folder[27:]
    groups = name.split("-")
    mode, cp_size, nodes, batch_size, num_tokens = groups
    print(f"{name}: {average_duration:.2f}ms")
    row = dict(
        name=name,
        mode=mode,
        cp_size=cp_size,
        nodes=nodes,
        batch_size=batch_size,
        num_tokens=num_tokens,
        # tolerance_factor=tolerance_factor,
        average_duration=average_duration,
    )
    for i, d in enumerate(durations):
        row[f"sample_{i}"] = d
    rows.append(row)
    # break
# %%
import pandas as pd
df = pd.DataFrame(rows)
df = df.sort_values(by=[
    "nodes", "batch_size", "num_tokens", "mode", "average_duration", "cp_size",
], ascending=False)
df.head(200)

# %%
