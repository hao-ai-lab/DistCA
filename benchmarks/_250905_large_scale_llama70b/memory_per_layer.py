# %%

import os
import json
import pandas as pd

a = !ls logs/


def parse_readme(readme_file):
    with open(readme_file, "r") as f:
        data = f.read()
    result = {}
    for line in data.split("\n"):
        if not line.startswith("-"):
            continue
        key, value = line.split(":", 1)
        key = key.replace("- ", "")
        result[key] = value
    return result

mem_rows = []

for folder in a:
    b = os.path.join("logs", folder, "mem-log", "mem.rank0.log.jsonl")
    desc_file = os.path.join("logs", folder, "README.md")
    # print(b)
    if not os.path.exists(b):
        continue
    desc = parse_readme(desc_file)
    with open(b, "r") as f:
        data = [json.loads(line) for line in f]
    df = pd.DataFrame(data)
    if df.empty:
        continue
    max_mem = df['pynvml_gpu_memory_usage'].max().item()
    desc['max_mem'] = max_mem
    desc['exp'] = folder
    mem_rows.append(desc)
    

mem_dfs = pd.DataFrame(mem_rows)
# move max_mem to the first column
front_columns = ['max_mem', 'NUM_LAYERS', 'NUM_TOKENS', 'BATCH_SIZE', 'MODEL_PATH', 'NNODES', "EXPERIMENT_ADD_SELECTIVE_CKPT", "EXPERIMENT_SHOULD_RESEND_QKV", "exp"]
front_columns = [c for c in front_columns if c in mem_dfs.columns]
front_columns.extend([col for col in mem_dfs.columns if col not in front_columns])
mem_dfs['max_mem'] = mem_dfs['max_mem'].astype(float).apply(lambda x: x / 1024 if x > 1024 else x).round(2)
mem_dfs = mem_dfs[front_columns]
mem_dfs = mem_dfs.rename(columns={
    "EXPERIMENT_ADD_SELECTIVE_CKPT": "ckpt",
    "MODEL_PATH": "model",
    "NUM_LAYERS": "layers",
    "NUM_TOKENS": "toks",
    "BATCH_SIZE": "bs",
    "NNODES": "nodes",
    "EXPERIMENT_SHOULD_RESEND_QKV": "resend",
})

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
mem_dfs
# %%
# mem_dfs.columns
# %%
a
# %%
