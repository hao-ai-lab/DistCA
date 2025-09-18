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

def status(folder):
    is_oom = False
    is_past_one_test = False
    is_finished = False
    iterations_finished = -1
    exp_total_time = -1 # Elapsed time
    avg_duration = -1
    sample_ids = []
    durations = []
    samples = [] 
    reason = ""
    
    # Parse log file for status
    log_files = os.listdir(os.path.join("logs", folder, "logs"))
    log_files = log_files[:4]
    for log_file_name in log_files:
        log_file_path = os.path.join("logs", folder, "logs", log_file_name)
        with open(log_file_path, "r") as f:
            log_text = f.read()
        if "avg_time_per_iteration" in log_text:
            is_past_one_test = True
        if "OutOfMemoryError" in log_text:
            is_oom = True
        
    # Parse benchmark.raw.jsonl for iterations reached
    benchmark_raw_file = os.path.join("logs", folder, "benchmark.raw.jsonl")
    if os.path.exists(benchmark_raw_file):
        with open(benchmark_raw_file, "r") as f:
            data = [json.loads(line) for line in f]
        sample_ids = [x['sample_id'] for x in data]
        durations = [x['duration_ms'] for x in data]
        avg_duration = sum(durations) / len(durations)
        samples = [x['samples'] for x in data]
        iterations_finished = len(durations)

    # Check if the experiment is finished
    is_finished = os.path.exists(os.path.join("logs", folder, "benchmark.json"))

    # Get the experiment total time, including all the setup and stuff
    if os.path.exists(os.path.join("logs", folder, "slurm.stdout")):
        with open(os.path.join("logs", folder, "slurm.stdout"), "r") as f:
            data = f.read()
        # Find the line that says "Elapsed time: X seconds"
        for line in data.split("\n"):
            if "elapsed_time" in line:
                exp_total_time = line.split("=")[1].strip()
                break
        
    # Reason about what happened
    if is_finished:
        reason = f"🟢{iterations_finished}"
    elif iterations_finished >= 0:
        reason = f"⚪️{iterations_finished}"
    elif is_oom:
        reason = "⚠️OOM"
    elif is_past_one_test:
        reason = "⚪️Past one test"
    else:
        reason = "❌Unknown"



    entry = {
        'reason': reason,
        'is_finished': is_finished,
        'is_oom': is_oom,
        'is_past_one_test': is_past_one_test,
        'it': iterations_finished,
        'exp_total_time': exp_total_time,
        'avg_duration': avg_duration,
    }
    if durations:
        for i, duration in enumerate(durations):
            entry[f'duration[{i}]'] = duration
    if samples:
        for i, sample in enumerate(samples):
            entry[f'sample[{i}]'] = sample
    return entry

mem_rows = []

for folder in a:
    mem_log_file = os.path.join("logs", folder, "mem-log", "mem.rank0.log.jsonl")
    desc_file = os.path.join("logs", folder, "README.md")
    if not os.path.exists(desc_file):
        continue
    desc = parse_readme(desc_file)
    desc['exp'] = folder
    if os.path.exists(mem_log_file):
        with open(mem_log_file, "r") as f:
            data = [json.loads(line) for line in f]
        df = pd.DataFrame(data)
        if not df.empty:
            max_mem = df['pynvml_gpu_memory_usage'].max().item()
            desc['max_mem'] = max_mem
    
    timed_item = status(folder)
    desc.update(timed_item)
    mem_rows.append(desc)

    

mem_dfs = pd.DataFrame(mem_rows)

# Now adding some statistical columns
# if 'max_mem' in mem_dfs.columns:
#     mem_dfs['max_mem'] = mem_dfs['max_mem'].astype(float).apply(lambda x: x / 1024 if x > 1024 else x).round(2)
mem_dfs = mem_dfs.rename(columns={
    "EXPERIMENT_ADD_SELECTIVE_CKPT": "ckpt",
    "MODEL_PATH": "model",
    "NUM_LAYERS": "layers",
    "NUM_TOKENS": "toks",
    "BATCH_SIZE": "bs",
    "NNODES": "nodes",
    "EXPERIMENT_SHOULD_RESEND_QKV": "resend",
})


# create_gid(mem_dfs, ['nodes', 'toks', 'bs', 'model', 'layers'])
mem_dfs['gid'] = mem_dfs['nodes'].apply(lambda x: f"{int(x):02d}") + "_" + mem_dfs['toks'].astype(str) + "_" + mem_dfs['bs'].astype(str) + "_" + mem_dfs['layers'].astype(str) + "_" + mem_dfs['model']

mem_dfs['mode'] = mem_dfs['MODE'].apply(lambda x: "🟦d2" if "d2" in x else "🟥wlbllm")

front_columns = ['gid', 'reason', 'is_finished', 'exp_total_time', 'mode', 'CP_SIZE', 'NUM_LAYERS', 'NUM_TOKENS', 'BATCH_SIZE', 'MODEL_PATH', 'NNODES', "EXPERIMENT_ADD_SELECTIVE_CKPT", "EXPERIMENT_SHOULD_RESEND_QKV", "exp", 'it', 'avg_duration', 'max_mem', ]
front_columns = [c for c in front_columns if c in mem_dfs.columns]
front_columns.extend([col for col in mem_dfs.columns if col not in front_columns])
mem_dfs = mem_dfs[front_columns]


pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
mem_dfs.to_csv("check_status.csv", sep="\t", index=False)
mem_dfs

# %%
# Filter for valid durations and create comparison groups
valid_df = mem_dfs[mem_dfs['avg_duration'] > 0].copy()

# Get min duration for each mode within each gid
grouped = valid_df.groupby(['gid', 'mode'])['avg_duration'].min().unstack()

# Calculate speedup (wlbllm time / d2 time)
grouped['speedup'] = grouped['🟥wlbllm'] / grouped['🟦d2']

# Sort by speedup
grouped = grouped.sort_values('speedup', ascending=False)
grouped.sort_values('gid', ascending=True, inplace=True)
grouped = grouped.reset_index()
grouped


# %%
grouped['wlbllm_entries'] = None
for gid in grouped['gid']:
    # print(gid)
    wlbllm_df = mem_dfs[
        (mem_dfs['gid'] == gid) & (mem_dfs['MODE'].apply(lambda x: "wlbllm" in x))
    ][['CP_SIZE', 'avg_duration', 'reason']]
    wlbllm_entries = wlbllm_df.to_dict(orient='records')
    wlbllm_entries = [
        dict(
            cp=r['CP_SIZE'],
            t=r['avg_duration'],
            r=r['reason'],
        )
        for r in wlbllm_entries
    ]
    # print(gid, wlbllm_entries)
    grouped.loc[
        grouped['gid'] == gid,
        'wlbllm_entries'
    ] = str(wlbllm_entries)

# %%
grouped.to_csv("check_status_grouped.csv", sep="\t", index=False)
grouped
# %%
# Dump success traces
mem_dfs
# %%
