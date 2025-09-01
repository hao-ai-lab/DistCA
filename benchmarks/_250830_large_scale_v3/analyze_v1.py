# %%
import os
import pandas as pd
# glo
import glob

# %%
log_path = "logs.v1"
a = os.listdir(log_path)
len(a)
# %%
import re
def parse_key_value_file(path: str) -> dict:
    """Parse a file with lines formatted as 'key: value' into a dictionary."""
    result = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line.startswith("-"):
                continue
            if not line or ":" not in line:
                continue  # skip empty or malformed lines
            key, value = line.split(":", 1)  # split only on the first colon
            key = key[2:]
            result[key.strip()] = value.strip()
    return result
# %%
b = f"{log_path}/{a[0]}/README.md"
b
parse_key_value_file(b)
os.path.exists(b)
# %%
import json

DF_INCLUDE_BENCHMARK = True
# DF_INCLUDE_BENCHMARK = False
# DF_ONLY_SHOW_SUCCESS = True
DF_ONLY_SHOW_SUCCESS = False
DF_ONLY_SHOW_FAILED = False
# DF_ONLY_SHOW_D2 = True
DF_ONLY_SHOW_D2 = False


import subprocess
active_jobs = subprocess.check_output(['squeue', '--me', '--noheader', '-o', '%A']).decode().splitlines()
# print(active_jobs)
# print(f"There are {len(active_jobs)} active jobs")

# %%
# scancel --name=d2-v3 -u $USER
# print("scancel "+ " ".join(active_jobs))

# %%

results = []
for exp_name in a:
    file_path = f"{log_path}/{exp_name}/README.md"
    if not os.path.exists(file_path):
        continue
    config = parse_key_value_file(file_path)

    slurm_env_file = f"{log_path}/{exp_name}/slurm.env"
    if not os.path.exists(slurm_env_file):
        print(f"Skipping {exp_name} because slurm.env file does not exist")
        continue
    with open(slurm_env_file, "r") as f:
        slurm_env = f.read()
        slurm_env = slurm_env.split("\n")
        slurm_env_dict = {}
        for line in slurm_env:
            line = line.strip()
            if not line or "=" not in line:
                continue
            key, value = line.split("=", 1)  # split only on the first colon
            slurm_env_dict[key.strip()] = value.strip()

        nnodes = int(slurm_env_dict["SLURM_NNODES"])
        batch_size = int(slurm_env_dict["BATCH_SIZE"])
        num_layers = int(slurm_env_dict["NUM_LAYERS"])
        EXPERIMENT_NVSHMEM_BUFFER_SIZE_GB = int(slurm_env_dict.get("EXPERIMENT_NVSHMEM_BUFFER_SIZE_GB", -1))
        MODEL_PATH = slurm_env_dict.get("MODEL_PATH", "")
        # rdzv_id = int(slurm_env_dict["RANDOM"])

    benchmark_file = f"{log_path}/{exp_name}/benchmark.json"
    run_successfully = os.path.exists(benchmark_file)
    benchmark_datas = {}
    if run_successfully and DF_INCLUDE_BENCHMARK:
        with open(benchmark_file, "r") as f:
            benchmark = json.load(f)
        samples = benchmark["samples"]
        for sample in samples:
            sample_id = sample["sample_id"]
            samples_duration = sample["duration_ms"]
            benchmark_datas[f"sample[{sample_id}]"] = samples_duration
    
    DP_SIZE = nnodes // (
        int(config.get("CP_SIZE")) * int(config.get("PP_SIZE"))
    )

    log_files = glob.glob(f"{log_path}/{exp_name}/logs/*")
    is_started = False
    is_comm_init = False # "Communication groups initialized"
    is_reached_flash_attn_3 = False # "flash_attn_3"
    is_past_warmup = False # "warmup done"
    is_past_one_test = False # "avg_time_per_iteration"
    iteration_reached = -1
    is_exited = False # "Finished test and exit"

    for i, log_file in enumerate(log_files):
        # print(f"\n\n=== {log_file} ===")
        with open(log_file, "r") as f:
            log_text = f.read()
            if "🟡" in log_text:
                is_started = True
            if "Communication groups initialized" in log_text:
                is_comm_init = True
            if "flash_attn_3" in log_text:
                is_reached_flash_attn_3 = True
            if "warmup done" in log_text:   
                is_past_warmup = True
            if "avg_time_per_iteration" in log_text:
                is_past_one_test = True
            # Search for [Sample ID=(%d)] and take the largest one
            try:
                sample_id_matches = re.findall(r"Sample ID=\(\d+\)", log_text)
                if sample_id_matches:
                    sample_id_matches = [
                        t.split("Sample ID=(")[1].split(")")[0]
                        for t in sample_id_matches
                    ]
                    sample_id_matches = [
                        int(t) for t in sample_id_matches
                    ]
                    sample_id = max(sample_id_matches)
                    iteration_reached = max(iteration_reached, sample_id)
            except Exception as e:
                print(e)
                pass
            if "Finished test and exit" in log_text:
                is_exited = True

    # Check if the job id is still active
    # squeue --me --noheader -o "%A"
    jobid = slurm_env_dict["SLURM_JOB_ID"]
    jobid = str(jobid)
    is_running = jobid in active_jobs
    
    
    results.append(dict(
        model=MODEL_PATH,
        is_running=is_running,
        batch_size=batch_size,
        num_layers=num_layers,
        success=run_successfully,
        is_started=is_started,
        is_comm_init=is_comm_init,
        is_reached_fa3=is_reached_flash_attn_3,
        is_past_warmup=is_past_warmup,
        is_past_one_test=is_past_one_test,
        it_reached=iteration_reached,
        is_exited=is_exited,
        exp_name=exp_name,
        buffer_size=EXPERIMENT_NVSHMEM_BUFFER_SIZE_GB,
        nnodes=nnodes,
        DP_SIZE=DP_SIZE,
        jobid=jobid,
        **config,
        # rdzv_id=rdzv_id,
        **benchmark_datas,
    ))

# %%
results[0]

# %%

df = pd.DataFrame(results)
df['NUM_TOKENS'] = df['NUM_TOKENS'].astype(int)
if DF_ONLY_SHOW_D2:
    df = df[df['MODE'] == 'd2']

sample_columns = [col for col in df.columns if col.startswith("sample[")]
df['total_time'] = df[sample_columns].sum(axis=1)

df_display = df.copy()

if DF_ONLY_SHOW_SUCCESS:
    df_display = df_display[df_display['success'] == True]
df_display['success'] = df_display['success'].map({True: '✅', False: '❌'})
df_display['modji'] = df_display['MODE'].map({
    'wlbllm': '🟥',
    'd2': '🟦',
})
# Sort by gid
df_display['gid'] = df_display['nnodes'].astype(str) + "_" + df_display['NUM_TOKENS'].astype(str) + "_" + df_display['BATCH_SIZE'].astype(str) + "_" + df_display['num_layers'].astype(str)
df_display['eid'] = df_display['nnodes'].astype(str) + "_" + df_display['NUM_TOKENS'].astype(str) + "_" + df_display['BATCH_SIZE'].astype(str) + "_" + df_display['MODE'].astype(str) + "_" + df_display['CP_SIZE'].astype(str) + "_" + df_display['num_layers'].astype(str)


# Then, aggregate by gid, and then compare the (only one) d2 row's total_time with all the wlbllm row's total_time in that aggregated group. Add a speedup = wlbllm min total time in that window / d2 total time
# Group by gid and calculate speedup
import numpy as np
speedups = []
for gid, group in df_display.groupby('gid'):
    d2_rows = group[group['modji'] == '🟦']
    wlbllm_rows = group[group['modji'] == '🟥']
    
    wlbllm_min_time = wlbllm_rows[wlbllm_rows['total_time'] > 0]['total_time'].min()
    # success
    is_wlbllm_all_success = wlbllm_rows['success'].all()

    if pd.isna(wlbllm_min_time):
        continue

    for idx, d2_row in d2_rows.iterrows():
        d2_time = d2_row['total_time']
        if d2_time == 0 or pd.isna(d2_time):
            break
        speedup = wlbllm_min_time / d2_time
        speedup = f"{speedup:.2f} (🟢)" if is_wlbllm_all_success else f"{speedup:.2f} (❌)"
        speedups.append((idx, speedup))

# Add speedup column
df_display['speedup'] = ""
for idx, speedup in speedups:
    df_display.loc[idx, 'speedup'] = speedup

# Format speedup to 2 decimal places where it exists
# df_display['speedup'] = df_display['speedup'].apply(lambda x: f'{x:.2f}' if pd.notnull(x) else '')

# Reorder columns to put key columns first
key_columns = ['gid', 'nnodes', 'NUM_TOKENS', 'BATCH_SIZE', 'modji',]
front_columns = ['gid', "eid", 'modji', 'is_running', 'success', 'speedup', 'total_time']
other_columns = [col for col in df_display.columns if col not in front_columns]
df_display = df_display[front_columns + other_columns]
df_display.sort_values(by=key_columns, ascending=True, inplace=True)


# When display the total_time column, make it %2f
df_display['total_time'] = df_display['total_time'].astype(float).round(2)
# df_display['total_time'] = df_display['total_time'].apply(lambda x: '' if pd.isna(x) or x == 0 else round(float(x), 2))
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)


# Filter some columns
# print("Dropped some columns: total_time, speedup")
# df_display.drop(columns=['total_time', 'speedup', 'nnodes', 'NUM_TOKENS', 'BATCH_SIZE', 'num_layers', 'modji', 'DP_SIZE'], inplace=True)

# Render True False values to emoji True: ✅, False: ❌
# - is_comm_init
# - is_reached_fa3
# - is_past_warmup
# - is_past_one_test
# - is_exited
df_display['is_started'] = df_display['is_started'].map({True: '🟢', False: '🔴'})
df_display['is_comm_init'] = df_display['is_comm_init'].map({True: '🟢', False: '🔴'})
df_display['is_reached_fa3'] = df_display['is_reached_fa3'].map({True: '🟢', False: '🔴'})
df_display['is_past_warmup'] = df_display['is_past_warmup'].map({True: '🟢', False: '🔴'})
df_display['is_past_one_test'] = df_display['is_past_one_test'].map({True: '🟢', False: '🔴'})
# df_display['it_reached'] = df_display['it_reached'].map({True: '🟢', False: '🔴', -1: '❌'})
df_display['is_exited'] = df_display['is_exited'].map({True: '🟢', False: '🔴'})
df_display['is_running'] = df_display['is_running'].map({True: '💨', False: '🛑'})


# df_display['gid'] = df_display['nnodes'].astype(str) + "_" + df_display['NUM_TOKENS'].astype(str) + "_" + df_display['BATCH_SIZE'].astype(str)
# df_display.sort_values(by=['exp_name', ], ascending=True)

df_display = df_display.sort_values(by=['num_layers', 'nnodes', 'NUM_TOKENS', 'BATCH_SIZE', 'MODE', 'CP_SIZE', ], ascending=True)
df_display

# %%
# Save this to a file
df_display.to_csv("analyze_status.tsv", index=False, sep="\t")
# %%
import numpy as np
speedups = []
for gid, group in df_display.groupby('gid'):
    print(gid)
    d2_rows = group[group['modji'] == '🟦']
    wlbllm_rows = group[group['modji'] == '🟥']
    
    wlbllm_min_time = wlbllm_rows[wlbllm_rows['total_time'] > 0]['total_time'].min()
    print(wlbllm_min_time)
    # success
    is_wlbllm_all_success = wlbllm_rows['success'].all()

    if pd.isna(wlbllm_min_time):
        continue

    for idx, d2_row in d2_rows.iterrows():
        d2_time = d2_row['total_time']
        if d2_time == 0 or pd.isna(d2_time):
            break
        speedup = wlbllm_min_time / d2_time
        speedup = f"{speedup:.2f} (🟢)" if is_wlbllm_all_success else f"{speedup:.2f} (❌)"
        speedups.append((idx, speedup))
    break

speedups
# %%
# Get all unique gids and create empty dataframe with both modji values for each
all_gids = df_display['gid'].unique()
all_modjis = ['🟦', '🟥']  # d2 and wlb
rows = []
for gid in all_gids:
    for modji in all_modjis:
        rows.append({'gid': gid, 'modji': modji})
template_df = pd.DataFrame(rows)

# Filter for valid times and calculate stats
valid_times = df_display[df_display['total_time'] > 0].groupby(['gid', 'modji'])['total_time']
min_times = valid_times.min().reset_index()
avg_times = valid_times.mean().reset_index()
std_times = valid_times.std().reset_index()

# Merge stats with template to ensure all gid/modji combinations exist
min_times = template_df.merge(min_times, on=['gid', 'modji'], how='left')
min_times = min_times.merge(avg_times[['gid', 'modji', 'total_time']].rename(columns={'total_time': 'avg_time'}), 
                           on=['gid', 'modji'], how='left')
min_times = min_times.merge(std_times[['gid', 'modji', 'total_time']].rename(columns={'total_time': 'std_time'}),
                           on=['gid', 'modji'], how='left')

# Add tuple column by splitting gid and sort
min_times['gid_tuple'] = min_times['gid'].apply(lambda x: tuple(map(int, x.split('_'))))
min_times = min_times.sort_values(['gid_tuple', 'modji'])

# Calculate speedup by comparing d2 and wlb in same group
speedups = []
for gid in all_gids:
    d2_time = min_times[(min_times['gid'] == gid) & (min_times['modji'] == '🟦')]['total_time'].iloc[0]
    wlb_time = min_times[(min_times['gid'] == gid) & (min_times['modji'] == '🟥')]['total_time'].iloc[0]
    
    if pd.isna(d2_time) or pd.isna(wlb_time):
        speedups.extend([None, None])
    else:
        speedup = wlb_time / d2_time
        speedups.extend([speedup, speedup])

min_times['speedup'] = speedups

# Sort the min_times by gid_tuple and modji
min_times = min_times.sort_values(['gid_tuple', 'modji'])

# Display results
print(min_times[['gid', 'modji', 'total_time', 'avg_time', 'std_time', 'speedup']].to_csv(index=False, sep="\t"))

# %%
