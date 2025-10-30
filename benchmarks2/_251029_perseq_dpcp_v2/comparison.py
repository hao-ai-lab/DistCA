# %%
import os
import json

# --------------------
# 3D Results
# --------------------

# root_path = "/mnt/weka/home/yonghao.zhuang/jd/d2/benchmarks2/_251020_perseq_dpcp/logs.v1-baseline-n8-8B"
# root_path =  "/mnt/weka/home/yonghao.zhuang/jd/d2/benchmarks2/_251020_perseq_dpcp/logs.v1-baseline-n16-8B"
# root_path =  "/mnt/weka/home/yonghao.zhuang/jd/d2/benchmarks2/_251020_perseq_dpcp/logs.v1-baseline-n32-8B"
# root_path = "/mnt/weka/home/yonghao.zhuang/jd/d2/benchmarks2/_251020_perseq_dpcp/logs.v2-n8-8B"
# root_path = "/mnt/weka/home/yonghao.zhuang/jd/d2/benchmarks2/_251020_perseq_dpcp/logs.v2-n8-34B"
# root_path = "/mnt/weka/home/yonghao.zhuang/jd/d2/benchmarks2/_251020_perseq_dpcp/logs.v2-n16-8B"
# root_path = "/mnt/weka/home/yonghao.zhuang/jd/d2/benchmarks2/_251020_perseq_dpcp/logs.v2-n16-34B"
# root_path = "/mnt/weka/home/yonghao.zhuang/jd/d2/benchmarks2/_251020_perseq_dpcp/logs.v2-n32-8B"
# root_path = "/mnt/weka/home/yonghao.zhuang/jd/d2/benchmarks2/_251020_perseq_dpcp/logs.v2-n32-34B"

# root_path = "/mnt/weka/home/yonghao.zhuang/jd/d2/benchmarks2/_251029_perseq_dpcp_v2/logs.v1-n8-8B"
root_path = "/mnt/weka/home/yonghao.zhuang/jd/d2/benchmarks2/_251029_perseq_dpcp_v2/logs.v1-n8-34B"


only_focus_on_sample_id = {}
# only_focus_on_sample_id = {0,1,2,3,4}
# only_focus_on_sample_id = {0,1,2,3,4,5,6,7,8}
# only_focus_on_sample_id = {0,1,2,3}

a = os.listdir(root_path)
a = sorted(a)

success_configs = {}
rows = []
for folder in a:

    if not os.path.isdir(os.path.join(root_path, folder)):
        continue
    # if 'old' in folder:
    #     continue
    
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
    print(folder)
    mode = readme_config["mode"].strip()
    cp_size = int(readme_config["cp_size"].strip())
    pp_size = int(readme_config["pp_size"].strip())
    nodes = int(readme_config["nnodes"].strip())
    batch_size = float(readme_config["batch_size"].strip())
    num_microbatch = 1 # int(readme_config["num_microbatch"].strip())
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
        print(f"Skip {folder} because file {file} does not exist")
        continue


    with open(file, "r") as f:
        data = [json.loads(line) for line in f]
    
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
        # average_duration=average_duration,
    )
    
    # # print(data)
    durations = [x['duration_ms'] for x in data]
    # # print(durations)
    # average_duration = sum(durations) / len(durations)
    # # groups = name.split("-")
    # # mode, cp_size, nodes, batch_size, num_tokens = groups
    # print(f"{name}: {average_duration:.2f}ms")

    if pp_size == 1 and num_microbatch > 1:
        print(f"Skip {folder} because pp_size == 1 and num_microbatch > 1")
        continue


    # only_focus_on_sample_id = {0}
    # only_focus_on_sample_id = {0,1,2,3}
    
    sample_durations = {}
    for i, d in enumerate(durations):
        if (only_focus_on_sample_id and i in only_focus_on_sample_id) or not only_focus_on_sample_id:
            sample_durations[f"sample_{i}"] = d
    num_samples = len(sample_durations)
    row['num_samples'] = num_samples
    average_duration = sum(sample_durations.values()) / len(sample_durations) if len(sample_durations) > 0 else 0
    row['average_duration'] = average_duration
    row.update(sample_durations)

    success_configs[name] = config
    rows.append(row)

# %%
row

# %%
import pandas as pd
df = pd.DataFrame(rows)
df = df.sort_values(by=[
    "model_size", "num_tokens", "ratio", "nodes", "mode",
    "average_duration", "total_batch_size",  "cp_size",
    'num_microbatch', 'batch_size',
], ascending=False)
df_to_save = df.copy()
# n  bs  mb    t       mode  cp  pp tp     comment        env_var
front_columns = [
    'model_size', 'nodes', 'batch_size',  'num_microbatch', 'num_tokens', 
    'mode', 'cp_size', 'pp_size', 
    'ratio', 'total_batch_size',  
]
back_columns = [x for x in df_to_save.columns if x not in front_columns]
df_to_save = df_to_save[front_columns + back_columns]
df_to_save = df_to_save.drop(columns=['name', 'group_id', 'model_path'])
df_to_save.rename(columns={
    'num_microbatch': 'mb',
    'batch_size': 'bs',
    'total_batch_size': 'tbs',
}, inplace=True)
df_to_save.to_csv("table.tsv", index=True, sep="\t")
df_to_save

# %%
# wlb_groups_best = df[
#     df['mode'] == 'wlbllm'
# ].groupby(["model_size",  "num_tokens", "ratio", "nodes", "total_batch_size", "dataset"]).agg({
#     'average_duration': ['min', lambda x: list(x)],
#     'pp_size': lambda x: list(x),
#     'cp_size': lambda x: list(x),
#     'num_microbatch': lambda x: list(x),
#     'batch_size': lambda x: list(x),
# }).reset_index()
# wlb_groups_best.columns = ['model_size', 'num_tokens', 'ratio', 'nodes', 'total_batch_size', 'dataset', 'average_duration', 'average_duration_list', 'pp_size_list', 'cp_size_list', 'num_microbatch_list', 'batch_size_list']

# wlb_groups_best['config_list'] = wlb_groups_best.apply(
#     lambda x: [
#         (batch_size, num_microbatch, cp, pp, round(duration, 2))
#         for cp, pp, duration, num_microbatch, batch_size in zip(
#             x['cp_size_list'],
#             x['pp_size_list'],
#             x['average_duration_list'],
#             x['num_microbatch_list'],
#             x['batch_size_list'],
#         )
#     ], axis=1
# )
# wlb_groups_best['config_list'] = wlb_groups_best['config_list'].apply(
#     lambda x: sorted(x, key=lambda x: x[-1])
# )
# wlb_groups_best['best_config'] = wlb_groups_best['config_list'].apply(
#     lambda x: sorted(x, key=lambda x: x[-1])[0]
# )
# wlb_groups_best = wlb_groups_best.drop(columns=['average_duration_list', 'pp_size_list', 'cp_size_list', 'num_microbatch_list', 'batch_size_list'])
# wlb_groups_best
# %%
wlb_perseq_groups_best = df[
    df['mode'] == 'wlbllm_perseq'
].groupby(["model_size",  "num_tokens", "ratio", "nodes", "total_batch_size", "dataset"]).agg({
    'average_duration': ['min', lambda x: list(x)],
    'pp_size': lambda x: list(x),
    'cp_size': lambda x: list(x),
    'num_microbatch': lambda x: list(x),
    'batch_size': lambda x: list(x),
}).reset_index()
wlb_perseq_groups_best.columns = ['model_size', 'num_tokens', 'ratio', 'nodes', 'total_batch_size', 'dataset', 'average_duration', 'average_duration_list', 'pp_size_list', 'cp_size_list', 'num_microbatch_list', 'batch_size_list']

wlb_perseq_groups_best['config_list'] = wlb_perseq_groups_best.apply(
    lambda x: [
        (batch_size, num_microbatch, cp, pp, round(duration, 2))
        for cp, pp, duration, num_microbatch, batch_size in zip(
            x['cp_size_list'],
            x['pp_size_list'],
            x['average_duration_list'],
            x['num_microbatch_list'],
            x['batch_size_list'],
        )
    ], axis=1
)
wlb_perseq_groups_best['config_list'] = wlb_perseq_groups_best['config_list'].apply(
    lambda x: sorted(x, key=lambda x: x[-1])
)
wlb_perseq_groups_best['best_config'] = wlb_perseq_groups_best['config_list'].apply(
    lambda x: sorted(x, key=lambda x: x[-1])[0]
)
wlb_perseq_groups_best = wlb_perseq_groups_best.drop(columns=['average_duration_list', 'pp_size_list', 'cp_size_list', 'num_microbatch_list', 'batch_size_list'])
wlb_perseq_groups_best


# %%
d2_groups_best = df[
    df['mode'] == 'd2'
].groupby(["model_size",  "num_tokens", "ratio", "nodes", "total_batch_size", "dataset"]).agg({
    'average_duration': ['min', lambda x: list(x)],
    'pp_size': lambda x: list(x),
    'cp_size': lambda x: list(x),
    'num_microbatch': lambda x: list(x),
    'batch_size': lambda x: list(x),
}).reset_index()
d2_groups_best.columns = ['model_size', 'num_tokens', 'ratio', 'nodes', 'total_batch_size', 'dataset', 'average_duration', 'average_duration_list', 'pp_size_list', 'cp_size_list', 'num_microbatch_list', 'batch_size_list']

d2_groups_best['config_list'] = d2_groups_best.apply(
    lambda x: [
        (batch_size, num_microbatch, cp, pp, round(duration, 2))
        for cp, pp, duration, num_microbatch, batch_size in zip(
            x['cp_size_list'],
            x['pp_size_list'],
            x['average_duration_list'],
            x['num_microbatch_list'],
            x['batch_size_list'],
        )
    ], axis=1
)
d2_groups_best['config_list'] = d2_groups_best['config_list'].apply(
    lambda x: sorted(x, key=lambda x: x[-1])
)
d2_groups_best['best_config'] = d2_groups_best['config_list'].apply(
    lambda x: sorted(x, key=lambda x: x[-1])[0]
)

d2_groups_best = d2_groups_best.drop(columns=['average_duration_list', 'pp_size_list', 'cp_size_list', 'num_microbatch_list', 'batch_size_list'])
d2_groups_best
# # %%
# # merge wlb_groups_best and d2_groups_best - 
# merged_wlb_vs_d2 = pd.merge(wlb_groups_best, d2_groups_best, on=['model_size', 'nodes', 'num_tokens', 'ratio', 'total_batch_size', 'dataset'], how='left', suffixes=('_wlb', '_d2'))
# merged_wlb_vs_d2['speedup'] = merged_wlb_vs_d2['average_duration_wlb'] / merged_wlb_vs_d2['average_duration_d2']
# merged_wlb_vs_d2['linear_speedup'] = merged_wlb_vs_d2['num_tokens'] / merged_wlb_vs_d2['total_batch_size']
# # merged_wlb_vs_d2['line_id'] = merged_wlb_vs_d2['num_tokens'] / merged_wlb_vs_d2['batch_size'] * merged_wlb_vs_d2['nodes']
# # merged_wlb_vs_d2['line_id'] = merged_wlb_vs_d2['line_id'].apply(
# #     lambda x: '128k' if x == 262144.0 else (
# #         '256k' if x == 1048576.0 else (
# #             '512k' if x == 4194304.0 else None
# #         )
# #     )
# # )
# merged_wlb_vs_d2['line_id'] = merged_wlb_vs_d2['num_tokens'].apply(
#     lambda x: (x // 1024)
# )
# merged_wlb_vs_d2_display = merged_wlb_vs_d2.sort_values(by=['model_size', 'num_tokens', 'nodes'], ascending=True)
# front_columns = ['model_size', 'nodes', 'num_tokens', 'ratio', 'total_batch_size', 
# 'speedup', 'dataset',
# 'average_duration_d2', 'best_config_d2', 
# 'average_duration_wlb','best_config_wlb', 
# ]
# back_columns = [x for x in merged_wlb_vs_d2_display.columns if x not in front_columns]
# merged_wlb_vs_d2_display = merged_wlb_vs_d2_display[front_columns + back_columns]
# merged_wlb_vs_d2_display__wlbllm = merged_wlb_vs_d2_display[merged_wlb_vs_d2_display['dataset'] == 'wlbllm']
# merged_wlb_vs_d2_display__prolong = merged_wlb_vs_d2_display[merged_wlb_vs_d2_display['dataset'] == 'prolong']

# # %%
# print("wlbllm")
# display(merged_wlb_vs_d2_display__wlbllm)
# print(
#     merged_wlb_vs_d2_display__wlbllm.to_csv("merged_wlb_vs_d2_display__wlbllm.tsv", index=False, sep="\t")
# )

# print("prolong")
# display(merged_wlb_vs_d2_display__prolong)
# print(
#     merged_wlb_vs_d2_display__prolong.to_csv("merged_wlb_vs_d2_display__prolong.tsv", index=False, sep="\t")
# )

# # %%
# pd.set_option('display.max_columns', None)
# pd.set_option('display.width', None)

# # %%
# # best config 
# print("pretrain best config:")
# merged_wlb_vs_d2_display__wlbllm[[
#     'best_config_d2', 'best_config_wlb', 'speedup', 
#     'num_tokens', 'model_size', 'nodes', 'total_batch_size',
# ]]

# # %%
# print("prolong best config")
# merged_wlb_vs_d2_display__prolong[[
#     'best_config_d2', 'best_config_wlb', 'speedup',
#     'num_tokens', 'model_size', 'nodes', 'total_batch_size',
# ]]
# # %%
# print(root_path)
# %%
merged_wlb_perseq_vs_d2 = pd.merge(wlb_perseq_groups_best, d2_groups_best, on=['model_size', 'nodes', 'num_tokens', 'ratio', 'total_batch_size', 'dataset'], how='left', suffixes=('_wlb_perseq', '_d2'))
merged_wlb_perseq_vs_d2['speedup'] = merged_wlb_perseq_vs_d2['average_duration_wlb_perseq'] / merged_wlb_perseq_vs_d2['average_duration_d2']


merged_wlb_perseq_vs_d2['linear_speedup'] = merged_wlb_perseq_vs_d2['num_tokens'] / merged_wlb_perseq_vs_d2['total_batch_size']
merged_wlb_perseq_vs_d2['line_id'] = merged_wlb_perseq_vs_d2['num_tokens'].apply(
    lambda x: (x // 1024)
)

merged_wlb_perseq_vs_d2_display = merged_wlb_perseq_vs_d2.sort_values(by=['model_size', 'num_tokens', 'nodes'], ascending=True)
front_columns = ['model_size', 'nodes', 'num_tokens', 'ratio', 'total_batch_size', 
'speedup', 'dataset',
'average_duration_d2', 'best_config_d2', 
'average_duration_wlb_perseq','best_config_wlb_perseq', 
]
back_columns = [x for x in merged_wlb_perseq_vs_d2_display.columns if x not in front_columns]
merged_wlb_perseq_vs_d2_display = merged_wlb_perseq_vs_d2_display[front_columns + back_columns]

merged_wlb_perseq_vs_d2_display__wlbllm_perseq = merged_wlb_perseq_vs_d2_display[merged_wlb_perseq_vs_d2_display['dataset'] == 'wlbllm']
merged_wlb_perseq_vs_d2_display__prolong = merged_wlb_perseq_vs_d2_display[merged_wlb_perseq_vs_d2_display['dataset'] == 'prolong']

# merged_wlb_perseq_vs_d2_display
# merged_wlb_perseq_vs_d2_display__wlbllm_perseq
# merged_wlb_perseq_vs_d2_display__prolong

# %%
print("wlbllm_perseq")
display(merged_wlb_perseq_vs_d2_display__wlbllm_perseq)
print(
    merged_wlb_perseq_vs_d2_display__wlbllm_perseq.to_csv("merged_wlb_perseq_vs_d2_display__wlbllm_perseq.tsv", index=False, sep="\t")
)

print("prolong")
display(merged_wlb_perseq_vs_d2_display__prolong)
print(
    merged_wlb_perseq_vs_d2_display__prolong.to_csv("merged_wlb_perseq_vs_d2_display__prolong.tsv", index=False, sep="\t")
)

# %%
