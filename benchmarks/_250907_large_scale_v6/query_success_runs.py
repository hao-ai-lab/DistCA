# %%
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--folder", type=str, required=True)
args = parser.parse_args()
root_folder = args.folder

# %%
import os

success_runs = []

exp_folders = os.listdir(root_folder)
for exp_folder in exp_folders:
    exp_folder = os.path.join(root_folder, exp_folder)
    if not os.path.isdir(exp_folder):
        continue
    if not os.path.exists(os.path.join(exp_folder, "benchmark.json")):
        continue
    success_runs.append(exp_folder)
print(success_runs)

# %%
eids = []

import json
for folder in success_runs:
    benchmark_file = os.path.join(folder, "benchmark.json")
    with open(benchmark_file, "r") as f:
        benchmark = json.load(f)
    config = benchmark['config']
    print(config)
    mode = config['mode']
    cp_size = config['cp_size']
    if mode == 'd2':
        cp_size = 1
    num_nodes = config['num_nodes']
    batch_size = config['batch_size']
    num_tokens = config['num_tokens']

    eid = f"{mode}-cp{cp_size}-n{num_nodes}-b{batch_size}-t{num_tokens}"
    eids.append(eid)
print(eids)

# %%
# output eids to a file
with open("success_eids.txt", "w") as f:
    all_eids = " ".join(eids)
    f.write(all_eids)

# %%