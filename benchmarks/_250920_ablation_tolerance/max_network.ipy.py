# %%
import os
import json

root_path = "./logs.v3-tolerance-full/"
a = os.listdir(root_path)



rows = []

for folder in a:
    if not os.path.isdir(os.path.join(root_path, folder)):
        continue
    readme_path = os.path.join(root_path, folder, "README.md")
    if not os.path.exists(readme_path):
        continue
    with open(readme_path, "r") as f:
        readme = f.read()
    readme = readme.split("\n")
    readme_dict = {}
    for line in readme:
        if not line.startswith("-"):
            continue
        key, value = line.split(":")
        key = key.strip().lower().replace("-", "").strip()
        value = value.strip()
        readme_dict[key] = value
    # print(readme_dict)  

    # network_inspect.summary.jsonl
    max_buffer_needed_for_each_case: list[int] = []
    with open(os.path.join(root_path, folder, "network_inspect.summary.jsonl"), "r") as f:
        for line in f:
            network_inspect = json.loads(line)
            max_buffer_needed_for_each_case.append(network_inspect["max_comm_budget_all_rank_mb"])

    max_buffer_needed = max(max_buffer_needed_for_each_case)
    row = dict(
        nnodes=readme_dict["nnodes"],
        num_tokens=readme_dict["num_tokens"],
        batch_size=readme_dict["batch_size"],
        model=readme_dict["model_path"],
        num_layers=readme_dict["num_layers"],
        max_buffer_needed=max_buffer_needed,
    )

    rows.append(row)
# %%    
import pandas as pd
df = pd.DataFrame(rows)
df
# %%

