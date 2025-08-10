# %% [markdown]

# # Anchor Golden Forward/Backward Result

# %%
import os
import json
import numpy as np
import pandas as pd
import rich

output_dir    = f"anchor_d2_baseline.results.1"
os.makedirs(output_dir, exist_ok=True)

# Collect data for all token sizes
data = []

for t in ["8k", "16k", "32k", "64k"]:
    d2_p0_path       = f"../data/20250810_014629.anchor.v1/d2.t{t}.p0.json"
    d2_p1_path       = f"../data/20250810_014629.anchor.v1/d2.t{t}.p1.json"
    baseline_path = f"../data/20250810_014629.anchor.v1/baseline.t{t}.json"

    with open(d2_p0_path, "r") as f:
        d2_p0_data = json.load(f)
    with open(d2_p1_path, "r") as f:
        d2_p1_data = json.load(f)
    with open(baseline_path, "r") as f:
        baseline_data = json.load(f)
    
    d2_p0_duration_ms = [sample['duration_ms'] for sample in d2_p0_data['samples']]
    d2_p1_duration_ms = [sample['duration_ms'] for sample in d2_p1_data['samples']]
    baseline_duration_ms = [sample['duration_ms'] for sample in baseline_data['samples']]
    
    d2_p0_mean = np.mean(d2_p0_duration_ms)
    d2_p0_std = np.std(d2_p0_duration_ms)
    d2_p1_mean = np.mean(d2_p1_duration_ms)
    d2_p1_std = np.std(d2_p1_duration_ms)
    baseline_mean = np.mean(baseline_duration_ms)
    baseline_std = np.std(baseline_duration_ms)

    # Add to data list
    data.append({
        'token_size': t,
        'd2p0': f"{d2_p0_mean:.2f} ± {d2_p0_std:.2f}",
        'd2p1': f"{d2_p1_mean:.2f} ± {d2_p1_std:.2f}",
        'baseline': f"{baseline_mean:.2f} ± {baseline_std:.2f}",
    })

# Create DataFrame
df = pd.DataFrame(data)
df = df.set_index('token_size')

# Print the pivot table (which is just the DataFrame in this case)
print("Duration (ms) by Token Size and Method (Mean ± Std):")
print(df.round(2))

# df.to_markdown(index=False)
print(df.to_markdown(index=True))

with open(f"{output_dir}/results.md", "w") as f:
    f.write(df.to_markdown(index=True))

# %%[markdown]

# | token_size   | d2p0          | d2p1          | baseline      |
# |:-------------|:--------------|:--------------|:--------------|
# | 8k           | 91.34 ± 1.48  | 88.29 ± 0.99  | 75.57 ± 0.56  |
# | 16k          | 156.14 ± 1.42 | 147.45 ± 0.80 | 141.59 ± 0.18 |
# | 32k          | 326.21 ± 0.86 | 308.71 ± 0.55 | 311.21 ± 0.61 |
# | 64k          | 794.77 ± 2.07 | 758.46 ± 1.58 | 786.11 ± 1.60 |

# %%
