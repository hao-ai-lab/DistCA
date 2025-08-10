# %% [markdown]

# # 32 K Comparison

# %%
import os

d2_path       = "../data/20250810_004013.v3/d2.json"
baseline_path = "../data/20250810_004013.v3/baseline.json"
output_dir    = "compare_d2_baseline.results.3.v3"
os.makedirs(output_dir, exist_ok=True)
# %%
import json

with open(d2_path, "r") as f:
    d2_data = json.load(f)

with open(baseline_path, "r") as f:
    baseline_data = json.load(f)

# %%
# First check the `config` field is the same except for `mode`.
# Then, look at the `samples` field.`
# for each sample, ensure `samples` is the same.
# Then record the `duration_ms` field for d2 and baseline.
# Finally, show the speedup of d2 over baseline.

# %%
d2_config = d2_data["config"]
assert d2_config["mode"] == "d2", "d2_config is not d2"
baseline_config = baseline_data["config"]
assert baseline_config["mode"] == "baseline", "baseline_config is not baseline"

d2_config.pop("mode")
baseline_config.pop("mode")
assert d2_config == baseline_config, "Config is not the same"

# %%
d2_samples = d2_data["samples"]
baseline_samples = baseline_data["samples"]
assert len(d2_samples) == len(baseline_samples), "Number of samples is not the same"

samples = [
    s["samples"]
    for s in d2_samples
]
# %%
for d2_sample, baseline_sample in zip(d2_samples, baseline_samples):
    assert d2_sample["samples"] == baseline_sample["samples"], "Samples are not the same"

# %%
d2_duration_ms = [sample["duration_ms"] for sample in d2_samples]
baseline_duration_ms = [sample["duration_ms"] for sample in baseline_samples]
speedup = [baseline_duration_ms[i] / d2_duration_ms[i] for i in range(len(d2_duration_ms))]
speedup_perc = [100 * (baseline_duration_ms[i] - d2_duration_ms[i]) / baseline_duration_ms[i] for i in range(len(d2_duration_ms))]
# %%
import matplotlib.pyplot as plt

plt.rcParams.update({'font.size': 14})

sorted_speedup_perc = sorted(speedup_perc)
plt.plot(sorted_speedup_perc, label="Speedup")
plt.title("Llama 8B (nlayer=4) Seq 32K DP2TP8 - speedup", fontsize=16)
plt.ylabel("Speedup % = (baseline - d2) / baseline", fontsize=14)
plt.xlabel("sample (sorted by speedup_perc)", fontsize=14)


plt.legend(fontsize=12)
plt.show()

# %%


# %%
import pandas as pd

df = pd.DataFrame({
    "d2_duration_ms": d2_duration_ms,
    "baseline_duration_ms": baseline_duration_ms,
    "speedup": speedup,
    "speedup_perc": speedup_perc,
    "samples": samples,
})
df2 = df.copy()


# Function to format samples for markdown display
def format_samples_for_markdown(samples_list):
    """Format samples list so each sublist appears on its own line in markdown"""
    if isinstance(samples_list, list):
        # Convert each sublist to string and join with <br> for line breaks
        formatted_sublists = [str(sublist) for sublist in samples_list]
        return "\n".join(formatted_sublists)
    else:
        return str(samples_list)

df["d2_duration_ms"] = df["d2_duration_ms"].round(2)
df["baseline_duration_ms"] = df["baseline_duration_ms"].round(2)
df["speedup"] = df["speedup"].round(2)
df["speedup_perc"] = df["speedup_perc"].round(2)

# Save to CSV with original samples format
df.rename(columns={
    "d2_duration_ms": "D2(ms)", 
    "baseline_duration_ms": "Base(ms)", 
    "speedup": "speedup=\nd2/base", 
    "speedup_perc": "speedup%=\n(base-d2)/base"
}, inplace=True)
df.to_csv(f"{output_dir}/d2_baseline_comparison.32k.tp8dp2.csv", index=False)

# Create a copy for markdown with formatted samples
df_markdown = df.copy()
df_markdown["samples"] = df_markdown["samples"].apply(format_samples_for_markdown)
df_markdown.to_markdown(f"{output_dir}/d2_baseline_comparison.32k.tp8dp2.md")

df2.drop(columns=["samples"], inplace=True)
df2.rename(columns={
    "d2_duration_ms": "D2(ms)", 
    "baseline_duration_ms": "Base(ms)", 
    "speedup": "speedup=\nd2/base", 
    "speedup_perc": "speedup%=\n(base-d2)/base"
}, inplace=True)

df2.to_csv(f"{output_dir}/d2_baseline_comparison.32k.tp8dp2.simple.csv", index=False)
df2.to_markdown(f"{output_dir}/d2_baseline_comparison.32k.tp8dp2.simple.md")



# %%
df