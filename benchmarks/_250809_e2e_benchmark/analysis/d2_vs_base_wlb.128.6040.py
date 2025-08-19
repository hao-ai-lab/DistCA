# %% [markdown]
# Baseline vs D2 benchmark analysis (plots + CSVs + reports)
# - Auto-detects which JSON is 'baseline' vs 'd2' via config.mode
# - Merges by sample_id
# - Saves full merged CSV and a report CSV (slower + top-5 fastest)
# - Prints average speedup and the two lists with sample configs

# %%
import json
import pandas as pd
import matplotlib.pyplot as plt
import os
from pathlib import Path

# User params

# Your two files (order doesn't matter)
pwd = os.getcwd()

seq_len = "128k (upsample = 4, 60%/40% long/short ratio)"
data_dir = f"{pwd}/../data/d2_vs_base_wlb.128.6040"

d2_path       = f"{data_dir}/benchmark.20250819_111803.d2.json"
baseline_path = f"{data_dir}/benchmark.20250819_110637.baseline.json"
wlbllm_path   = f"{data_dir}/benchmark.20250819_122919.wlbllm.json"

# Output dir (same ../data as inputs)
out_dir = Path(data_dir).resolve()
out_dir.mkdir(parents=True, exist_ok=True)


# %%
# Load JSONs
with open(d2_path, "r") as f:
    file_a = json.load(f)
with open(baseline_path, "r") as f:
    file_b = json.load(f)
with open(wlbllm_path, "r") as f:
    file_c = json.load(f)
# %%

def normalize_role(x):
    # Some dumps store mode in root ("mode"), others under "config"->"mode".
    if isinstance(x, dict):
        if "mode" in x: 
            return x["mode"]
        if "config" in x and isinstance(x["config"], dict) and "mode" in x["config"]:
            return x["config"]["mode"]
    return None

mode_a = normalize_role(file_a)
mode_b = normalize_role(file_b)
mode_c = normalize_role(file_c)

# Validate and assign roles
if mode_a is None or mode_b is None or mode_c is None:
    raise ValueError("Could not find 'mode' in one or both files.")

d2_data = file_a
baseline_data = file_b
wlbllm_data = file_c


print(f"[info] baseline mode: {normalize_role(baseline_data)}, d2 mode: {normalize_role(d2_data)}")

# %%
# Convert to DataFrames (keep the 'samples' payload so we can show per-sample configs)
baseline_df = pd.DataFrame([
    {"sample_id": s["sample_id"], 
     "Baseline Duration (ms)": s["duration_ms"],
     "samples_baseline": s.get("samples")}
    for s in baseline_data["samples"]
])

d2_df = pd.DataFrame([
    {"sample_id": s["sample_id"], 
     "D2 Duration (ms)": s["duration_ms"],
     "samples_d2": s.get("samples")}
    for s in d2_data["samples"]
])

wlbllm_df = pd.DataFrame([
    {"sample_id": s["sample_id"], 
     "WLBLLM Duration (ms)": s["duration_ms"],
     "samples_wlbllm": s.get("samples")}
    for s in wlbllm_data["samples"]
])

# Merge on sample_id
merged_df = pd.merge(baseline_df, d2_df, on="sample_id", how="inner")
merged_df = pd.merge(merged_df, wlbllm_df, on="sample_id", how="inner")

# Compute metrics
merged_df["Difference (ms)"] = merged_df["D2 Duration (ms)"] - merged_df["Baseline Duration (ms)"]
merged_df["WLBLLM Difference (ms)"] = merged_df["D2 Duration (ms)"] - merged_df["WLBLLM Duration (ms)"]
merged_df["Speedup (%)"] = (
    (merged_df["Baseline Duration (ms)"] - merged_df["D2 Duration (ms)"])
    / merged_df["Baseline Duration (ms)"] * 100
)
merged_df["WLBLLM Speedup (%)"] = (
    (merged_df["WLBLLM Duration (ms)"] - merged_df["D2 Duration (ms)"])
    / merged_df["WLBLLM Duration (ms)"] * 100
)

# Pretty helper (optional): shorten giant sample configs for printing
def shorten_cfg(cfg, max_len=200):
    text = json.dumps(cfg, separators=(",", ":"))
    return text if len(text) <= max_len else text[:max_len] + " …"

# # A single column showing the baseline sample config (you could diff with samples_d2 if needed)
# merged_df["sample_config"] = merged_df["samples_baseline"].apply(shorten_cfg)

# # Save full merged CSV
# full_csv = out_dir.joinpath(f"baseline_vs_d2_full.{seq_len}.csv")
# merged_df.to_csv(full_csv, index=False)
# print(f"[save] full comparison -> {full_csv}")

# %%
# Plots
plt.figure(figsize=(12, 6))
bar_width = 0.25
indices = range(len(merged_df))
plt.bar([i - bar_width for i in indices], merged_df["Baseline Duration (ms)"], width=bar_width, label="Baseline")
plt.bar([i for i in indices], merged_df["D2 Duration (ms)"], width=bar_width, label="D2")
plt.bar([i + bar_width for i in indices], merged_df["WLBLLM Duration (ms)"], width=bar_width, label="WLBLLM")
plt.xlabel("Sample ID")
plt.ylabel("Duration (ms)")
plt.title(f"Baseline vs D2 vs WLBLLM Duration Comparison - {seq_len}")
plt.xticks(list(indices), merged_df["sample_id"])
plt.legend()
plt.tight_layout()
plt.savefig(out_dir.joinpath(f"plot_abs_time.png"))
plt.show()

# %%
plt.figure(figsize=(12, 6))
bar_width = 0.35
indices = range(len(merged_df))
plt.bar([i - bar_width/2 for i in indices], -merged_df["Difference (ms)"], width=bar_width, label="Baseline - D2")
plt.bar([i + bar_width/2 for i in indices], -merged_df["WLBLLM Difference (ms)"], width=bar_width, label="WLBLLM - D2")
plt.axhline(0, linewidth=1)
plt.xlabel("Sample ID")
plt.ylabel("Difference (ms)")
plt.title(f"Difference in Duration (Baseline - D2) or (WLBLLM - D2) - {seq_len}")
plt.xticks(indices, merged_df["sample_id"])
plt.legend()
# Add minor ticks at 1/5 of major tick intervals
from matplotlib.ticker import MultipleLocator, AutoMinorLocator
plt.gca().yaxis.set_minor_locator(AutoMinorLocator(5))
plt.grid(True, which='minor', alpha=0.3, linestyle=':')
plt.tight_layout()
plt.savefig(out_dir.joinpath(f"plot_diff.png"))
plt.show()

# %%
# Summary stats
avg_speedup = merged_df["Speedup (%)"].mean()
avg_wlbllm_speedup = merged_df["WLBLLM Speedup (%)"].mean()
print(f"Average speedup of D2 over Baseline - {seq_len}: {avg_speedup:.4f}%")
print(f"Average speedup of D2 over WLBLLM - {seq_len}: {avg_wlbllm_speedup:.4f}%")

# Plot speedup
plt.figure(figsize=(12, 6))
bar_width = 0.35
indices = range(len(merged_df))
plt.bar([i - bar_width/2 for i in indices], merged_df["Speedup (%)"], width=bar_width, color='blue', label='D2 vs Baseline')
plt.bar([i + bar_width/2 for i in indices], merged_df["WLBLLM Speedup (%)"], width=bar_width, color='red', label='D2 vs WLBLLM')
plt.axhline(0, linewidth=1)
plt.xlabel("Sample ID")
plt.ylabel("Speedup (%)")
plt.title(f"Speedup of D2 over Baseline or WLBLLM - {seq_len}")
plt.xticks(indices, merged_df["sample_id"])
plt.legend()
# add horizontal lines at means
plt.axhline(avg_speedup, color='blue', linewidth=1, linestyle='--')
plt.axhline(avg_wlbllm_speedup, color='red', linewidth=1, linestyle='--')
# add annotations for the average speedups
plt.text(-1.0, avg_speedup, f"avg={avg_speedup:.2f}%", color='blue', ha='center', va='bottom')
plt.text(-1.0, avg_wlbllm_speedup, f"avg={avg_wlbllm_speedup:.2f}%", color='red', ha='center', va='bottom')
# Add minor ticks every 1%
from matplotlib.ticker import MultipleLocator
plt.gca().yaxis.set_minor_locator(MultipleLocator(1))
plt.grid(True, which='minor', alpha=0.3, linestyle=':')
plt.tight_layout()
plt.savefig(out_dir.joinpath(f"plot_speedup.png"))
plt.show()

# %%
samples_a = file_b['samples']
samples_a, len(samples_a)

# %%
up_sample_factor = file_a['config']['up_sample_factor']
dp_size = file_a['config']['dp_size']
elongate_factor = 4
filter_threshold = 10 * 1024
filter_ratio = 0.90
total_seq_len = 128 * 1024
N = len(file_a['samples'])
K = 1024

def get_speedup_from_simulation():
    # Simulated configs
    base_seq_len = K * 64
    attn_base_time = 12.5020
    mlp_base_time = 13.5  # assume expert parallel
    qkvo_base_time = 8.5 
    linear_base_time = (mlp_base_time + qkvo_base_time)  # mlp + qkvo
    # --------

    def get_attn_time(batch):
        total_time = 0
        for l in batch:
            ratio = l / base_seq_len
            total_time += attn_base_time * (ratio ** 2)
        return total_time

    def flatten(batch):
        return [item for sublist in batch for item in sublist]
    
    from d2.simulator.optimizers.samples import (
        sample_wlbllm_docs_upsample, 
        batch_documents
    )

    all_speedups = []
    for i, sample in enumerate(samples_a):
        batches = sample['samples']
        # 
        mlp_time = linear_base_time * (total_seq_len / base_seq_len)
        
        # Calculate Baseline
        dp_batches = []
        for rid in range(dp_size):
            dp_batches.append(
                batches[rid] + batches[rid + dp_size]
            )
        
        baseline_attn_time = [
            get_attn_time(batch)
            for batch in dp_batches
        ]
        max_baseline_attn_time = max(baseline_attn_time)
        baseline_time = max_baseline_attn_time + mlp_time
        
        # Calculate d2
        d2_attn_time = get_attn_time(flatten(batches)) / dp_size 
        d2_time = d2_attn_time + mlp_time

        # Calculate speedup
        speedup = - (d2_time - baseline_time) / baseline_time
        all_speedups.append(speedup)
        print(f"Sample {i}: Speedup: {speedup:.2%}; Baseline: {baseline_time:.2f} ms; D2: {d2_time:.2f} ms. Attn ratio: {(max_baseline_attn_time / baseline_time):.2%}. Baseline attention time: {max_baseline_attn_time:.2f} ms, D2 attention time: {d2_attn_time:.2f} ms")
        # print(f"Sample {i}: Batches: {batches}")

    import numpy as np
    import rich
    rich.print(f"""
    Experiment Config:
    - dp_size: {dp_size}
    - total_seq_len: {total_seq_len}
    - up_sample_factor: {up_sample_factor}
    - filter_threshold: {filter_threshold}
    - filter_ratio: {filter_ratio}

    Speedup: d2 vs baseline
    - Average: {np.mean(all_speedups):.2%}
    - Max: {max(all_speedups):.2%}
    - Min: {min(all_speedups):.2%}
    - Median: {np.median(all_speedups):.2%}
    - Std: {np.std(all_speedups):.2%}
    """)
    # import matplotlib.pyplot as plt
    # plt.hist(all_speedups, bins=20)
    # plt.show()
    return all_speedups


# Plot the same speedup %
# # Plot speedup
import numpy as np
plt.figure(figsize=(12, 6))
all_speedups = get_speedup_from_simulation()
all_speedups = [i * 100 for i in all_speedups]
sample_ids = list(range(len(all_speedups)))
plt.bar(sample_ids, all_speedups, color='darkgreen')
plt.axhline(0, linewidth=1)
plt.xlabel("Sample ID")
plt.ylabel("Speedup (%)")
plt.title(f"Simulation: Speedup of D2 over Baseline - {seq_len}")
# add an average
simulated_avg_speedup = np.mean(all_speedups)
plt.axhline(simulated_avg_speedup, color='red', linewidth=1, linestyle='--')
plt.text(0.0, simulated_avg_speedup, f"avg={simulated_avg_speedup:.2f}%", color='red', ha='center', va='bottom')
# Add minor ticks every 1%
from matplotlib.ticker import MultipleLocator
plt.gca().yaxis.set_minor_locator(MultipleLocator(1))
plt.grid(True, which='minor', alpha=0.3, linestyle=':')
plt.tight_layout()
plt.show()





# %%
# Reporting: slower (negative speedup) and top-5 fastest
slower_samples = merged_df[merged_df["Speedup (%)"] < 0].copy()
slower_samples = slower_samples.sort_values(by="Speedup (%)", ascending=True)
fastest_samples = merged_df.sort_values(by="Speedup (%)", ascending=False).head(5).copy()
fastest_samples = fastest_samples.sort_values(by="Speedup (%)", ascending=False)

cols_to_print = [
    "sample_id", "Speedup (%)",
    "Baseline Duration (ms)", "D2 Duration (ms)",
    "Difference (ms)", "sample_config"
]

print("📉 Samples where D2 is slower (negative speedup):")
print(slower_samples[cols_to_print].to_string(index=False))

print("\n🚀 Top 5 fastest samples (highest speedup):")
print(fastest_samples[cols_to_print].to_string(index=False))

# Save a report CSV for quick filtering
report_csv = out_dir.joinpath(f"d2_speedup_analysis.{seq_len}.csv")
pd.concat([slower_samples[cols_to_print], fastest_samples[cols_to_print]]).to_csv(report_csv, index=False)
# sort by speedup (%)
print(f"\n[save] report (slower + top-5 fastest) -> {report_csv}")

# %%
# (Optional) sanity check: warn if some sample_ids exist in one run but not the other
only_base = set(baseline_df["sample_id"]) - set(d2_df["sample_id"])
only_d2   = set(d2_df["sample_id"]) - set(baseline_df["sample_id"])
if only_base:
    print(f"[warn] sample_ids only in baseline: {sorted(list(only_base))[:20]}{' …' if len(only_base)>20 else ''}")
if only_d2:
    print(f"[warn] sample_ids only in d2: {sorted(list(only_d2))[:20]}{' …' if len(only_d2)>20 else ''}")

# %%

# %% Plotly Interactive Histogram
import plotly.graph_objects as go
import plotly.express as px

# Create interactive histogram with hover data
fig = go.Figure()

fig.add_trace(go.Histogram(
    x=merged_df["Speedup (%)"],
    nbinsx=50,
    name="Speedup Distribution",
    hovertemplate='<b>Speedup Range:</b> %{x}<br><b>Count:</b> %{y}<extra></extra>'
))

fig.update_layout(
    title=f"Distribution of Speedup (%) - {seq_len}",
    xaxis_title="Speedup (%)",
    yaxis_title="Frequency",
    showlegend=False
)

# Add vertical line for average
fig.add_vline(x=avg_speedup, line_dash="dash", line_color="red", 
              annotation_text=f"avg={avg_speedup:.2f}%")

fig.show()

# %% Plotly Interactive Scatter Plot
# Create scatter plot with sample config in tooltip
fig = go.Figure()

fig.add_trace(go.Scatter(
    x=list(range(len(merged_df))),
    y=merged_df["Speedup (%)"],
    mode='markers',
    marker=dict(
        size=5,
        color=merged_df["Speedup (%)"],
        colorscale='RdYlBu',
        showscale=True,
        colorbar=dict(title="Speedup (%)")
    ),
    customdata=merged_df["sample_config"],
    hovertemplate='<b>Sample ID:</b> %{text}<br>' +
                  '<b>Speedup:</b> %{y:.2f}%<br>' +
                  '<b>Sample Config:</b> %{customdata}<br>' +
                  '<extra></extra>',
    text=merged_df["sample_id"],
    name="Samples"
))

fig.update_layout(
    title=f"Speedup of D2 over Baseline - {seq_len}",
    xaxis_title="Sample Index",
    yaxis_title="Speedup (%)",
    showlegend=False
)

# Add horizontal line for average
fig.add_hline(y=avg_speedup, line_dash="dash", line_color="red",
              annotation_text=f"avg={avg_speedup:.2f}%")

fig.show()

# %% Plotly Box Plot by Sample Config Pattern
# Extract first few elements of sample config for grouping
merged_df['config_pattern'] = merged_df['sample_config'].apply(
    lambda x: str(x)[:50] + "..." if len(str(x)) > 50 else str(x)
)

fig = go.Figure()

# Get unique config patterns
unique_patterns = merged_df['config_pattern'].unique()

for pattern in unique_patterns:
    pattern_data = merged_df[merged_df['config_pattern'] == pattern]
    
    fig.add_trace(go.Box(
        y=pattern_data["Speedup (%)"],
        name=pattern,
        customdata=pattern_data["sample_config"],
        hovertemplate='<b>Config Pattern:</b> %{x}<br>' +
                      '<b>Speedup:</b> %{y:.2f}%<br>' +
                      '<b>Full Config:</b> %{customdata}<br>' +
                      '<extra></extra>'
    ))

fig.update_layout(
    title=f"Speedup Distribution by Sample Config Pattern - {seq_len}",
    xaxis_title="Config Pattern",
    yaxis_title="Speedup (%)",
    xaxis={'categoryorder': 'total descending'}
)

# Add horizontal line for average
fig.add_hline(y=avg_speedup, line_dash="dash", line_color="red",
              annotation_text=f"avg={avg_speedup:.2f}%")

fig.show()
# %%