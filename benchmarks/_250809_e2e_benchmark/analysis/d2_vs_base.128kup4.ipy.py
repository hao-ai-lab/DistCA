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

# seq_len = "64k"
# d2_path       = f"{pwd}/../data/benchmark.20250813_014357.d2.json"
# baseline_path = f"{pwd}/../data/benchmark.20250813_015721.baseline.json"
# benchmark.20250813_014357.d2.json
# seq_len = "128k"
# d2_path       = f"{pwd}/../data/benchmark.20250813_104749.d2.json"
# baseline_path = f"{pwd}/../data/benchmark.20250813_110401.baseline.json"

seq_len = "128k (upsample = 4)"
d2_path       = f"{pwd}/../data/128kup4/benchmark.20250813_235630.d2.json"
baseline_path = f"{pwd}/../data/128kup4/benchmark.20250814_000831.baseline.json"

# Output dir (same ../data as inputs)
out_dir = Path(pwd).joinpath("../data").resolve()
out_dir.mkdir(parents=True, exist_ok=True)

# %%
# Load JSONs
with open(d2_path, "r") as f:
    file_a = json.load(f)
with open(baseline_path, "r") as f:
    file_b = json.load(f)

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

# Validate and assign roles
if mode_a is None or mode_b is None:
    raise ValueError("Could not find 'mode' in one or both files.")

if mode_a == "d2" and mode_b == "baseline":
    d2_data = file_a
    baseline_data = file_b
elif mode_a == "baseline" and mode_b == "d2":
    d2_data = file_b
    baseline_data = file_a
else:
    # Fallback to filenames if both claim same mode (rare)
    if "d2" in Path(d2_path).name:
        d2_data = file_a
        baseline_data = file_b
    else:
        d2_data = file_b
        baseline_data = file_a

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

# Merge on sample_id
merged_df = pd.merge(baseline_df, d2_df, on="sample_id", how="inner")

# Compute metrics
merged_df["Difference (ms)"] = merged_df["D2 Duration (ms)"] - merged_df["Baseline Duration (ms)"]
merged_df["Speedup (%)"] = (
    (merged_df["Baseline Duration (ms)"] - merged_df["D2 Duration (ms)"])
    / merged_df["Baseline Duration (ms)"] * 100
)

# Pretty helper (optional): shorten giant sample configs for printing
def shorten_cfg(cfg, max_len=200):
    text = json.dumps(cfg, separators=(",", ":"))
    return text if len(text) <= max_len else text[:max_len] + " …"

# A single column showing the baseline sample config (you could diff with samples_d2 if needed)
merged_df["sample_config"] = merged_df["samples_baseline"].apply(shorten_cfg)

# Save full merged CSV
full_csv = out_dir.joinpath(f"baseline_vs_d2_full.{seq_len}.csv")
merged_df.to_csv(full_csv, index=False)
print(f"[save] full comparison -> {full_csv}")

# %%
# Plots
plt.figure(figsize=(12, 6))
bar_width = 0.35
indices = range(len(merged_df))
plt.bar([i - bar_width/2 for i in indices], merged_df["Baseline Duration (ms)"], width=bar_width, label="Baseline")
plt.bar([i + bar_width/2 for i in indices], merged_df["D2 Duration (ms)"], width=bar_width, label="D2")
plt.xlabel("Sample ID")
plt.ylabel("Duration (ms)")
plt.title(f"Baseline vs D2 Duration Comparison - {seq_len}")
plt.xticks(list(indices), merged_df["sample_id"])
plt.legend()
plt.tight_layout()
plt.show()

# %%
plt.figure(figsize=(12, 6))
plt.bar(merged_df["sample_id"], - merged_df["Difference (ms)"])
plt.axhline(0, linewidth=1)
plt.xlabel("Sample ID")
plt.ylabel("Difference (ms)")
plt.title(f"Difference in Duration (Baseline - D2) - {seq_len}")
# Add minor ticks at 1/5 of major tick intervals
from matplotlib.ticker import MultipleLocator, AutoMinorLocator
plt.gca().yaxis.set_minor_locator(AutoMinorLocator(5))
plt.grid(True, which='minor', alpha=0.3, linestyle=':')
plt.tight_layout()
plt.show()

# %%


# Summary stats
avg_speedup = merged_df["Speedup (%)"].mean()
print(f"Average speedup of D2 over Baseline - {seq_len}: {avg_speedup:.4f}%")

# Plot speedup
plt.figure(figsize=(12, 6))
plt.bar(merged_df["sample_id"], merged_df["Speedup (%)"], color='green')
plt.axhline(0, linewidth=1)
plt.xlabel("Sample ID")
plt.ylabel("Speedup (%)")
plt.title(f"Speedup of D2 over Baseline - {seq_len}")
# add another horizontal line at mean
plt.axhline(avg_speedup, color='red', linewidth=1, linestyle='--')
# add annotation that this is the average speedup
plt.text(0.0, avg_speedup, f"avg={avg_speedup:.2f}%", color='red', ha='center', va='bottom')
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