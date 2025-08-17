# %%
import json
import pandas as pd
import matplotlib.pyplot as plt
import os

pwd = os.getcwd()
# Load the uploaded JSON files
with open(f"{pwd}/../data/64k/benchmark.20250813_014357.d2.json") as f:
    d2_data = json.load(f)

with open(f"{pwd}/../data/64k/benchmark.20250813_015721.baseline.json") as f:
    baseline_data = json.load(f)

# Convert samples to DataFrames
baseline_df = pd.DataFrame([{"sample_id": s["sample_id"], "Baseline Duration (ms)": s["duration_ms"]} for s in baseline_data["samples"]])
d2_df = pd.DataFrame([{"sample_id": s["sample_id"], "D2 Duration (ms)": s["duration_ms"]} for s in d2_data["samples"]])

# Merge on sample_id
merged_df = pd.merge(baseline_df, d2_df, on="sample_id")

# Calculate difference and speedup
merged_df["Difference (ms)"] = merged_df["D2 Duration (ms)"] - merged_df["Baseline Duration (ms)"]
merged_df["Speedup (%)"] = (merged_df["Baseline Duration (ms)"] - merged_df["D2 Duration (ms)"]) / merged_df["Baseline Duration (ms)"] * 100

# Save to CSV
csv_path_json_compare = f"{pwd}/../data/baseline_vs_d2_from_json.csv"
merged_df.to_csv(csv_path_json_compare, index=False)

# Plot 1: Duration comparison
plt.figure(figsize=(12, 6))
bar_width = 0.35
indices = range(len(merged_df))
plt.bar([i - bar_width/2 for i in indices], merged_df["Baseline Duration (ms)"], width=bar_width, label="Baseline")
plt.bar([i + bar_width/2 for i in indices], merged_df["D2 Duration (ms)"], width=bar_width, label="D2")
plt.xlabel("Sample ID")
plt.ylabel("Duration (ms)")
plt.title("Baseline vs D2 Duration Comparison (from JSON)")
plt.xticks(indices, merged_df["sample_id"])
plt.legend()
plt.tight_layout()
plt.show()
# %%
# Plot 2: Difference
plt.figure(figsize=(12, 6))
plt.bar(merged_df["sample_id"], merged_df["Difference (ms)"], color="orange")
plt.axhline(0, color='black', linewidth=1)
plt.xlabel("Sample ID")
plt.ylabel("Difference (ms)")
plt.title("Difference in Duration (D2 - Baseline)")
plt.tight_layout()
plt.show()
# %%
# Plot 3: Speedup
plt.figure(figsize=(12, 6))
plt.bar(merged_df["sample_id"], merged_df["Speedup (%)"], color="green")
plt.axhline(0, color='black', linewidth=1)
plt.xlabel("Sample ID")
plt.ylabel("Speedup (%)")
plt.title("Speedup of D2 over Baseline")
plt.tight_layout()
plt.show()

# %%
