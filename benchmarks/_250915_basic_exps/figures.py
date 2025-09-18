# %%
"""
3. Attention Divisibility (Priority: High. EST: 3 hr?)
   1. With xx K total tokens different combination of CP shards have the same throughput
   2. x axis: size of each CP shards (KV context size is randomly sampled); y axis: throughput(FLOPs per GPU)

Note: 
- 32K? 64K? we don't need to make this number very high), 
"""

# %%
import matplotlib.pyplot as plt
import numpy as np
from textwrap import fill

# --- Synthetic template data (MFU vs shard length in tokens) ---
shard_len = np.array([16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384])
# MFU rises quickly and plateaus after ~128 tokens
mfu = np.array([0.28, 0.42, 0.58, 0.70, 0.78, 0.79, 0.80, 0.80, 0.80, 0.80, 0.80])

plt.figure(figsize=(7, 4.2))
ax = plt.gca()

# Plot curve
ax.plot(shard_len, mfu, marker="o", linewidth=1.5, label="Core Attention MFU (template)")

# Academic styling
ax.set_xscale('log', base=2)  # Set x-axis to log scale with base 2
ax.set_xticks(shard_len)  # Show all values
ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{int(x/1000)}k' if x >= 1000 else str(int(x))))  # Format as 1k, 2k etc for x>1000
ax.set_xlabel("Shard Length (tokens)")
ax.set_ylabel("MFU (FlashAttention Core)")
ax.grid(True, which="both", linestyle="--", linewidth=0.6, alpha=0.7)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.legend(frameon=False)

# # Visual cue for the 128-token tiling boundary
# ax.axvline(128, linestyle=":", linewidth=1.2)
# ax.annotate("Tile size ≈ 128 tokens",
#             xy=(128, 0.78), xytext=(180, 0.68),
#             arrowprops=dict(arrowstyle="->", lw=1.0),
#             fontsize=10)

# Place a neat, wrapped text box inside the figure area
# Add "FAKE DATA" watermark
ax.text(0.5, 0.5, "FAKE DATA - See FA Profile", 
        transform=ax.transAxes,
        fontsize=36,
        color='red',
        alpha=0.8,
        ha='center',
        va='center',
        rotation=30)

# Create wrapped text for the note
wrapped = fill("Note: This is template data for illustration only. "
              "Real measurements will be added later.", width=80)

plt.gcf().text(0.00, 1.00, wrapped, fontsize=12, va="bottom", color='red')

plt.tight_layout(rect=[0, 0.08, 1, 1])

# Save for reuse in papers
out_path = "item_03.attention_divisibility_MFU.png"
plt.savefig(out_path, dpi=300, bbox_inches="tight")
plt.show()

out_path


# %%

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# =============================
# (1) Stacked BAR CHART TEMPLATE
# =============================
# X-axis: total tokens per batch (fixed ticks requested)
batch_tokens = np.array([16_000, 32_000, 64_000])

# Real measured values (GB)
mem_weights = np.array([9.5, 9.5, 9.5], dtype=float)
mem_opt_states = np.array([7, 7, 7], dtype=float)
mem_lin_act = np.array([12.8, 25.6, 51.2], dtype=float)
mem_attn_core = np.array([4.096, 8.192, 16.384], dtype=float)
mem_misc_act = np.array([2.32, 4.64, 9.28], dtype=float)

# Save a CSV so you can overwrite with real profiler outputs
df = pd.DataFrame({
    "total_tokens": batch_tokens,
    "weights_gb": mem_weights,
    "opt_states_gb": mem_opt_states,
    "linear_acts_gb": mem_lin_act,
    "core_attention_acts_gb": mem_attn_core,
    "misc_acts_gb": mem_misc_act,
})

# Plot
# Increase figure width to accommodate legend on left
fig, ax = plt.subplots(figsize=(5, 4))

indices = np.arange(len(batch_tokens))
bar_width = 0.85


"""
Linear layers -> FFN? 因为按我理解，qkv_proj和o_proj这两个linear，其实对应的是红色那段》
噢，我明白了，这里其实有个可以优化的点，就是qkv proj存的是concat_qkv，但是attn存的是split qkv
这里比较tricky，我的想法是在这里plot的时候别说CA有存activation（因为理论上确实什么都不用额外存），然后图上给：
Weight & Optimizer States,
MISC Layers Activation,
FFN Activation,
qkv_proj and o_proj Activation
"""

# Stacked bars
b1 = ax.bar(indices, mem_weights, bar_width, label="Weights")
b2 = ax.bar(indices, mem_opt_states, bar_width, bottom=mem_weights, label="Optimizer States")
b3 = ax.bar(indices, mem_lin_act, bar_width, bottom=mem_weights + mem_opt_states, label="FFN Activation")
b4 = ax.bar(indices, mem_attn_core, bar_width, bottom=mem_weights + mem_opt_states + mem_lin_act, label="QKVO Proj Activation")
b5 = ax.bar(indices, mem_misc_act, bar_width, bottom=mem_weights + mem_opt_states + mem_lin_act + mem_attn_core, label="MISC Activation")

# Labels & ticks with larger font
ax.set_xlabel("Total tokens per batch", fontsize=16)
ax.set_ylabel("Memory (GB)", fontsize=16)
ax.set_xticks(indices, [f"{t//1000}k" for t in batch_tokens])
ax.tick_params(axis='both', which='major', labelsize=14)
ax.grid(True, axis="y", linestyle="--", linewidth=0.6, alpha=0.7)

# Academic styling
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

# Legend in single column on left with larger font
ax.legend(frameon=False, loc='upper left', bbox_to_anchor=(0, 1.15), fontsize=14)

plt.tight_layout()
out_path = "item_04.memory_breakdown_bars.png"
plt.savefig(out_path, dpi=300, bbox_inches="tight")
plt.show()

# png_path, csv_path
# %%
# Calculate ratios for each component
total_memory = mem_weights + mem_opt_states + mem_lin_act + mem_misc_act + mem_attn_core
ratios_weights = (mem_weights + mem_opt_states) / total_memory * 100
ratios_lin_act = mem_lin_act / total_memory * 100
ratios_misc_act = mem_misc_act / total_memory * 100
ratios_attn_core = mem_attn_core / total_memory * 100

# Create new figure for ratio breakdown
fig, ax = plt.subplots(figsize=(6, 4))

indices = np.arange(len(batch_tokens))
bar_width = 0.75

# Stacked bars with percentages
b1 = ax.bar(indices, ratios_weights, bar_width, label="Weight & Optimizer States")
b2 = ax.bar(indices, ratios_lin_act, bar_width, bottom=ratios_weights, label="FFN Activation")
b3 = ax.bar(indices, ratios_misc_act, bar_width, bottom=ratios_weights + ratios_lin_act, label="MISC Activation")
b4 = ax.bar(indices, ratios_attn_core, bar_width, bottom=ratios_weights + ratios_lin_act + ratios_misc_act, label="QKVO Proj Activation")

# Labels & ticks
ax.set_xticks(indices, [f"{t//1000}k" for t in batch_tokens])
ax.grid(True, axis="y", linestyle="--", linewidth=0.6, alpha=0.7)
# Make axis labels larger
ax.set_xlabel("Total tokens per batch", fontsize=16)
ax.set_ylabel("Memory breakdown (%)", fontsize=16)
ax.tick_params(axis='both', which='major', labelsize=14)


# Academic styling
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

# Place legend directly above figure
ax.legend(frameon=False, loc='upper center', bbox_to_anchor=(0.5, 1.25), ncol=2, fontsize=12)

plt.tight_layout()
# out_path = "item_04.memory_breakdown_bars_ratios.png"
out_path = "memory_breakdown.png"
plt.savefig(out_path, dpi=300, bbox_inches="tight")

plt.show()


# %%

# Create new figure for total memory consumption
fig, ax = plt.subplots(figsize=(5, 3))

# Convert total memory to GB and create box plot data
total_memory_gb = total_memory / (1024 * 1024 * 1024)  # Convert to GB
box_data = []
selected_tokens = [64000, 128000, 256000]
selected_indices = [i for i, t in enumerate(batch_tokens) if t in selected_tokens]

for i in selected_indices:
    # Create synthetic data points around the total memory value
    # Using normal distribution with small variance
    data_points = np.random.normal(total_memory_gb[i], total_memory_gb[i]*0.1, 20)
    data_points = np.clip(data_points, 0, None)  # Ensure no negative values
    box_data.append(data_points)

# Create box plot
positions = np.arange(len(selected_indices))
ax.boxplot(box_data, positions=positions, widths=bar_width,
           patch_artist=True,
           boxprops=dict(facecolor='lightblue', color='black'),
           medianprops=dict(color='red'),
           whiskerprops=dict(color='black'),
           capprops=dict(color='black'))

# Labels & ticks
ax.set_xlabel("Total tokens per batch")
ax.set_ylabel("Total Memory (GB)")
ax.set_xticks(positions, ["64k", "128k", "256k"])
ax.grid(True, axis="y", linestyle="--", linewidth=0.6, alpha=0.7)

# Academic styling
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

# Add "FAKE DATA" watermark
ax.text(0.5, 0.5, "FAKE DATA", 
        transform=ax.transAxes,
        fontsize=36,
        color='red',
        alpha=0.8,
        ha='center',
        va='center',
        rotation=30)

plt.tight_layout()
plt.show()
# %%

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


tokens = np.array([16_000, 32_000, 64_000, 128_000, 256_000])
ratios_tok_per_dev = [4_000, 8_000, 16_000]
labels = [f"{r//1000}k : 1 (tok:dev)" for r in ratios_tok_per_dev]

def synth_time_seconds(tok, ratio):
    devices = tok / ratio
    a = 0.035
    base = 0.08
    poly = (tok / 32_000)**0.6
    t = base * np.exp(a * devices) ** poly
    return t

times = {r: np.array([synth_time_seconds(t, r) for t in tokens]) for r in ratios_tok_per_dev}

rows = []
for i, t in enumerate(tokens):
    row = {"total_tokens": t}
    for r in ratios_tok_per_dev:
        row[f"solver_time_tok_per_dev_{r}"] = times[r][i]
    rows.append(row)
df = pd.DataFrame(rows)
csv_path = "flexsp_ilp_solver_time_template.csv"
df.to_csv(csv_path, index=False)

fig, ax = plt.subplots(figsize=(7.8, 4.6))
for r, label in zip(ratios_tok_per_dev, labels):
    ax.plot(tokens, times[r], marker="o", linewidth=1.4, label=label)

ax.set_xlabel("Total tokens per batch")
ax.set_ylabel("ILP solver time (s)")
ax.set_xscale("log", base=2)
ax.set_yscale("log")
ax.grid(True, which="both", linestyle="--", linewidth=0.6, alpha=0.7)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.set_xticks(tokens, [f"{t//1000}k" if t<1_000_000 else "1M" for t in tokens])
ax.legend(frameon=False, title="tokens : devices")

# Add "FAKE DATA" watermark
ax.text(0.5, 0.5, "FAKE DATA", 
        transform=ax.transAxes,
        fontsize=36,
        color='red',
        alpha=0.8,
        ha='center',
        va='center',
        rotation=30)

plt.tight_layout()
png_path = "flexsp_ilp_solver_time.png"
plt.savefig(png_path, dpi=300, bbox_inches="tight")
plt.show()

png_path, csv_path
# %%
