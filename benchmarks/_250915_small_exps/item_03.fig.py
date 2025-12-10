# %%
import json
import matplotlib.pyplot as plt
import pandas as pd

# %%
with open("item_03.result.nh32.hdim128.L16384.jsonl", "r") as f:
    data = [json.loads(line) for line in f]
# %%
data
# %%
df = pd.DataFrame(data)
# %%
df
# %%
a = data[0]
flops = 4 * a['num_seq'] * a['L'] ** 2 * a['num_heads'] * a['head_dim'] // (2 if a['causal'] else 1)
# %%
# TFLOPS
flops / 1e12
# %%
df['tflops'] = flops / 1e12
df['tflops_per_sec_acheived'] = df['tflops'] / (df['duration_ms'] / 1000)
df['utilization'] = df['tflops_per_sec_acheived'] / 989 * 100
df
# %%
x = df['cp_shard_len']
y = df['utilization']
plt.plot(x, y)
plt.show()
# %%
import matplotlib.pyplot as plt
import numpy as np
from textwrap import fill

plt.figure(figsize=(7, 4.2))
ax = plt.gca()

# Plot curve using actual data
ax.plot(df['cp_shard_len'], df['utilization'], marker="o", linewidth=1.5, label="Core Attention MFU")

# Academic styling
ax.set_xscale('log', base=2)  # Set x-axis to log scale with base 2
ax.set_xticks(df['cp_shard_len'])  # Show all values
ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{int(x/1000)}k' if x >= 1000 else str(int(x))))  # Format as 1k, 2k etc for x>1000
ax.set_xlabel("Shard Length (tokens)")
ax.set_ylabel("MFU (FlashAttention Core)")
ax.grid(True, which="both", linestyle="--", linewidth=0.6, alpha=0.7)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.legend(frameon=False)

plt.tight_layout(rect=[0, 0.08, 1, 1])

# Save for reuse in papers
out_path = "attention_divisibility_MFU.png"
plt.savefig(out_path, dpi=300, bbox_inches="tight")
plt.show()

out_path


# %%
