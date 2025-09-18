# TODO: given a dict of {shard_size: [throughput for each sample]}, plot a curve figure
# x axis: shard size, use str instead of values
# y axis: throughput (TFLOPs) (this is the averaged throughput. No error bars)
# title: FlashAttention Throughput

import matplotlib.pyplot as plt
import numpy as np

with open("fa_profile_h4_d128_tok32768_ss1024.pkl", "rb") as f:
    import pickle
    results = pickle.load(f)

shard_sizes = sorted(results.keys())
throughputs = [np.mean(results[ss]) for ss in shard_sizes]  # average throughput for each shard size
plt.figure(figsize=(10, 6))
plt.plot([str(ss) for ss in shard_sizes], throughputs, marker='o')
plt.yscale('linear')
plt.xlabel('Shard Size')
plt.ylabel('Throughput (TFLOPs)')
plt.title('FlashAttention Throughput')
plt.grid(True, which="both", ls="--", linewidth=0.5)
plt.savefig('fa_throughput.png')
