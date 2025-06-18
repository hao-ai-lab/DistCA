# %%

from d2.simulator.optimizers.attnserver import AttnServerSolver
from d2.simulator.optimizers.wlbllm import WlbLlmSolver

import numpy as np
import time
from typing import Deque, Iterable, List, Sequence
from collections import deque
import matplotlib.pyplot as plt

# %%
K = 1024
M = 1024 ** 2

# %%

def wlbllm_vs_attnserver(
    batch, 
    num_workers = 4, # DP Degree
    num_total_devices = 32, # Number of GPUs
):
    # WLB-LLM Results
    wlbllm_results = {}
    for tp in [8, 4, 2, 1]:
        wlbllm_results[tp] = {}
        for cp in [8, 4, 2, 1]:
            wlbllm_results[tp][cp] = 1e15

    wlb_start_time = time.time()
    wlbllm_best_solution = (1e15, None, None)
    # print("WLB-LLM Solution:")
    for tp in [8, 4, 2, 1]:
        for cp in [8, 4, 2, 1]:
            if tp * cp > num_total_devices:
                continue
            parallel_plan = (tp, cp)
            num_workers = num_total_devices // (tp * cp)
            assert (
                num_workers * tp * cp == num_total_devices, 
                "num_workers * tp * cp != num_total_devices"
            )
            
            solver = WlbLlmSolver()
            solution = solver.solve(
                batch, 
                max_length=sum(batch),
                num_workers=num_workers,
                parallel_plan=parallel_plan,
            )
            lat_max = solution.lat_max
            wlbllm_results[tp][cp] = lat_max
            wlbllm_best_solution = min(wlbllm_best_solution, (lat_max, parallel_plan, solution))
    wlbllm_lat = wlbllm_best_solution[0]
    wlb_end_time = time.time()
    wlb_time = wlb_end_time - wlb_start_time
    # print(f"WLB-LLM Best solution: {wlbllm_best_solution[0]} ms <- {wlbllm_best_solution[1]}")


    # AttnServer Results
    attn_start_time = time.time()
    solver = AttnServerSolver()
    attnserver_solution = solver.solve(
        batch, 
        num_workers=num_total_devices, 
        num_total_devices=num_total_devices,
    )
    attnserver_lat = attnserver_solution.lat_max
    attn_end_time = time.time()
    attn_time = attn_end_time - attn_start_time
    # attnserver_solution.print_solution()

    # Compare WLB-LLM and AttnServer
    speedup = (wlbllm_lat - attnserver_lat) / wlbllm_lat
    return speedup, wlbllm_best_solution[2], attnserver_solution, wlb_time, attn_time


# %%

import matplotlib.pyplot as plt
from d2.simulator.optimizers.samples import sample_wlbllm_docs_altered
for i in [64,128,256,512,1024,2048]:
    docs = sample_wlbllm_docs_altered(size=i)
    plt.hist(docs, bins=100, edgecolor='black')
    plt.show()


# %%

# %%

full_results = []
display_results = []

from d2.simulator.optimizers.samples import (
    sample_multimodal_gaussian, 
    sample_random_docs,
    sample_wlbllm_docs,
    batch_documents,
)



num_batches = 1
num_workers = 4
num_total_devices = 8

# sample_size = 64
# sample_size = 128
# sample_size = 256
# sample_size = 512
# sample_size = 1024
sample_size = 2048

# name = "multimodal_gaussian"
name = "wlbllm_docs"
# name = "random_docs"

if name == "multimodal_gaussian":
    docs = sample_multimodal_gaussian(
        means=[1*K,  8*K, 16*K], 
        sigmas=[0.1*K, 0.2*K, 0.2*K], weights=[0.3, 0.25, 0.45], 
        size=sample_size, seed=42,
    )
elif name == "random_docs":
    docs = sample_random_docs(max_ctx_length = 64 * K, size = sample_size, seed=42,)
elif name == "wlbllm_docs":
    docs = sample_wlbllm_docs(
        size = sample_size,
        seed=42,
    )


batches = batch_documents(
    docs, 
    max_ctx_length = 64 * K * num_batches,
    # max_ctx_length = 16 * K,
)
batches = list(batches)
flatted_batches = [x for xs in batches for x in xs]

# Plot the distribution of docs and flatted_batches
plt.figure(figsize=(10, 8))

# Upper subplot for docs
plt.subplot(2, 1, 1)
plt.hist(docs, bins=100, edgecolor='black')
plt.title('Distribution of Docs')
plt.xlabel('Document Length')
plt.ylabel('Frequency')

# Lower subplot for flatted_batches
plt.subplot(2, 1, 2)
plt.hist(flatted_batches, bins=100, edgecolor='black')
plt.title('Distribution of Flatted Batches')
plt.xlabel('Batch Length')
plt.ylabel('Frequency')

plt.tight_layout()
plt.show()

print(len(batches))

# batches



# %%
print(batches[0])

# %%


cumulative_batches = []
for i, batch in enumerate(batches):
    cumulative_batches.extend(batch)
    (
        speedup, 
        wlbllm_best_solution, attnserver_solution, 
        wlb_time, attn_time
    ) = wlbllm_vs_attnserver(
        batch = batch,
        num_workers = num_workers,
        num_total_devices = num_total_devices,
    )
    display_results.append(speedup * 100)
    full_results.append(
        {
            "batch": batch,
            "speedup": speedup,
            "wlbllm_best_solution": wlbllm_best_solution,
            "attnserver_solution": attnserver_solution,
            "wlb_time": wlb_time,
            "attn_time": attn_time,
        }
    )

    print(f"Iteration {i+1}: WLB-LLM Time: {wlb_time:.2f}s, AttnServer Time: {attn_time:.2f}s")
    
    # Plot figure every 10 iterations
    # if (i + 1) % 10 == 0:
    #     plt.figure(figsize=(10, 6))
    #     plt.plot(display_results, marker='o')
    #     plt.title(f'Speedup after {i+1} iterations')
    #     plt.xlabel('Iteration')
    #     plt.ylabel('Speedup (%)')
    #     plt.ylim(-50, 50)
    #     plt.grid(True, which='major', linestyle='--', alpha=0.7)
    #     plt.yticks(range(-50, 50, 5))  # Set y-ticks every 5
    #     plt.tight_layout()
    #     plt.show()
    def plot_speedup_distribution():
        plt.figure(figsize=(10, 10))  # Increased figure height to accommodate two subplots
        
        # Top subplot for speedup distribution
        plt.subplot(3, 1, 1)
        bin_edges = [-20, -18, -16, -14, -12, -10, -8, -6, -4, -2, 0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30]
        n, bins, patches = plt.hist(display_results, bins=bin_edges, edgecolor='black')
        
        # Add labels for each bar
        for count, x in zip(n, bins[:-1]):
            plt.text(x + (bins[1]-bins[0])/2, count, str(int(count)), 
                     ha='center', va='bottom')
        
        plt.title(f'Speedup (Workers: {num_workers}, Devices: {num_total_devices})')
        plt.xlabel('Speedup (%)')
        plt.ylabel('Frequency')
        plt.grid(True, axis='y', linestyle='--', alpha=0.7)
        
        # Bottom subplot for cumulative batches distribution
        plt.subplot(3, 1, 2)
        batch_bin_edges = list(range(0, int(sum(docs))+1000, 1000))
        plt.hist(docs, bins=batch_bin_edges, edgecolor='black')
        plt.title(f'Post-pack Document Distribution (total docs: {len(docs)})')
        plt.xlabel('Document length (tokens)')
        plt.ylabel('Frequency')
        plt.grid(True, axis='y', linestyle='--', alpha=0.7)

        plt.subplot(3, 1, 3)
        

        
        plt.tight_layout()
        plt.show()
    
    # if (i + 1) % 10 == 0:
    plot_speedup_distribution()

plot_speedup_distribution()

# %%
