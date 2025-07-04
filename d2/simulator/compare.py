import numpy as np
import time
from typing import Deque, Iterable, List, Sequence
from collections import deque
import matplotlib.pyplot as plt

import d2.timemodule.compute as tm

from d2.simulator.optimizers.attnserver import AttnServerSolver
from d2.simulator.optimizers.wlbllm import WlbLlmSolver
from d2.simulator.optimizers.samples import (
    sample_multimodal_gaussian, 
    sample_random_docs,
    sample_wlbllm_docs,
    batch_documents,
)

K = 1024
M = 1024 ** 2

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


def run_comparison(
    batches: list[list[int]], 
    num_workers: int, 
    num_total_devices: int, 
    docs: list[int],
    plot_every: int = -1,
):
    cumulative_batches = []
    display_results = []
    full_results = []
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
                "did_time_out": attnserver_solution.did_time_out,
            }
        )

        print(f"Iteration {i+1}: WLB-LLM Time: {wlb_time:.2f}s, AttnServer Time: {attn_time:.2f}s")
        
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
            plt.title(f'Pre-pack Document Distribution (total docs: {len(docs)})')
            plt.xlabel('Document length (tokens)')
            plt.ylabel('Frequency')
            plt.grid(True, axis='y', linestyle='--', alpha=0.7)

            plt.subplot(3, 1, 3)
            flat_batch = [item for sublist in batch for item in sublist]
            plt.hist(flat_batch, bins=batch_bin_edges, edgecolor='black')
            plt.title(f'Post-pack Document Distribution (total docs: {len(flat_batch)})')
            plt.xlabel('Document length (tokens)')
            plt.ylabel('Frequency')
            plt.grid(True, axis='y', linestyle='--', alpha=0.7)

            plt.tight_layout()
            plt.show()
        
        if plot_every >= 0 and (plot_every == 0 or (i + 1) % plot_every == 0):
            plot_speedup_distribution()

    if plot_every > 0:
        plot_speedup_distribution()
    

    return full_results
