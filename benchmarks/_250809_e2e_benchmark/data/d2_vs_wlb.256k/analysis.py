# %%
from typing import Iterable, List, Optional
from d2.simulator.optimizers.samples import sample_wlbllm_docs_upsample, batch_documents

ITERATION_ID = 0
GLOBAL_BATCH: Optional[Iterable[List[int]]] = None

K = 1024
iterated_samples = []

def setup_global_batch(
    total_seq_len, 
    up_sample_factor=2,
    elongate_factor=2,
    filter_threshold=64 * 1024,
    filter_ratio=0.90,
    should_add_debug_cases=False,
):
    global GLOBAL_BATCH
    if GLOBAL_BATCH is not None:
        return

    GLOBAL_BATCH = batch_documents(
        sample_wlbllm_docs_upsample(
            size=10000,
            filter_threshold=filter_threshold,
            filter_ratio=filter_ratio,
            upsample_long_factor=up_sample_factor,
            elongate_factor=elongate_factor,
        ), max_ctx_length=total_seq_len
    )

    
    if should_add_debug_cases:
        GLOBAL_BATCH = list(GLOBAL_BATCH)
        manual_case = [
            [total_seq_len // 4 * 3 - 512, 512, total_seq_len // 4],
            [total_seq_len // 4 * 3 - 512, 512, total_seq_len // 4],
        ]
        GLOBAL_BATCH = manual_case + GLOBAL_BATCH
        GLOBAL_BATCH = iter(GLOBAL_BATCH)
    return


def get_next_batch(dp_size) -> Iterable[List[List[int]]]:
    global GLOBAL_BATCH
    global ITERATION_ID
    global iterated_samples
    # get dp_size number of batches 
    batches = []
    for _ in range(dp_size):    
        batches.append(next(GLOBAL_BATCH))
    ITERATION_ID += 1
    iterated_samples.append(batches)
    return batches

# %%
def flatten(a):
    return [item for sublist in a for item in sublist]
# %%
setup_global_batch(256 * 1024)
all_batches = []
while True:
    try:
        batches = get_next_batch(1)
    except StopIteration:
        break
    all_batches.extend(batches)
all_batches = flatten(all_batches)
# %%
max(all_batches)
# %%
import matplotlib.pyplot as plt
plt.hist(all_batches, bins=100)
plt.ylim(0, 10)
plt.show()
# %%[markdown]
# Compare each group

# %%
import json
"""
group1.d2.dpcp8.bs1.json
group1.wlbllm.dp1cp8.bs1.json
group2.d2.dpcp8.bs2.json
group2.wlbllm.dp2cp4.bs2.json
group3.d2.dpcp8.bs4.json
group3.wlbllm.dp4cp2.bs4.json
group4.d2.dpcp8.bs4.json
group4.wlbllm.dp8cp1.bs4.json
group2.wlbllm.dp2cp4.bs1.json
group4.wlbllm.dp8cp1.bs8.json
"""
# Group 1
# file_d2 = "group1.d2.dpcp8.bs1.json"
# file_wlb = "group1.wlbllm.dp1cp8.bs1.json"

# Group 2
# file_d2 = "group2.d2.dpcp8.bs2.json"
# file_wlb = "group2.wlbllm.dp2cp4.bs2.json"

# # Group 3
# file_d2 = "group3.d2.dpcp8.bs4.json"
# file_wlb = "group3.wlbllm.dp4cp2.bs4.json"

# # Group 4
file_d2 = "group4.d2.dpcp8.bs4.json"
file_wlb = "group4.wlbllm.dp8cp1.bs4.json"

# %%
_data_d2 = json.load(open(file_d2))
_data_wlb = json.load(open(file_wlb))

data_d2 = _data_d2["samples"]
data_wlb = _data_wlb["samples"]

durations_d2 = [sample["duration_ms"] for sample in data_d2]
durations_wlb = [sample["duration_ms"] for sample in data_wlb]

diff_d2_wlb = [durations_wlb[i] - durations_d2[i] for i in range(len(durations_d2))]
speedup_d2_wlb = [durations_wlb[i] / durations_d2[i] for i in range(len(durations_d2))]

from rich.console import Console
from rich.table import Table

# Print a table using rich
console = Console()
table = Table(title="Comparison: D2 vs WLB")

table.add_column("Row ID", justify="right", style="white", no_wrap=True)
table.add_column("D2", justify="right", style="white")
table.add_column("WLB", justify="right", style="white") 
table.add_column("Diff", justify="right", style="white")
table.add_column("Speedup", justify="right", style="white")

for i in range(len(durations_d2)):
    table.add_row(str(i), f"{durations_d2[i]:.2f}", f"{durations_wlb[i]:.2f}", f"{diff_d2_wlb[i]:.2f}", f"{speedup_d2_wlb[i]:.2f}")

console.print(table)


e2e_d2 = sum(durations_d2)
e2e_wlb = sum(durations_wlb)
print(f"E2E D2: {e2e_d2:.2f}ms")
print(f"E2E WLB: {e2e_wlb:.2f}ms")
print(f"Speedup: {e2e_wlb / e2e_d2:.2f}x")



# %%
