# %%

# %%
def flatten(a):
    return [item for sublist in a for item in sublist]

# %%
import json


file_groups = {
    "D2 (BS4) vs WLB DP4CP2 (BS4) [512K]": {
        "d2": "d2.bs2.json",
        "wlb": "wlbllm.dp4cp2.bs2.json",
    },
    "D2 (BS4) vs WLB DP2CP4 (BS4) [512K]": {
        "d2": "d2.bs2.json",
        "wlb": "wlbllm.dp2cp4.bs2.json",
    },
    "D2 (BS4) vs WLB DP1CP8 (BS4) [512K]": {
        "d2": "d2.bs2.json",
        "wlb": "wlbllm.dp1cp8.bs2.json",
    },
    "D2 (BS8) vs WLB DP1CP8 (BS8) [512K]": {
        "d2": "d2.bs4.json",
        "wlb": "wlbllm.dp1cp8.bs4.json",
    },
    
    
}

# %%
for i, (name, file_group) in enumerate(file_groups.items()):
    file_d2 = file_group["d2"]
    file_wlb = file_group["wlb"]

    _data_d2 = json.load(open(file_d2))
    _data_wlb = json.load(open(file_wlb))

    data_d2 = _data_d2["samples"]
    data_wlb = _data_wlb["samples"]

    durations_d2 = [sample["duration_ms"] for sample in data_d2]
    durations_wlb = [sample["duration_ms"] for sample in data_wlb]

    N = min(len(durations_d2), len(durations_wlb))
    durations_d2 = durations_d2[:N]
    durations_wlb = durations_wlb[:N]

    diff_d2_wlb = [durations_wlb[i] - durations_d2[i] for i in range(N)]
    speedup_d2_wlb = [durations_wlb[i] / durations_d2[i] for i in range(N)]

    from rich.console import Console
    from rich.table import Table

    # Print a table using rich
    console = Console()
    table = Table(
        title=name,
        caption=f"D2 = {file_d2}\nWLB = {file_wlb}"
    )

    table.add_column("Row ID", justify="right", style="white", no_wrap=True)
    table.add_column("D2", justify="right", style="white")
    table.add_column("WLB", justify="right", style="white") 
    table.add_column("Diff", justify="right", style="white")
    table.add_column("Speedup", justify="right", style="white")

    for i in range(N):
        table.add_row(str(i), f"{durations_d2[i]:.2f}", f"{durations_wlb[i]:.2f}", f"{diff_d2_wlb[i]:.2f}", f"{speedup_d2_wlb[i]:.2f}")

    console.print(table)


    e2e_d2 = sum(durations_d2)
    e2e_wlb = sum(durations_wlb)
    print(f"E2E D2: {e2e_d2:.2f}ms")
    print(f"E2E WLB: {e2e_wlb:.2f}ms")
    print(f"Speedup: {e2e_wlb / e2e_d2:.2f}x")




# %%

