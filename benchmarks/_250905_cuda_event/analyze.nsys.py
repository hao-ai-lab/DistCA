"""
Analyze nsys export sqlite database to get the NVTX hierarchy and kernel runtimes.

References
https://docs.nvidia.com/nsight-systems/2022.1/nsys-exporter/examples.html#thread-summary-composite
https://docs.nvidia.com/nsight-systems/2022.1/nsys-exporter/exported_data.html
https://docs.nvidia.com/cupti/main/main.html?highlight=nvtx#nvidia-tools-extension-callbacks

Usage

```bash
    python analyze.nsys.py \
        --db fs-mbz-gpu-017.sqlite \
        --target_thread_name "NVSHMEM PE 0" \
        --gpu-id 0
```
"""
# %% [markdown]
# NVTX hierarchy (per-thread) from Nsight Systems SQLite
# -----------------------------------------------------
# - Works with push/pop (59) and optionally start/end (60) ranges
# - Given GPU device id and a thread id you saw in the GUI (OS TID),
#   builds the NVTX hierarchy on that thread only (no cross-thread parenting).
# - Also tags each NVTX range with whether it launched kernels on the chosen GPU.
#
## Usage:
#   - Set DB, GPU_ID, OS_TID below. If you also know OS_PID, set it (optional).
#   - Run cells top → bottom.
## Notes:
#   - 59 = NvtxPushPopRange (strictly nested on one thread)
#   - 60 = NvtxStartEndRange (can close on another thread; we only keep 60 whose end is on the same thread)
#   - PyTorch exposes both via torch.cuda.nvtx: range_push/pop (59) and range_start/end (60).
# -----------------------------------------------------

# %%
import sqlite3
import pandas as pd
import numpy as np
from collections import defaultdict

pd.set_option("display.max_rows", 200)
pd.set_option("display.max_columns", 200)
pd.set_option("display.width", 180)

# %% [markdown]
# Parameters


# %%
import argparse

parser = argparse.ArgumentParser(description='Analyze NSYS SQLite database')
parser.add_argument('--db', type=str, default="fs-mbz-gpu-017.sqlite",
                    help='path to your nsys export')
parser.add_argument('--target-thread-name', type=str, default="NVSHMEM PE 0",
                    help='target thread name to analyze')
parser.add_argument('--gpu-id', type=int, default=0,
                    help='target GPU deviceId to check kernel launches')
parser.add_argument('--os-tid', type=int, default=None,
                    help='the thread id you saw in the GUI (e.g. "[1964964] NVSHMEM PE0")')
parser.add_argument('--os-pid', type=int, default=None,
                    help='optional: if you know the OS process id, set it; else leave None')
parser.add_argument('--include-nvtx-start-end', type=bool, default=True,
                    help='include eventType=60 *only if it also ends on the same thread*')

args = parser.parse_args()

DB = args.db
TARGET_THREAD_NAME = args.target_thread_name
GPU_ID = args.gpu_id
OS_TID = args.os_tid
OS_PID = args.os_pid
INCLUDE_NVTX_START_END = args.include_nvtx_start_end

# %%
# %% [markdown]
# Connect and basic helpers

# %%
conn = sqlite3.connect(DB)

def bring_front(df, cols):
    rest = [c for c in df.columns if c not in cols]
    return df[cols + rest]

def load_string_map(conn):
    sid = pd.read_sql_query("SELECT id, value FROM StringIds", conn)
    return dict(zip(sid.id, sid.value))

def coalesce_name(text, textId, sid):
    return text if text is not None else sid.get(textId)

def decode_pid(globalTid):
    return (int(globalTid) >> 24) & 0xFFFFFF

def decode_tid(globalTid):
    return int(globalTid) & 0xFFFFFF

def make_globalTid(pid, tid):
    return (int(pid) << 24) | (int(tid) & 0xFFFFFF)

sid = load_string_map(conn)

# %% [markdown]
# Find candidate globalTid(s) matching the OS TID.
# If OS_PID is given, compute the exact globalTid; else discover candidates from NVTX and Runtime tables.

# # %%

stream_name_to_device_info_mapping = pd.read_sql_query("""
SELECT * , 
    globalTid / 0x1000000 % 0x1000000 AS PID, 
    globalTid % 0x1000000 AS TID
FROM ThreadNames
JOIN StringIds ON ThreadNames.nameId = StringIds.id
JOIN TARGET_INFO_CUDA_CONTEXT_INFO ON PID = TARGET_INFO_CUDA_CONTEXT_INFO.processId
;
""", conn)

# filtered_stream_name_to_device_info_mapping['value'] = filtered_stream_name_to_device_info_mapping['value'].str.strip()

filtered_stream_name_to_device_info_mapping = stream_name_to_device_info_mapping[
    (stream_name_to_device_info_mapping['value'].str.strip() == (TARGET_THREAD_NAME)) & 
    (stream_name_to_device_info_mapping['deviceId'] == GPU_ID)
]

assert len(filtered_stream_name_to_device_info_mapping) == 1
if OS_TID is None:
    OS_TID = filtered_stream_name_to_device_info_mapping['TID'].iloc[0].item()
else:
    OS_TID = OS_TID


# %%
candidates = []

if OS_PID is not None:
    candidates = [make_globalTid(OS_PID, OS_TID)]
else:
    q_nvtx = """
    SELECT DISTINCT globalTid AS gtid FROM NVTX_EVENTS
    WHERE (globalTid & 0xFFFFFF) = :os_tid
    """
    q_rt = """
    SELECT DISTINCT globalTid AS gtid FROM CUPTI_ACTIVITY_KIND_RUNTIME
    WHERE (globalTid & 0xFFFFFF) = :os_tid
    """
    g1 = pd.read_sql_query(q_nvtx, conn, params={"os_tid": OS_TID})
    g2 = pd.read_sql_query(q_rt,   conn, params={"os_tid": OS_TID})
    candidates = sorted(set(g1["gtid"]).union(set(g2["gtid"])))

print("OS_TID:", OS_TID, "| OS_PID:", OS_PID)
print("Candidate globalTid(s):", candidates)

# %% [markdown]
# If multiple candidates, pick the one that actually launches kernels on the target GPU most often.

# %%
def score_globalTid(gtid, gpu_id):
    q = """
    SELECT COUNT(*) AS c
    FROM CUPTI_ACTIVITY_KIND_RUNTIME api
    JOIN CUPTI_ACTIVITY_KIND_KERNEL k
      ON k.correlationId = api.correlationId
    WHERE api.globalTid = :gtid AND k.deviceId = :gpu
    """
    row = pd.read_sql_query(q, conn, params={"gtid": int(gtid), "gpu": int(gpu_id)}).iloc[0]
    return int(row["c"])

if not candidates:
    raise RuntimeError("No candidate globalTid found for OS_TID. Are you sure the DB matches the GUI run?")

scores = [(gtid, score_globalTid(gtid, GPU_ID)) for gtid in candidates]
scores = sorted(scores, key=lambda x: x[1], reverse=True)
GLOBAL_TID = scores[0][0]

print("Selected GLOBAL_TID:", GLOBAL_TID, " (score=", scores[0][1], ")")
if len(scores) > 1:
    print("Other candidates:", scores[1:])

# %% [markdown]
# Load NVTX events for the selected thread.
# - Always include push/pop (59).
# - Optionally include start/end (60) **only if** end is on the same thread (endGlobalTid == globalTid).

# %%
evt_types = (59,) if not INCLUDE_NVTX_START_END else (59, 60)

nvtx = pd.read_sql_query(
    f"""
    SELECT rowid AS nid, eventType, globalTid, endGlobalTid, start, end, text, textId
    FROM NVTX_EVENTS
    WHERE globalTid = :gtid
      AND eventType IN ({",".join(map(str, evt_types))})
    ORDER BY start ASC, end DESC
    """,
    conn,
    params={"gtid": int(GLOBAL_TID)},
)

# keep 60 only if ends on same thread
if INCLUDE_NVTX_START_END:
    nvtx = nvtx[(nvtx["eventType"] == 59) | ((nvtx["eventType"] == 60) & (nvtx["endGlobalTid"] == nvtx["globalTid"]))]

nvtx["name"] = [coalesce_name(t, tid, sid) for t, tid in zip(nvtx["text"], nvtx["textId"])]
nvtx = bring_front(nvtx, ["nid", "eventType", "name", "start", "end", "globalTid", "endGlobalTid"])
nvtx

# %%
nvtx[
    nvtx['name'].str.contains('sample')
]

# %% [markdown]
# Build strict hierarchy using **push/pop (59)** (stack-based).
# Then, if 60s are included, attach each 60 as a child of the *smallest enclosing* 59 on this thread (if any).

# %%
nvtx59 = nvtx[nvtx["eventType"] == 59].copy()
nvtx60 = nvtx[nvtx["eventType"] == 60].copy()

nvtx59

# %%
nvtx59 = nvtx59.sort_values(["start", "end"], ascending=[True, False])

parent_of = {}
children_of = defaultdict(list)
depth_of = {}

# stack nesting for 59
stack = []
for _, r in nvtx59.iterrows():
    while stack and not (r["start"] >= stack[-1]["start"] and r["end"] <= stack[-1]["end"]):
        stack.pop()
    pid = stack[-1]["nid"] if stack else None
    parent_of[r["nid"]] = pid
    if pid is not None:
        children_of[pid].append(r["nid"])
        depth_of[r["nid"]] = depth_of[pid] + 1
    else:
        depth_of[r["nid"]] = 0
    stack.append(r)

# attach 60 under smallest enclosing 59 (if exists)
def find_enclosing_59(s, e):
    # among 59s where start<=s and end>=e, pick with minimal (end-start)
    g = nvtx59[(nvtx59["start"] <= s) & (nvtx59["end"] >= e)]
    if g.empty:
        return None
    idx = (g["end"] - g["start"]).idxmin()
    return int(nvtx59.loc[idx, "nid"])

for _, r in nvtx60.iterrows():
    pid = find_enclosing_59(r["start"], r["end"])
    parent_of[r["nid"]] = pid
    if pid is not None:
        children_of[pid].append(r["nid"])
        depth_of[r["nid"]] = depth_of[pid] + 1
    else:
        depth_of[r["nid"]] = 0

# %%

# %%

# %% [markdown]
# Compute paths (root→leaf) and assemble a nodes DataFrame.

# %%
def path_ids(nid):
    chain = []
    cur = nid
    while cur is not None:
        chain.append(cur)
        cur = parent_of.get(cur)
    return list(reversed(chain))

id2name = dict(zip(nvtx["nid"], nvtx["name"]))
id2etype = dict(zip(nvtx["nid"], nvtx["eventType"]))
id2start = dict(zip(nvtx["nid"], nvtx["start"]))
id2end   = dict(zip(nvtx["nid"], nvtx["end"]))

rows = []
for nid in nvtx["nid"]:
    chain = path_ids(int(nid))
    rows.append({
        "nid": int(nid),
        "eventType": id2etype[nid],
        "name": id2name[nid],
        "start": id2start[nid],
        "end": id2end[nid],
        "duration_us": (id2end[nid] - id2start[nid]) / 1e3,
        "parent_nid": parent_of.get(nid),
        "depth": depth_of.get(nid, 0),
        "path_ids": "|".join(map(str, chain)),
        "path_names": "|".join(id2name[c] for c in chain),
    })

nodes = pd.DataFrame(rows).sort_values(["start", "end"], ascending=[True, False])
bring_front(nodes, ["nid", "eventType", "name", "depth", "parent_nid", "duration_us", "start", "end"])

# %% [markdown]
# (Optional) Tag each NVTX range with whether it launched kernels on GPU `GPU_ID`
# by checking if there exists a CUDA Runtime API on this thread within the range
# whose correlationId links to a kernel on the device.

# %%
q_range_kernels = f"""
SELECT r.rowid AS nid, api.correlationId
FROM NVTX_EVENTS r
JOIN CUPTI_ACTIVITY_KIND_RUNTIME api
  ON api.globalTid = r.globalTid
 AND api.start >= r.start AND api.end <= r.end
JOIN CUPTI_ACTIVITY_KIND_KERNEL k
  ON k.correlationId = api.correlationId
 AND k.deviceId = :gpu
WHERE r.globalTid = :gtid
  AND r.eventType IN ({",".join(map(str, evt_types))})
"""
rk = pd.read_sql_query(q_range_kernels, conn, params={"gpu": int(GPU_ID), "gtid": int(GLOBAL_TID)})
rk_grp = rk.groupby("nid")["correlationId"].nunique().rename("num_kernels_on_gpu").reset_index()

nodes2 = nodes.merge(rk_grp, on="nid", how="left")
nodes2["num_kernels_on_gpu"] = nodes2["num_kernels_on_gpu"].fillna(0).astype(int)
nodes2["launched_kernels_on_gpu"] = nodes2["num_kernels_on_gpu"] > 0

bring_front(nodes2, ["nid", "name", "depth", "launched_kernels_on_gpu", "num_kernels_on_gpu", "start", "end"])

# %% [markdown]
# Inspect only the subtree(s) that launched kernels on the chosen GPU (quick glance).

# %%
sub = nodes2[nodes2["launched_kernels_on_gpu"]].copy()
sub = sub.sort_values(["start", "end"], ascending=[True, False])
sub

# %% [markdown]
# Pretty-print a compact tree (indentation) for quick debugging.

# %%
# Compute a time origin for pretty relative times
T0 = int(nodes2["start"].min()) if not nodes2.empty else 0

def print_tree(df_nodes: pd.DataFrame, only_with_kernels=False, max_rows=None):
    df = df_nodes.copy()
    if only_with_kernels:
        keep = set(df[df["launched_kernels_on_gpu"]]["nid"].tolist())
        def ancestors(nid):
            out, cur = [], nid
            while True:
                p = parent_of.get(cur)
                if p is None: break
                out.append(p); cur = p
            return out
        for nid in list(keep):
            keep.update(ancestors(nid))
        df = df[df["nid"].isin(keep)]

    df = df.sort_values(["start", "end"], ascending=[True, False])
    if max_rows is not None:
        df = df.head(max_rows)
    for _, r in df.iterrows():
        indent = "  " * int(r["depth"])
        mark = "★ " if r.get("launched_kernels_on_gpu", False) else "  "
        et = int(r["eventType"])
        ns_start = int(r["start"])
        ns_end   = int(r["end"])
        rel_ms   = (ns_start - T0) / 1e6
        dur_ms   = (ns_end - ns_start) / 1e6
        print(
            f"{indent}{mark}[{et}] nid={int(r['nid'])}  {r['name']}\n"
            f"{indent}    abs: start={ns_start/1e6:.3f} ms  end={ns_end/1e6:.3f} ms  rel: t0+{rel_ms:.3f} ms  dur={dur_ms:.3f} ms"
        )

print("── NVTX tree (all nodes) ──")
print_tree(nodes2, only_with_kernels=False)

print(f"\n── NVTX tree (nodes that launched kernels on GPU {GPU_ID} + ancestors) ──")
print_tree(nodes2, only_with_kernels=True)

# %% [markdown]
# Export to CSV for downstream analysis.

# %%
OUT_CSV = f"nvtx_thread_{GLOBAL_TID}_gpu{GPU_ID}.csv"
nodes2.to_csv(OUT_CSV, index=False)
print("Wrote:", OUT_CSV)


# %%
# %% [markdown]
# Build kernel list for this thread and map each to innermost NVTX (by API launch window)
# Join Runtime (only this thread) -> Kernel (optionally filter by GPU_ID)
kernels_thread = pd.read_sql_query(
    """
    SELECT
      api.rowid            AS rid,
      api.globalTid        AS api_tid,
      api.start            AS api_start,
      api.end              AS api_end,
      api.correlationId    AS corr,

      k.rowid              AS kid,
      k.start              AS k_start,
      k.end                AS k_end,
      k.deviceId,
      k.streamId,
      k.shortName,
      k.demangledName
    FROM CUPTI_ACTIVITY_KIND_RUNTIME api
    JOIN CUPTI_ACTIVITY_KIND_KERNEL k
      ON k.correlationId = api.correlationId
    WHERE api.globalTid = :gtid
    """,
    conn,
    params={"gtid": int(GLOBAL_TID)},
)

# %%
# Keep only chosen GPU (comment this line if you want all GPUs)
kernels_thread = kernels_thread[kernels_thread["deviceId"] == GPU_ID].copy()

# %%
# Resolve kernel names
kernels_thread["kernel_short"] = kernels_thread["shortName"].map(sid)
kernels_thread["kernel_full"]  = kernels_thread["demangledName"].map(sid)

# Fallbacks
kernels_thread["kernel_name"] = kernels_thread["kernel_short"].fillna(kernels_thread["kernel_full"])
kernels_thread["kernel_full"] = kernels_thread["kernel_full"].fillna(kernels_thread["kernel_name"])

# %%
# Pre-sort NVTX59 for candidate lookup
nvtx59_min = nvtx59[["nid", "start", "end"]].sort_values(["start", "end"], ascending=[True, False]).copy()

# %%
# Fast candidate: NVTX with start <= api_start, then verify containment; climb parents if needed
nvtx59_min = nvtx59_min.sort_values("start").reset_index(drop=True)
cand = pd.merge_asof(
    kernels_thread.sort_values("api_start"),
    nvtx59_min.rename(columns={"start": "nv_start"}),
    left_on="api_start",
    right_on="nv_start",
    direction="backward",
)

# %%
def find_innermost_enclosing(nid, api_end):
    # Ensure chosen candidate encloses [api_start, api_end]; if not, walk up parents
    if pd.isna(nid):
        return None
    cur = int(nid)
    while cur is not None and nvtx59.loc[nvtx59["nid"] == cur, "end"].item() < api_end:
        cur = parent_of.get(cur)
    return cur

cand["innermost_nid"] = [
    find_innermost_enclosing(nid, e) for nid, e in zip(cand["nid"], cand["api_end"])
]

# Build mapping: innermost_nid -> list of kernels (sorted by k_start)
kernels_by_nvtx = {}
for _, r in cand.iterrows():
    key = r["innermost_nid"]
    kernels_by_nvtx.setdefault(key, []).append(
        {
            "kid": int(r["kid"]),
            "k_start": int(r["k_start"]), 
            "k_end": int(r["k_end"]),
            "deviceId": int(r["deviceId"]),
            "streamId": int(r["streamId"]),
            "kernel_name": r["kernel_name"],        # short (or fallback)
            "kernel_full": r["kernel_full"],        # full demangled (or fallback)
            "api_start": int(r["api_start"]),
            "api_end": int(r["api_end"]),
        }
    )

# sort each child list by kernel start time
for key in kernels_by_nvtx:
    kernels_by_nvtx[key].sort(key=lambda x: x["k_start"])

# For convenience, compute an "unscoped" bucket: kernels with no enclosing NVTX on this thread
unscoped_kernels = kernels_by_nvtx.get(None, [])
len(kernels_thread), sum(len(v) for v in kernels_by_nvtx.values())


    
# %% [markdown]
# v2 printer: show NVTX nodes and kernels as children (sorted by start time).
# Includes absolute ns and relative ms times.

def fmt_rel(ns):
    return (ns - T0) / 1e6


def print_tree_v2(
    df_nodes: pd.DataFrame,
    show_only_kernel_paths: bool = False,
    max_rows: int | None = None,
    show_full_kernel_name: bool = False,
    show_time: bool = False,
    kernel_name_filters: list[str] | None = None,
    case_insensitive: bool = True,
):
    """
    Print NVTX tree with kernels as children of their innermost NVTX.

    Args:
        df_nodes: NVTX nodes dataframe (nodes2).
        show_only_kernel_paths: If True, only show NVTX nodes that are on a path to at least one
            *printed* kernel (after applying kernel_name_filters).
        max_rows: Limit number of NVTX nodes printed (after filtering and sort).
        show_full_kernel_name: Print demangled kernel name line.
        show_time: Print absolute and relative times for NVTX and kernels.
        kernel_name_filters: Optional list of substrings; when provided, only kernels whose
            short or full name contains ANY of these substrings are printed. NVTX nodes with no
            matching child kernels are hidden when show_only_kernel_paths=True.
        case_insensitive: Match filters case-insensitively (default True).
    """

    def _match_kernel(krec) -> bool:
        if not kernel_name_filters:
            return True
        name_short = krec.get("kernel_name") or ""
        name_full  = krec.get("kernel_full") or ""
        if case_insensitive:
            name_short_l = name_short.lower()
            name_full_l  = name_full.lower()
            for pat in kernel_name_filters:
                pat_l = pat.lower()
                if pat_l in name_short_l or pat_l in name_full_l:
                    return True
            return False
        else:
            for pat in kernel_name_filters:
                if pat in name_short or pat in name_full:
                    return True
            return False

    # Build a filtered view of kernels_by_nvtx according to kernel_name_filters
    filtered_kernels_by_nvtx: dict[int, list[dict]] = {}
    for nid, lst in kernels_by_nvtx.items():
        if nid is None:
            # handled separately for unscoped list
            continue
        kept = [k for k in lst if _match_kernel(k)]
        if kept:
            filtered_kernels_by_nvtx[int(nid)] = kept

    # Optionally reduce to nodes on paths that include (filtered) kernels
    df = df_nodes.copy()
    if show_only_kernel_paths:
        keep: set[int] = set()
        for nid in filtered_kernels_by_nvtx.keys():
            cur = int(nid)
            keep.add(cur)
            # include its ancestors
            while True:
                p = parent_of.get(cur)
                if p is None:
                    break
                keep.add(p)
                cur = p
        if keep:
            df = df[df["nid"].isin(keep)]
        else:
            df = df.iloc[0:0]

    # Sort and cap
    df = df.sort_values(["start", "end"], ascending=[True, False])
    if max_rows is not None:
        df = df.head(max_rows)

    for _, r in df.iterrows():
        indent = "  " * int(r["depth"])
        et = int(r["eventType"])
        ns0 = int(r["start"]) ; ns1 = int(r["end"])
        rel0 = fmt_rel(ns0) ; dur_ms = (ns1 - ns0) / 1e6
        star = "★ " if r.get("launched_kernels_on_gpu", False) else "  "
        print(f"{indent}{star}[{et}] nid={int(r['nid'])}  {r['name']}")
        if show_time:
            print(f"{indent}    abs: start={ns0/1e6:.3f} ms  end={ns1/1e6:.3f} ms")
            print(f"{indent}    rel: t0+{rel0:.3f} ms  dur={dur_ms:.3f} ms")

        # Print only filtered kernels under this NVTX
        kids = filtered_kernels_by_nvtx.get(int(r["nid"]), []) if kernel_name_filters else (kernels_by_nvtx.get(int(r["nid"])) or [])
        for krec in kids:
            k_rel = fmt_rel(krec["k_start"]) ; k_dur = (krec["k_end"] - krec["k_start"]) / 1e6
            print(f"{indent}    └─ (kernel) {krec['kernel_name']}  [dev={krec['deviceId']}, stream={krec['streamId']}]")
            if show_full_kernel_name:
                print(f"{indent}       full: {krec['kernel_full']}")
            if show_time:
                print(f"{indent}       abs:  start={krec['k_start']} ns  end={krec['k_end']} ns")
                print(f"{indent}       rel:  t0+{k_rel:.3f} ms  dur={k_dur:.3f} ms")

    # Unscoped kernels (no enclosing NVTX on this thread)
    if unscoped_kernels:
        unscoped = [k for k in unscoped_kernels if _match_kernel(k)] if kernel_name_filters else unscoped_kernels
        if unscoped:
            print("\n(unscoped kernels on this thread; no enclosing push/pop NVTX)")
            for krec in unscoped:
                k_rel = fmt_rel(krec["k_start"]) ; k_dur = (krec["k_end"] - krec["k_start"]) / 1e6
                print(f"  (kernel) {krec['kernel_name']}  [dev={krec['deviceId']}, stream={krec['streamId']}]")
                if show_full_kernel_name:
                    print(f"     full: {krec['kernel_full']}")
                if show_time:
                    print(f"     abs:  start={krec['k_start']/1e6:.3f} ms  end={krec['k_end']/1e6:.3f} ms")
                    print(f"     rel:  t0+{k_rel:.3f} ms  dur={k_dur:.3f} ms")

print("── NVTX + kernels (all nodes) ──")
# print_tree_v2(nodes2, show_only_kernel_paths=False, show_full_kernel_name=False, show_time=True)
# Only show alltoall-related kernels (short or full demangled) and their parents
print_tree_v2(
    nodes2,
    show_only_kernel_paths=True,
    show_full_kernel_name=True,
    show_time=True,
    kernel_name_filters=["alltoall", "spreadout_alltoallv_internode_kernel"]
)

# print(f"\n── NVTX + kernels (only paths that include kernels on GPU {GPU_ID}) ──")
# print_tree_v2(nodes2, show_only_kernel_paths=True)

# %% [markdown]
# Done. Close the connection.

# %%
