"""
Enhanced NCCL‑tests explorer
===========================
Launch with
```
streamlit run nccl_latency_app.py
```
This revision adds:

"""

from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Tuple

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).parent
CSV_PATH = ROOT / "nccl_perf.csv"

df = pd.read_csv(CSV_PATH)

# ────────────────────────────────────────────────────────────────────────────────
# Helpers
# ────────────────────────────────────────────────────────────────────────────────
_SUFFIX = {"k": 1 << 10, "m": 1 << 20, "g": 1 << 30, "": 1}


def parse_qty(txt: str) -> int:
    """Parse "1.5M", "256k", "1024" → int."""
    m = re.fullmatch(r"\s*([0-9]*\.?[0-9]+)\s*([kKmMgG]?)\s*", txt)
    if not m:
        raise ValueError("Invalid quantity")
    val, suf = m.groups()
    return int(float(val) * _SUFFIX[suf.lower()])


def fmt_qty(x: int) -> str:
    for suf, mul in reversed(_SUFFIX.items()):
        if x >= mul and mul != 1:
            return f"{x / mul:.2f}{suf.upper()}"
    return str(x)

def fmt_qty_byte(x: int) -> str:
    for suf, mul in reversed(_SUFFIX.items()):
        if x >= mul and mul != 1:
            return f"{x / mul:.2f}{suf.upper()}B"
    return str(x) + "B"


def interp_latency(size_req: int, sizes: np.ndarray, times: np.ndarray) -> Tuple[float, str]:
    """Linear interpolation / extrapolation (µs). Returns (lat, equation str)."""
    idx = np.searchsorted(sizes, size_req)
    if idx == 0:
        x0, x1 = sizes[0], sizes[1]
        y0, y1 = times[0], times[1]
    elif idx == len(sizes):
        x0, x1 = sizes[-2], sizes[-1]
        y0, y1 = times[-2], times[-1]
    else:
        x0, x1 = sizes[idx - 1], sizes[idx]
        y0, y1 = times[idx - 1], times[idx]
    slope = (y1 - y0) / (x1 - x0)
    est = y0 + slope * (size_req - x0)
    eq = f"t = {y0:.1f} + ({slope:.6f})*(bytes-{x0})"
    return est, eq

DTYPE_B = {"half": 2, "float": 4}




# ────────────────────────────────────────────────────────────────────────────────
# Sidebar inputs
# ────────────────────────────────────────────────────────────────────────────────
st.set_page_config(layout="wide", initial_sidebar_state="expanded", page_title="NCCL Latency Estimator", page_icon="📊")
st.markdown("<style>body{padding-top: 0px;}</style>", unsafe_allow_html=True)


with st.sidebar:
    st.header("NCCL Latency estimator")

    ops_sel = st.multiselect("Collective ops", sorted(df.op.unique()), default=sorted(df.op.unique()))
    gpus_sel = st.multiselect("GPU counts", sorted(df.ngpu.unique()), default=sorted(df.ngpu.unique()))

    dtype_key = st.selectbox("Data type", list(DTYPE_B.keys()), index=0)
    bpe = DTYPE_B[dtype_key]

    mode = st.radio("Describe payload as", ["tokens × hidden", "elements"], horizontal=True)

    if mode.startswith("elements"):
        elems_txt = st.text_input("Elements (k/m/g allowed)", "1M")
        total_elems = parse_qty(elems_txt)
    else:
        tok_txt = st.text_input("Tokens", "16384")
        hid_txt = st.text_input("Hidden size", "512")
        total_elems = parse_qty(tok_txt) * parse_qty(hid_txt)
        total_size = total_elems * bpe
        st.text_input("Total elements", value=fmt_qty(total_elems), disabled=True)
        st.text_input("Total size", value=fmt_qty(total_size)+"B", disabled=True)

    per_gpu = st.radio("Elements are", ["Total", "Per‑GPU"], horizontal=True)

    log_x = st.checkbox("Log X", value=False)
    log_y = st.checkbox("Log Y", value=False)
    kind = st.selectbox("Chart type", ["Line", "Bar"], index=0)

# ────────────────────────────────────────────────────────────────────────────────
# Derived quantities & latency calc
# ────────────────────────────────────────────────────────────────────────────────
chart_df = df[(df.op.isin(ops_sel)) & (df.ngpu.isin(gpus_sel))].copy()
if chart_df.empty:
    st.error("No data for chosen combination.")
    st.stop()

# compute latency for first op/gpu pair for headline
op0, n0 = ops_sel[0], gpus_sel[0]
seg = chart_df[(chart_df.op == op0) & (chart_df.ngpu == n0)]
req_bytes = (total_elems // (1 if per_gpu == "Per‑GPU" else n0)) * bpe
sizes = seg.size_bytes.values
lat_col = "t_us_out"  # use out‑of‑place for metric
latencies = seg[lat_col].values
est_us, equation = interp_latency(req_bytes, sizes, latencies)

# Determine the number of metrics to print
st.metric("Total Size", f"{fmt_qty(total_size)}B", help="Total size of the data in bytes")

num_ops = len(ops_sel)
num_gpus = len(gpus_sel)
metric_grid = [
    st.columns(num_ops) for _ in range(num_gpus)
]

# Prepare a grid for metrics display
for j, ngpu in enumerate(gpus_sel):
    for i, op in enumerate(ops_sel):
        seg = chart_df[(chart_df.op == op) & (chart_df.ngpu == ngpu)]
        if not seg.empty:
            sizes = seg.size_bytes.values
            latencies = seg[lat_col].values
            est_us, equation = interp_latency(req_bytes, sizes, latencies)
            with metric_grid[j][i]:
                st.metric(
                    f"{op} GPU={ngpu} (µs)", 
                    f"{est_us:.2f} µs", 
                    help=f"Bytes per element = {bpe} for {dtype_key}"
                )

# ────────────────────────────────────────────────────────────────────────────────
# Table view
# ────────────────────────────────────────────────────────────────────────────────
st.subheader("Benchmark table")
fmt_cols = {
    "size_bytes": "size_bytes",
    "t_us_out": "t_out (µs)",
    "t_us_in": "t_in (µs)",
}

# view = chart_df[[*fmt_cols]].rename(columns=fmt_cols)
view = chart_df.rename(columns=fmt_cols)
view = view.sort_values(["op", "ngpu", "size_bytes"])
# view["size_bytes"] = view["size_bytes"].apply(fmt_qty)
# view["count_elems"] = view["count_elems"].apply(fmt_qty)


column_details = """
- **op**: The type of NCCL operation being performed (e.g., all_gather, all_reduce, reduce_scatter).
- **ngpu**: The number of GPUs involved in the operation.
- **dtype**: The data type used in the operation (e.g., half, float).
- **size_bytes**: The size of bytes expected after the operation is completed.
  - AllGather, ReduceScatter: per-GPU size * number of GPUs
  - AllReduce: Just the size of the data.
- **count_elems**: The number of elements per rank.
- **t_us_out**: The time taken for the out-of-place operation in microseconds.
- **algbw_out_gbs**: The algorithmic bandwidth for the out-of-place operation in gigabytes per second.
- **busbw_out_gbs**: The bus bandwidth for the out-of-place operation in gigabytes per second.
- **t_us_in**: The time taken for the in-place operation in microseconds.
- **algbw_in_gbs**: The algorithmic bandwidth for the in-place operation in gigabytes per second.
- **busbw_in_gbs**: The bus bandwidth for the in-place operation in gigabytes per second.

See also:
- Meaning of size in NCCL tests: https://forums.developer.nvidia.com/t/meaning-of-size-in-nccl-tests/289806
- NCCL tests performance documentation: https://github.com/NVIDIA/nccl-tests/blob/master/doc/PERFORMANCE.md
"""

with st.expander("Column Details"):
    st.markdown(column_details)

# Add sequence length column if in tokens × hidden mode
if mode == "tokens × hidden":
    hidden_size = parse_qty(hid_txt)
    view['seq_len'] = (view['size_bytes'] / (hidden_size * bpe)).astype(int)

fmt = {
    "size_bytes": lambda x: str(fmt_qty_byte(x)),
    "count_elems": lambda x: str(fmt_qty(x)),
    "seq_len": lambda x: str(fmt_qty(x)),
}
fmt.update({
    col: "{:,.2f}"
    for col in ["algbw_out_gbs", "busbw_out_gbs", "algbw_in_gbs", "busbw_in_gbs", "t_out (µs)", "t_in (µs)"]
})
view = view.style.format(fmt)
st.dataframe(view, use_container_width=True)

# ────────────────────────────────────────────────────────────────────────────────
# Chart
# ────────────────────────────────────────────────────────────────────────────────
chart_df["lat_ms_out"] = chart_df.t_us_out / 1000
chart_df["size_MiB"] = chart_df.size_bytes / (1 << 20)
chart_df["label"] = chart_df.op + "‑" + chart_df.ngpu.astype(str) + "GPU"

base = alt.Chart(chart_df).encode(
    x=alt.X("size_MiB", title="Bytes per GPU (MiB)", scale=alt.Scale(type="log" if log_x else "linear")),
    y=alt.Y("lat_ms_out", title="Latency (ms)", scale=alt.Scale(type="log" if log_y else "linear")),
    color="label",
    tooltip=["label", "size_MiB", "lat_ms_out"],
)

if kind == "Line":
    chart = base.mark_line(point=True)
else:
    chart = base.mark_bar()

st.subheader("Latency curves")
st.altair_chart(chart, use_container_width=True)

st.caption("Linear interpolation/extrapolation is used between benchmark points to estimate latency. Bytes per element = %d for %s." % (bpe, dtype_key))
