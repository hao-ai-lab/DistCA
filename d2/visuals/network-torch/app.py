import pandas as pd
import numpy as np
import streamlit as st
from pathlib import Path
import re

# ────────────────────────────────────────────────────────────────────────────────
# Config & data load
# ────────────────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent
CSV_PATH = ROOT / "network-allgather-H100-2.csv"

df_raw = pd.read_csv(CSV_PATH)

# ────────────────────────────────────────────────────────────────────────────────
# Helper functions
# ────────────────────────────────────────────────────────────────────────────────
_SUFFIX = {"k": 1 << 10, "m": 1 << 20, "g": 1 << 30, "": 1}

def parse_qty(txt: str) -> int:
    """Parse quantities like "1M", "256k", "1024" → int."""
    m = re.fullmatch(r"\s*([0-9]*\.?[0-9]+)\s*([kKmMgG]?)\s*", txt)
    if not m:
        raise ValueError("Invalid quantity format")
    val, suf = m.groups()
    return int(float(val) * _SUFFIX[suf.lower()])

def fmt_qty(x: int) -> str:
    for suf, mul in reversed(_SUFFIX.items()):
        if x >= mul and mul != 1:
            return f"{x / mul:.2f}{suf.upper()}"
    return str(x)

def interp(xq: int, xs: np.ndarray, ys: np.ndarray) -> float:
    """Linear interpolation/extrapolation for a monotonic x-axis."""
    idx = np.searchsorted(xs, xq)
    if idx == 0:
        # extrapolate below range
        x0, x1, y0, y1 = xs[0], xs[1], ys[0], ys[1]
    elif idx == len(xs):
        # extrapolate above range
        x0, x1, y0, y1 = xs[-2], xs[-1], ys[-2], ys[-1]
    else:
        x0, x1, y0, y1 = xs[idx - 1], xs[idx], ys[idx - 1], ys[idx]
    slope = (y1 - y0) / (x1 - x0)
    return y0 + slope * (xq - x0)

# ────────────────────────────────────────────────────────────────────────────────
# Streamlit UI
# ────────────────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="AllGather Latency Explorer", page_icon="📈", layout="wide")

st.title("📈 AllGather Latency Explorer (H100, world size=2)")

with st.sidebar:
    st.header("Query latency")
    # Operation selector (only allgather exists but keep UI flexible)
    op_sel = st.selectbox("Collective op", sorted(df_raw.op.unique()))
    dtype_sel = st.selectbox("Data type", sorted(df_raw.dtype.unique()))
    nelem_txt = st.text_input("Elements per GPU (k/m/g suffix ok)", value="1024")

    submit = st.button("Estimate latency")

# Main area – table
st.subheader("Benchmark table")
st.dataframe(df_raw, use_container_width=True)

if submit:
    try:
        nelem_req = parse_qty(nelem_txt)
    except ValueError as e:
        st.error(str(e))
        st.stop()

    seg = df_raw[(df_raw.op == op_sel) & (df_raw.dtype == dtype_sel)]
    if seg.empty:
        st.error("No data for chosen op / dtype combination.")
        st.stop()

    xs = seg.nelem.values
    ys = seg["latency(ms)"].values
    est_ms = interp(nelem_req, xs, ys)

    st.subheader("Estimation result")
    st.write(f"Requested elements per GPU: {fmt_qty(nelem_req)}")
    st.metric(label="Estimated latency (ms)", value=f"{est_ms:.3f} ms")
    # Throughput estimation (nelem per ms)
    ys_tp = seg["throughput(nelem_per_ms)"].values
    est_tp = interp(nelem_req, xs, ys_tp)
    st.metric(label="Estimated throughput (elem/ms)", value=f"{est_tp:,.1f}")

    # Plot curve with point
    import altair as alt
    chart_df = seg.copy()
    chart_df["lat_ms"] = chart_df["latency(ms)"]
    chart_df["label"] = "data"
    point_df = pd.DataFrame({
        "nelem": [nelem_req],
        "lat_ms": [est_ms],
        "label": ["estimate"],
    })
    full_df = pd.concat([chart_df[["nelem", "lat_ms", "label"]], point_df])
    base = alt.Chart(full_df).encode(
        x=alt.X("nelem", title="Elements per GPU", scale=alt.Scale(type="log")),
        y=alt.Y("lat_ms", title="Latency (ms)"),
        color="label"
    )
    st.altair_chart(base.mark_line(point=True), use_container_width=True)
