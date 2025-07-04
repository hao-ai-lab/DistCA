import re
import json
from datetime import datetime

import streamlit as st
from streamlit_cookies_manager import EncryptedCookieManager
import pandas as pd
import plotly.graph_objects as go
import timemodule as tm

st.set_page_config(page_title="Latency Sandbox", layout="wide")
st.title("Latency Sandbox")

# ------------------------------------------------------------------
# Cookie manager for persisting configuration across sessions
# ------------------------------------------------------------------
# A small symmetric key is sufficient – change if stronger security required.
cookies = EncryptedCookieManager(prefix="latency_sandbox", password="__secret_key__")
# The first time the component loads, it returns no cookies; stop and let it reload once ready.
if not cookies.ready():
    st.stop()


# ------------------------------------------------------------------
# Config helpers (apply / snapshot)
# ------------------------------------------------------------------


def _apply_config(cfg: dict):
    """Populate st.session_state from a previously-saved config dict."""
    if not isinstance(cfg, dict):
        return
    st.session_state.workers = cfg.get("workers", [])
    # UI controls (populate prior to drawing widgets)
    st.session_state["cols_slider"] = cfg.get("cols_per_row", 4)
    st.session_state["metrics_slider"] = cfg.get("metric_cols_per_row", 2)
    st.session_state["shared_mlp_toggle"] = cfg.get("use_shared_mlp", False)
    st.session_state["mlp_tp"] = cfg.get("mlp_tp", 1)
    st.session_state["mlp_cp"] = cfg.get("mlp_cp", 1)
    st.session_state["mlp_dp"] = cfg.get("mlp_dp", 1)
    st.session_state["zero_ag_toggle"] = cfg.get("zero_ag", False)


def _current_config() -> dict:
    """Return a JSON-serialisable snapshot of the current UI state."""
    return {
        "workers": st.session_state.get("workers", []),
        "cols_per_row": st.session_state.get("cols_slider", 4),
        "metric_cols_per_row": st.session_state.get("metrics_slider", 2),
        "use_shared_mlp": st.session_state.get("shared_mlp_toggle", False),
        "mlp_tp": st.session_state.get("mlp_tp", 1),
        "mlp_cp": st.session_state.get("mlp_cp", 1),
        "mlp_dp": st.session_state.get("mlp_dp", 1),
        "zero_ag": st.session_state.get("zero_ag_toggle", False),
    }


# ------------------------------------------------------------------
# Deferred config application (used when loading after widgets exist)
# ------------------------------------------------------------------

# If a previous interaction requested to load a new configuration _after_
# widgets were already instantiated, we stash it in ``st.session_state``
# under the key ``pending_config`` and trigger ``st.rerun()``. On the next
# run – which reaches this point **before** any widgets are re-created –
# we safely apply the configuration.

if "pending_config" in st.session_state:
    try:
        _apply_config(st.session_state.pop("pending_config"))
    except Exception as err:  # pragma: no cover – best-effort load
        st.warning(f"Failed to apply pending config: {err}")


# ------------------------------------------------------------------
# One-time load of config from cookie, if available
# ------------------------------------------------------------------

if "config_loaded" not in st.session_state:
    saved = cookies.get("saved_config")
    if saved:
        try:
            _apply_config(json.loads(saved))
        except Exception as err:  # pragma: no cover – best-effort load
            st.warning(f"Failed to load saved config: {err}")
    st.session_state.config_loaded = True

# ------------------------------------------------------------------
# Session-state initialisation
# ------------------------------------------------------------------
if "workers" not in st.session_state:
    st.session_state.workers = []  # each {'id','tp','cp','seq': [{'id','len'}]}

workers = st.session_state.workers

# --------------- Sidebar settings ---------------
st.sidebar.header("Settings")
cols_per_row = st.sidebar.slider("Workers per row", 1, 16, 4, 1, key="cols_slider")
metric_cols_per_row = st.sidebar.slider("Metrics per row", 1, 4, 2, 1, key="metrics_slider")

# Shared MLP settings
use_shared_mlp = st.sidebar.checkbox("Use shared MLP worker", value=False, key="shared_mlp_toggle")
mlp_tp = st.sidebar.selectbox("Shared MLP tp", [1, 2, 4, 8], index=0, key="mlp_tp", disabled=not use_shared_mlp)
mlp_cp = st.sidebar.selectbox("Shared MLP cp", [1, 2, 4, 8], index=0, key="mlp_cp", disabled=not use_shared_mlp)
mlp_dp = st.sidebar.selectbox("Shared MLP dp", [1, 2, 4, 8, 16], index=0, key="mlp_dp", disabled=not use_shared_mlp)

# Option to zero AllGather on attention workers
zero_ag = st.sidebar.checkbox("Zero AllGather on attention workers", value=False, key="zero_ag_toggle")


# ------------------------------------------------------------------
# Helper functions
# ------------------------------------------------------------------

def next_worker_id():
    return 0 if not workers else max(w["id"] for w in workers) + 1


def next_seq_id(worker):
    return 0 if not worker["seq"] else max(s["id"] for s in worker["seq"]) + 1


def seq_times(tp, cp, length):
    attn = tm.get_attn_time(tp, cp, length)
    mlp = tm.get_mlp_time(tp, cp, length)
    ag = tm.get_allgather_time(tp, cp, length)
    return attn, mlp, ag, attn + mlp + ag


# expression parser (suffixes + math)
def parse_length(expr: str) -> int:
    expr = expr.lower().strip().replace(' ', '')
    expr = re.sub(r'(\d+)k', lambda m: f'({m.group(1)}*1024)', expr)
    expr = re.sub(r'(\d+)m', lambda m: f'({m.group(1)}*1024*1024)', expr)
    if not re.match(r'^[0-9+\-*/().]*$', expr):
        raise ValueError("Invalid characters in expression")
    return int(eval(expr, {"__builtins__": None}))


# ------------------------------------------------------------------
# Render each worker panel in grid
# ------------------------------------------------------------------
overall_worker_totals = []
shared_mlp_total = 0.0
if not workers:
    st.info("Add a worker to begin.")
else:
    # chunk workers
    chunks = [workers[i:i + cols_per_row] for i in range(0, len(workers), cols_per_row)]
    for row in chunks:
        columns = st.columns(cols_per_row)
        for col, w in zip(columns, row):
            with col:
                tp, cp = w["tp"], w["cp"]
                seqs = w["seq"]

                # compute totals
                total_attn = total_mlp = total_ag = total_total = 0.0
                rows = []
                for s in seqs:
                    attn, mlp, ag, tot = seq_times(tp, cp, s["len"])
                    if use_shared_mlp:
                        mlp = 0.0  # offloaded
                        tot = attn + ag
                    if zero_ag:
                        ag = 0.0
                        tot = attn + mlp
                    total_attn += attn
                    total_mlp += mlp
                    total_ag += ag
                    total_total += tot
                    rows.append({"Seq ID": s["id"], "Length": s["len"], "Attention": attn, "MLP": mlp, "AllGather": ag,
                                 "Total": tot})

                exp = st.expander(f"Worker {w['id']}  |  tp={tp}  cp={cp}  |  Seq={len(seqs)}  |  {total_total:.1f} ms",
                                  expanded=True)
                with exp:
                    # record for global metric
                    overall_worker_totals.append(total_total)

                    metrics = [
                        ("Total Attention", total_attn),
                        ("Total MLP", total_mlp),
                        ("Total AllGather", total_ag),
                        ("Total Latency", total_total),
                    ]
                    # chunk metrics into rows based on sidebar setting
                    for i in range(0, len(metrics), metric_cols_per_row):
                        row_metrics = metrics[i:i + metric_cols_per_row]
                        cols_m = st.columns(len(row_metrics))
                        for col_m, (label, val) in zip(cols_m, row_metrics):
                            col_m.metric(label, f"{val:.1f} ms")

                    # --- editable tp/cp ---
                    edit_tp, edit_cp, edit_btn = st.columns([1, 1, 1])
                    new_tp = edit_tp.selectbox("tp", [1, 2, 4, 8], index=[1, 2, 4, 8].index(tp),
                                               key=f"edit_tp_{w['id']}")
                    new_cp = edit_cp.selectbox("cp", [1, 2, 4, 8], index=[1, 2, 4, 8].index(cp),
                                               key=f"edit_cp_{w['id']}")
                    if edit_btn.button("Apply", key=f"apply_tp_cp_{w['id']}"):
                        w['tp'] = new_tp
                        w['cp'] = new_cp
                        st.rerun()

                    st.markdown("#### Add Sequence")
                    c_len, c_btn = st.columns([2, 1])
                    expr = c_len.text_input("Seq len expr", value="1024", key=f"len_w{w['id']}")
                    if c_btn.button("Add", key=f"add_seq_{w['id']}"):
                        try:
                            length_val = parse_length(expr)
                            w["seq"].append({"id": next_seq_id(w), "len": length_val})
                            st.rerun()
                        except Exception as e:
                            st.error(str(e))

                    if rows:
                        df = pd.DataFrame(rows)
                        st.dataframe(df.style.format("{:.2f}"), use_container_width=True)

                        # multi-select delete
                        del_ids = st.multiselect("Delete Sequences", [r["Seq ID"] for r in rows],
                                                 key=f"multi_del_{w['id']}")
                        if st.button("Delete Selected", key=f"del_selected_{w['id']}") and del_ids:
                            w["seq"] = [s for s in seqs if s["id"] not in del_ids]
                            st.rerun()

                        with st.expander("Show Plot", expanded=False):
                            fig = go.Figure()
                            for comp, color in zip(["Attention", "MLP", "AllGather"],
                                                   ["#3498db", "#2ecc71", "#e74c3c"]):
                                fig.add_trace(go.Bar(name=comp, x=[f"S{r['Seq ID']}" for r in rows], y=df[comp],
                                                     marker_color=color))
                            fig.update_layout(barmode="stack", height=250, margin=dict(l=0, r=0, t=30, b=0))
                            st.plotly_chart(fig, use_container_width=True, key=f"plot_{w['id']}")
                    else:
                        st.info("No sequences yet for this worker.")

                    if st.button("🗑️ Delete Worker", key=f"del_worker_{w['id']}"):
                        workers.remove(w)
                        st.rerun()

# ---------------- Add Worker in sidebar (after helpers so functions exist) ----------------
st.sidebar.markdown("---")
st.sidebar.subheader("Add Worker")
tp_val = st.sidebar.selectbox("tp", [1, 2, 4, 8], key="add_tp")
cp_val = st.sidebar.selectbox("cp", [1, 2, 4, 8], key="add_cp")
if st.sidebar.button("➕ Add Worker", key="btn_sidebar_add_worker"):
    workers.append({"id": next_worker_id(), "tp": tp_val, "cp": cp_val, "seq": []})
    st.rerun()

# ---------------- Config persistence controls ----------------
st.sidebar.markdown("---")
st.sidebar.subheader("Config Persistence")

# Helper – gather saved sessions (new multi-cookie format)
def _list_session_keys():
    return [k for k in cookies.keys() if k.startswith("cfg_")]

# ---------- Save current session ----------
st.sidebar.markdown("**Save Current Session**")
new_name = st.sidebar.text_input("Name", value="", key="save_session_name")
if st.sidebar.button("💾 Save Snapshot", key="btn_save_snapshot"):
    timestamp = datetime.now().strftime("%Y%m%dT%H%M%S_%f")
    key = f"cfg_{timestamp}"
    cfg = _current_config()
    cfg["name"] = new_name.strip() or timestamp
    cookies[key] = json.dumps(cfg)
    cookies.save()
    st.sidebar.success("Session saved!")

st.sidebar.markdown("---")

# ---------- Manage saved sessions ----------
session_keys = _list_session_keys()
if session_keys:
    # Build mapping key -> display name
    key_to_name = {}
    for k in session_keys:
        try:
            key_to_name[k] = json.loads(cookies.get(k)).get("name", k)
        except Exception:
            key_to_name[k] = k

    selected_key = st.sidebar.selectbox(
        "Saved Sessions",
        session_keys,
        format_func=lambda k: key_to_name.get(k, k),
        key="select_saved_session",
    )

    sel_cfg_raw = cookies.get(selected_key)
    sel_cfg = None
    if sel_cfg_raw:
        try:
            sel_cfg = json.loads(sel_cfg_raw)
        except Exception:
            sel_cfg = None

    # Load button
    if st.sidebar.button("⏪ Load Selected", key="btn_load_selected") and sel_cfg is not None:
        st.session_state["pending_config"] = sel_cfg
        st.sidebar.success("Loading session – applying…")
        st.rerun()

    # Rename section
    with st.sidebar.expander("Rename Selected", expanded=False):
        new_label = st.text_input("New name", value=key_to_name.get(selected_key, selected_key), key="rename_input")
        if st.button("✏️ Rename", key="btn_rename_session"):
            if sel_cfg is not None:
                sel_cfg["name"] = new_label.strip() or sel_cfg.get("name", "")
                cookies[selected_key] = json.dumps(sel_cfg)
                cookies.save()
                st.success("Session renamed!")
                st.rerun()

    # Export / Delete
    with st.sidebar.expander("Export / Delete", expanded=False):
        if st.button("🗑️ Delete", key="btn_delete_session"):
            del cookies[selected_key]
            cookies.save()
            st.rerun()

        if sel_cfg is not None:
            st.download_button(
                label="⬇️ Export JSON",
                data=json.dumps(sel_cfg, indent=2),
                mime="application/json",
                file_name=f"{key_to_name.get(selected_key, selected_key)}.json",
                key="btn_export_selected_json",
            )
else:
    st.sidebar.info("No saved sessions yet.")

# ---------- Import JSON ----------
st.sidebar.markdown("---")
st.sidebar.markdown("**Import Session JSON**")
uploaded_cfg = st.sidebar.file_uploader("⬆️ Import JSON", type=["json"], key="upload_json")
if uploaded_cfg is not None:
    try:
        imported_cfg = json.load(uploaded_cfg)
        if not isinstance(imported_cfg, dict):
            raise ValueError("Uploaded JSON is not a config dictionary")
        # Save under new timestamp key
        timestamp = datetime.now().strftime("%Y%m%dT%H%M%S_%f")
        key = f"cfg_{timestamp}"
        imported_cfg.setdefault("name", timestamp)
        cookies[key] = json.dumps(imported_cfg)
        cookies.save()
        st.sidebar.success("Session imported!")
        st.session_state["pending_config"] = imported_cfg
        st.rerun()
    except Exception as e:
        st.sidebar.error(f"Failed to import configuration: {e}")

# ---------------- Shared MLP worker panel ----------------
if use_shared_mlp and workers:
    # gather all sequences
    merged_seqs = []
    for w in workers:
        merged_seqs.extend(w["seq"])
    if merged_seqs:
        total_len = sum(s["len"] for s in merged_seqs)
        shared_mlp_latency = tm.get_mlp_time(mlp_tp, mlp_cp, total_len / mlp_dp)
        # allgather cost per seq remains, accumulate
        total_ag = 0.0
        rows = []
        for s in merged_seqs:
            ag = tm.get_allgather_time(mlp_tp, mlp_cp, s["len"])
            rows.append({"Seq": s["id"], "Len": s["len"], "AllGather": ag})
            total_ag += ag
        total_total = shared_mlp_latency + total_ag
        shared_mlp_total = total_total
        exp = st.expander(f"SHARED MLP WORKER | tp={mlp_tp} cp={mlp_cp} | {len(rows)} seq | {total_total:.1f} ms",
                          expanded=True)
        with exp:
            cols_m = st.columns(metric_cols_per_row)
            metrics = [("MLP Latency", shared_mlp_latency), ("Total AllGather", total_ag),
                       ("Total Latency", total_total)]
            for col, (lab, val) in zip(cols_m, metrics):
                col.metric(lab, f"{val:.1f} ms")
            df = pd.DataFrame(rows)
            st.dataframe(df.style.format("{:.2f}"), use_container_width=True)
            
# ---------------- Overall Latency Metric ----------------
if overall_worker_totals:
    max_attn = max(overall_worker_totals)
else:
    max_attn = 0.0

# shared_mlp_total set within shared panel computation

grand_total = max_attn + shared_mlp_total

st.metric("Estimated End-to-End Latency", f"{grand_total:.1f} ms")

# ------------------------------------------------------------------
# (End of file)
