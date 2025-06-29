import re, ast, operator, json, requests, streamlit as st

# ──────────────────────────────────────────────  Setup  ─────────────────────────────────────────────
st.set_page_config(layout="wide")
st.markdown(
    """
    <style>
    .block-container{padding-top:1rem;padding-bottom:1rem;}
    div[data-testid="stTextInput"] input{
        max-width:6rem;padding:2px 4px;text-align:center;font-size:0.85rem;}
    div[data-testid="stTextInput"] .st-cg{background:transparent;}
    </style>
    """,
    unsafe_allow_html=True,
)
st.title("FLOPs & Latency Estimator")

# ──────────────────────────────  1.  Smart integer parser (k, kk, 64k*16, …)  ──────────────────────
_UNIT_EXP = dict(k=1, m=2, g=3, t=4, p=5, e=6)               # IEC powers
_NUM_UNIT = re.compile(r'(\d+(?:\.\d+)?)\s*([a-zA-Z]+)')

def _unit_pow(u: str) -> int:
    u = u.lower()
    if set(u) == {'k'}:               # 'k', 'kk', …
        return len(u)
    if len(u) == 1 and u in _UNIT_EXP:
        return _UNIT_EXP[u]
    raise ValueError(u)

def _rewrite(expr: str) -> str:
    def repl(m):
        num, unit = m.groups()
        return f"({num}*{1024**_unit_pow(unit)})"
    return _NUM_UNIT.sub(repl, expr)

_OP = {ast.Add:operator.add, ast.Sub:operator.sub, ast.Mult:operator.mul,
       ast.Div:operator.truediv, ast.FloorDiv:operator.floordiv,
       ast.Pow:operator.pow, ast.USub:operator.neg}

def _safe_eval(expr: str) -> float:
    expr = _rewrite(expr.replace("^", "**"))
    node = ast.parse(expr, mode="eval")
    def _eval(n):
        if isinstance(n, ast.Constant): return n.value
        if isinstance(n, ast.BinOp):    return _OP[type(n.op)](_eval(n.left), _eval(n.right))
        if isinstance(n, ast.UnaryOp):  return _OP[type(n.op)](_eval(n.operand))
        raise ValueError
    return _eval(node.body)

def get_int(label: str, default: str) -> int:
    txt = st.text_input(label, default, key=label)
    try:      return int(round(_safe_eval(txt)))
    except:   st.error(f"{label}: invalid"); return int(round(_safe_eval(default)))

def iec(n: int) -> str:                              # pretty-print
    for u, p in reversed(_UNIT_EXP.items()):
        if n % (1024**p) == 0:
            return f"{n // (1024**p)} {u.upper()}"
    return f"{n:,}"

# ──────────────────────────────  2.  Top-level meta inputs  ────────────────────────────────────────
mcol1, mcol2 = st.columns(2)
with mcol1:
    model_name = st.text_input("Model Name", "Qwen3MoeForCausalLM")
with mcol2:
    config_path = st.text_input("Config Path",
        "https://huggingface.co/Qwen/Qwen3-235B-A22B/raw/main/config.json")

# optional JSON viewer
with st.expander("🗂 Model config (click)"):
    try:
        cache_key = f"_cfg::{config_path}"
        if cache_key not in st.session_state:
            st.session_state[cache_key] = (
                requests.get(config_path, timeout=10).json()
                if config_path.startswith(("http", "https"))
                else json.load(open(config_path))
            )
        st.json(st.session_state[cache_key], expanded=True)
    except Exception as e:
        st.error(f"Could not load config: {e}")

# ──────────────────────────────  3.  Core parameter inputs  ────────────────────────────────────────
param_specs = [
    ("Hidden Size",        "4096"),
    ("Qo Heads",           "64"),
    ("Kv Heads",           "4"),
    ("Head Dim",           "128"),
    ("MoE Intermediate",   "1536"),
    ("Activated Experts",  "8"),
]
seq_specs = [
    ("Sequence Length", "64k"),
    ("Batch Size",      "1"),
    ("TP",              "1"),
    ("CP",              "1"),
]

def make_row(specs):
    out, cols = {}, st.columns(len(specs))
    for (label, default), col in zip(specs, cols):
        with col:
            out[label] = get_int(label, default)
    return out

cfg  = make_row(param_specs)
seq  = make_row(seq_specs)

# handy aliases
hs, hq, hk, hd, dexp, kexp = (cfg[k] for k in
    ("Hidden Size", "Qo Heads", "Kv Heads", "Head Dim", "MoE Intermediate", "Activated Experts"))

# ──────────────────────────────  5.  FLOPs per op ───────────────────────────────────────────
def flops(tag: str, seq_len: int) -> int:
    t = seq_len * seq["Batch Size"]
    if tag == "Q":           return 2 * t * hs * hd * hq
    if tag == "KV":          return 2 * t * hs * hd * hk
    if tag == "LinearProj":  return 2 * t * hs * hs
    if tag == "FC1":         return 2 * t * kexp * hs * dexp
    if tag == "Activation":  return 4 * t * kexp * dexp
    if tag == "FC2":         return 2 * t * kexp * dexp * hs
    raise ValueError(tag)

# ──────────────────────────────  6.  Formula row ───────────────────────────────────────────

est_ms = dict(
    Q=(32768, 8.1972732544),
    KV=(32768, 2.0493183136),
    LinearProj=(32768, 4.9398078918),
    FC1=(32768, 23.0062084198),
    Activation=(32768, 2.6169919968),
    FC2=(32768, 19.2186565399),
)

def formula_row(tag, coef, factors):
    """Draw a row, allow per-row calibration, show est latency."""

    N_fields = len(factors)

    # tag | coef | factors | ctx | ms | est | FLOPs
    cols = st.columns([0.6] + [1] * (N_fields + 5 + 1)) 

    cols[0].markdown(f"**{tag}**")
    cols[1].text_input("coef", iec(coef), disabled=True, label_visibility="visible",
                       key=f"{tag}_coef")
    for (lab, val), c in zip(factors, cols[2:2 + len(factors)]):
        c.text_input(lab, iec(val), disabled=True, label_visibility="visible",
                     key=f"{tag}_{lab}")

    est_col = cols[-1]
    ms_col = cols[-2]
    ctx_col = cols[-3]
    f_col = cols[-5]
    adjusted_f_col = cols[-4]

    ctx  = ctx_col.number_input("measured ctx", value=est_ms[tag][0], step=1, key=f"{tag}_ctx")
    # ctx = 64 * 1024
    ms   = ms_col.number_input("measured ms", value=est_ms[tag][1], step=0.1, key=f"{tag}_ms")

    flop_val = coef
    for _, v in factors: 
        flop_val *= v
    
    f_col.text_input("Total FLOPs", iec(flop_val), disabled=True,
                     label_visibility="visible", key=f"{tag}_flops")
    
    adjusted_f_val = flop_val
    if tag == "KV":
        num_kv_head = 4
        adjusted_f_val /= min(seq["TP"], num_kv_head)
    else:
        adjusted_f_val /= seq["TP"]
    
    adjusted_f_val /= seq["CP"]
    
    adjusted_f_col.text_input("FLOPs per GPU", iec(adjusted_f_val), disabled=True,
                     label_visibility="visible", key=f"{tag}_adjusted_flops")

    if ctx > 0 and ms > 0:
        est = ms * (seq["Sequence Length"] / ctx)
        if tag == "KV":
            est = est / min(seq["TP"], num_kv_head)
        else:
            est = est / seq["TP"]
            pass
        est = est / seq["CP"]

        est_col.text_input("≈ ms", f"{est:.3f}", disabled=True,
                           label_visibility="visible", key=f"{tag}_est")
    else:
        est = 0
        est_col.text_input("≈ ms", "", disabled=True, label_visibility="visible",
                           key=f"{tag}_est_blank")
    
    
    est_time = est
    return adjusted_f_val, est_time

st.divider(); st.subheader("Per-layer FLOPs & Latency")

q_fl, q_est = formula_row("Q", 2, [("batch_size", seq["Batch Size"]),
                            ("seq_len", seq["Sequence Length"]),
                            ("hidden_size", hs), ("head_dim", hd),
                            ("num_qo_head", hq),
                        ])

kv_fl, kv_est = formula_row("KV", 2, [("batch_size", seq["Batch Size"]),
                             ("seq_len", seq["Sequence Length"]),
                             ("hidden_size", hs), ("head_dim", hd),
                             ("num_kv_head", hk),
                            ])

proj_fl, proj_est = formula_row("LinearProj", 2, [("batch_size", seq["Batch Size"]),
                                     ("seq_len", seq["Sequence Length"]),
                                     ("hidden_size", hs), ("hidden_size_", hs)])

fc1_fl, fc1_est = formula_row("FC1", 2, [("batch_size", seq["Batch Size"]),
                              ("seq_len", seq["Sequence Length"]),
                              ("activated_experts", kexp),
                              ("hidden_size", hs), ("dexp", dexp)])

act_fl, act_est = formula_row("Activation", 4, [("batch_size", seq["Batch Size"]),
                                     ("seq_len", seq["Sequence Length"]),
                                     ("activated_experts", kexp),
                                     ("dexp", dexp)])

fc2_fl, fc2_est = formula_row("FC2", 2, [("batch_size", seq["Batch Size"]),
                              ("seq_len", seq["Sequence Length"]),
                              ("activated_experts", kexp),
                              ("dexp", dexp), ("hidden_size", hs)])

total_flops = q_fl + kv_fl + proj_fl + fc1_fl + act_fl + fc2_fl
total_est = q_est + kv_est + proj_est + fc1_est + act_est + fc2_est

cols = st.columns(2)
cols[0].text_input("Total linear FLOPs", iec(total_flops), disabled=True,
                   label_visibility="visible", key="total_lin")
cols[1].text_input("Total estimated latency", f"{total_est:.3f} ms", disabled=True,
                   label_visibility="visible", key="total_est")