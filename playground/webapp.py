import streamlit as st
import pandas as pd

@st.cache_data
def attn_table():
    from d2.profiling import get_attn_data
    df = get_attn_data()
    return df

@st.cache_data
def mlp_table():
    from d2.profiling import get_mlp_data
    df = get_mlp_data()
    return df

def get_mlp_time(batch: list[int], tp: int, cp: int):
    pass

st.set_page_config(layout="wide")

# Inject CSS to reduce top spacing
st.markdown("""
    <style>
        .block-container {
            padding-top: 1rem;
            padding-bottom: 1rem;
        }
    </style>
""", unsafe_allow_html=True)

st.title("WLBLLM Optimizer Playground")

# Side-by-side input for number of workers and total GPUs
col1, col2 = st.columns(2)
with col1:
    num_workers = st.number_input("Number of Workers (W)", min_value=1, max_value=32, value=1, step=1)
with col2:
    total_gpus = st.number_input("Total GPUs (G)", min_value=1, max_value=256, value=8, step=1)

# st.markdown("---")
# st.subheader("Worker Configurations")

# Worker config table
total_used_gpus = 0

for i in range(num_workers):
    # cols = st.columns(8)
    cols = st.columns([0.6, 3, 0.7, 0.7, 1, 1.5, 1.5, 1.5])

    with cols[0]:
        st.markdown(f"W{i + 1}")

    with cols[1]:
        batch_str = st.text_input("Batch (list)", key=f"batch_{i}", value="")
        try:
            batch_list = [int(x.strip()) for x in batch_str.split(",") if x.strip()]
        except ValueError:
            batch_list = []
            st.warning("⚠️ Please enter a comma-separated list of integers.")
    with cols[2]:
        tp = st.number_input("TP", key=f"tp_{i}", min_value=1, max_value=8, value=1)
    with cols[3]:
        cp = st.number_input("CP", key=f"cp_{i}", min_value=1, value=1)

    # Computed values
    num_gpus = tp * cp
    attn_time = float(sum(batch_list) * tp)
    mlp_time = float(sum(batch_list) * cp)
    total_used_gpus += num_gpus

    with cols[4]:
        st.number_input("Num GPUs", value=num_gpus, disabled=True, key=f"gpu_{i}")
    with cols[5]:
        st.number_input("Attn Time (ms)", value=attn_time, format="%.2f", disabled=True, key=f"attn_{i}")
    with cols[6]:
        st.number_input("MLP Time (ms)", value=mlp_time, format="%.2f", disabled=True, key=f"mlp_{i}")
    with cols[7]:
        st.number_input("Total Time (ms)", value=attn_time + mlp_time, format="%.2f", disabled=True, key=f"total_{i}")


if total_used_gpus > total_gpus:
    st.error(f"⚠️ Total used GPUs ({total_used_gpus}) exceeds available GPUs ({total_gpus})!")
else:
    st.success(f"✅ Total used GPUs: {total_used_gpus} / {total_gpus}")