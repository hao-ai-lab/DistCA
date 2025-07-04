# %%
import pandas as pd

filename = "compute-attn-H100.0.psv"
df = pd.read_csv(filename, sep="|")
df['ctx_len'] = df['total_len'] / df['batch_size']
df['ctx_len'] = df['ctx_len'].astype(int)
df['latency_us'] = df['latency(ms)'].astype(float) * 1000
df.drop(columns=['latency(ms)', 'total_latency(ms)', 'total_len', 'batch_size', 'hqo', 'hkv', 'd', 'batch'], inplace=True)
df.head()
# %%
import plotly.express as px

fig = px.scatter(
    df,
    x='ctx_len',
    y='latency_us',
    facet_row='tp',
    facet_col='cp',
    title='Latency vs Sequence Length for Different TP and CP Configurations',
    labels={'ctx_len': 'Sequence Length', 'latency_us': 'Latency (us)'},
    height=800,
    width=800,
    log_x=True,
    log_y=True,
)

fig.update_layout(
    margin=dict(l=20, r=20, t=40, b=20),
    title_x=0.5
)

fig.update_xaxes(matches=None, showticklabels=True)
fig.update_yaxes(matches=None, showticklabels=True)

fig.show()



# %%
