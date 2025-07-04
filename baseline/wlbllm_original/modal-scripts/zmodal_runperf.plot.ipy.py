# %%

import json
import pandas as pd
df0 = pd.read_json("zmodal_runperf_H100_1.jsonl", lines=True)
df1 = pd.read_json("zmodal_runperf_H100_2.jsonl", lines=True)
df2 = pd.read_json("zmodal_runperf_H100_4.jsonl", lines=True)
df3 = pd.read_json("zmodal_runperf_H100_8.jsonl", lines=True)
df = pd.concat([df0, df1, df2, df3])
# df = pd.read_json("zmodal_runperf.jsonl", lines=True)
# df = pd.read_json("zmodal_runperf_H100_2.jsonl", lines=True)

def product(xs): 
    from functools import reduce
    return reduce(lambda x, y: x * y, xs)

allreduce_size = df["0"].map(lambda x: product(x["output_shape"]))
df['allreduce_size_per_gpu_per_seq'] = allreduce_size / df['num_seq_in_batch']

from network import get_allreduce_time
df['allreduce_time'] = df.apply(
    lambda x: get_allreduce_time(x['allreduce_size_per_gpu_per_seq'], x['tp']),
    axis=1,
)
df['per_seq_latency'] += df['allreduce_time']
df['per_doc_latency'] += df['allreduce_time']


df = df.drop(columns=[col for col in ["0", "1", "2", "3", "4", "5", "6", "7", "comment"] if col in df.columns])


df.head()

# %%


# df.head()
# %%

import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Create a subplot figure
fig = make_subplots()

# Add traces for each combination of tp and cp
for (tp, cp), group in df.groupby(['tp', 'cp']):
    # per_seq_latency
    fig.add_trace(
        go.Scatter(
            x=group['seq_len'],
            y=group['per_seq_latency'],
            mode='lines',
            name=f'Per-Seq Latency (tp={tp}, cp={cp})'
        )
    )
    # per_doc_latency
    fig.add_trace(
        go.Scatter(
            x=group['seq_len'],
            y=group['per_doc_latency'],
            mode='lines',
            name=f'Per-Doc Latency (tp={tp}, cp={cp})'
        )
    )
    # per_seq_allgather_latency
    fig.add_trace(
        go.Scatter(
            x=group['seq_len'],
            y=group['per_seq_allgather_latency'],
            mode='lines',
            name=f'Per-Seq Allgather Latency (tp={tp}, cp={cp})'
        )
    )
    # per_seq_attn_latency
    fig.add_trace(
        go.Scatter(
            x=group['seq_len'],
            y=group['per_seq_attn_latency'],
            mode='lines',
            name=f'Per-Seq Attn Latency (tp={tp}, cp={cp})'
        )
    )
    # per_doc_allgather_latency
    fig.add_trace(
        go.Scatter(
            x=group['seq_len'],
            y=group['per_doc_allgather_latency'],
            mode='lines',
            name=f'Per-Doc Allgather Latency (tp={tp}, cp={cp})'
        )
    )
    # per_doc_attn_latency
    fig.add_trace(
        go.Scatter(
            x=group['seq_len'],
            y=group['per_doc_attn_latency'],
            mode='lines',
            name=f'Per-Doc Attn Latency (tp={tp}, cp={cp})'
        )
    )
    # allreduce_time
    fig.add_trace(
        go.Scatter(
            x=group['seq_len'],
            y=group['allreduce_time'],
            mode='lines',
            name=f'Per-Seq Allreduce Latency (tp={tp}, cp={cp})'
        )
    )
    fig.add_trace(
        go.Scatter(
            x=group['seq_len'],
            y=group['allreduce_time'],
            mode='lines',
            name=f'Per-Doc Allreduce Latency (tp={tp}, cp={cp})'
        )
    )

# Update layout with buttons to toggle visibility
fig.update_layout(
    updatemenus=[
        dict(
            type="buttons",
            direction="left",
            buttons=list([
                dict(
                    args=[{"visible": [True if 'Per-Seq Latency' in trace.name else False for trace in fig.data]}],
                    label="Show Per-Seq Latency",
                    method="update"
                ),
                dict(
                    args=[{"visible": [True if 'Per-Doc Latency' in trace.name else False for trace in fig.data]}],
                    label="Show Per-Doc Latency",
                    method="update"
                ),
                dict(
                    args=[{"visible": [True if 'Per-Seq Allgather Latency' in trace.name else False for trace in fig.data]}],
                    label="Show Per-Seq Allgather Latency",
                    method="update"
                ),
                dict(
                    args=[{"visible": [True if 'Per-Seq Attn Latency' in trace.name else False for trace in fig.data]}],
                    label="Show Per-Seq Attn Latency",
                    method="update"
                ),
                dict(
                    args=[{"visible": [True if 'Per-Doc Allgather Latency' in trace.name else False for trace in fig.data]}],
                    label="Show Per-Doc Allgather Latency",
                    method="update"
                ),
                dict(
                    args=[{"visible": [True if 'Per-Doc Attn Latency' in trace.name else False for trace in fig.data]}],
                    label="Show Per-Doc Attn Latency",
                    method="update"
                ),
                dict(
                    args=[{"visible": [True if 'Per-Seq Allreduce Latency' in trace.name else False for trace in fig.data]}],
                    label="Show Per-Seq Allreduce Latency",
                    method="update"
                ),
                dict(
                    args=[{"visible": [True if 'Per-Doc Allreduce Latency' in trace.name else False for trace in fig.data]}],
                    label="Show Per-Doc Allreduce Latency",
                    method="update"
                ),
                dict(
                    args=[{"visible": [True for _ in fig.data]}],
                    label="Show All",
                    method="update"
                )
            ]),
            pad={"r": 0, "t": 0},
            showactive=True,
            x=0.1,
            xanchor="left",
            y=1.1,
            yanchor="bottom"
        ),
    ],
    # Set default visibility to only show Per-Seq Latency
    showlegend=True
)

# # Set default visibility to only show Per-Seq Latency
# for trace in fig.data:
#     trace.visible = 'legendonly' if 'Per-Doc' in trace.name else True

# Update axes labels
fig.update_xaxes(title_text="Sequence Length")
fig.update_yaxes(title_text="Latency (ms)")

# Add title
fig.update_layout(
    title_text=(
        "WLB-LLM CP Latency<br>"
        "Network latency only include all-gather, no all-reduce"
    )
)


# Show the figure
fig.show()
# save figure to html
fig.write_html("zmodal_runperf.plot.html")

# %%
