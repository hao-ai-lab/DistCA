# %%

import json
import pandas as pd

# df1 = pd.read_json("zmodal_runperf_H100_2.jsonl", lines=True)
# df2 = pd.read_json("zmodal_runperf_H100_4.jsonl", lines=True)
# df3 = pd.read_json("zmodal_runperf_H100_8.jsonl", lines=True)
# df = pd.concat([df1, df2, df3])
# df = pd.read_json("zmodal_runperf.jsonl", lines=True)
df = pd.read_json("zmodal_runperf_H100_2.jsonl", lines=True)
# %%

df = df.drop(columns=[col for col in ["0", "1", "2", "3", "4", "5", "6", "7", "comment"] if col in df.columns])

df.head()
# %%

import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Create a subplot figure
fig = make_subplots()

# Add traces for each combination of tp and cp
for (tp, cp), group in df.groupby(['tp', 'cp']):
    fig.add_trace(
        go.Scatter(
            x=group['seq_len'],
            y=group['per_seq_latency'],
            mode='lines',
            name=f'Per-Seq Latency (tp={tp}, cp={cp})'
        )
    )
    fig.add_trace(
        go.Scatter(
            x=group['seq_len'],
            y=group['per_doc_latency'],
            mode='lines',
            name=f'Per-Doc Latency (tp={tp}, cp={cp})'
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
                    args=[{"visible": [True if 'Per-Seq' in trace.name else False for trace in fig.data]}],
                    label="Show Per-Seq Latency",
                    method="update"
                ),
                dict(
                    args=[{"visible": [True if 'Per-Doc' in trace.name else False for trace in fig.data]}],
                    label="Show Per-Doc Latency",
                    method="update"
                ),
                dict(
                    args=[{"visible": [True for _ in fig.data]}],
                    label="Show All",
                    method="update"
                )
            ]),
            pad={"r": 10, "t": 10},
            showactive=True,
            x=0.1,
            xanchor="left",
            y=1.1,
            yanchor="top"
        ),
    ],
    # Set default visibility to only show Per-Seq Latency
    showlegend=True
)

# Set default visibility to only show Per-Seq Latency
for trace in fig.data:
    trace.visible = 'legendonly' if 'Per-Doc' in trace.name else True

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
