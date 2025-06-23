from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import d2.timemodule as tm
import plotly.graph_objects as go
import plotly.subplots as sp

K = 1024
M = 1024 * 1024

plt_dir = Path(__file__).parent / "test_timemodule_plots"
plt_dir.mkdir(parents=True, exist_ok=True)

def test_get_attn_time_monotonically_increase():
    for tp in [1, 2, 4, 8]:
        for cp in [1, 2, 4, 8]:
            prev = 0
            for seq_len in [
                32, 64, 128, 256, 512,
                1 * K, 2 * K, 4 * K, 8 * K, 
                16 * K, 32 * K, 64 * K, 128 * K, 
            ]:
                a = tm.get_attn_time(seq_len, tp, cp)
                assert a >= prev, f"tp={tp}, cp={cp}, seq_len={seq_len}, a={a}, prev={prev} is not increasing: {a - prev = }"
                # print(f"tp={tp}, cp={cp}, seq_len={seq_len}, a={a}")
    
def test_plot_get_attn_time():
    """Use matplotlib to plot the result of attn time"""
    # Plotly plot
    fig = go.Figure()

    for tp in [1, 2, 4, 8]:
        for cp in [1, 2, 4, 8]:
            x = np.concatenate([
                np.arange(4, 1 * K, 4),
                np.arange(1 * K, 1 * M, 1 * K)
            ])
            x = np.sort(x)
            y = [tm.get_attn_time(x, tp, cp) for x in x]
            fig.add_trace(go.Scatter(x=x, y=y, mode='lines', name=f"tp={tp}, cp={cp}"))

    fig.update_xaxes(type="log")
    fig.update_yaxes(type="log")
    # add the toggle buttons
    fig.update_layout(
        updatemenus=[
            dict(
                type="buttons",
                direction="right",
                x=1.0, y=1.15,
                showactive=True,
                buttons=[
                    dict(
                        label="Log-Log",
                        method="relayout",
                        args=[{"xaxis.type": "log", "yaxis.type": "log"}]
                    ),
                    dict(
                        label="Linear",
                        method="relayout",
                        args=[{"xaxis.type": "linear", "yaxis.type": "linear"}]
                    ),
                ],
            )
        ]
    )

    # add title
    fig.update_layout(title="Attention Time (ms)")
    # add x-axis and y-axis labels
    fig.update_xaxes(title="Sequence Length (tokens)")
    fig.update_yaxes(title="Time (ms)")

    fig.write_html(plt_dir / "attn_time.html")
    fig.write_image(plt_dir / "attn_time.png")

    plt.close()


def test_plot_get_mlp_time():
    """Use matplotlib to plot the result of mlp time"""
    fig = go.Figure()
    
    for tp in [1, 2, 4, 8]:
        for cp in [1, 2, 4, 8]:
            x = np.concatenate([
                np.arange(4, 1 * K, 4),
                np.arange(1 * K, 1 * M, 1 * K)
            ])
            x = np.sort(x)
            y = [tm.get_mlp_time(x, tp, cp) for x in x]
            fig.add_trace(go.Scatter(x=x, y=y, mode='lines', name=f"tp={tp}, cp={cp}"))

    fig.update_xaxes(type="log")
    fig.update_yaxes(type="log")

    # add the toggle buttons
    fig.update_layout(
        updatemenus=[
            dict(
                type="buttons",
                direction="right",
                x=1.0, y=1.15,
                showactive=True,
                buttons=[
                    dict(
                        label="Log-Log",
                        method="relayout",
                        args=[{"xaxis.type": "log", "yaxis.type": "log"}]
                    ),
                    dict(
                        label="Linear",
                        method="relayout",
                        args=[{"xaxis.type": "linear", "yaxis.type": "linear"}]
                    ),
                ],
            )
        ]
    )

    # add title
    fig.update_layout(title="MLP Time (ms)")
    # add x-axis and y-axis labels
    fig.update_xaxes(title="Sequence Length (tokens)")
    fig.update_yaxes(title="Time (ms)")

    fig.write_html(plt_dir / "mlp_time.html")
    fig.write_image(plt_dir / "mlp_time.png")

    plt.close()
    

def test_mlp_attn_ratio():
    # Plotly plot
    fig = go.Figure()
    # Create a 4x4 subplot
    fig = sp.make_subplots(
        rows=4, cols=4, 
        subplot_titles=[f"tp={tp}, cp={cp}" for tp in [1, 2, 4, 8] for cp in [1, 2, 4, 8]],
        x_title="Sequence Length (tokens)", 
        y_title="Time (ms)",
        specs=[[{"secondary_y": True} for _ in range(4)] for _ in range(4)],  # Enable secondary y-axis
        horizontal_spacing=0.05,
        vertical_spacing=0.05,
    )

    for i, tp in enumerate([1, 2, 4, 8], start=1):
        for j, cp in enumerate([1, 2, 4, 8], start=1):
            x = np.concatenate([
                np.arange(4, 1 * K, 4),
                np.arange(1 * K, 128 * K, 1 * K)
            ])
            x = np.sort(x)
            y_mlp = [tm.get_mlp_time(x, tp, cp) for x in x]
            y_attn = [tm.get_attn_time(x, tp, cp) for x in x]
            y_ratio = [a / m if m != 0 else 0 for a, m in zip(y_attn, y_mlp)]

            fig.add_trace(go.Scatter(x=x, y=y_mlp, mode='lines', name=f"MLP (tp={tp}, cp={cp})", line=dict(color='blue')), row=i, col=j, secondary_y=False)
            fig.add_trace(go.Scatter(x=x, y=y_attn, mode='lines', name=f"Attn (tp={tp}, cp={cp})", line=dict(color='red')), row=i, col=j, secondary_y=False)
            fig.add_trace(go.Scatter(x=x, y=y_ratio, mode='lines', name=f"Ratio (tp={tp}, cp={cp})", line=dict(color='green')), row=i, col=j, secondary_y=True)

            # Draw horizontal line at ratio == 1
            fig.add_hline(y=1, line_dash="dash", line_color="gray", row=i, col=j, secondary_y=True)

            # Draw vertical lines where ratio == 1
            for idx, ratio in enumerate(y_ratio):
                if ratio == 1:
                    fig.add_vline(x=x[idx], line_dash="dash", line_color="gray", row=i, col=j, secondary_y=True)

    fig.update_layout(
        height=1200, 
        width=1200, 
        title_text="MLP and Attn Time with Ratio",
        title="MLP vs Attn Time and Ratio (Batch Size = 1)",
        margin=dict(l=20, r=20, t=50, b=0),  # Reduce margins to make padding smaller
    )

    # Configure x-axis and y-axis labels visibility
    for i in range(1, 5):  # Assuming 4 rows and 4 columns
        for j in range(1, 5):
            show_x = False
            show_y = False
            show_y2 = False
            # show_x = (i == 4)  # Show x-axis only if it's the last row
            # show_y = (j == 1)  # Show y-axis only if it's the first column
            # show_y2 = (j == 4)
            fig.update_xaxes(title="Sequence Length (tokens)" if show_x else None, row=i, col=j)
            fig.update_yaxes(title="Time (ms)" if show_y else None, row=i, col=j, secondary_y=False)
            fig.update_yaxes(title="Ratio" if show_y2 else None, range=[0, 2], row=i, col=j, secondary_y=True)

    # save as html and image
    fig.write_html(plt_dir / "mva_bs1.html")
    fig.write_image(plt_dir / "mva_bs1.png")


def test_mlp_attn_ratio_rand():
    # Plotly plot
    fig = go.Figure()
    # Create a 4x4 subplot
    fig = sp.make_subplots(
        rows=4, cols=4, 
        subplot_titles=[f"tp={tp}, cp={cp}" for tp in [1, 2, 4, 8] for cp in [1, 2, 4, 8]],
        x_title="Sequence Length (tokens)", 
        y_title="Time (ms)",
        specs=[[{"secondary_y": True} for _ in range(4)] for _ in range(4)],  # Enable secondary y-axis
        horizontal_spacing=0.05,
        vertical_spacing=0.05,
    )

    from d2.simulator.optimizers.samples import (
        sample_wlbllm_docs, batch_documents
    )
    wlbllm_docs = sample_wlbllm_docs(size=10000)
    wlbllm_docs = batch_documents(wlbllm_docs, max_ctx_length=128 * K)
    wlbllm_docs = list(wlbllm_docs)

    x = [i for i in range(len(wlbllm_docs))]
    
    for i, tp in enumerate([1, 2, 4, 8], start=1):
        for j, cp in enumerate([1, 2, 4, 8], start=1):

            y_mlp = [
                tm.get_mlp_time(sum(batch), tp, cp)
                for batch in wlbllm_docs
            ]
            y_attn = [
                sum([tm.get_attn_time(doc, tp, cp) for doc in batch])
                for batch in wlbllm_docs
            ]
            y_ratio = [
                a / m if m != 0 else 0
                for a, m in zip(y_attn, y_mlp)
            ]

            fig.add_trace(go.Scatter(x=x, y=y_mlp, mode='lines', name=f"MLP (tp={tp}, cp={cp})", line=dict(color='blue')), row=i, col=j, secondary_y=False)
            fig.add_trace(go.Scatter(x=x, y=y_attn, mode='lines', name=f"Attn (tp={tp}, cp={cp})", line=dict(color='red')), row=i, col=j, secondary_y=False)
            fig.add_trace(go.Scatter(x=x, y=y_ratio, mode='lines', name=f"Ratio (tp={tp}, cp={cp})", line=dict(color='green')), row=i, col=j, secondary_y=True)

            # Draw horizontal line at ratio == 1
            fig.add_hline(y=1, line_dash="dash", line_color="gray", row=i, col=j, secondary_y=True)

            # Draw vertical lines where ratio == 1
            for idx, ratio in enumerate(y_ratio):
                if ratio == 1:
                    fig.add_vline(x=x[idx], line_dash="dash", line_color="gray", row=i, col=j, secondary_y=True)
            pass
    
    fig.update_layout(
        height=1200, 
        width=1500, 
        title_text="MLP and Attn Time with Ratio",
        title="MLP vs Attn Time and Ratio (WLBLLM, 64K tokens per batch)",
        margin=dict(l=20, r=20, t=50, b=0),  # Reduce margins to make padding smaller
    )

    # Configure x-axis and y-axis labels visibility
    for i in range(1, 5):  # Assuming 4 rows and 4 columns
        for j in range(1, 5):
            show_x = False
            show_y = False
            show_y2 = False
            # show_x = (i == 4)  # Show x-axis only if it's the last row
            # show_y = (j == 1)  # Show y-axis only if it's the first column
            # show_y2 = (j == 4)
            fig.update_xaxes(title="Sequence Length (tokens)" if show_x else None, row=i, col=j)
            fig.update_yaxes(title="Time (ms)" if show_y else None, row=i, col=j, secondary_y=False)
            fig.update_yaxes(title="Ratio" if show_y2 else None, range=[0, 2], row=i, col=j, secondary_y=True)

    # INSERT_YOUR_CODE
    # Add buttons to toggle visibility of MLP, Attn, and Ratio lines
    fig.update_layout(
        updatemenus=[
            dict(
                type="buttons",
                direction="left",
                buttons=list([
                    dict(
                        args=[{"visible": [True, False, False] * 4}],  # Show only MLP lines
                        label="Show MLP",
                        method="update"
                    ),
                    dict(
                        args=[{"visible": [False, True, False] * 4}],  # Show only Attn lines
                        label="Show Attn",
                        method="update"
                    ),
                    dict(
                        args=[{"visible": [False, False, True] * 4}],  # Show only Ratio lines
                        label="Show Ratio",
                        method="update"
                    ),
                    dict(
                        args=[{"visible": [True, True, True] * 4}],  # Show all lines
                        label="Show All",
                        method="update"
                    )
                ]),
                pad={"r": 10, "t": 10},
                showactive=True,
                x=0.8,
                xanchor="left",
                y=1.1,
                yanchor="top"
            ),
        ]
    )

    # save as html and image
    fig.write_html(plt_dir / "mva_rand.html")
    fig.write_image(plt_dir / "mva_rand.png")

    
def test_mlp_attn_ratio_rand_cdf():
    # Plotly plot
    fig = go.Figure()
    # Create a 4x4 subplot
    fig = sp.make_subplots(
        rows=4, cols=4, 
        subplot_titles=[f"tp={tp}, cp={cp}" for tp in [1, 2, 4, 8] for cp in [1, 2, 4, 8]],
        x_title="Value", 
        y_title="CDF",
        specs=[[{"secondary_y": False} for _ in range(4)] for _ in range(4)],  # No secondary y-axis needed for CDF
        horizontal_spacing=0.05,
        vertical_spacing=0.05,
    )

    from d2.simulator.optimizers.samples import (
        sample_wlbllm_docs, batch_documents
    )
    wlbllm_docs = sample_wlbllm_docs(size=10000)
    wlbllm_docs = batch_documents(wlbllm_docs, max_ctx_length=128 * K)
    wlbllm_docs = list(wlbllm_docs)

    for i, tp in enumerate([1, 2, 4, 8], start=1):
        for j, cp in enumerate([1, 2, 4, 8], start=1):

            y_mlp = [
                tm.get_mlp_time(sum(batch), tp, cp)
                for batch in wlbllm_docs
            ]
            y_attn = [
                sum([tm.get_attn_time(doc, tp, cp) for doc in batch])
                for batch in wlbllm_docs
            ]
            y_ratio = [
                a / m if m != 0 else 0
                for a, m in zip(y_attn, y_mlp)
            ]

            # Calculate CDFs
            y_mlp_sorted = np.sort(y_mlp)
            y_attn_sorted = np.sort(y_attn)
            y_ratio_sorted = np.sort(y_ratio)

            cdf_mlp = np.arange(1, len(y_mlp_sorted) + 1) / len(y_mlp_sorted)
            cdf_attn = np.arange(1, len(y_attn_sorted) + 1) / len(y_attn_sorted)
            cdf_ratio = np.arange(1, len(y_ratio_sorted) + 1) / len(y_ratio_sorted)

            fig.add_trace(go.Scatter(x=y_mlp_sorted, y=cdf_mlp, mode='lines', name=f"MLP CDF (tp={tp}, cp={cp})", line=dict(color='blue')), row=i, col=j)
            fig.add_trace(go.Scatter(x=y_attn_sorted, y=cdf_attn, mode='lines', name=f"Attn CDF (tp={tp}, cp={cp})", line=dict(color='red')), row=i, col=j)
            fig.add_trace(go.Scatter(x=y_ratio_sorted, y=cdf_ratio, mode='lines', name=f"Ratio CDF (tp={tp}, cp={cp})", line=dict(color='green')), row=i, col=j)

    fig.update_layout(
        height=1200, 
        width=1500, 
        title_text="CDF of MLP, Attn, and Ratio",
        title="CDF of MLP vs Attn Time and Ratio (WLBLLM, 64K tokens per batch)",
        margin=dict(l=20, r=20, t=50, b=0),  # Reduce margins to make padding smaller
    )

    # Add buttons to toggle visibility of MLP, Attn, and Ratio traces
    fig.update_layout(
        updatemenus=[
            dict(
                type="buttons",
                direction="right",
                buttons=list([
                    dict(
                        args=[{"visible": [False, False, True] * 16}],  # Only show Ratio traces
                        label="Show Ratio",
                        method="update"
                    ),
                    dict(
                        args=[{"visible": [True, False, False] * 16}],  # Only show MLP traces
                        label="Show MLP",
                        method="update"
                    ),
                    dict(
                        args=[{"visible": [False, True, False] * 16}],  # Only show Attn traces
                        label="Show Attn",
                        method="update"
                    ),
                    dict(
                        args=[{"visible": [True, True, True] * 16}],  # Show all traces
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
        ]
    )

    # Add a vertical line at x=1 when showing Ratio
    for i in range(1, 5):
        for j in range(1, 5):
            fig.add_shape(
                type="line",
                x0=1, y0=0, x1=1, y1=1,
                line=dict(color="Gray", width=2, dash="dash"),
                row=i, col=j
            )

    # Set default visibility to show Ratio and the vertical line
    fig.update_traces(visible=False)
    for trace in fig.data:
        if "Ratio" in trace.name:
            trace.visible = True

    # save as html and image
    fig.write_html(plt_dir / "mva_rand_cdf.html")
    fig.write_image(plt_dir / "mva_rand_cdf.png")

    



if __name__ == "__main__":
    print("Running tests...")
    # print("test_get_attn_time_monotonically_increase")
    # test_get_attn_time_monotonically_increase()
    # print("test_plot_get_attn_time")
    # test_plot_get_attn_time()
    # print("test_plot_get_mlp_time")
    # test_plot_get_mlp_time()
    # print("test_mlp_attn_ratio")
    # test_mlp_attn_ratio()
    # print("test_mlp_attn_ratio_rand")
    # test_mlp_attn_ratio_rand()
    print("test_mlp_attn_ratio_rand_cdf")
    test_mlp_attn_ratio_rand_cdf()