# %%
# name = "attn-dist_wlbllm-64k"

import sys
import subprocess

args = sys.argv[1:]
args = args or ["attn-dist_wlbllm-64k", "attn-dist_multimodal-64k"]

# If multiple names are provided, recursively call the script
if len(args) > 1:
    for name in args:
        subprocess.run([sys.executable, __file__, name])
    sys.exit(0)

# Else use the single provided name, or default
name = args[0]

print(f"Processing: {name}")

# %%
import pandas as pd
filename = f"{name}.psv"
# filename = f"attn-dist_multimodal-64k.psv"
df = pd.read_csv(f"{filename}", sep="|")
df.head()

# %%
size = len(df['batch'].unique())
max_ctx_length = df['total_len'].max().item()
size, max_ctx_length

# %%

# %%
# Use plotly, for each (tp, cp), plot a scatter plot of (x = real_attn_time_ms, y = sim_attn_time_ms)
import plotly.express as px
import plotly.graph_objs as go

# Create a figure
fig = go.Figure()

# Iterate through unique (tp, cp) combinations
for (tp, cp) in df[['tp', 'cp']].drop_duplicates().itertuples(index=False):
    subset = df[(df['tp'] == tp) & (df['cp'] == cp)]
    
    # Add scatter trace for each (tp, cp) combination
    fig.add_trace(go.Scatter(
        x=subset['real_attn_time_ms'], 
        # y=subset['real_attn_time_ms'] - subset['sim_attn_time_ms'],
        y=subset['sim_attn_time_ms'],
        mode='markers',
        name=f'tp={tp}, cp={cp}',
        hovertemplate='Real Time: %{x:.2f} ms<br>Sim Time: %{y:.2f} ms<br>Total Len: %{customdata[0]}<br>Batch Size: %{customdata[1]}<extra></extra>',
        customdata=subset[['total_len', 'bs']].values
    ))

# Customize the layout
fig.update_layout(
    title=(
        f'Real vs Simulated Attention Time - {name} Dataset<br>'
        f'Dataset: {name}, Sample Size: {size}, Max Context Length: {max_ctx_length}'
    ),
    xaxis_title='Real Attention Time (ms)',
    yaxis_title='Simulated Attention Time (ms)',
    legend_title='TP and CP'
)

# Add a diagonal line for perfect prediction
min_val = df['real_attn_time_ms'].min()
max_val = df['real_attn_time_ms'].max()
fig.add_trace(go.Scatter(
    x=[min_val, max_val], 
    y=[min_val, max_val],
    mode='lines',
    name='Perfect Prediction',
    line=dict(color='red', dash='dash')
))

# Save the plot as an interactive HTML file
fig.write_html(f"{filename}.scatter.html")
# Save the plot as a PNG file
fig.write_image(f"{filename}.scatter.png")


# Show the plot
fig.show()



# %%
# Create a figure for the difference between real and simulated attention times
fig_diff = go.Figure()

# Get unique tp and cp values to determine color scales
tp_values = sorted(df['tp'].unique())
cp_values = sorted(df['cp'].unique())

# Iterate through unique (tp, cp) combinations
for (tp, cp) in df[['tp', 'cp']].drop_duplicates().itertuples(index=False):
    subset = df[(df['tp'] == tp) & (df['cp'] == cp)]
    
    # Calculate color based on tp (grey to red)
    tp_color_val = (tp_values.index(tp) / (len(tp_values) - 1)) if len(tp_values) > 1 else 0
    tp_color = f'rgb({int(255 * tp_color_val)}, {int(100 * (1 - tp_color_val))}, {int(100 * (1 - tp_color_val))})'
    
    # Calculate color based on cp (grey to green)
    cp_color_val = (cp_values.index(cp) / (len(cp_values) - 1)) if len(cp_values) > 1 else 0
    cp_color = f'rgb({int(100 * (1 - cp_color_val))}, {int(255 * cp_color_val)}, {int(100 * (1 - cp_color_val))})'
    
    # Blend the colors
    blended_color = f'rgb({(int(tp_color.split("(")[1].split(",")[0]) + int(cp_color.split("(")[1].split(",")[0]))//2}, ' \
                    f'{(int(tp_color.split(",")[1]) + int(cp_color.split(",")[1]))//2}, ' \
                    f'{(int(tp_color.split(",")[2].split(")")[0]) + int(cp_color.split(",")[2].split(")")[0]))//2})'
    
    # Add scatter trace for each (tp, cp) combination
    # Now plotting the difference between real and simulated attention times
    fig_diff.add_trace(go.Scatter(
        x=subset['real_attn_time_ms'], 
        y=subset['real_attn_time_ms'] - subset['sim_attn_time_ms'],
        mode='markers',
        name=f'tp={tp}, cp={cp}',
        marker=dict(color=blended_color),
        hovertemplate='Real Time: %{x:.2f} ms<br>Difference: %{y:.2f} ms<br>Total Len: %{customdata[0]}<br>Batch Size: %{customdata[1]}<extra></extra>',
        customdata=subset[['total_len', 'bs']].values
    ))

# Customize the layout
fig_diff.update_layout(
    title=(
        f'Difference Between Real and Simulated Attention Time - {name} Dataset<br>'
        f'Dataset: {name}, Size: {size}, Max Context Length: {max_ctx_length}'
    ),
    xaxis_title='Real Attention Time (ms)',
    yaxis_title='Real - Simulated Attention Time (ms)',
    legend_title='TP and CP'
)

# Save the plot as an interactive HTML file
fig_diff.write_html(f"{filename}.diff_scatter.html")
# Save the plot as a PNG file
fig_diff.write_image(f"{filename}.diff_scatter.png")

# Show the plot
fig_diff.show()



# %%
# Create a new figure for batch size vs attention time difference
fig_bs_diff = go.Figure()

# Iterate through unique (tp, cp) combinations
for (tp, cp) in df[['tp', 'cp']].drop_duplicates().itertuples(index=False):
    subset = df[(df['tp'] == tp) & (df['cp'] == cp)]
    
    # Calculate color based on tp (grey to red)
    tp_color_val = (tp_values.index(tp) / (len(tp_values) - 1)) if len(tp_values) > 1 else 0
    tp_color = f'rgb({int(255 * tp_color_val)}, {int(100 * (1 - tp_color_val))}, {int(100 * (1 - tp_color_val))})'
    
    # Calculate color based on cp (grey to green)
    cp_color_val = (cp_values.index(cp) / (len(cp_values) - 1)) if len(cp_values) > 1 else 0
    cp_color = f'rgb({int(100 * (1 - cp_color_val))}, {int(255 * cp_color_val)}, {int(100 * (1 - cp_color_val))})'
    
    # Blend the colors
    blended_color = f'rgb({(int(tp_color.split("(")[1].split(",")[0]) + int(cp_color.split("(")[1].split(",")[0]))//2}, ' \
                    f'{(int(tp_color.split(",")[1]) + int(cp_color.split(",")[1]))//2}, ' \
                    f'{(int(tp_color.split(",")[2].split(")")[0]) + int(cp_color.split(",")[2].split(")")[0]))//2})'
    
    # Add scatter trace for each (tp, cp) combination
    # Plotting the difference between real and simulated attention times vs batch size
    fig_bs_diff.add_trace(go.Scatter(
        x=subset['bs'], 
        y=subset['real_attn_time_ms'] - subset['sim_attn_time_ms'],
        mode='markers',
        name=f'tp={tp}, cp={cp}',
        marker=dict(color=blended_color),
        hovertemplate='Batch Size: %{x}<br>Difference: %{y:.2f} ms<br>Total Len: %{customdata[0]}<br>Real Time: %{customdata[1]:.2f} ms<extra></extra>',
        customdata=subset[['total_len', 'real_attn_time_ms']].values
    ))

# Customize the layout
fig_bs_diff.update_layout(
    title=(
        f'Difference Between Real and Simulated Attention Time vs Batch Size - {name} Dataset<br>'
        f'Dataset: {name}, Size: {size}, Max Context Length: {max_ctx_length}'
    ),
    xaxis_title='Batch Size',
    yaxis_title='Real - Simulated Attention Time (ms)',
    legend_title='TP and CP'
)

# Save the plot as an interactive HTML file
fig_bs_diff.write_html(f"{filename}.bs_diff_scatter.html")
# Save the plot as a PNG file
fig_bs_diff.write_image(f"{filename}.bs_diff_scatter.png")

# Show the plot
fig_bs_diff.show()

# %%
# Create a new figure for relative error
fig_bs_rel_error = go.Figure()

# Iterate through unique (tp, cp) combinations
for (tp, cp) in df[['tp', 'cp']].drop_duplicates().itertuples(index=False):
    subset = df[(df['tp'] == tp) & (df['cp'] == cp)]
    
    # Calculate color based on tp (grey to red)
    tp_color_val = (tp_values.index(tp) / (len(tp_values) - 1)) if len(tp_values) > 1 else 0
    tp_color = f'rgb({int(255 * tp_color_val)}, {int(100 * (1 - tp_color_val))}, {int(100 * (1 - tp_color_val))})'
    
    # Calculate color based on cp (grey to green)
    cp_color_val = (cp_values.index(cp) / (len(cp_values) - 1)) if len(cp_values) > 1 else 0
    cp_color = f'rgb({int(100 * (1 - cp_color_val))}, {int(255 * cp_color_val)}, {int(100 * (1 - cp_color_val))})'
    
    # Blend the colors
    blended_color = f'rgb({(int(tp_color.split("(")[1].split(",")[0]) + int(cp_color.split("(")[1].split(",")[0]))//2}, ' \
                    f'{(int(tp_color.split(",")[1]) + int(cp_color.split(",")[1]))//2}, ' \
                    f'{(int(tp_color.split(",")[2].split(")")[0]) + int(cp_color.split(",")[2].split(")")[0]))//2})'
    
    # Calculate relative error
    subset['relative_error'] = (subset['real_attn_time_ms'] - subset['sim_attn_time_ms']) / subset['real_attn_time_ms']
    
    # Add scatter trace for each (tp, cp) combination
    # Plotting the relative error vs batch size
    fig_bs_rel_error.add_trace(go.Scatter(
        x=subset['bs'], 
        y=subset['relative_error'],
        mode='markers',
        name=f'tp={tp}, cp={cp}',
        marker=dict(color=blended_color),
        hovertemplate='Batch Size: %{x}<br>Relative Error: %{y:.2%}<br>Total Len: %{customdata[0]}<br>Real Time: %{customdata[1]:.2f} ms<br>Sim Time: %{customdata[2]:.2f} ms<extra></extra>',
        customdata=subset[['total_len', 'real_attn_time_ms', 'sim_attn_time_ms']].values
    ))

# Customize the layout
fig_bs_rel_error.update_layout(
    title=(
        f'Relative Error of Simulated Attention Time vs Batch Size - {name} Dataset<br>'
        f'Dataset: {name}, Size: {size}, Max Context Length: {max_ctx_length}'
    ),
    xaxis_title='Batch Size',
    yaxis_title='Relative Error (Real - Sim) / Real',
    yaxis_tickformat='.1%',
    legend_title='TP and CP'
)

# Save the plot as an interactive HTML file
fig_bs_rel_error.write_html(f"{filename}.bs_rel_error_scatter.html")
# Save the plot as a PNG file
fig_bs_rel_error.write_image(f"{filename}.bs_rel_error_scatter.png")

# Show the plot
fig_bs_rel_error.show()

# %%
# Create a subplot for real and simulated attention time distribution
from plotly.subplots import make_subplots
fig_time_dist = make_subplots(
    rows=2, cols=1, 
    subplot_titles=('Real Attention Time Distribution', 'Simulated Attention Time Distribution'),
    vertical_spacing=0.1
)

# Flatten the data across all (tp, cp) combinations
real_times = []
sim_times = []

for tp in tp_values:
    for cp in cp_values:
        subset = df[(df['tp'] == tp) & (df['cp'] == cp)]
        real_times.extend(subset['real_attn_time_ms'])
        sim_times.extend(subset['sim_attn_time_ms'])

# Add histogram for real attention times
fig_time_dist.add_trace(
    go.Histogram(
        x=real_times, 
        name='Real Attention Time',
        marker_color='blue',
        opacity=0.7
    ),
    row=1, col=1
)

# Add histogram for simulated attention times
fig_time_dist.add_trace(
    go.Histogram(
        x=sim_times, 
        name='Simulated Attention Time',
        marker_color='red',
        opacity=0.7
    ),
    row=2, col=1
)

# Update layout
fig_time_dist.update_layout(
    title=(
        f'Distribution of Real vs Simulated Attention Times - {name} Dataset<br>'
        f'Dataset: {name}, Size: {size}, Max Context Length: {max_ctx_length}'
    ),
    height=600,
    showlegend=True
)

# Update x-axis labels
fig_time_dist.update_xaxes(title_text='Time (ms)', row=1, col=1)
fig_time_dist.update_xaxes(title_text='Time (ms)', row=2, col=1)

# Update y-axis labels
fig_time_dist.update_yaxes(title_text='Frequency', row=1, col=1)
fig_time_dist.update_yaxes(title_text='Frequency', row=2, col=1)

# Save the plot as an interactive HTML file
fig_time_dist.write_html(f"{filename}.time_distribution.html")

# Save the plot as a PNG file
fig_time_dist.write_image(f"{filename}.time_distribution.png")

# Show the plot
fig_time_dist.show()

# %%
