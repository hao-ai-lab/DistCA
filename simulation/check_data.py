# %%
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import ipywidgets as widgets
from IPython.display import display

# %%
# Check the Attention Time Data


# %%
# Load the attention time data
attn_time = np.load("data/attn_time.npy")

# %%

def plot_heatmap_for_tp(attn_time, tp_index):
    """
    Plot heatmap for a specific tp index from the attn_time data.
    
    Parameters:
    - attn_time: 3D numpy array of shape (tp_degree_log, cp_degree_log, seq_len_log)
    - tp_index: Index for the tp_degree_log dimension
    """
    if tp_index < 0 or tp_index >= attn_time.shape[0]:
        raise ValueError(
            "tp_index is out of bounds for the attn_time array."
        )
    
    data = attn_time[tp_index]
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        data, annot=True, fmt=".2f", cmap="viridis", 
        cbar_kws={'label': 'Attention Time'}
    )
    plt.title(f"Heatmap for TP Index: {tp_index}")
    plt.xlabel("Seq Len Log")
    plt.ylabel("CP Degree Log")
    plt.show()

def interactive_heatmap(attn_time):
    """
    Create an interactive widget to select tp index and plot the corresponding heatmap.
    
    Parameters:
    - attn_time: 3D numpy array of shape (tp_degree_log, cp_degree_log, seq_len_log)
    """
    tp_slider = widgets.IntSlider(value=0, min=0, max=attn_time.shape[0] - 1, step=1, description='TP Log Index')
    widgets.interact(plot_heatmap_for_tp, attn_time=widgets.fixed(attn_time), tp_index=tp_slider)

# %%
interactive_heatmap(attn_time)

# %%
# Plot the MLP Time Data

# %%
# Load the MLP Time Data
mlp_time = np.load("data/mlp_time.npy")

# %%

# (tp_degree_log, num_token_log)    # 2D array

# %%

def plot_heatmap(mlp_time):
    plt.figure(figsize=(14, 8))
    sns.heatmap(mlp_time, annot=True, fmt=".2f", cmap="viridis", cbar_kws={'label': 'MLP Time'})
    plt.title("MLP Time Heatmap")
    plt.xlabel("Num Token Log")
    plt.ylabel("TP Degree Log")
    plt.show()


# %%
plot_heatmap(mlp_time)

# %%

mlp_time[:, 0] = mlp_time[:, 1] - 0.01


# %%
# np.save("data/mlp_time.npy", mlp_time)

# %%
import simulator
# %%

import pickle
import numpy as np
    
sim = simulator.Simulator(
    attn_time,
    mlp_time,
)
# %%
sim
# %%
sim.get_batch_attn_time(np.array([100]), 1, 2, do_sum=True)

# %%
sim.get_batch_mlp_time(np.array([128]), 1, 1, do_sum=True)

# %%
sim.mlp_time[0]