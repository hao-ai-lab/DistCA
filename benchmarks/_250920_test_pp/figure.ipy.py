# %%
import pandas as pd

_prolong = """dataset	model_size	nodes	num_tokens	total_batch_size	speedup	dataset	average_duration_wlb	average_duration_d2	linear_speedup	line_id
prolong	34b	16	131072	4	1.615178758	prolong	16724.13301	10354.35424	32768	128
prolong	34b	32	131072	8	1.139922334	prolong	15667.64374	13744.48352	16384	128
prolong	34b	32	262144	4	1.560729905	prolong	24018.92223	15389.54444	65536	256
prolong	34b	32	524288	2	1.325928129	prolong	49193.84508	37101.44162	262144	512
prolong	8b	8	131072	8	1.476820846	prolong	18342.44924	12420.22638	16384	128
prolong	8b	16	131072	16	1.421374747	prolong	19506.52556	13723.70348	8192	128
prolong	8b	32	131072	16	1.629771636	prolong	12044.61212	7390.367985	8192	128
prolong	8b	8	262144	4	1.160499573	prolong	27629.45538	23808.24261	65536	256
prolong	8b	16	262144	8	1.431076499	prolong	27564.59336	19261.43947	32768	256
prolong	8b	32	262144	8	1.515814952	prolong	16271.43188	10734.44476	32768	256
prolong	8b	16	524288	8	0.9851165391	prolong	85526.58601	86818.74948	65536	512
prolong	8b	32	524288	8	1.39416343	prolong	46867.20905	33616.7253	65536	512"""


_wlbllm = """dataset	model_size	nodes	num_tokens	total_batch_size	speedup	dataset	average_duration_wlb	average_duration_d2	linear_speedup	line_id
wlbllm	34b	16	131072	4	1.241475335	wlbllm	12930.15885	10415.15565	32768	128
wlbllm	34b	32	131072	8	1.057584996	wlbllm	14834.91735	14027.16322	16384	128
wlbllm	34b	32	262144	4	1.420188492	wlbllm	21890.23023	15413.60908	65536	256
wlbllm	8b	8	131072	8	1.140968644	wlbllm	14041.25147	12306.43063	16384	128
wlbllm	8b	16	131072	16	1.217272956	wlbllm	15946.84467	13100.4674	8192	128
wlbllm	8b	32	131072	16	1.421914704	wlbllm	10442.96007	7344.29431	8192	128
wlbllm	8b	8	262144	4	0.9685579358	wlbllm	23451.83034	24213.14149	65536	256
wlbllm	8b	16	262144	8	1.19607699	wlbllm	22936.50713	19176.44711	32768	256
wlbllm	8b	32	262144	8	1.283353258	wlbllm	13794.13414	10748.50907	32768	256
wlbllm	8b	32	524288	8	1.230581383	wlbllm	41414.92987	33654.76713	65536	512"""


"""dataset	model_size	nodes	num_tokens	total_batch_size	speedup	dataset	average_duration_wlb	average_duration_d2	linear_speedup	line_id

"""
from io import StringIO
prolong = pd.read_csv(StringIO(_prolong), sep="\t")
wlbllm = pd.read_csv(StringIO(_wlbllm), sep="\t")

# %%
prolong
# %%
wlbllm
# %%

comb = [
    (prolong, "prolong"),
    (wlbllm, "wlbllm"),
]

for merged_wlb_vs_d2, dataset_name in comb:
    # Plot speedup vs nodes for each num_tokens/batch_size combination
    import matplotlib.pyplot as plt

    # Get unique model sizes
    model_sizes = merged_wlb_vs_d2['model_size'].unique()

    # Create subplot for each model size
    fig, axes = plt.subplots(1, len(model_sizes), figsize=(5*len(model_sizes), 4))
    if len(model_sizes) == 1:
        axes = [axes]

    # Plot data for each model size
    for idx, model_size in enumerate(sorted(model_sizes)):
        ax = axes[idx]
        
        # Get data for this model size
        model_data = merged_wlb_vs_d2[merged_wlb_vs_d2['model_size'] == model_size]
        line_ids = model_data['line_id'].unique()
        
        # Plot a line for each line_id
        for line_id in sorted(line_ids):
            data = model_data[model_data['line_id'] == line_id]
            nodes_str = data['nodes'].astype(str)
            
            ax.plot(nodes_str, data['speedup'],
                    marker='o', markersize=8,
                    label=f'NTokens {line_id:.0f}k',
                    linewidth=2)
        
        # Set labels and title
        ax.set_xlabel('Number of Nodes')
        ax.set_ylabel('Speedup (WLBLLM/D2)')
        ax.set_title(f'Dataset: {dataset_name}, Model Size: {model_size}')
        
        # Add legend and grid
        ax.legend()
        ax.grid(True, which="both", ls="-", alpha=0.2)

    # Adjust layout
    plt.tight_layout()
    plt.show()

