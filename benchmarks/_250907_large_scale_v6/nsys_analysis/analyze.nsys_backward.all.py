# %%
import traceback
import time
import os
import sqlite3
import pandas as pd
import analyze_nsys_backward
import tqdm

# %%
input_dir = '/mnt/weka/home/yonghao.zhuang/jd/d2/benchmarks/_250907_large_scale_v6/logs.v-sweep/20250907_205136.job-703746.d2-cp1-n32-b4-t524288/nsys-reps-sqlite'

database_files = os.listdir(input_dir)
database_files

pbar = tqdm.tqdm(database_files)
for i in range(len(database_files)):
    pbar.update(1)
    name = database_files[i]
    DB = os.path.join(input_dir, name)
    dfs = []
    for gpu_id in range(0, 7):
        conn = sqlite3.connect(DB)
        thread_name = f"pt_autograd_{gpu_id}"
        try:
            df = analyze_nsys_backward.get_all_to_all_kernel_df(
                conn, DB, 
                TARGET_THREAD_NAME = thread_name, 
                GPU_ID = gpu_id, 
                OS_TID = None, OS_PID = None, 
            )
            df['thread_name'] = thread_name
            df['host_id'] = i
            df['host_name'] = name
            df['call_id'] = list(range(len(df)))
            dfs.append(df)
        except Exception as e:
            traceback.print_exc()
            print(f"Error when reading database {name} @ {gpu_id = }")
            time.sleep(1)
            continue

# %%
for df in dfs:
    print(len(df))

# %%





# %%
import pandas as pd

cat_dfs = pd.concat(dfs, axis=0)

# %%
def bring_to_front(columns):
    rest = [c for c in cat_dfs.columns if c not in columns]
    return cat_dfs[columns + rest]

cat_dfs = bring_to_front(['thread_name', 'k_duration_ms',  'k_start', 'k_end', 'is_send'])

cat_dfs = cat_dfs.sort_values(by=['k_start'])
# %%
cat_dfs.to_csv("analyzed-nsys-backward-all.csv", index=False)

# %%
from IPython.display import display
with pd.option_context('display.max_rows', None, 'display.max_columns', None):
    display(cat_dfs)


# %%
with open("analyzed-nsys-backward-all.csv") as f:
    pass

# %%
cat_dfs.head(10)
# %%
short_cat_df = cat_dfs[
    [
        'call_id', 'k_start', 'k_end', 'k_duration_ms', 
        'api_start', 'api_end', 'is_send', 'host_id', 'host_name', 'thread_name'
    ]
]
# %%
short_cat_df.head(10)
# %%
short_cat_df = short_cat_df.sort_values(by=['call_id', 'k_end'])
# %%
short_cat_df


# %%
short_cat_df[
    (short_cat_df['is_send'] == False)
]
# %%
# Get stats for receive operations grouped by call_id
recv_stats = short_cat_df[short_cat_df['is_send'] == False].groupby('call_id').agg({
    'k_start': ['min', 'max'],
    'k_end': ['min', 'max'], 
    'k_duration_ms': ['min', 'max']
})

# Make the column names more readable
recv_stats.columns = [f'{col[0]}_{col[1]}' for col in recv_stats.columns]

print("\nStats for receive operations by call_id:")
print(recv_stats)

# %%
# Plot receive operations timeline for a specific call_id
import matplotlib.pyplot as plt

def plot_recv_timeline(df, call_id):
    # Filter data for the given call_id
    call_data = df[
        (df['is_send'] == False) & 
        (df['call_id'] == call_id)
    ].sort_values('k_start')
    
    if len(call_data) == 0:
        print(f"No receive operations found for call_id {call_id}")
        return
        
    # Create figure and axis
    fig, ax = plt.subplots(figsize=(15, 5))
    
    # Plot horizontal bars
    y_positions = range(len(call_data))
    
    # Convert timestamps to milliseconds for better readability
    start_times = call_data['k_start'].values / 1e6  # Convert to ms
    end_times = call_data['k_end'].values / 1e6      # Convert to ms
    durations = end_times - start_times
    
    # Plot bars
    bars = ax.barh(y_positions, durations, left=start_times, height=0.3)
    
    # Customize the plot
    ax.set_title(f'Timeline of Receive Operations for call_id {call_id}')
    ax.set_xlabel('Time (ms)')
    ax.set_ylabel('Operation Index')
    
    # Add host/thread info as y-tick labels
    labels = [f"Host {row['host_id']}\n{row['thread_name']}" 
             for _, row in call_data.iterrows()]
    ax.set_yticks(y_positions)
    ax.set_yticklabels(labels)
    
    # Add duration annotations
    for i, (start, duration) in enumerate(zip(start_times, durations)):
        if duration > 0.1:  # Only show duration if bar is wide enough
            ax.text(start + duration/2, i, f'{duration:.2f}ms', 
                   ha='center', va='center')
    
    plt.tight_layout()
    plt.show()

# Example usage - plot for call_id 1
plot_recv_timeline(short_cat_df, 1)

# %%
