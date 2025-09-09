
# %%
folder = "/mnt/weka/home/yonghao.zhuang/jd/d2/benchmarks/_250907_large_scale_v6/logs.v5-compare/0908_142924_PST/20250908_142925.job-703746.d2-cp1-n8-b4-t131072-normal/nsys-reps"
# folder = '/mnt/weka/home/yonghao.zhuang/jd/d2/benchmarks/_250907_large_scale_v6/logs.v5-compare/0908_142924_PST/20250908_143113.job-703746.d2-cp1-n8-b4-t131072-signal/nsys-reps'
# %%
import os
import tqdm

a = !ls $folder/*.nsys-rep
for file in tqdm.tqdm(a):
    nsys_file = os.path.join(folder, file)
    os.system(f"cd {folder} && nsys export --type sqlite {nsys_file}")
# %%
import os
import pandas as pd
import sqlite3
import tqdm

a = os.listdir(folder)
# %%
a 
# %%

dfs = []


for file in tqdm.tqdm(os.listdir(folder)):
    full_file_path = os.path.join(folder, file)
    if not file.endswith('.sqlite'):
        continue
    try:
        print(full_file_path)
        with sqlite3.connect(full_file_path) as conn:
            df = pd.read_sql_query("""
            WITH A AS (
                SELECT * 
                FROM CUPTI_ACTIVITY_KIND_KERNEL
                JOIN StringIds ON CUPTI_ACTIVITY_KIND_KERNEL.mangledName = StringIds.id
                JOIN (
                    SELECT busLocation, cuDevice FROM TARGET_INFO_GPU
                ) AS TARGET_INFO_GPU ON TARGET_INFO_GPU.cuDevice = deviceId
            )
            SELECT value as name, start, end, deviceId as gpuId, streamId, (
                CASE WHEN value like '%Lb1ELb0%' THEN 1 ELSE 0 END
            ) as is_send, busLocation
            FROM A
            WHERE name like '%alltoallv%'
            ;
            """, conn)
            df['start'] = df['start'] / 1e6 # ns -> ms
            df['end'] = df['end'] / 1e6 # ns -> ms
            df['duration'] = df['end'] - df['start']
            df['host_id'] = file.split('.')[0]
            df['deviceId'] = df['host_id'] + '-' + df['busLocation'].astype(str) + df['gpuId'].astype(str)
            # sort by start and add another column that is the "call_id" representing the index
            df['is_send'] = df['is_send'].astype(bool)
            df = df.sort_values(by=['start'])
            df['call_id'] = df.groupby('deviceId').cumcount()
            dfs.append(df)
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"Error when reading database {file}")
        continue


# %%
df = pd.concat(dfs, axis=0)
df
# %%
df.name.unique().tolist()
# %%
len(df)
# %%
import matplotlib.pyplot as plt

# Create a figure and axis
plt.figure(figsize=(15, 8))
# Filter by call_id range
is_send = [True, False]
should_align_start_time = True
call_ids = df['call_id'].max().item() + 1

for call_id in range(0, call_ids, 2):
    call_id_min = call_id
    call_id_max = call_id + 1
    filtered_df = df[(df['call_id'] >= call_id_min) & (df['call_id'] <= call_id_max)]

    # align start time of each deviceId. For each group (deviceId), take the smallest call_id's event
    if should_align_start_time:
        filtered_df = filtered_df.groupby('deviceId').apply(lambda x: x.sort_values(by=['start'])).reset_index(drop=True)

    filtered_df = filtered_df[
        filtered_df['is_send'].isin(is_send)
    ]
    filtered_df.sort_values(by=['deviceId'], inplace=True)

    # Get unique device IDs from filtered data
    device_ids = filtered_df['deviceId'].unique()

    # Plot timeline for each device
    for i, device_id in enumerate(device_ids):
        device_data = filtered_df[filtered_df['deviceId'] == device_id]
        
        # Plot bars for each start-end pair
        for _, row in device_data.iterrows():
            color = 'blue' if row['is_send'] else 'red'
            plt.hlines(y=i, xmin=row['start'], xmax=row['end'], 
                    linewidth=8, color=color, alpha=0.5)

    # Customize plot
    plt.yticks(range(len(device_ids)), [f'Device {d}' for d in device_ids])
    plt.xlabel('Time (ms)')
    plt.ylabel('Device ID')
    plt.title('Kernel Timeline by Device (Call IDs {}-{})'.format(call_id_min, call_id_max))
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


# %%
import plotly.graph_objects as go

# Create figure
fig = go.Figure()

# Filter by call_id range
is_send = [True, False]
call_id_min = 0
call_id_max = float('inf')
filtered_df = df[(df['call_id'] >= call_id_min) & (df['call_id'] <= call_id_max)]
filtered_df = filtered_df[filtered_df['is_send'].isin(is_send)]
filtered_df.sort_values(by=['deviceId'], inplace=True)

# Get unique device IDs from filtered data
device_ids = filtered_df['deviceId'].unique()

# Plot timeline for each device
for i, device_id in enumerate(device_ids):
    device_data = filtered_df[filtered_df['deviceId'] == device_id]
    
    # Plot bars for each start-end pair
    for _, row in device_data.iterrows():
        color = 'blue' if row['is_send'] else 'red'
        fig.add_trace(go.Scatter(
            x=[row['start'], row['end']],
            y=[i, i],
            mode='lines',
            line=dict(color=color, width=8),
            opacity=0.5,
            showlegend=False
        ))

# Customize plot
fig.update_layout(
    title=f'Kernel Timeline by Device (Call IDs {call_id_min}-{call_id_max})',
    xaxis_title='Time (ms)',
    yaxis_title='Device ID',
    yaxis=dict(
        ticktext=[f'Device {d}' for d in device_ids],
        tickvals=list(range(len(device_ids))),
    ),
    height=600,
    width=1200,
    showlegend=False,
    plot_bgcolor='rgba(0,0,0,0.05)'
)

fig.show()

# %%
df[
    df['host_id'].str.contains('130')
].head(40)

# %%
