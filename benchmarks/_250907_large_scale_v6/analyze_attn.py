# %%

folder = "/mnt/weka/home/yonghao.zhuang/jd/d2/benchmarks/_250907_large_scale_v6/logs.v5-compare/0908_142924_PST/20250908_142925.job-703746.d2-cp1-n8-b4-t131072-normal/nsys-reps"

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
dfs = []
for file in tqdm.tqdm(os.listdir(folder)):
    full_file_path = os.path.join(folder, file)
    if not file.endswith('.sqlite'):
        continue
    try:
        print(full_file_path)
        with sqlite3.connect(full_file_path) as conn:
            pattern = '%flash%'
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
                CASE WHEN value like '%fwd%' THEN 0 ELSE 1 END
            ) as is_bwd, busLocation
            FROM A
            WHERE name like :pattern
            ;
            """, conn, params={'pattern': pattern})
            df['start'] = df['start'] / 1e6 # ns -> ms
            df['end'] = df['end'] / 1e6 # ns -> ms
            df['duration'] = df['end'] - df['start']
            df['host_id'] = file.split('.')[0]
            df['deviceId'] = df['host_id'] + '-' + df['busLocation'].astype(str) + df['gpuId'].astype(str)
            df['is_bwd'] = df['is_bwd'].astype(bool)
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
len(df)
# %%
df.name.unique().tolist()

# %%
df
# %%
import matplotlib.pyplot as plt

# Create a figure and axis
# Filter by call_id range
is_bwds = [True, False]
should_align_start_time = True
call_ids = df['call_id'].max().item() + 1

num_call_ids = 30
for call_id in range(0, call_ids, num_call_ids):
    plt.figure(figsize=(20, 10))
    call_id_min = call_id
    call_id_max = call_id + num_call_ids
    filtered_df = df[(df['call_id'] >= call_id_min) & (df['call_id'] <= call_id_max)]

    # align start time of each deviceId. For each group (deviceId), take the smallest call_id's event
    if should_align_start_time:
        filtered_df = filtered_df.groupby('deviceId').apply(lambda x: x.sort_values(by=['start'])).reset_index(drop=True)

    filtered_df = filtered_df[
        filtered_df['is_bwd'].isin(is_bwds)
    ]
    filtered_df.sort_values(by=['deviceId'], inplace=True)

    # Get unique device IDs from filtered data
    device_ids = filtered_df['deviceId'].unique()

    # Plot timeline for each device
    for i, device_id in enumerate(device_ids):
        device_data = filtered_df[filtered_df['deviceId'] == device_id]
        
        # Plot bars for each start-end pair
        for _, row in device_data.iterrows():
            color = 'blue' if row['is_bwd'] else 'red'
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
