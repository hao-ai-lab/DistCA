# %%

import sqlite3
import pandas as pd
import numpy as np
from collections import defaultdict

pd.set_option("display.max_rows", 200)
pd.set_option("display.max_columns", 200)
pd.set_option("display.width", 180)

def get_all_to_all_kernel_df(
    conn, DB, 
    TARGET_THREAD_NAME = "pt_autograd_0", 
    GPU_ID = 0, 
    OS_TID = None, OS_PID = None, 
):
    conn = sqlite3.connect(DB)

    def bring_front(df, cols):
        rest = [c for c in df.columns if c not in cols]
        return df[cols + rest]

    def load_string_map(conn):
        sid = pd.read_sql_query("SELECT id, value FROM StringIds", conn)
        return dict(zip(sid.id, sid.value))

    def coalesce_name(text, textId, sid):
        return text if text is not None else sid.get(textId)

    def decode_pid(globalTid):
        return (int(globalTid) >> 24) & 0xFFFFFF

    def decode_tid(globalTid):
        return int(globalTid) & 0xFFFFFF

    def make_globalTid(pid, tid):
        return (int(pid) << 24) | (int(tid) & 0xFFFFFF)

    sid = load_string_map(conn)

    stream_name_to_device_info_mapping = pd.read_sql_query("""
    SELECT * , 
        globalTid / 0x1000000 % 0x1000000 AS PID, 
        globalTid % 0x1000000 AS TID
    FROM ThreadNames
    JOIN StringIds ON ThreadNames.nameId = StringIds.id
    JOIN TARGET_INFO_CUDA_CONTEXT_INFO ON PID = TARGET_INFO_CUDA_CONTEXT_INFO.processId
    ;
    """, conn)

    stream_name_to_device_info_mapping['value'].unique().tolist()

    filtered_stream_name_to_device_info_mapping = stream_name_to_device_info_mapping[
        (stream_name_to_device_info_mapping['value'].str.contains(TARGET_THREAD_NAME)) 
        &  (stream_name_to_device_info_mapping['deviceId'] == GPU_ID)
    ]


    filtered_stream_name_to_device_info_mapping


    num_tries = len(filtered_stream_name_to_device_info_mapping)
    num_tries

    for try_idx in range(num_tries):
        # assert len(filtered_stream_name_to_device_info_mapping) == 1
        OS_TID = filtered_stream_name_to_device_info_mapping['TID'].iloc[try_idx].item()
        
        # OS_TID
        # stream_name_to_device_info_mapping

        candidates = []

        if OS_PID is not None:
            candidates = [make_globalTid(OS_PID, OS_TID)]
        else:
            q_nvtx = """
            SELECT DISTINCT globalTid AS gtid FROM NVTX_EVENTS
            WHERE (globalTid & 0xFFFFFF) = :os_tid
            """
            q_rt = """
            SELECT DISTINCT globalTid AS gtid FROM CUPTI_ACTIVITY_KIND_RUNTIME
            WHERE (globalTid & 0xFFFFFF) = :os_tid
            """
            g1 = pd.read_sql_query(q_nvtx, conn, params={"os_tid": OS_TID})
            g2 = pd.read_sql_query(q_rt,   conn, params={"os_tid": OS_TID})
            candidates = sorted(set(g1["gtid"]).union(set(g2["gtid"])))

        print("OS_TID:", OS_TID, "| OS_PID:", OS_PID)
        print("Candidate globalTid(s):", candidates)
        if candidates:
            break

    def score_globalTid(gtid, gpu_id):
        q = """
        SELECT COUNT(*) AS c
        FROM CUPTI_ACTIVITY_KIND_RUNTIME api
        JOIN CUPTI_ACTIVITY_KIND_KERNEL k
        ON k.correlationId = api.correlationId
        WHERE api.globalTid = :gtid AND k.deviceId = :gpu
        """
        row = pd.read_sql_query(q, conn, params={"gtid": int(gtid), "gpu": int(gpu_id)}).iloc[0]
        return int(row["c"])

    if not candidates:
        raise RuntimeError("No candidate globalTid found for OS_TID. Are you sure the DB matches the GUI run?")

    scores = [(gtid, score_globalTid(gtid, GPU_ID)) for gtid in candidates]
    scores = sorted(scores, key=lambda x: x[1], reverse=True)
    GLOBAL_TID = scores[0][0]

    print("Selected GLOBAL_TID:", GLOBAL_TID, " (score=", scores[0][1], ")")
    if len(scores) > 1:
        print("Other candidates:", scores[1:])


    # Build kernel list for this thread and map each to innermost NVTX (by API launch window)
    # Join Runtime (only this thread) -> Kernel (optionally filter by GPU_ID)
    kernels_thread = pd.read_sql_query(
        """
        SELECT
        api.rowid            AS rid,
        api.globalTid        AS api_tid,
        api.start            AS api_start,
        api.end              AS api_end,
        api.correlationId    AS corr,

        k.rowid              AS kid,
        k.start              AS k_start,
        k.end                AS k_end,
        k.deviceId,
        k.streamId,
        k.shortName,
        k.demangledName
        FROM CUPTI_ACTIVITY_KIND_RUNTIME api
        JOIN CUPTI_ACTIVITY_KIND_KERNEL k
        ON k.correlationId = api.correlationId
        WHERE api.globalTid = :gtid
        """,
        conn,
        params={"gtid": int(GLOBAL_TID)},
    )


    # Keep only chosen GPU (comment this line if you want all GPUs)
    kernels_thread = kernels_thread[kernels_thread["deviceId"] == GPU_ID].copy()


    # Resolve kernel names
    kernels_thread["kernel_short"] = kernels_thread["shortName"].map(sid)
    kernels_thread["kernel_full"]  = kernels_thread["demangledName"].map(sid)

    # Fallbacks
    kernels_thread["kernel_name"] = kernels_thread["kernel_short"].fillna(kernels_thread["kernel_full"])
    kernels_thread["kernel_full"] = kernels_thread["kernel_full"].fillna(kernels_thread["kernel_name"])

    filtered_all_to_all_kernel_df = kernels_thread[
        kernels_thread['kernel_short'].str.contains('alltoall')
    ][[
        'k_start', 'k_end', 'kernel_short', 'kernel_full', 'api_start', 'api_end'
    ]].sort_values(by=['k_start'])
    filtered_all_to_all_kernel_df['kernel_full'] = filtered_all_to_all_kernel_df['kernel_full'].apply(lambda x: x[40:])


    filtered_all_to_all_kernel_df['is_send'] = filtered_all_to_all_kernel_df['kernel_full'].apply(
        lambda x: "<(bool)1, (bool)0>" in x
    )
    filtered_all_to_all_kernel_df['k_duration'] = filtered_all_to_all_kernel_df['k_end'] - filtered_all_to_all_kernel_df['k_start']
    filtered_all_to_all_kernel_df['k_duration_ms'] = filtered_all_to_all_kernel_df['k_duration'] / 1e6
    return filtered_all_to_all_kernel_df