# %%
mem_log_root = "/mnt/weka/home/yonghao.zhuang/jd/d2/benchmarks/_250905_large_scale_llama70b/sweep/20250906_133414.job-697093/mem-log"

import pandas as pd
import json 
import os
mem_logs = os.listdir(mem_log_root)
mem_logs = [os.path.join(mem_log_root, x) for x in mem_logs]
mem_logs


def find_nth_record(data, key, n):
    count = 0
    for i, record in enumerate(data):
        if key in record['message']:
            if count == n:
                return record['pynvml_gpu_memory_usage']
            count += 1
    return None


for mem_log in mem_logs:
    with open(mem_log, "r") as f:
        data = [json.loads(line) for line in f]
    # print([d['message'] for d in data])
    # wlbllm
    init_done = find_nth_record(data, "warmup start", 0)
    first_attn_start = find_nth_record(data, "PerDocumentCPAttention.forward(init)", 1)
    first_attn_end = find_nth_record(data, "PerDocumentCPAttention.forward(end)", 1)
    second_attn_start = find_nth_record(data, "PerDocumentCPAttention.forward(init)", 2)
    loss_func = find_nth_record(data, "loss_func", 0)
    done = find_nth_record(data, "forward_backward_batch:done", 0)

    attn_usage = (first_attn_end - first_attn_start)
    mlp_usage = (second_attn_start - first_attn_end)
    peak = (done or loss_func)
    is_oom = not done

    print(f"{mem_log}: init: {init_done} MB, attn_usage: {attn_usage} MB, mlp_usage: {mlp_usage} MB, peak: {peak} MB, is_oom: {is_oom}")

# %%


