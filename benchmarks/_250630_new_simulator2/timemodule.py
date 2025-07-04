import pandas as pd
import numpy as np
from functools import lru_cache

K = 1024
M = K ** 2
G = K ** 3
T = K ** 4

hidden_size = 4 * K
num_qo_head = 64
num_kv_head = 4
head_dim = 128
expert_dim = int(1.5 * K)
num_activate_experts = 8
dtype_size = 2

INF = int(1e10)

flops_per_ms = 989 * G # 1TFlops = 1e12 Flops/s

def get_mlp_time(tp, cp, total_tokens_across_all_gpus: int):
    if tp == 0 and cp == 0:
        return INF
    
    t = total_tokens_across_all_gpus
    q_proj_flops = dtype_size * t * hidden_size * num_qo_head * head_dim
    k_proj_flops = dtype_size * t * hidden_size * num_kv_head * head_dim
    v_proj_flops = dtype_size * t * hidden_size * num_kv_head * head_dim

    o_proj_flops = dtype_size * t * hidden_size * num_qo_head * head_dim

    mlp_fc1_flops = 2 * t * hidden_size * (expert_dim * num_activate_experts)
    mlp_gate_flops = 2 * t * hidden_size * (expert_dim * num_activate_experts)
    mlp_activation_flops = 4 * t * (expert_dim * num_activate_experts)
    mlp_fc2_flops = 2 * t * hidden_size * (expert_dim * num_activate_experts)

    linear_flops = (
        q_proj_flops + k_proj_flops + v_proj_flops + o_proj_flops 
        + mlp_fc1_flops + mlp_gate_flops + mlp_activation_flops + mlp_fc2_flops
    )

    linear_latency_ms = linear_flops / (tp * cp * flops_per_ms)

    component_flops = dict(
        q_proj_flops=q_proj_flops,
        k_proj_flops=k_proj_flops,
        v_proj_flops=v_proj_flops,
        o_proj_flops=o_proj_flops,
        mlp_fc1_flops=mlp_fc1_flops,
        mlp_gate_flops=mlp_gate_flops,
        mlp_activation_flops=mlp_activation_flops,
        mlp_fc2_flops=mlp_fc2_flops,
        linear_flops=linear_flops,
    )

    return linear_latency_ms, component_flops


@lru_cache(maxsize=None)
def setup_attn_time():
    df = pd.read_csv("attn_time.csv")
    df["latency_ms"] = df["latency_ms"].astype(float)
    
    attn_time_dict = {}
    for _, row in df.iterrows():
        key = (row['tp'], row['cp'])
        if key not in attn_time_dict:
            attn_time_dict[key] = {}
        attn_time_dict[key][row['t']] = row['latency_ms']
    return attn_time_dict

def interpolate_value(scoped_dict, key, squared=False):
    """Interpolate the value for a given key using log linear interpolation."""
    if key in scoped_dict:
        return scoped_dict[key]
    if key < min(scoped_dict.keys()) or key > max(scoped_dict.keys()):
        max_key = max(scoped_dict.keys())
        max_value = max(scoped_dict.values())

        scaling = (key / max_key)
        if squared:
            scaling = scaling ** 2
        return max_value * scaling

    sorted_keys = sorted(scoped_dict.keys())
    key1 = max(x for x in sorted_keys if x < key)
    key2 = min(x for x in sorted_keys if x > key)
    if key1 is None or key2 is None or key1 == key2:
        raise ValueError(f"key={key} not found in scoped_dict")

    log_y1 = np.log(scoped_dict[key1])
    log_y2 = np.log(scoped_dict[key2])
    log_x1 = np.log(key1)
    log_x2 = np.log(key2)
    log_x = np.log(key)
    log_y = log_y1 + (log_y2 - log_y1) * (log_x - log_x1) / (log_x2 - log_x1)
    return np.exp(log_y)


@lru_cache(maxsize=None)
def setup_allreduce_time():
    df = pd.read_csv("allreduce_time.csv")
    df["latency_ms"] = df["latency(ms)"].astype(float)
    allreduce_time_dict = {}
    for _, row in df.iterrows():
        key = row['world_size']
        if key not in allreduce_time_dict:
            allreduce_time_dict[key] = {}
        allreduce_time_dict[key][row['nelem']] = row['latency_ms']
    return allreduce_time_dict

@lru_cache(maxsize=None)
def setup_allgather_time():
    df = pd.read_csv("allgather_time.csv")
    df["latency_ms"] = df["latency(ms)"].astype(float)
    allgather_time_dict = {}
    for _, row in df.iterrows():
        key = row['world_size']
        if key not in allgather_time_dict:
            allgather_time_dict[key] = {}
        allgather_time_dict[key][row['nelem']] = row['latency_ms']
    return allgather_time_dict

def get_attn_time(tp, cp, t: int):
    if tp == 0 and cp == 0:
        return INF
    
    attn_time_dict = setup_attn_time()
    key = (tp, cp)
    scoped_attn_time_dict = attn_time_dict[key]
    return interpolate_value(scoped_attn_time_dict, t)

def get_allreduce_time(world_size, nelem_per_rank: int):
    if world_size == 1:
        return 0
    
    allreduce_time_dict = setup_allreduce_time()
    key = world_size
    scoped_allreduce_time_dict = allreduce_time_dict[key]
    return interpolate_value(scoped_allreduce_time_dict, nelem_per_rank)


def get_allreduce_time_with_config(world_size, num_tokens, hidden_dim):
    if world_size == 1:
        return 0
    
    nelem_per_rank = num_tokens * hidden_dim // world_size
    return get_allreduce_time(world_size, nelem_per_rank)

def get_allgather_time(world_size, nelem_per_rank: int):
    if world_size == 1:
        return 0
    
    allgather_time_dict = setup_allgather_time()
    key = world_size
    scoped_allgather_time_dict = allgather_time_dict[key]
    return interpolate_value(scoped_allgather_time_dict, nelem_per_rank)

def get_allgather_time_with_config(world_size, num_tokens, hidden_dim):
    if world_size == 1:
        return 0
    
    nelem_per_rank = num_tokens * hidden_dim // world_size
    return get_allgather_time(world_size, nelem_per_rank)