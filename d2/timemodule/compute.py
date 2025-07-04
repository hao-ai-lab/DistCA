# TODO: Add more models.

INF = int(1e15)

MLP_plan2dict = None

def setup_mlp_data():
    global MLP_plan2dict
    if MLP_plan2dict is not None:
        return
    
    from d2.profiling.get_mlp_data import get_mlp_data
    mlp_mapping = get_mlp_data()
    MLP_plan2dict = mlp_mapping
    return


def get_mlp_time_1_interpolate_all(x: int, tp: int, cp: int) -> float:
    setup_mlp_data()
    scoped_time = MLP_plan2dict[(tp, cp)]

    y = x
    min_key = min(scoped_time.keys())
    max_key = max(scoped_time.keys())

    if y < min_key:
        return scoped_time[min_key]  * (y / min_key)
    
    if y > max_key:
        return scoped_time[max_key] * (y / max_key)
    
    if y in scoped_time:
        return scoped_time[y]
    
    # Interpolate
    import bisect
    sorted_keys = list(scoped_time.keys())
    lower_key = max(k for k in sorted_keys if k <= y)
    upper_key = min(k for k in sorted_keys if k >= y)
    lower_time = scoped_time[lower_key]
    upper_time = scoped_time[upper_key]

    result = lower_time + (upper_time - lower_time) * (y - lower_key) / (upper_key - lower_key)
    return result

def get_mlp_time_2_remain_highest(x: int, tp: int, cp: int) -> float:
    """Remain only the highest and interpolate all rest."""
    setup_mlp_data()
    scoped_time = MLP_plan2dict[(tp, cp)]

    max_key = max(scoped_time.keys())
    max_time = scoped_time[max_key]

    duration = max_time * (x / max_key)
    return duration

# get_mlp_time = get_mlp_time_1_interpolate_all
get_mlp_time = get_mlp_time_2_remain_highest

# (tp, cp) -> {seq_len: latency(ms)}
ATTN_plan2dict = None


def setup_attn_data():
    global ATTN_plan2dict
    if ATTN_plan2dict is not None:
        return
    
    from d2.profiling.get_attn_data import get_attn_data
    attn_mapping = get_attn_data()
    ATTN_plan2dict = attn_mapping
    return 


def get_attn_time(
    x: int, tp: int, cp: int,
    hqo: int = 64, hkv: int = 4, d: int = 128,
) -> float:
    setup_attn_data()
    scoped_time = ATTN_plan2dict[(tp, cp)]

    y = x
    min_key = min(scoped_time.keys())
    max_key = max(scoped_time.keys())

    if y < min_key:
        return scoped_time[min_key]
    
    if y > max_key:
        return scoped_time[max_key] * (y / max_key) ** 2
    
    if y in scoped_time:
        return scoped_time[y]
    
    # Interpolate
    import bisect

    sorted_keys = list(scoped_time.keys())

    # lower_index = bisect.bisect_right(sorted_keys, y) - 1
    # upper_index = bisect.bisect_left(sorted_keys, y)
    # lower_key = sorted_keys[lower_index]
    # upper_key = sorted_keys[upper_index]

    lower_key = max(k for k in sorted_keys if k <= y)
    upper_key = min(k for k in sorted_keys if k >= y)

    lower_time = scoped_time[lower_key]
    upper_time = scoped_time[upper_key]
    # map the sequence lengths into “quadratic space”
    y2, lo2, hi2 = y * y, lower_key * lower_key, upper_key * upper_key
    ratio = (y2 - lo2) / (hi2 - lo2)           # 0 → 1 as y² moves between the anchors

    result = lower_time + (upper_time - lower_time) * ratio
    return result

    # result = lower_time + (upper_time - lower_time) * (y - lower_key) / (upper_key - lower_key)
    # return result
