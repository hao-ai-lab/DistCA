INF = int(1e15)

network_time_dict = {
    ('H100','allreduce', 2): {
        4294967296: 24.94,
        8589934592: 49.21,
        17179869184: 98.21
        # 16G elements per GPU, fp16, 98 ms
    },
    ('H100','allreduce', 4): {
        4294967296: 35.35,
        8589934592: 70.59,
        17179869184: 140.53
        # 16G elements per GPU, fp16, 140 ms
    },
    ('H100','allreduce', 8): {
        4294967296: 31.45,
        8589934592: 62.93,
        17179869184: 125.28
        # 16G elements per GPU, fp16, 125.28 ms
    },
    ('H100','allgather', 2): {
        1073741824: 10.37,
        2147483648: 45.1,
        4294967296: 145.34,
        8589934592: 314.83
        # 8G elements per GPU, fp16, 314.83 ms
    },
    ('H100','allgather', 4): {
        1073741824: 23.12,
        2147483648: 107.77,
        4294967296: 282.06,
        8589934592: 701.06
    },
    ('H100','allgather', 8): {
        1073741824: 109.26,
        2147483648: 304.8,
        4294967296: 969.17
    }
}

def get_allreduce_time(x: int, tp: int) -> float:
    if tp == 1:
        return 0
    key = ('H100', 'allreduce', tp)
    if key not in network_time_dict:
        return 0
    scoped_time = network_time_dict[key]
    y = x
    
    if y < min(scoped_time.keys()):
        return scoped_time[min(scoped_time.keys())]
    
    if y > max(scoped_time.keys()):
        return scoped_time[max(scoped_time.keys())] * (y / max(scoped_time.keys()))
    
    if y in scoped_time:
        return scoped_time[y]
    
    # interpolate if not exists
    lower_key = max(k for k in scoped_time.keys() if k <= y)
    upper_key = min(k for k in scoped_time.keys() if k >= y)

    lower_time = scoped_time[lower_key]
    upper_time = scoped_time[upper_key]
    
    return lower_time + (upper_time - lower_time) * (y - lower_key) / (upper_key - lower_key)


def get_allgather_time(x: int, cp: int) -> float:
    if cp == 1:
        return 0
    
    key = ('H100', 'allgather', cp)
    if key not in network_time_dict:
        return 0
    scoped_time = network_time_dict[key]
    y = x
    
    if y < min(scoped_time.keys()):
        return scoped_time[min(scoped_time.keys())]
    
    if y > max(scoped_time.keys()):
        return scoped_time[max(scoped_time.keys())] * (y / max(scoped_time.keys()))
    
    if y in scoped_time:
        return scoped_time[y]
    
    # interpolate if not exists
    lower_key = max(k for k in scoped_time.keys() if k <= y)
    upper_key = min(k for k in scoped_time.keys() if k >= y)

    lower_time = scoped_time[lower_key]
    upper_time = scoped_time[upper_key]
    
    return lower_time + (upper_time - lower_time) * (y - lower_key) / (upper_key - lower_key)
    pass
