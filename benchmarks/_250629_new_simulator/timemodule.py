import pandas as pd
from functools import lru_cache
import numpy as np

@lru_cache()
def setup_allgather_time():
    df = pd.read_csv("attn_time.csv")
    # fwd.allgather_kv
    allgather_df = df[df["name"] == "fwd.allgather_kv"]
    # (tp, cp) -> {seq_len: duration_ms}
    result = allgather_df.groupby(["tp", "cp"]).apply(lambda x: x.set_index("seq_len")["duration_ms"].to_dict()).to_dict()
    return result

@lru_cache()
def setup_attn_time():
    # Read the CSV file
    df = pd.read_csv("attn_time.csv")
    
    # Create a dictionary to store results
    # Format will be: result[(tp, cp)][seq_len] = total_time
    result = {}
    
    # Get unique combinations of tp, cp
    tp_cp_combinations = set(zip(df['tp'], df['cp']))
    
    # Process each combination
    for tp, cp in tp_cp_combinations:
        result[(tp, cp)] = {}
        
        # Get all sequence lengths for this tp, cp combination
        seq_lens = sorted(set(df[(df['tp'] == tp) & (df['cp'] == cp)]['seq_len']))
        
        for seq_len in seq_lens:
            # Filter data for current tp, cp, seq_len
            data = df[
                (df['tp'] == tp) & 
                (df['cp'] == cp) & 
                (df['seq_len'] == seq_len)
            ]
            
            # Sum up the times for each component
            total_time = 0
            required_components = {
                'fwd.init': False,
                'fwd.attn': False,
                'fwd.epilogue': False,
                'fwd.unshuffle': False
            }
            
            for _, row in data.iterrows():
                if row['name'] in required_components:
                    total_time += row['duration_ms']
                    required_components[row['name']] = True
            
            # Check if we have all required components
            if not all(required_components.values()):
                missing = [k for k, v in required_components.items() if not v]
                raise Exception(f"Missing components {missing} for tp={tp}, cp={cp}, seq_len={seq_len}")
            
            result[(tp, cp)][seq_len] = total_time
    
    return result

@lru_cache()
def setup_mlp_time():
    # Read the CSV file
    df = pd.read_csv("mlp_time.csv")
    
    # Create a dictionary to store results
    # Format will be: result[(tp, cp)][seq_len] = total_time
    result = {}
    
    # Process each row
    for _, row in df.iterrows():
        tp = int(row['tp'])
        cp = int(row['cp'])
        seq_len = int(row['seq_len'])  # Using seq_len from CSV
        
        if (tp, cp) not in result:
            result[(tp, cp)] = {}
        
        # Sum up MLP time components (mlp + mlp_bda)
        total_time = float(row['mlp']) + float(row['mlp_bda']) + float(row['linear_proj']) + float(row['qkv']) + float(row['self_attn_bda'])
        result[(tp, cp)][seq_len] = total_time
    
    return result

def _interpolate_log_linear(data_dict, tp, cp, x):
    """Common helper for log-linear interpolation of timing data.
    
    Args:
        data_dict: Dictionary with format {(tp,cp): {seq_len: time}}
        tp: Tensor parallel size
        cp: Context parallel size 
        x: Sequence length to interpolate for
    
    Returns:
        float: Interpolated time value
    """
    seq_lens = sorted(data_dict[(tp, cp)].keys())
    x1 = max([s for s in seq_lens if s <= x], default=None)
    x2 = min([s for s in seq_lens if s >= x], default=None)
    
    # If exact match found, return it directly
    if x1 == x:
        return float(data_dict[(tp, cp)][x1])
    
    if x1 is None or x2 is None:
        raise ValueError(f"Cannot interpolate for x={x} with available sequence lengths {seq_lens}")
    
    y1 = float(data_dict[(tp, cp)][x1])
    y2 = float(data_dict[(tp, cp)][x2])
    
    # Interpolate using logarithmic scale
    log_y1 = np.log(y1)
    log_y2 = np.log(y2)
    log_y = log_y1 + (log_y2 - log_y1) * ((np.log(x) - np.log(x1)) / (np.log(x2) - np.log(x1)))
    
    return float(np.exp(log_y))

def get_attn_time(tp, cp, x):
    collected_data = setup_attn_time()
    return _interpolate_log_linear(collected_data, tp, cp, x)

def _interpolate_linear_from_zero(data_dict, tp, cp, x):
    """Special linear interpolation for MLP time that assumes (0,0) as starting point.
    Since MLP only has one sequence length measurement, we do linear interpolation
    from origin (0,0) through the single measured point.
    """
    seq_lens = sorted(data_dict[(tp, cp)].keys())
    if len(seq_lens) != 1:
        raise ValueError(f"Expected exactly one sequence length for MLP data, got {len(seq_lens)}")
    
    # Get the single measurement point
    x1 = seq_lens[0]
    y1 = float(data_dict[(tp, cp)][x1])
    
    # Linear interpolation from (0,0) through (x1,y1)
    # y = mx where m = y1/x1
    return float(y1 * x / x1)

def get_mlp_time(tp, cp, x):
    collected_data = setup_mlp_time()
    return _interpolate_linear_from_zero(collected_data, tp, cp, x)

def get_allgather_time(tp, cp, x):
    collected_data = setup_allgather_time()
    return _interpolate_log_linear(collected_data, tp, cp, x)

def get_allreduce_time(tp, cp, x):
    raise RuntimeError("Allreduce time is included in MLP linear_proj time.")