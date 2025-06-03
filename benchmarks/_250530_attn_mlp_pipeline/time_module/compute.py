import enum
import os
import pandas as pd
from functools import lru_cache

from .utils import dtype_to_str, DTYPE, CommType

KB = 1024

this_dir = os.path.dirname(os.path.abspath(__file__))
GEMM_TMPL = os.path.join(this_dir, "profile/{gpu}/comp/gemm.csv")
MHA_TMPL = os.path.join(this_dir, "profile/{gpu}/comp/mha.csv")
BI_MHA_TMPL = os.path.join(this_dir, "profile/{gpu}/comp/bimha.csv")


@lru_cache(maxsize=512)
def _load_table(op_kind: str, gpu: str) -> pd.DataFrame:
    """
    op_kind ∈ {"gemm", "mha", "bimha"}
    """
    name = {"gemm": GEMM_TMPL, "mha": MHA_TMPL, "bimha": BI_MHA_TMPL}[op_kind].format(
        gpu = gpu
    )
    df = pd.read_csv(name)
    # assert not df.empty, f"Cannot find {op_kind} time for {gpu} in path: {name}"
    print(f"Loaded {op_kind} time for {gpu} from {name}")

    return df


def _gemm_df(gpu: str) -> pd.DataFrame:
    return _load_table("gemm", gpu)


def _mha_df(gpu: str) -> pd.DataFrame:
    return _load_table("mha", gpu)


def _bimha_df(gpu: str) -> pd.DataFrame:
    return _load_table("bimha", gpu)


def _interpolate(
    df: pd.DataFrame,
    col1: str,
    col1_val: int,
    col2: str,
    col2_val: int,
    target_col: str,
) -> float:
    small = df[(df[col1] <= col1_val) & (df[col2] <= col2_val)]
    large = df[(df[col1] >= col1_val) & (df[col2] >= col2_val)]
    if len(small) == 0 or len(large) == 0:
        if len(small) == 0 and len(large) != 0:
            return large.iloc[0][target_col]
        if len(large) == 0 and len(small) != 0:
            return small.iloc[-1][target_col]
        else:
            raise ValueError(
                "Cannot interpolate. "
                f"col1: {col1}, col1_val: {col1_val}, "
                f"col2: {col2}, col2_val: {col2_val}."
            )

    small = small.iloc[-1]
    large = large.iloc[0]
    if small[col1] == large[col1] and small[col2] == large[col2]:
        return small[target_col]
    elif small[col1] == large[col1]:
        r2 = (col2_val - small[col2]) / (large[col2] - small[col2])
        return small[target_col] * (1 - r2) + large[target_col] * r2
    elif small[col2] == large[col2]:
        r1 = (col1_val - small[col1]) / (large[col1] - small[col1])
        return small[target_col] * (1 - r1) + large[target_col] * r1
    else:
        r1 = (col1_val - small[col1]) / (large[col1] - small[col1])
        r2 = (col2_val - small[col2]) / (large[col2] - small[col2])
        r = (r1 * r2) ** 0.5
        return small[target_col] * (1 - r) + large[target_col] * r
    
def _interpolate_1d(
    df: pd.DataFrame,
    col: str,
    val: int,
    target_col: str,
) -> float:
    small = df[df[col] <= val]
    large = df[df[col] >= val]
    if len(small) == 0 or len(large) == 0:
        if len(small) == 0 and len(large) != 0:
            return large.iloc[0][target_col]
        if len(large) == 0 and len(small) != 0:
            return small.iloc[-1][target_col]
        else:
            raise ValueError(
                f"Cannot interpolate. {col}={val}"
            )

    small = small.iloc[-1]
    large = large.iloc[0]
    if small[col] == large[col]:
        return small[target_col]

    r = (val - small[col]) / (large[col] - small[col])
    return small[target_col] * (1 - r) + large[target_col] * r



@lru_cache(maxsize=512)
def _gemm_time(
    gpu: str,
    m: int,
    k: int,
    n: int,
    dtype: str,
) -> float:
    df = _gemm_df(gpu)
    df = df[df["dtype"] == dtype]
    df = df[df["n"] == n]
    assert (
        not df.empty
    ), f"Cannot find gemm time for {gpu}, {dtype},{m},{k},{n}"
    exe_time = _interpolate(df, "m", m, "k", k, "time(us)")
    return exe_time


def round_to_power_of_2(n):
    # If n is already a power of 2, return n
    if (n & (n - 1)) == 0:
        return n
    # Find the closest power of 2 greater than or equal to n
    power_of_2_greater = 1
    while power_of_2_greater < n:
        power_of_2_greater <<= 1
    # Find the closest power of 2 less than n
    power_of_2_less = power_of_2_greater >> 1
    # Return the closest power of 2
    if (n - power_of_2_less) < (power_of_2_greater - n):
        return power_of_2_less
    else:
        return power_of_2_greater


def gemm_time(
    gpu: str,
    m: int,
    k: int,
    n: int,
    dtype: DTYPE = DTYPE.BFLOAT16,
) -> float:
    # Round up to the nearest multiple of 128.

    n = (n + 127) // 128 * 128 if n > 64 else 64

    if dtype == "float8":
        m = 16 if m < 16 else m
        k = 16 if k < 16 else k
        m = round_to_power_of_2(m)
        k = round_to_power_of_2(k)
        n = round_to_power_of_2(n)
    return _gemm_time(gpu, m, k, n, dtype)


@lru_cache(maxsize=512)
def attn_time(
    gpu: str,
    cp: int,
    head_dim: int,
    nhead: int,
    tokens: int,
    dtype: str,
    is_fwd:bool=True,
) -> float:
    df = _mha_df(gpu)
    df = df[
        (df["cp"] == cp) &
        (df["dtype"] == dtype) &
        (df["head_dim"] == head_dim)
    ]
    assert (
        not df.empty
    ), f"Cannot find attn time for {gpu}, {dtype}, {head_dim}"
    # Round up to the nearest multiple of 16.

    fwdbwd = "fwd(us)" if is_fwd else "bwd(us)"
    exe_time = _interpolate(
        df, "tokens", tokens, "nhead", nhead, fwdbwd
    )
    return exe_time

