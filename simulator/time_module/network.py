"""
Estimate the time of attention, mlp and communication.
"""
import enum
import pandas as pd
from functools import lru_cache

from .utils import dtype_to_str, DTYPE, CommType

KB = 1024

ALL_GATHER_TMPL = "profile/{gpu}/comm/all_gather.csv"
ALL_REDUCE_TMPL = "profile/{gpu}/comm/all_reduce.csv"
ALL_TO_ALL_TMPL = "profile/{gpu}/comm/alltoall.csv"
REDUCE_SCATTER_TMPL = "profile/{gpu}/comm/reduce_scatter.csv"
SEND_RECV_TMPL = "profile/{gpu}/comm/sendrecv.csv"


@lru_cache(maxsize=512)
def _load_table(op_kind: str, gpu: str) -> pd.DataFrame:
    """
    op_kind ∈ {"allgather", "allreduce", "alltoall", "reducescatter", "sendrecv}
    """
    name = {"allgather": ALL_GATHER_TMPL, "allreduce": ALL_REDUCE_TMPL, "alltoall": ALL_TO_ALL_TMPL, "reducescatter": REDUCE_SCATTER_TMPL, "sendrecv": SEND_RECV_TMPL}[op_kind].format(
        gpu = gpu
    )
    df = pd.read_csv(name)

    return df


def _allgather_df(gpu: str) -> pd.DataFrame:
    return _load_table("allgather", gpu)

def _allreduce_df(gpu: str) -> pd.DataFrame:
    return _load_table("allreduce", gpu)

def _alltoall_df(gpu: str) -> pd.DataFrame:
    return _load_table("alltoall", gpu)

def _reducescatter_df(gpu: str) -> pd.DataFrame:
    return _load_table("reducescatter", gpu)

def _sendrecv_df(gpu: str) -> pd.DataFrame:
    return _load_table("sendrecv", gpu)

def _interpolate(
    df: pd.DataFrame,
    col: str,
    val: int,
    target_col: str,
) -> float:
    large = df[df[col] >= val]
    if large.empty:
        r = val / df[col].max()
        return df[target_col].max() * r

    small = df[df[col] <= val]
    if len(small) == 0:
        raise ValueError(f"Cannot interpolate. {col}={val}")

    small = small.iloc[-1]
    large = large.iloc[0]
    if small[col] == large[col]:
        return small[target_col]

    r = (val - small[col]) / (large[col] - small[col])
    return small[target_col] * (1 - r) + large[target_col] * r


@lru_cache(maxsize=512)
def get_comm_time(
    comm_type: CommType,
    gpu: str,
    num_nodes: int,
    num_gpus_per_node: int,
    dtype: DTYPE,
    num_elements: int,
) -> float:
    if num_nodes == 1 and num_gpus_per_node == 1:
        return 0.0

    if comm_type == CommType.AllGather:
        df = _allgather_df(gpu)
    elif comm_type == CommType.AllReduce:
        df = _allreduce_df(gpu)
    elif comm_type == CommType.AllToAll:
        df = _alltoall_df(gpu)
    elif comm_type == CommType.ReduceScatter:
        df = _reducescatter_df(gpu)
    else:
        raise ValueError(f"Unknown comm type: {comm_type}")

    df = df[df["num_nodes"] == num_nodes]
    df = df[df["num_gpus_per_node"] == num_gpus_per_node]

    dtype_str = "half" if dtype == DTYPE.FLOAT8 else dtype_to_str(dtype)
    df = df[df["dtype"] == dtype_str]
    assert not df.empty, (
        f"Cannot find comm time for {comm_type} "
        f"gpu={gpu}, num_nodes={num_nodes}, "
        f"num_gpus_per_node={num_gpus_per_node}, dtype={dtype_str}"
    )

    size = num_elements * dtype.size // KB
    return _interpolate(df, "size(kb)", size, "time(us)")


@lru_cache(maxsize=256)
def get_p2p_comm_time(
    gpu: str,
    num_nodes: int,
    num_gpus_per_node: int,
    dtype: DTYPE,
    num_elements: int,
) -> float:
    if num_nodes == 1 and num_gpus_per_node == 1:
        return 0.0
    if num_nodes * num_gpus_per_node != 2:
        raise ValueError("P2P communication only supports 2 GPUs.")

    df = _sendrecv_df(gpu)

    df = df[df["num_nodes"] == num_nodes]
    df = df[df["num_gpus_per_node"] == num_gpus_per_node]
    dtype_str = "half" if dtype == DTYPE.FLOAT8 else dtype_to_str(dtype)
    df = df[df["dtype"] == dtype_str]
    assert not df.empty, (
        f"Cannot find Send Recv comm time for "
        f"gpu={gpu}, num_nodes={num_nodes}, "
        f"num_gpus_per_node={num_gpus_per_node}, dtype={dtype_str}"
    )

    size = num_elements * dtype.size // KB
    return _interpolate(df, "size(kb)", size, "time(us)")

