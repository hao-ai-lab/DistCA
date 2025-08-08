"""Create communication group"""

from datetime import timedelta
from typing import Callable, List, Optional, Tuple

from megatron.core import parallel_state as mpu
import torch

_ATTN_SERVER_GROUP = None
_ATTN_SERVER_GROUP_GLOO = None
_ATTN_SERVER_GLOBAL_RANKS = None


def initialize_attention_server_comm(
    distributed_timeout_minutes: int = 10,
):
    rank = torch.distributed.get_rank()
    global _ATTN_SERVER_GROUP
    global _ATTN_SERVER_GROUP_GLOO
    global _ATTN_SERVER_GLOBAL_RANKS
    # Create Gloo group only to send the uid in order to create the NVSHMEM group.
    timeout = timedelta(minutes=distributed_timeout_minutes)

    data_parallel_size = mpu.get_data_parallel_world_size() # TODO: double check

    # CP
    assert (
        mpu.get_context_parallel_group(check_initialized=False) is None or
        mpu.get_context_parallel_world_size() == 1
    )
    context_parallel_size = 1
    # TP
    if mpu.get_tensor_model_parallel_group(check_initialized=False) is None:
        tensor_model_parallel_size = 1
    else:
        tensor_model_parallel_size = mpu.get_tensor_model_parallel_world_size()
    # PP
    if mpu.get_pipeline_model_parallel_group() is None:
        pipeline_model_parallel_size = 1
    else:
        pipeline_model_parallel_size = mpu.get_pipeline_model_parallel_world_size()

    # FIXME: this is a hack of a fixed order, should be instead detected.
    order = "tp-cp-ep-dp-pp"
    decoder_rank_generator = mpu.RankGenerator(
        tp=tensor_model_parallel_size,
        ep=1,
        dp=data_parallel_size,
        pp=pipeline_model_parallel_size,
        cp=context_parallel_size,
        order=order,
        rank_offset=0,  # encoder world size = 0
    )

    for ranks in decoder_rank_generator.get_ranks('dp-pp'):
        group_gloo = mpu.create_group(
            ranks, timeout=timeout, backend="gloo",
            group_desc="ATTN_SERVER_GROUP_GLOO"
        )
        group = mpu.create_group(
            ranks, timeout=timeout, backend="nccl",
            group_desc="ATTN_SERVER_GROUP"
        )
        if rank in ranks:
            _ATTN_SERVER_GROUP_GLOO = group_gloo
            _ATTN_SERVER_GLOBAL_RANKS = ranks
            _ATTN_SERVER_GROUP = group


def destroy_attention_server_comm():
    """Destroy the attention server communication group."""
    global _ATTN_SERVER_GROUP
    global _ATTN_SERVER_GROUP_GLOO
    global _ATTN_SERVER_GLOBAL_RANKS

    _ATTN_SERVER_GROUP = None
    _ATTN_SERVER_GROUP_GLOO = None
    _ATTN_SERVER_GLOBAL_RANKS = None


######## Tool functions ########
def get_attn_server_group(check_initialized: bool = True):
    if check_initialized:
        assert _ATTN_SERVER_GROUP is not None, "attention server communication group is not initialized."
    return _ATTN_SERVER_GROUP


def get_attn_server_group_gloo(check_initialized: bool = True):
    if check_initialized:
        assert _ATTN_SERVER_GROUP_GLOO is not None, "attention server communication group is not initialized."
    return _ATTN_SERVER_GROUP_GLOO


def get_attn_server_global_ranks(check_initialized: bool = True):
    if check_initialized:
        assert _ATTN_SERVER_GLOBAL_RANKS is not None, "attention server global ranks are not initialized."
    return _ATTN_SERVER_GLOBAL_RANKS


def get_attn_server_rank():
    """Get the global rank of the attention server."""
    return torch.distributed.get_rank(
        group=get_attn_server_group_gloo(check_initialized=True)
    )


def get_attn_server_group_src_rank():
    """Get the source rank of the attention server group."""
    return get_attn_server_global_ranks(check_initialized=True)[0]
