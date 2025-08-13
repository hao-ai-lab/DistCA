from dataclasses import dataclass
import os
import socket
from typing import Optional
import math


from megatron.core import parallel_state as mpu
import ray
import torch

from d2.runtime.attn_kernels.ops import (
    nvshmem_get_unique_id, nvshmem_alloc_empty_unique_id, FastDispatcherWrapper
)
from d2.runtime.fast_alltoall_metadata import (
    compute_fa2a_metadata_from_logical_metadata,
    forward_backward_with_resend_e2e_metadata,
)
from d2.runtime.inplace_metadata import (
    Metadata, compute_attn_layout_seqlens, compute_metadata,
    compute_metadata_kv, exclusive_cumsum
)
from d2.runtime.megatron_patch.create_group import (
    initialize_attention_server_comm, get_attn_server_group_gloo,
    get_attn_server_rank, get_attn_server_group_src_rank
)


######## Workers
@dataclass
class ParallelConfig:
    tensor_model_parallel_size: int = 1
    pipeline_model_parallel_size: int = 1
    virtual_pipeline_model_parallel_size: Optional[int] = None
    context_parallel_size: int = 1
    expert_model_parallel_size: int = 1
    expert_tensor_parallel_size: Optional[int] = None


# NOTE: the Worker abstraction is to make it compatible with ray.
# However, since ray has some issue with nsys, our default launch is torchrun.
class BaseWorker:
    def __init__(self, rank: int, world_size: int):
        self.rank = rank
        self.world_size = world_size
        self.device = None

    def init_torch_distributed(self,):
        if not torch.distributed.is_initialized():
            torch.distributed.init_process_group(backend="cpu:gloo,cuda:nccl", rank=self.rank, world_size=self.world_size)
            local_rank = int(os.environ.get("LOCAL_RANK"))
            self.device = torch.device(f"cuda:{local_rank}")

    def init_nvshmem(self, buffer_size: int, local_rank: int = None):
        if self.rank == 0:
            uid = nvshmem_get_unique_id()
        else:
            uid = nvshmem_alloc_empty_unique_id()
        torch.distributed.broadcast(uid, src=0)

        FastDispatcherWrapper.init(
            self.rank, local_rank, self.world_size, buffer_size, uid
        )

    def init_comm(self, buffer_size: int, local_rank: int = None):
        if local_rank is None:
            local_rank = int(os.getenv("LOCAL_RANK"))

        self.init_torch_distributed()
        self.init_nvshmem(buffer_size, local_rank)

    #### General init functions for ray.
    def get_node_ip_port(self):
        host_ipv4 = os.getenv("MY_HOST_IP", None)
        host_ipv6 = os.getenv("MY_HOST_IPV6", None)
        host_ip_by_env = host_ipv4 or host_ipv6
        host_ip_by_sdk = ray._private.services.get_node_ip_address()

        host_ip = host_ip_by_env or host_ip_by_sdk

        with socket.socket() as sock:
            sock.bind(("", 0))
            port = sock.getsockname()[1]
        return host_ip, str(port)

    def set_master_addr_port(self, master_addr: str, master_port: str):
        os.environ["MASTER_ADDR"] = master_addr
        os.environ["MASTER_PORT"] = master_port

    # def shutdown(self):
    #     torch.distributed.destroy_process_group()

    # def __del__(self):
    #     self.shutdown()


class MegatronBaseWorker(BaseWorker):
    """Worker base class to init communication groups (megatron and nvshmem)."""
    def __init__(self, rank: int, world_size: int):
        super().__init__(rank, world_size)
        self.as_world_size = None
        self.as_rank = None

    def init_nvshmem(self, buffer_size: int, parallel_config: ParallelConfig, local_rank: int=None):
        if local_rank is None:
            local_rank = int(os.getenv("LOCAL_RANK"))

        tp_size = parallel_config.tensor_model_parallel_size
        if tp_size == 1:
            super().init_nvshmem(buffer_size, local_rank)
            self.as_world_size = self.world_size
            self.as_rank = self.rank

        initialize_attention_server_comm()
        group = get_attn_server_group_gloo()
        as_world_size = torch.distributed.get_world_size(group=group)
        as_rank = get_attn_server_rank()
        as_src_rank = get_attn_server_group_src_rank()

        self.as_world_size = as_world_size
        self.as_rank = as_rank
        if as_rank == 0:
            uid = nvshmem_get_unique_id()
        else:
            uid = nvshmem_alloc_empty_unique_id()
        torch.distributed.broadcast(uid, src=as_src_rank, group=group)
        FastDispatcherWrapper.init(
            as_rank, local_rank, as_world_size, buffer_size, uid
        )


    def init_comm(self, buffer_size: int, parallel_config: ParallelConfig, local_rank: Optional[int] = None):
        # Init megatron communication.
        self.init_torch_distributed()
        # NOTE: do not set to local_rank here because the cuda visible device is set by ray.
        mpu.initialize_model_parallel(
            tensor_model_parallel_size=parallel_config.tensor_model_parallel_size,
            pipeline_model_parallel_size=parallel_config.pipeline_model_parallel_size,
            virtual_pipeline_model_parallel_size=parallel_config.virtual_pipeline_model_parallel_size,
            pipeline_model_parallel_split_rank=None,
            use_sharp=False,
            context_parallel_size=parallel_config.context_parallel_size,
            expert_model_parallel_size=parallel_config.expert_model_parallel_size,
            expert_tensor_parallel_size=parallel_config.expert_tensor_parallel_size,
            nccl_communicator_config_path=None,
        )
        self.init_nvshmem(buffer_size, parallel_config, local_rank)


def init_worker_torch_distributed(
    world_size, buffer_size, worker_cls=BaseWorker, parallel_config=None
):
    assert world_size == int(os.environ.get("WORLD_SIZE"))
    rank = int(os.environ.get("RANK"))
    local_rank = int(os.environ.get("LOCAL_RANK"))
    worker = worker_cls(
        rank, world_size
    )
    if parallel_config is not None:
        worker.init_comm(buffer_size, parallel_config, local_rank)
    else:
        worker.init_comm(buffer_size, local_rank)
    return worker


######## Data construction
def test_local_proj_metadata(world_size: int, seq_len: int, offset: int):
    """
    Test tool generating the easiest case: Each rank holds a sequence,
    and is sent to a rank with an offset. (0 means sending to itself)
    """
    num_recv_tokens = torch.zeros((world_size, world_size + 1), dtype=torch.int64)
    num_recv_tokens[:, -1] = seq_len
    i_diag = torch.arange(world_size, dtype=torch.int64)
    j_diag = (i_diag + offset) % world_size
    # the diagonal of num_recv_tokens is the seq_len
    fwd_recv_tokens = num_recv_tokens.clone()
    fwd_recv_tokens[j_diag, i_diag] = seq_len

    bwd_j_diag = (i_diag - offset) % world_size
    bwd_recv_tokens = num_recv_tokens.clone()
    bwd_recv_tokens[bwd_j_diag, i_diag] = seq_len

    mlp_to_attn_metadata = Metadata(
        dst_rank=torch.tensor(
            [(i + offset) % world_size for i in range(world_size)]
        ).reshape(world_size, 1),
        dst_offset=torch.tensor(
            [0] * world_size
        ).reshape(world_size, 1),
        seq_len=torch.tensor(
            [seq_len] * world_size
        ).reshape(world_size, 1),
        num_recv_tokens=fwd_recv_tokens,
        num_seqs=torch.tensor([1] * world_size).reshape(world_size, 1),
        world_size=world_size,
        num_total_recv_tokens=[seq_len] * world_size,
    )
    attn_to_mlp_metadata = Metadata(
        dst_rank=torch.tensor(
            [(i - offset) % world_size for i in range(world_size)]
        ).reshape(world_size, 1),
        dst_offset=mlp_to_attn_metadata.dst_offset.clone(),
        seq_len=mlp_to_attn_metadata.seq_len.clone(),
        num_recv_tokens=bwd_recv_tokens,
        num_seqs=mlp_to_attn_metadata.num_seqs.clone(),
        world_size=world_size,
        num_total_recv_tokens=[seq_len] * world_size,
    )
    mlp_to_attn_kv_metadata = Metadata(
        dst_rank=mlp_to_attn_metadata.dst_rank.clone().unsqueeze(-1),
        dst_offset=mlp_to_attn_metadata.dst_offset.clone().unsqueeze(-1),
        seq_len=mlp_to_attn_metadata.seq_len.clone(),
        num_recv_tokens=fwd_recv_tokens.clone(),
        num_seqs=mlp_to_attn_metadata.num_seqs.clone(),
        world_size=world_size,
        num_total_recv_tokens=[seq_len] * world_size,
    )
    mlp_to_attn_kv_grad_metadata = Metadata(
        dst_rank=attn_to_mlp_metadata.dst_rank.clone(),
        dst_offset=attn_to_mlp_metadata.dst_offset.clone(),
        seq_len=attn_to_mlp_metadata.seq_len.clone(),
        num_recv_tokens=bwd_recv_tokens.clone(),
        num_seqs=attn_to_mlp_metadata.num_seqs.clone(),
        world_size=world_size,
        seq_recv_mask=torch.ones(world_size, 1, 1),
        recv_seq_lens=mlp_to_attn_metadata.seq_len.clone(),
        num_total_recv_tokens=[seq_len] * world_size,
    )
    return mlp_to_attn_metadata, attn_to_mlp_metadata, mlp_to_attn_kv_metadata, mlp_to_attn_kv_grad_metadata


def gen_seq_lens(world_size: int, num_seqs: int, total_len: int) -> torch.Tensor:
    ratio = torch.rand((world_size, num_seqs)) + 0.25 / num_seqs   # Use a min value to guarantee that the sequence is not too short (0 after rounding)
    ratio = ratio / ratio.sum(dim=1, keepdim=True)
    seq_len = (ratio * total_len).round().int()
    seq_len_total = seq_len.sum(dim=1)
    seq_len_total_error = seq_len_total - total_len
    seq_len[:, -1] -= seq_len_total_error
    return seq_len


def create_raw_qkv_dispatch(
    world_size: int, total_seq_len: int, num_seqs: int, max_cp_degree: int,
    return_mlp_no_shard_seq_lens: bool=False,
    reverse_seq_lens: Optional[torch.Tensor] = None,
    last_seq_lens: Optional[torch.Tensor] = None,
    dummy_first: bool = False,
    dummy_except_first: bool = False,
    reverse: bool = False,
):
    '''
    reverse_seq_lens: the forward seq_lens used to reverse (see reverse)
    last_seq_lens: the seq_lens from last microbatch in pp, ensuring the same microbatch has the same seq_lens on different ranks
    dummy_first: the first microbatch will have minimal seq_len, used for dummy forward in pp cooldown
    dummy_except_first: all microbatches except the first one will have minimal seq_len, used for dummy forward in pp warmup
    reverse: comparing pp forward and backward, the same group of microbatches have reversed ranks,
             so we reverse the seq_lens in forwad pass to get that in backward pass.
    '''
    """NOTE: this is currently a dispatch tensor of not consider the 2CP optimization."""
    # init sequence
    assert total_seq_len % (max_cp_degree) == 0
    _num_tokens_shard = total_seq_len // (max_cp_degree)
    seq_lens = gen_seq_lens(world_size, num_seqs, _num_tokens_shard).long()
    if dummy_first:
        seq_lens[:1] = gen_seq_lens(1, 1, 1).long()
    if dummy_except_first:
        seq_lens[1:] = gen_seq_lens(world_size - 1, 1, 1).long()
    # make sure each sequence is divisible by max_cp_degree.
    seq_lens *= max_cp_degree
    if last_seq_lens is not None:
        seq_lens[1:] = last_seq_lens[:-1]
    if reverse:
        seq_lens = reverse_seq_lens.flip(0)

    # init cp degree for each sequence
    log_cp_num = torch.randint(0, int(math.log2(max_cp_degree)) + 1, (world_size, num_seqs))
    cp_num = torch.pow(2, log_cp_num)

    # init cp send dstination.
    cp_dst_helper = torch.rand((world_size, num_seqs, world_size)).argsort(dim=2)
    cp_dst = cp_dst_helper[:, :, :max_cp_degree]
    mask = torch.arange(max_cp_degree).expand(world_size, num_seqs, max_cp_degree)
    cp_num_expanded = cp_num.unsqueeze(-1)
    mask = mask >= cp_num_expanded
    cp_dst[mask] = -1

    # q_global_dispatch tensor:
    num_cp_shards = cp_num.sum(dim=1)
    pad_len = torch.max(num_cp_shards)
    cp_seq_lens = torch.zeros(world_size, pad_len, dtype=torch.int64)
    cp_query_dst = torch.ones(world_size, pad_len, dtype=torch.int64) * -1
    kv_to_q_mapping = torch.ones((world_size, pad_len, max_cp_degree, 2), dtype=torch.int64) * -1
    kv_to_q_rank = torch.ones((world_size, pad_len, max_cp_degree), dtype=torch.int64) * -1
    kv_context_size = torch.zeros((world_size, pad_len), dtype=torch.int64)
    q_to_num_kv_seq = torch.zeros((world_size, pad_len), dtype=torch.int64)

    # cumulative number of cp shards before this one.
    num_cul_cp_shards = exclusive_cumsum(cp_num, dim=1)

    for i in range(world_size):
        cp_seq_lens_local = []
        cp_query_dst_local = []
        kv_to_q_mapping_local = []
        kv_to_q_rank_local = []
        kv_context_size_local = []
        q_to_num_kv_seq_local = []

        for j in range(num_seqs):
            num_cp = int((cp_num[i, j]).item())
            seq_len = seq_lens[i, j]
            seq_shard_len = seq_len // num_cp

            cp_seq_lens_local.append(seq_shard_len.reshape(1,).repeat(num_cp))
            cp_query_dst_local.append(cp_dst[i, j, :num_cp].flatten())
            #### Compute kv_to_q_mapping.
            row_indices = torch.arange(num_cp).view(-1, 1)
            col_indices = torch.arange(max_cp_degree).view(1, -1)
            mask = col_indices < (num_cp - row_indices)
            kv_to_q_mapping_seq = torch.empty((num_cp, max_cp_degree, 2), dtype=torch.int64)
            # All q shards are on this node (TODO: we are testing MLP-DP. For MLP-CP, this is different).
            kv_to_q_mapping_seq[..., 0] = torch.where(mask, i, -1)
            vals_ch1 = row_indices + col_indices + num_cul_cp_shards[i, j]
            kv_to_q_mapping_seq[..., 1] = torch.where(mask, vals_ch1, -1)
            kv_to_q_mapping_local.append(kv_to_q_mapping_seq)
            #### Compute kv_to_q_rank (Index of this KV to the query's dst).
            kv_to_q_rank_seq = torch.arange(num_cp).view(-1, 1).repeat(1, max_cp_degree) * mask + (mask.int() - 1)
            kv_to_q_rank_local.append(kv_to_q_rank_seq)
            #### Compute kv context size (For this kv, how many tokens are in the context).
            kv_context_size_seq = torch.arange(num_cp) * seq_shard_len
            kv_context_size_local.append(kv_context_size_seq)
            #### Compute q_to_num_kv_seq (For this kv, how many shards are in the context).
            q_to_num_kv_seq_seq = torch.arange(num_cp) + 1
            q_to_num_kv_seq_local.append(q_to_num_kv_seq_seq)

        cp_seq_lens_local = torch.cat(cp_seq_lens_local, dim=0)
        cp_query_dst_local = torch.cat(cp_query_dst_local, dim=0)
        kv_to_q_mapping_local = torch.cat(kv_to_q_mapping_local, dim=0)
        kv_to_q_rank_local = torch.cat(kv_to_q_rank_local, dim=0)
        kv_context_size_local = torch.cat(kv_context_size_local, dim=0)
        q_to_num_kv_seq_local = torch.cat(q_to_num_kv_seq_local, dim=0)
        # shape check:
        seq_shards = cp_seq_lens_local.shape[0]
        assert cp_seq_lens_local.shape == (seq_shards,)
        assert cp_query_dst_local.shape == (seq_shards,)
        assert kv_to_q_mapping_local.shape == (seq_shards, max_cp_degree, 2)
        assert kv_to_q_rank_local.shape == (seq_shards, max_cp_degree)
        assert kv_context_size_local.shape == (seq_shards,)
        assert q_to_num_kv_seq_local.shape == (seq_shards,)

        cp_seq_lens[i, :seq_shards] = cp_seq_lens_local
        cp_query_dst[i, :seq_shards] = cp_query_dst_local
        kv_to_q_mapping[i, :seq_shards] = kv_to_q_mapping_local
        kv_to_q_rank[i, :seq_shards] = kv_to_q_rank_local
        kv_context_size[i, :seq_shards] = kv_context_size_local
        q_to_num_kv_seq[i, :seq_shards] = q_to_num_kv_seq_local

    q_to_num_kv_tokens = kv_context_size + cp_seq_lens
    return (
        cp_seq_lens, num_cp_shards, cp_query_dst,
        kv_to_q_mapping, kv_to_q_rank, kv_context_size,
        q_to_num_kv_seq, q_to_num_kv_tokens,
        (seq_lens, ) if return_mlp_no_shard_seq_lens else (),
    )

def create_qkv_dispath_with_backward(
    world_size: int, total_seq_len: int, num_seqs: int, max_cp_degree: int,
    hidden_size_q: int, hidden_size_k: int,
    element_size: int, # dtype's size
    softmax_lse_size: int, # size of the softmax_lse tensor, should be num_heads
    return_mlp_no_shard_seq_lens: bool=False,
    last_seq_lens: Optional[torch.Tensor] = None,
    dummy_first: bool = False,
    dummy_except_first: bool = False,
    reverse: bool = False,
):
    '''
    reverse_seq_lens, last_seq_lens, dummy_first, dummy_except_first, reverse:
    see create_raw_qkv_dispatch for details.

    reverse:
    comparing pp forward and backward, the same group of microbatches have reversed ranks,
    so we reverse the seq_lens in forwad pass to get that in backward pass.

    Thus, we first run a pass with reverse=False to get the forward metadata,
    then run anothe pass, forcing the seqlens to be the reverse,
    '''
    if reverse:
        f1, _, f2, _, f3, _, f4, _, f5, _, seq_lens = create_qkv_dispath_with_backward(
            world_size, total_seq_len, num_seqs, max_cp_degree,
            hidden_size_q, hidden_size_k,
            element_size, softmax_lse_size,
            return_mlp_no_shard_seq_lens,
            last_seq_lens=last_seq_lens,
            dummy_first=dummy_first,
            dummy_except_first=dummy_except_first,
            reverse=False,
        )
        (mlp_seq_len, mlp_num_seqs, mlp_q_dispatch_fwd,
        kv_to_q_mapping, kv_to_q_rank, kv_context_size,
        q_to_num_kv_seq, q_to_num_kv_tokens,
        _) = create_raw_qkv_dispatch(
            world_size, total_seq_len, num_seqs, max_cp_degree,
            return_mlp_no_shard_seq_lens,
            last_seq_lens=last_seq_lens,
            reverse_seq_lens=seq_lens,
            dummy_first=dummy_first,
            dummy_except_first=dummy_except_first,
            reverse=True
        )
        mlp_q_dispatch_bwd = torch.randint_like(mlp_q_dispatch_fwd, low=0, high=world_size)
        mask = torch.arange(mlp_q_dispatch_fwd.shape[1])[None].repeat_interleave(world_size, dim=0) >= mlp_num_seqs.reshape(world_size, 1)
        mlp_q_dispatch_bwd[mask] = -1

        _, r1, _, r2, _, r3, _, r4, _, r5 = forward_backward_with_resend_e2e_metadata(
            mlp_seq_len, mlp_num_seqs, mlp_q_dispatch_fwd, mlp_q_dispatch_bwd,
            kv_to_q_mapping, kv_to_q_rank, kv_context_size,
            q_to_num_kv_seq, q_to_num_kv_tokens,
            hidden_size_q, hidden_size_k, element_size, softmax_lse_size,
        )
        return f1, r1, f2, r2, f3, r3, f4, r4, f5, r5, seq_lens
    (mlp_seq_len, mlp_num_seqs, mlp_q_dispatch_fwd,
     kv_to_q_mapping, kv_to_q_rank, kv_context_size,
     q_to_num_kv_seq, q_to_num_kv_tokens,
     seq_lens) = create_raw_qkv_dispatch(
        world_size, total_seq_len, num_seqs, max_cp_degree,
        return_mlp_no_shard_seq_lens,
        last_seq_lens=last_seq_lens,
        dummy_first=dummy_first,
        dummy_except_first=dummy_except_first,
    )
    mlp_q_dispatch_bwd = torch.randint_like(mlp_q_dispatch_fwd, low=0, high=world_size)
    mask = torch.arange(mlp_q_dispatch_fwd.shape[1])[None].repeat_interleave(world_size, dim=0) >= mlp_num_seqs.reshape(world_size, 1)
    mlp_q_dispatch_bwd[mask] = -1

    ret = forward_backward_with_resend_e2e_metadata(
        mlp_seq_len, mlp_num_seqs, mlp_q_dispatch_fwd, mlp_q_dispatch_bwd,
        kv_to_q_mapping, kv_to_q_rank, kv_context_size,
        q_to_num_kv_seq, q_to_num_kv_tokens,
        hidden_size_q, hidden_size_k, element_size, softmax_lse_size,
    )
    ret += seq_lens
    return ret


def create_qkv_dispatch(
    world_size: int, total_seq_len: int, num_seqs: int, max_cp_degree: int,
    return_intermediate: bool=False, return_mlp_no_shard_seq_lens: bool=False
):
    (cp_seq_lens, num_cp_shards, cp_query_dst,
     kv_to_q_mapping, kv_to_q_rank, kv_context_size,
     q_to_num_kv_seq, q_to_num_kv_tokens,
     seq_lens) = create_raw_qkv_dispatch(
        world_size, total_seq_len, num_seqs, max_cp_degree,
        return_mlp_no_shard_seq_lens
    )
    fwd_q_metadata, rev_q_metadata, q_intermediates = compute_metadata(
        cp_seq_lens, cp_query_dst, return_intermediate=True
    )
    _, q_seq_to_dst, _ = q_intermediates
    pad_len = torch.max(num_cp_shards)
    fwd_k_metadata, rev_k_metadata, kv_intermediates = compute_metadata_kv(
        kv_to_q_mapping, kv_to_q_rank, kv_context_size, q_to_num_kv_seq,
        q_to_num_kv_tokens, cp_seq_lens, num_cp_shards, cp_query_dst,
        q_seq_to_dst.squeeze(2), pad_len,
        return_intermediate=True
    )
    attention_metadata = compute_attn_layout_seqlens(
        cp_seq_lens, q_to_num_kv_tokens, cp_query_dst, shard_to_tuple=True
    )
    ret = (
        fwd_q_metadata, rev_q_metadata, fwd_k_metadata, rev_k_metadata, attention_metadata
    )
    if return_intermediate:
        intermediates = q_intermediates + kv_intermediates
        ret += (intermediates,)
    ret += seq_lens
    return ret


import rich
def create_qkv_dispatch_2cp(
    world_size: int, total_seq_len: int, num_seqs: int, max_cp_degree: int,
    return_intermediate: bool=False, return_mlp_no_shard_seq_lens: bool=False,
    verbose: bool=False,
):
    """
    Test a case with (not strictly balanced) 2CP.
    This case is mainly testing the case where world_size < max_cp_degree. 
    """
    VERBOSE = verbose
    def print_if_verbose(*args, **kwargs):
        if VERBOSE:
            rich.print(*args, **kwargs)

    assert max_cp_degree == world_size * 2, "2CP is only supported for max_cp_degree = world_size * 2"

    # init sequence
    assert total_seq_len % (max_cp_degree) == 0
    _num_tokens_shard = total_seq_len // (max_cp_degree)
    seq_lens = gen_seq_lens(world_size, num_seqs, _num_tokens_shard).long()
    # make sure each sequence is divisible by max_cp_degree.
    seq_lens *= max_cp_degree
    print_if_verbose("seq_lens =", seq_lens)

    # init cp degree for each sequence
    log_cp_num = torch.randint(0, int(math.log2(max_cp_degree)) + 1, (world_size, num_seqs))
    cp_num = torch.pow(2, log_cp_num)
    print_if_verbose("cp_num =", cp_num)

    # init cp send dstination.
    cp_dst_helper = torch.rand((world_size, num_seqs, max_cp_degree)).argsort(dim=2)
    cp_dst_helper = cp_dst_helper % world_size # ensure everything is in the range of world_size
    cp_dst = cp_dst_helper[:, :, :max_cp_degree]
    mask = torch.arange(max_cp_degree).expand(world_size, num_seqs, max_cp_degree)
    cp_num_expanded = cp_num.unsqueeze(-1)
    mask = mask >= cp_num_expanded
    cp_dst[mask] = -1
    print_if_verbose("cp_dst =", cp_dst)

    # q_global_dispatch tensor:
    num_cp_shards = cp_num.sum(dim=1)
    pad_len = torch.max(num_cp_shards)
    print_if_verbose("num_cp_shards =", num_cp_shards)

    cp_seq_lens = torch.zeros(world_size, pad_len, dtype=torch.int64)
    cp_query_dst = torch.ones(world_size, pad_len, dtype=torch.int64) * -1
    kv_to_q_mapping = torch.ones((world_size, pad_len, max_cp_degree, 2), dtype=torch.int64) * -1
    kv_to_q_rank = torch.ones((world_size, pad_len, max_cp_degree), dtype=torch.int64) * -1
    kv_context_size = torch.zeros((world_size, pad_len), dtype=torch.int64)
    q_to_num_kv_seq = torch.zeros((world_size, pad_len), dtype=torch.int64)

    # cumulative number of cp shards before this one.
    num_cul_cp_shards = exclusive_cumsum(cp_num, dim=1)
    print_if_verbose("num_cul_cp_shards =", num_cul_cp_shards)

    for i in range(world_size):
        cp_seq_lens_local = []
        cp_query_dst_local = []
        kv_to_q_mapping_local = []
        kv_to_q_rank_local = []
        kv_context_size_local = []
        q_to_num_kv_seq_local = []

        for j in range(num_seqs):
            num_cp = int((cp_num[i, j]).item())
            seq_len = seq_lens[i, j]
            seq_shard_len = seq_len // num_cp

            cp_seq_lens_local.append(seq_shard_len.reshape(1,).repeat(num_cp))
            cp_query_dst_local.append(cp_dst[i, j, :num_cp].flatten())
            #### Compute kv_to_q_mapping.
            row_indices = torch.arange(num_cp).view(-1, 1)
            col_indices = torch.arange(max_cp_degree).view(1, -1)
            mask = col_indices < (num_cp - row_indices)
            kv_to_q_mapping_seq = torch.empty((num_cp, max_cp_degree, 2), dtype=torch.int64)
            # All q shards are on this node (TODO: we are testing MLP-DP. For MLP-CP, this is different).
            kv_to_q_mapping_seq[..., 0] = torch.where(mask, i, -1)
            vals_ch1 = row_indices + col_indices + num_cul_cp_shards[i, j]
            kv_to_q_mapping_seq[..., 1] = torch.where(mask, vals_ch1, -1)
            kv_to_q_mapping_local.append(kv_to_q_mapping_seq)
            #### Compute kv_to_q_rank (Index of this KV to the query's dst).
            kv_to_q_rank_seq = torch.arange(num_cp).view(-1, 1).repeat(1, max_cp_degree) * mask + (mask.int() - 1)
            kv_to_q_rank_local.append(kv_to_q_rank_seq)
            #### Compute kv context size (For this kv, how many tokens are in the context).
            kv_context_size_seq = torch.arange(num_cp) * seq_shard_len
            kv_context_size_local.append(kv_context_size_seq)
            #### Compute q_to_num_kv_seq (For this kv, how many shards are in the context).
            q_to_num_kv_seq_seq = torch.arange(num_cp) + 1
            q_to_num_kv_seq_local.append(q_to_num_kv_seq_seq)

        cp_seq_lens_local = torch.cat(cp_seq_lens_local, dim=0)
        cp_query_dst_local = torch.cat(cp_query_dst_local, dim=0)
        kv_to_q_mapping_local = torch.cat(kv_to_q_mapping_local, dim=0)
        kv_to_q_rank_local = torch.cat(kv_to_q_rank_local, dim=0)
        kv_context_size_local = torch.cat(kv_context_size_local, dim=0)
        q_to_num_kv_seq_local = torch.cat(q_to_num_kv_seq_local, dim=0)
        # shape check:
        seq_shards = cp_seq_lens_local.shape[0]
        assert cp_seq_lens_local.shape == (seq_shards,)
        assert cp_query_dst_local.shape == (seq_shards,)
        assert kv_to_q_mapping_local.shape == (seq_shards, max_cp_degree, 2)
        assert kv_to_q_rank_local.shape == (seq_shards, max_cp_degree)
        assert kv_context_size_local.shape == (seq_shards,)
        assert q_to_num_kv_seq_local.shape == (seq_shards,)

        cp_seq_lens[i, :seq_shards] = cp_seq_lens_local
        cp_query_dst[i, :seq_shards] = cp_query_dst_local
        kv_to_q_mapping[i, :seq_shards] = kv_to_q_mapping_local
        kv_to_q_rank[i, :seq_shards] = kv_to_q_rank_local
        kv_context_size[i, :seq_shards] = kv_context_size_local
        q_to_num_kv_seq[i, :seq_shards] = q_to_num_kv_seq_local

    q_to_num_kv_tokens = kv_context_size + cp_seq_lens

    print_if_verbose("cp_seq_lens =", cp_seq_lens)
    print_if_verbose("cp_query_dst =", cp_query_dst)
    print_if_verbose("kv_to_q_mapping =", kv_to_q_mapping)
    print_if_verbose("kv_to_q_rank =", kv_to_q_rank)
    print_if_verbose("kv_context_size =", kv_context_size)
    print_if_verbose("q_to_num_kv_seq =", q_to_num_kv_seq)
    print_if_verbose("q_to_num_kv_tokens =", q_to_num_kv_tokens)

    # TODO: Use the compute_metadata_e2e() instead.
    fwd_q_metadata, rev_q_metadata, q_intermediates = compute_metadata(
        cp_seq_lens, cp_query_dst, return_intermediate=True
    )
    _, q_seq_to_dst, _ = q_intermediates
    fwd_k_metadata, rev_k_metadata, kv_intermediates = compute_metadata_kv(
        kv_to_q_mapping, kv_to_q_rank, kv_context_size, q_to_num_kv_seq,
        q_to_num_kv_tokens, cp_seq_lens, num_cp_shards, cp_query_dst,
        q_seq_to_dst.squeeze(2), pad_len,
        return_intermediate=True
    )
    attention_metadata = compute_attn_layout_seqlens(
        cp_seq_lens, q_to_num_kv_tokens, cp_query_dst, shard_to_tuple=True
    )
    ret = (
        fwd_q_metadata, rev_q_metadata, fwd_k_metadata, rev_k_metadata, attention_metadata
    )
    print_if_verbose("fwd_q_metadata =", fwd_q_metadata)
    print_if_verbose("rev_q_metadata =", rev_q_metadata)
    print_if_verbose("fwd_k_metadata =", fwd_k_metadata)
    print_if_verbose("rev_k_metadata =", rev_k_metadata)
    print_if_verbose("attention_metadata =", attention_metadata)

    if return_intermediate:
        intermediates = q_intermediates + kv_intermediates
        ret += (intermediates,)
    if return_mlp_no_shard_seq_lens:
        ret += (seq_lens,)
    return ret



def create_qkv_dispatch_with_custom_mapping(
    world_size: int, 
    seq_lens: 'torch.Tensor[world_size, max_num_seqs]',
    # cp_num[src_rank, seq_id] = num_cp_shards
    cp_num: 'torch.Tensor[world_size, max_num_seqs]',
    # cp_dst[src_rank, seq_id, shard_id] = dst_rank (or -1 as null)
    cp_dst: 'torch.Tensor[world_size, max_num_seqs, max_cp_degree]',
    # seq_shard_lens[src_rank, seq_id, shard_id] = shard_len (or 0 as null)
    seq_shard_lens: 'torch.Tensor[world_size, max_num_seqs, max_cp_degree]',
    return_intermediate: bool=False, return_mlp_no_shard_seq_lens: bool=False,
    verbose: bool=False,
):
    """
    Test a case where we deliberately construct a flops-balanced 2CP case..
    """
    VERBOSE = verbose
    def print_if_verbose(*args, **kwargs):
        if VERBOSE:
            rich.print(*args, **kwargs)

    num_seqs = seq_lens.shape[1]
    max_cp_degree = cp_dst.shape[2]
    
    assert seq_lens.shape == (world_size, num_seqs)
    assert seq_lens.min() >= 0
    assert cp_num.shape == (world_size, num_seqs)
    assert cp_num.min() >= 0
    assert cp_dst.shape == (world_size, num_seqs, max_cp_degree)

    
    print_if_verbose("seq_lens =", seq_lens)
    print_if_verbose("cp_num =", cp_num)

    def check_seq_lens():
        for i in range(world_size):
            for j in range(num_seqs):
                if seq_lens[i, j] == 0:
                    # Then everything k > j should also be 0
                    for k in range(j + 1, num_seqs):
                        assert seq_lens[i, k] == 0, f"seq_lens[{i}, {k}] = {seq_lens[i, k]} is not 0"
        return

    check_seq_lens()
    print_if_verbose("seq_lens =", seq_lens)

    # # init cp send dstination.
    # cp_dst_helper = torch.rand((world_size, num_seqs, max_cp_degree)).argsort(dim=2)
    # cp_dst_helper = cp_dst_helper % world_size # ensure everything is in the range of world_size
    # cp_dst = cp_dst_helper[:, :, :max_cp_degree]
    # mask = torch.arange(max_cp_degree).expand(world_size, num_seqs, max_cp_degree)
    # cp_num_expanded = cp_num.unsqueeze(-1)
    # mask = mask >= cp_num_expanded
    # cp_dst[mask] = -1

    # check: cp_dst[src_rank, seq_id, shard_id] >= 0 if shard_id < cp_num[src_rank, seq_id]
    def check_cp_dst():
        for i in range(world_size):
            for j in range(num_seqs):
                num_cp = int((cp_num[i, j]).item())
                for k in range(max_cp_degree):
                    if k < num_cp:
                        assert cp_dst[i, j, k] >= 0, f"cp_dst[{i}, {j}, {k}] = {cp_dst[i, j, k]} is not >= 0"
                    else:
                        assert cp_dst[i, j, k] == -1, f"cp_dst[{i}, {j}, {k}] = {cp_dst[i, j, k]} is not -1"
        return
    
    check_cp_dst()
    print_if_verbose("cp_dst =", cp_dst)

    # Prepare the sequence length for each shard
    # seq_shard_lens = torch.zeros((world_size, num_seqs, max_cp_degree), dtype=torch.int64)
    # for i in range(world_size):
    #     for j in range(num_seqs):
    #         num_cp = int((cp_num[i, j]).item())
    #         seq_len = seq_lens[i, j]
    #         seq_shard_lens[i, j, :num_cp] = seq_len // num_cp
    def check_seq_shard_lens():
        for i in range(world_size):
            for j in range(num_seqs):
                # Check if the shard length is valid.
                num_cp = int((cp_num[i, j]).item())
                for k in range(max_cp_degree):
                    if k < num_cp:
                        assert seq_shard_lens[i, j, k] > 0, f"seq_shard_lens[{i}, {j}, {k}] = {seq_shard_lens[i, j, k]} is not >= 0"
                    else:
                        assert seq_shard_lens[i, j, k] == 0, f"seq_shard_lens[{i}, {j}, {k}] = {seq_shard_lens[i, j, k]} is not 0"
                # Check the sum of this sequence matches seq_len[i, j]
                assert seq_shard_lens[i, j, :num_cp].sum() == seq_lens[i, j], f"seq_shard_lens[{i}, {j}, :{num_cp}].sum() = {seq_shard_lens[i, j, :num_cp].sum()} is not equal to seq_lens[{i}, {j}] = {seq_lens[i, j]}"
        return
    
    check_seq_shard_lens()
    print_if_verbose("seq_shard_lens =", seq_shard_lens)

    # q_global_dispatch tensor:
    num_cp_shards = cp_num.sum(dim=1)
    pad_len = torch.max(num_cp_shards)
    print_if_verbose("num_cp_shards =", num_cp_shards)

    cp_seq_lens = torch.zeros(world_size, pad_len, dtype=torch.int64)
    cp_query_dst = torch.ones(world_size, pad_len, dtype=torch.int64) * -1
    kv_to_q_mapping = torch.ones((world_size, pad_len, max_cp_degree, 2), dtype=torch.int64) * -1
    kv_to_q_rank = torch.ones((world_size, pad_len, max_cp_degree), dtype=torch.int64) * -1
    kv_context_size = torch.zeros((world_size, pad_len), dtype=torch.int64)
    q_to_num_kv_seq = torch.zeros((world_size, pad_len), dtype=torch.int64)


    # cumulative number of cp shards before this one.
    num_cul_cp_shards = exclusive_cumsum(cp_num, dim=1)
    print_if_verbose("num_cul_cp_shards =", num_cul_cp_shards)

    # breakpoint()

    for i in range(world_size):
        cp_seq_lens_local = []
        cp_query_dst_local = []
        kv_to_q_mapping_local = []
        kv_to_q_rank_local = []
        kv_context_size_local = []
        q_to_num_kv_seq_local = []

        for j in range(num_seqs):
            num_cp = int((cp_num[i, j]).item())
            seq_len = seq_lens[i, j]
            if seq_len == 0:
                break
            # seq_shard_len = seq_len // num_cp
            _seq_shard_len = seq_shard_lens[i, j, :num_cp]
            try:
                _kv_context_size_seq = exclusive_cumsum(_seq_shard_len, dim=0)
            except Exception as e:
                breakpoint()

            cp_seq_lens_local.append(_seq_shard_len)
            cp_query_dst_local.append(cp_dst[i, j, :num_cp].flatten())
            #### Compute kv_to_q_mapping.
            row_indices = torch.arange(num_cp).view(-1, 1)
            col_indices = torch.arange(max_cp_degree).view(1, -1)
            mask = col_indices < (num_cp - row_indices)
            kv_to_q_mapping_seq = torch.empty((num_cp, max_cp_degree, 2), dtype=torch.int64)
            # All q shards are on this node (TODO: we are testing MLP-DP. For MLP-CP, this is different).
            kv_to_q_mapping_seq[..., 0] = torch.where(mask, i, -1)
            vals_ch1 = row_indices + col_indices + num_cul_cp_shards[i, j]
            kv_to_q_mapping_seq[..., 1] = torch.where(mask, vals_ch1, -1)
            kv_to_q_mapping_local.append(kv_to_q_mapping_seq)
            #### Compute kv_to_q_rank (Index of this KV to the query's dst).
            kv_to_q_rank_seq = torch.arange(num_cp).view(-1, 1).repeat(1, max_cp_degree) * mask + (mask.int() - 1)
            kv_to_q_rank_local.append(kv_to_q_rank_seq)
            #### Compute kv context size (For this kv, how many tokens are in the context).
            kv_context_size_seq = _kv_context_size_seq
            kv_context_size_local.append(kv_context_size_seq)
            #### Compute q_to_num_kv_seq (For this kv, how many shards are in the context).
            q_to_num_kv_seq_seq = torch.arange(num_cp) + 1
            q_to_num_kv_seq_local.append(q_to_num_kv_seq_seq)

        cp_seq_lens_local = torch.cat(cp_seq_lens_local, dim=0)
        cp_query_dst_local = torch.cat(cp_query_dst_local, dim=0)
        kv_to_q_mapping_local = torch.cat(kv_to_q_mapping_local, dim=0)
        kv_to_q_rank_local = torch.cat(kv_to_q_rank_local, dim=0)
        kv_context_size_local = torch.cat(kv_context_size_local, dim=0)
        q_to_num_kv_seq_local = torch.cat(q_to_num_kv_seq_local, dim=0)
        # shape check:
        seq_shards = cp_seq_lens_local.shape[0]
        assert cp_seq_lens_local.shape == (seq_shards,)
        assert cp_query_dst_local.shape == (seq_shards,)
        assert kv_to_q_mapping_local.shape == (seq_shards, max_cp_degree, 2)
        assert kv_to_q_rank_local.shape == (seq_shards, max_cp_degree)
        assert kv_context_size_local.shape == (seq_shards,)
        assert q_to_num_kv_seq_local.shape == (seq_shards,)

        cp_seq_lens[i, :seq_shards] = cp_seq_lens_local
        cp_query_dst[i, :seq_shards] = cp_query_dst_local
        kv_to_q_mapping[i, :seq_shards] = kv_to_q_mapping_local
        kv_to_q_rank[i, :seq_shards] = kv_to_q_rank_local
        kv_context_size[i, :seq_shards] = kv_context_size_local
        q_to_num_kv_seq[i, :seq_shards] = q_to_num_kv_seq_local

    q_to_num_kv_tokens = kv_context_size + cp_seq_lens

    print_if_verbose("cp_seq_lens =", cp_seq_lens)
    print_if_verbose("cp_query_dst =", cp_query_dst)
    print_if_verbose("kv_to_q_mapping =", kv_to_q_mapping)
    print_if_verbose("kv_to_q_rank =", kv_to_q_rank)
    print_if_verbose("kv_context_size =", kv_context_size)
    print_if_verbose("q_to_num_kv_seq =", q_to_num_kv_seq)
    print_if_verbose("q_to_num_kv_tokens =", q_to_num_kv_tokens)

    # TODO: Use the compute_metadata_e2e() instead.
    fwd_q_metadata, rev_q_metadata, q_intermediates = compute_metadata(
        cp_seq_lens, cp_query_dst, return_intermediate=True
    )
    _, q_seq_to_dst, _ = q_intermediates
    fwd_k_metadata, rev_k_metadata, kv_intermediates = compute_metadata_kv(
        kv_to_q_mapping, kv_to_q_rank, kv_context_size, q_to_num_kv_seq,
        q_to_num_kv_tokens, cp_seq_lens, num_cp_shards, cp_query_dst,
        q_seq_to_dst.squeeze(2), pad_len,
        return_intermediate=True
    )
    attention_metadata = compute_attn_layout_seqlens(
        cp_seq_lens, q_to_num_kv_tokens, cp_query_dst, shard_to_tuple=True
    )
    ret = (
        fwd_q_metadata, rev_q_metadata, fwd_k_metadata, rev_k_metadata, attention_metadata
    )
    print_if_verbose("fwd_q_metadata =", fwd_q_metadata)
    print_if_verbose("rev_q_metadata =", rev_q_metadata)
    print_if_verbose("fwd_k_metadata =", fwd_k_metadata)
    print_if_verbose("rev_k_metadata =", rev_k_metadata)
    print_if_verbose("attention_metadata =", attention_metadata)

    if return_intermediate:
        intermediates = q_intermediates + kv_intermediates
        ret += (intermediates,)
    if return_mlp_no_shard_seq_lens:
        ret += (seq_lens,)
    return ret





# FIXME: remove this function.
def create_fast_a2a_metadata_from_qkv_dispatch(
    fwd_q_metadata, rev_q_metadata, fwd_k_metadata, rev_k_metadata, intermediates, element_size: int, hidden_size_q: int, hidden_size_k: int, mlp_total_token: int,
    create_attn_outs: bool=False
):
    (qkv_fwd_fa2a_metadata, qkv_rev_fa2a_metadata,
     attn_out_fwd_fa2a_metadata, attn_out_rev_fa2a_metadata,
    ) = compute_fa2a_metadata_from_logical_metadata(
        fwd_q_metadata, rev_q_metadata, fwd_k_metadata, rev_k_metadata,
        intermediates, mlp_total_token, hidden_size_q, hidden_size_k, element_size,
    )
    if create_attn_outs:
        return (qkv_fwd_fa2a_metadata, qkv_rev_fa2a_metadata,
                attn_out_fwd_fa2a_metadata, attn_out_rev_fa2a_metadata,)
    return (qkv_fwd_fa2a_metadata, qkv_rev_fa2a_metadata)


@torch.no_grad()
def orchestrate_simulate(
    tensor: torch.Tensor, output_tensor: torch.Tensor, metadata: Metadata
):
    """Simulate a communication based on the metadata."""
    assert tensor.dim() == 3    # (world_size, num_tokens, hidden_dim)
    world_size = tensor.shape[0]
    # handle sending rank-by-rank:
    for src_rank in range(world_size):
        dst_rank = metadata.dst_rank[src_rank]
        dst_offset = metadata.dst_offset[src_rank]
        seq_lens = metadata.seq_len[src_rank]
        acu_tokens = 0
        for j, rs in enumerate(dst_rank):
            seq_len = seq_lens[j]
            seq = tensor[src_rank][acu_tokens:acu_tokens + seq_len]
            if dst_rank.dim() == 1:
                rank = rs
                if rank >= 0:
                    try:
                        output_tensor[rank][dst_offset[j]: dst_offset[j] + seq_len] = seq
                    except RuntimeError as e:
                        print(f"{src_rank=}, {rank=}, {dst_offset[j]=}, {dst_offset[j] + seq_len=}, {seq_len=}, {output_tensor.shape, seq.shape, acu_tokens, tensor.shape}")
                        raise e
            else:
                for k, rank in enumerate(rs):
                    if rank >= 0:
                        output_tensor[rank][dst_offset[j][k]: dst_offset[j][k] + seq_len] = seq
            acu_tokens += seq_len
    return output_tensor


def simulate_communication(tensors: list[torch.Tensor], metadata: Metadata):
    """
    Simulate a communication based on the metadata, but with all input paddings
    already removed.
    """
    world_size = len(tensors)
    assert world_size == metadata.world_size
    output_seq_len = int(metadata.num_recv_tokens.max().item())
    input_pad_len = max(tensor.shape[0] for tensor in tensors)
    pad_tensors = [
        torch.cat([
            tensor,
            torch.zeros(
                (input_pad_len - tensor.shape[0], *tensor.shape[1:]),
                dtype=tensor.dtype, device=tensor.device
            )
        ], dim=0).unsqueeze(0) for tensor in tensors
    ]
    input_tensor = torch.cat(pad_tensors, dim=0)
    output_tensor = torch.zeros(
        (world_size, output_seq_len, *input_tensor.shape[2:]),
        dtype=input_tensor.dtype, device=input_tensor.device
    )
    output_tensor = orchestrate_simulate(
        input_tensor.reshape(world_size, input_pad_len, -1),
        output_tensor.reshape(world_size, output_seq_len, -1),
        metadata
    ).reshape(world_size, output_seq_len, *input_tensor.shape[2:])
    output_tensors = torch.split(output_tensor, 1, dim=0)
    output_tensors_split = [
        t[0, :metadata.num_recv_tokens[rank].max()] for rank, t in enumerate(output_tensors)
    ]
    return output_tensors_split
