from dataclasses import dataclass
import os
import socket
from typing import Optional
import math


from megatron.core import parallel_state as mpu
import ray
import torch

from d2.runtime.attn_kernels.ops import nvshmem_get_unique_id, nvshmem_alloc_empty_unique_id, FastDispatcherWrapper
from d2.runtime.inplace_metadata import (
    Metadata, compute_attn_layout_seqlens, compute_metadata, compute_metadata_kv,
    exclusive_cumsum
)
from d2.runtime.fast_alltoall_metadata import  compute_fa2a_metadata_from_logical_metadata


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

    def init_comm(self, buffer_size: int, local_rank: int = None):
        # Init megatron communication.
        if local_rank is None:
            local_rank = int(os.getenv("LOCAL_RANK"))

        if not torch.distributed.is_initialized():
            torch.distributed.init_process_group(backend="cpu:gloo,cuda:nccl", rank=self.rank, world_size=self.world_size)

        if self.rank == 0:
            uid = nvshmem_get_unique_id()
        else:
            uid = nvshmem_alloc_empty_unique_id()
        torch.distributed.broadcast(uid, src=0)

        FastDispatcherWrapper.init(
            self.rank, local_rank, self.world_size, buffer_size, uid
        )

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


class MegatronBaseWorker(BaseWorker):
    """Worker base class to init communication groups (megatron and nvshmem)."""

    def init_comm(self, buffer_size: int, parallel_config: ParallelConfig, local_rank: Optional[int] = None):
        # Init megatron communication.
        super().init_comm(buffer_size, local_rank)
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


def create_qkv_dispatch(
    world_size: int, total_seq_len: int, num_seqs: int, max_cp_degree: int,
    return_intermediate: bool=False, return_mlp_no_shard_seq_lens: bool=False
):
    """NOTE: this is currently a dispatch tensor of not consider the 2CP optimization."""
    # init sequence
    assert total_seq_len % (max_cp_degree) == 0
    _num_tokens_shard = total_seq_len // (max_cp_degree)
    seq_lens = gen_seq_lens(world_size, num_seqs, _num_tokens_shard).long()
    # make sure each sequence is divisible by max_cp_degree.
    seq_lens *= max_cp_degree

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

# This function can be merged with gen_seq_lens later. Seperate for now to avoid DP test.
# Output:[cp_group][cp_degree * seq]. Sequence should be equally divided in CP group.
def gen_cp_seq_lens(world_size: int, num_seqs: int, total_len: int, cp_degree: int=1) -> torch.Tensor:
    if world_size % cp_degree != 0:
        raise ValueError("cp_degree must divide world_size")
    num_cp_group = world_size // cp_degree
    # For one sequence, each cp group has similar length of the sequence. For cp_degree=1, this also works.
    ratio = torch.rand((num_cp_group, num_seqs)) + 0.25 / num_seqs   # Use a min value to guarantee that the sequence is not too short (0 after rounding)
    ratio = ratio / ratio.sum(dim=1, keepdim=True)
    seq_lens = (ratio * total_len).round().int()
    seq_len_total = seq_lens.sum(dim=1)
    seq_len_total_error = seq_len_total - total_len
    seq_lens[:, -1] -= seq_len_total_error
    seq_lens = seq_lens.repeat_interleave(cp_degree, dim=0)  # [num_cp_group * cp_degree, num_seqs]

    # seq_lens = seq_lens.view(num_cp_group * cp_degree, num_seqs)   # [num_cp_group, cp_degree, num_seqs]
    return seq_lens



# TODO(Pb): Solved
# Do we need max_cp_degree for CP dispatch? Solved. For this test, we use max_cp_degree = cp_degree.
# We will create another test when extreme long sequences dispatch across max_cp_degree.
def create_cp_qkv_dispatch(
    world_size: int, total_seq_len: int, num_seqs: int, max_cp_degree: int, cp_degree: int, shards_per_cp: int=2,
    return_intermediate: bool=False
):
    # init sequence. On default, each cp core own 2 shards. ([0]th and [2*CP-1]th). Can we hardcode shards_per_cp = 2! Yes.
    assert total_seq_len % (max_cp_degree * shards_per_cp) == 0
    assert max_cp_degree == cp_degree
    assert shards_per_cp == 2   # Only support head-tail CP.

    _num_tokens_shard = total_seq_len // (max_cp_degree * shards_per_cp)
    num_cp_group = world_size // cp_degree
    seq_lens = gen_cp_seq_lens(world_size, num_seqs, _num_tokens_shard, cp_degree).long()   #[cp_group_size, cp_degree, num_seqs]
    seq_lens *= max_cp_degree * shards_per_cp

    
    # CP degree. All Sequences should own the same CP_Degree.(Different from DP, where we random the CP degree for each sequence.)
    # Currently, each sequence use cp = cp_degree.
    # TODO(Pb): Confirm each sequence use [cp = cp_degree] or [cp = max_cp_degree].
    # Tradeoff:
    #   cp = cp_degree. Less communication(Q stay still). But unbalanced flops cross cp_group.
    #   cp = cp_max. Balanced flops. But more communication.(Dispatch Q.)
    
    
    # log_cp_num = torch.randint(0, int(math.log2(max_cp_degree)) + 1, (world_size, num_seqs))
    # cp_num = torch.pow(2, log_cp_num)

    # seq_cp_degree = torch.ones((num_cp_group, cp_degree, num_seqs)) * cp_degree   # TODO(pb): Maybe we don't need this. Delete later.
    
    # cp_num = seq_cp_degree.reshape(num_cp_group * cp_degree, num_seqs)  # number of shards for each sequence.
    cp_num = torch.ones((num_cp_group * cp_degree, num_seqs)) * shards_per_cp

    # init cp send destination. Shards all computed on current cp_core.
    rank_ids = torch.arange(num_cp_group * cp_degree).reshape(num_cp_group, cp_degree)
    cp_dst = rank_ids.unsqueeze(-1).unsqueeze(-1).expand(num_cp_group, cp_degree, num_seqs, shards_per_cp)  # which rank, this shard go to.

    cp_dst = cp_dst.view(num_cp_group * cp_degree, num_seqs, shards_per_cp)
    
    # cp_dst_helper = torch.rand((world_size, num_seqs, world_size)).argsort(dim=2)
    # cp_dst = cp_dst_helper[:, :, :max_cp_degree]
    # mask = torch.arange(max_cp_degree).expand(world_size, num_seqs, max_cp_degree)
    # cp_num_expanded = cp_num.unsqueeze(-1)
    # mask = mask >= cp_num_expanded
    # cp_dst[mask] = -1

    # cumsum cp_shards number for each rank. Should be: num_seq * shards_per_cp
    num_cp_shards = cp_num.sum(dim=1)
    
    pad_len = int(torch.max(num_cp_shards))      # We don' t need this for cp. Because all rank has the same number of shards.
    total_shard_per_seq = num_seqs * cp_degree  * shards_per_cp     # total shards for all sequence in one cp_group.
    pad_len = total_shard_per_seq       # This total shards for a complete sequence.
    pad_len = int(torch.max(num_cp_shards))     # Total shards for a rank, for a sequence.

    cp_seq_lens = torch.zeros(world_size, pad_len, dtype=torch.int64)       # Sequence length for each shards.
    cp_query_dst = torch.ones(world_size, pad_len, dtype=torch.int64) * -1  # destination for each shards.
    # kv to q for each shards.
    kv_to_q_mapping = torch.ones((world_size, pad_len, max_cp_degree * shards_per_cp, 2), dtype=torch.int64) * -1
    kv_to_q_rank = torch.ones((world_size, pad_len, max_cp_degree * shards_per_cp ), dtype=torch.int64) * -1
    kv_context_size = torch.zeros((world_size, pad_len), dtype=torch.int64)
    q_to_num_kv_seq = torch.zeros((world_size, pad_len), dtype=torch.int64)

    # cumulative number of cp shards before this one.
    num_cul_cp_shards = exclusive_cumsum(cp_num, dim=1)
    # cp_location. This can be difference in the future.
    max_num_shard = max_cp_degree * shards_per_cp
    shard_location = torch.zeros(num_seqs, max_num_shard, 2)
    max_shard_index = torch.arange(max_num_shard)
    rank_id = torch.minimum(max_shard_index, 2 * cp_degree - 1 - max_shard_index)               # [0,1,2,3,3,2,1,0]
    shard_id = torch.arange(shards_per_cp).repeat_interleave(max_num_shard // shards_per_cp)    # [0,0,0,0,1,1,1,1]
    shard_location[..., 0] = rank_id
    shard_location[..., 1] = shard_id

    
    for i in range(num_cp_group):   # Dispatch happen inside CP group.
        cp_seq_lens_local = []
        cp_query_dst_local = []
        kv_to_q_mapping_local = []
        kv_to_q_rank_local = []
        kv_context_size_local = []
        q_to_num_kv_seq_local = []

        for j in range(num_seqs):
            # TODO: Discuss with Junda, how to locate a sequence. 
            # For cp, we can't locate a sequence just use rank_id.
            num_cp = cp_degree #int((cp_num[i, j]).item())         # cp degree for current sequence.

            seq_len = sum(seq_lens[:cp_degree, j])       # sum sequence length for current sequence. 
            seq_shard_len = seq_len // (num_cp * shards_per_cp)       # shard length for current sequence.
            #num_shards = num_cp * shards_per_cp
            num_shards = num_cp * shards_per_cp
            
            #### Compute cp_seq_lens.      
            cp_seq_lens_seq = seq_shard_len.reshape(1,).repeat(num_shards).view(cp_degree, shards_per_cp)
            cp_seq_lens_local.append(cp_seq_lens_seq)      # shape: [cp_degree, shards_per_cp]

            # [rank_range, seq_id, all_shards]
            cp_q_dst = cp_dst[i*cp_degree:(i+1)*cp_degree, j, :].flatten().view(cp_degree, shards_per_cp)
            cp_query_dst_local.append(cp_q_dst)    # [num_cp_group * cp_degree, num_seqs, shards_per_cp] = [ws, num_seqs, shards]
            # cp_query_dst_local.append(cp_dst[i, j, :num_cp].flatten())  #  [ws, seq_len, shards]
            

            # # All q shards are on this node (TODO: we are testing MLP-DP. For MLP-CP, this is different).
            
            even_indices = torch.arange(0, max_cp_degree * shards_per_cp, 2)        # [0,2,4,6] => [0,1,2,3]
            odd_indices = torch.arange(1, max_cp_degree * shards_per_cp + 1, 2)     # [1,3,5,7] => [7,6,5,4]
            # torch.where(mask[:], i, -1)
            # kv_to_q_mapping_seq[even_indices, :, 0] = torch.where(mask[:num_cp][:], i, -1)  # [0,1,2,3]
            # kv_to_q_mapping_seq[odd_indices, :, 0] = torch.where(mask[num_cp:][:], i, -1).flip(0)   #[7,6,5,4]
            # #kv_to_q_mapping_seq[..., 0] = torch.where(mask, i, -1)
            
            # vals_ch1 = row_indices + col_indices + num_cul_cp_shards[i, j].int()    # convert float to int
            
            # #kv_to_q_mapping_seq[..., 1] = torch.where(mask, vals_ch1, -1)
            # kv_to_q_mapping_seq[even_indices, :, 1] = torch.where(mask, vals_ch1, -1)[:num_cp]  # [0,1,2,3]
            # kv_to_q_mapping_seq[odd_indices, :, 1] = torch.where(mask, vals_ch1, -1)[num_cp:].flip(0)   #[7,6,5,4]
            # kv_to_q_mapping_local.append(kv_to_q_mapping_seq)
            
            
            #### Compute kv_to_q_mapping.
            kv_to_q_mapping_seq = torch.empty((num_shards, max_num_shard, 2), dtype=torch.int64)
            row_indices = torch.arange(num_shards).view(-1, 1)
            col_indices = torch.arange(max_num_shard).view(1, -1)
            mask = col_indices < (num_shards - row_indices)
            kv_to_q_idx_mapping_origin = (row_indices + col_indices) * mask + (mask.int() - 1)
            kv_to_q_idx_mapping = torch.empty_like(kv_to_q_idx_mapping_origin)

            
            kv_to_q_idx_mapping[even_indices] = kv_to_q_idx_mapping_origin[:num_cp]
            kv_to_q_idx_mapping[odd_indices] = kv_to_q_idx_mapping_origin[num_cp:].flip(0)

            #max_shard_index = row_indices.expand(num_shards, max_num_shard)
            #masked_q_shard_idx = mask * max_shard_index + (mask.int() - 1)
            invalid = (kv_to_q_idx_mapping == -1).unsqueeze(-1)
            valid_values = shard_location[j][kv_to_q_idx_mapping]
            fill_neg = torch.tensor([-1, -1], dtype=valid_values.dtype)
            result = torch.where(invalid, fill_neg, valid_values).int()
            kv_to_q_mapping_seq[..., :] = result
            
            kv_to_q_mapping_seq = kv_to_q_mapping_seq.view(cp_degree, shards_per_cp, *kv_to_q_mapping_seq.shape[1:])

            kv_to_q_mapping_local.append(kv_to_q_mapping_seq)
            

            #### Compute kv_to_q_rank (Index of this KV to the query's dst).
            kv_to_q_rank_seq = torch.arange(num_shards).view(-1, 1).repeat(1, max_cp_degree * shards_per_cp) * mask + (mask.int() - 1)
            kv_to_q_rank_seq[even_indices, :] = kv_to_q_rank_seq[:num_cp].clone()
            kv_to_q_rank_seq[odd_indices, :] = kv_to_q_rank_seq[num_cp:].flip(0).clone()
            kv_to_q_rank_seq = kv_to_q_rank_seq.view(cp_degree, shards_per_cp, *kv_to_q_rank_seq.shape[1:])
            kv_to_q_rank_local.append(kv_to_q_rank_seq)

            #### Compute kv context size (For this kv, how many tokens are in the context).
            continuous_kv_context = torch.arange(num_shards) * seq_shard_len
            kv_context_size_seq = torch.empty(num_shards, dtype=torch.long)
            kv_context_size_seq[even_indices] = continuous_kv_context[:num_cp]
            kv_context_size_seq[odd_indices] = continuous_kv_context[num_cp:].flip(0)
            kv_context_size_seq = kv_context_size_seq.view(cp_degree, shards_per_cp)
            kv_context_size_local.append(kv_context_size_seq)

            #### Compute q_to_num_kv_seq (For this kv, how many shards are in the context).
            continuous_q_to_num_kv_seq = torch.arange(num_shards) + 1
            q_to_num_kv_seq_seq = torch.empty(num_shards, dtype=torch.long)
            q_to_num_kv_seq_seq[even_indices] = continuous_q_to_num_kv_seq[:num_cp]
            q_to_num_kv_seq_seq[odd_indices] = continuous_q_to_num_kv_seq[num_cp:].flip(0)
            q_to_num_kv_seq_seq = q_to_num_kv_seq_seq.view(cp_degree, shards_per_cp)
            q_to_num_kv_seq_local.append(q_to_num_kv_seq_seq)
        
        cp_seq_lens_local = torch.cat(cp_seq_lens_local, dim=1)             # Before Concat: [rank_id, 2]. After concat: [rank_id, 2 * num_seq]
        cp_query_dst_local = torch.cat(cp_query_dst_local, dim=1)           # Before Concat: [rank_id, 2]. After concat: [rank_id, 2 * num_seq]
        kv_to_q_mapping_local = torch.cat(kv_to_q_mapping_local, dim=1)     # Before Concat: [rank_id, 2, total_q_shards_id, 2]. After Concat: [rank_id, 2 * num_seq, total_q_shards_id, 2]
        kv_to_q_rank_local = torch.cat(kv_to_q_rank_local, dim=1)           # Before Concat: [rank_id, 2, total_q_shards_id]. After concat: [rank_id, 2 * num_seq, total_q_shards_id]
        kv_context_size_local = torch.cat(kv_context_size_local, dim=1)     # Before Concat: [rank_id, 2]. After concat: [rank_id, 2 * num_seq]
        q_to_num_kv_seq_local = torch.cat(q_to_num_kv_seq_local, dim=1)     # Before Concat: [rank_id, 2]. After concat: [rank_id, 2 * num_seq]

        # shape check:
        # Shards per rank [num_seq * [cp_degree, 2]]
        
        seq_shards = cp_seq_lens_local.shape[1]
        max_cp_shards_per_seq = cp_degree * shards_per_cp
        assert cp_seq_lens_local.shape == (cp_degree, seq_shards,)
        assert cp_query_dst_local.shape == (cp_degree, seq_shards,)
        assert kv_to_q_mapping_local.shape == (cp_degree, seq_shards, max_cp_shards_per_seq, 2)
        assert kv_to_q_rank_local.shape == (cp_degree, seq_shards, max_cp_shards_per_seq)
        assert kv_context_size_local.shape == (cp_degree, seq_shards,)
        assert q_to_num_kv_seq_local.shape == (cp_degree, seq_shards,)
        
        cp_seq_lens[i*cp_degree:(i+1)*cp_degree, :seq_shards] = cp_seq_lens_local
        cp_query_dst[i*cp_degree:(i+1)*cp_degree, :seq_shards] = cp_query_dst_local
        kv_to_q_mapping[i*cp_degree:(i+1)*cp_degree, :seq_shards] = kv_to_q_mapping_local
        kv_to_q_rank[i*cp_degree:(i+1)*cp_degree, :seq_shards] = kv_to_q_rank_local
        kv_context_size[i*cp_degree:(i+1)*cp_degree, :seq_shards] = kv_context_size_local
        q_to_num_kv_seq[i*cp_degree:(i+1)*cp_degree, :seq_shards] = q_to_num_kv_seq_local

    q_to_num_kv_tokens = kv_context_size + cp_seq_lens
    
    fwd_q_metadata, rev_q_metadata, q_intermediates = compute_metadata(
        cp_seq_lens, cp_query_dst, return_intermediate=True
    )
    _, q_seq_to_dst, _ = q_intermediates
    
    pad_len = int(torch.max(num_cp_shards))  # hardcode here. means number of shards per rank?
    fwd_k_metadata, rev_k_metadata, kv_intermediates = compute_metadata_kv(
        kv_to_q_mapping, kv_to_q_rank, kv_context_size, q_to_num_kv_seq,
        q_to_num_kv_tokens, cp_seq_lens, num_cp_shards, cp_query_dst,
        q_seq_to_dst.squeeze(2), pad_len,
        return_intermediate=True
    )
    attention_metadata = compute_attn_layout_seqlens(
        cp_seq_lens, q_to_num_kv_tokens, cp_query_dst
    )
    if return_intermediate:
        intermediates = q_intermediates + kv_intermediates
        return (
            fwd_q_metadata, rev_q_metadata, fwd_k_metadata, rev_k_metadata,
            attention_metadata, intermediates
        )
    return (
        fwd_q_metadata, rev_q_metadata, fwd_k_metadata, rev_k_metadata, attention_metadata
    )


def test_gen_cp_seq_lens():
    # Example test for gen_cp_seq_lens
    world_size = 8
    num_seqs = 4
    total_len = 64
    cp_degree = 4
    max_cp_degree = 4
    # total token 64; shard length : 64/(2*4) = 8; cp_group has 4 cores, each core has 2 shards.
    fwd_q_metadata, rev_q_metadata, fwd_k_metadata, rev_k_metadata, attention_metadata = create_cp_qkv_dispatch(world_size, total_len, num_seqs, max_cp_degree, cp_degree)
    print("fwd_q_metadata.seq_len.shape:", fwd_q_metadata.seq_len.shape)
    print("rev_q_metadata.seq_len.shape:", rev_q_metadata.seq_len.shape)
    print("fwd_k_metadata.seq_len.shape:", fwd_k_metadata.seq_len.shape)
    print("rev_k_metadata.seq_len.shape:", rev_k_metadata.seq_len.shape)
    print("attention_metadata[0].shape:", attention_metadata[0].shape if isinstance(attention_metadata, (list, tuple)) else attention_metadata.shape)

    
if __name__ == "__main__":
    test_gen_cp_seq_lens()