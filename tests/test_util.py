from dataclasses import dataclass
import logging
import math
import os
import random
import socket
from typing import Optional


from megatron.core import parallel_state as mpu
from transformers import AutoConfig
import ray
import rich
import torch

from d2.runtime.attn_kernels.ops import (
    nvshmem_get_unique_id, nvshmem_alloc_empty_unique_id, FastDispatcherWrapper
)
from d2.runtime.compute_metadata import (
    from_planner_output, backward_from_planner_output,
)
from d2.runtime.fast_alltoall_metadata import (
    compute_backward_resend_e2e_metadata,
    compute_e2e_fa2a_metadata,
)
from d2.runtime.inplace_metadata import (
    compute_attn_layout_seqlens, compute_metadata,
    compute_metadata_kv, exclusive_cumsum
)
from d2.runtime.shard_info import ShardInfo
from d2.runtime.megatron_patch.create_group import (
    initialize_attention_server_comm, get_attn_server_group_gloo,
    get_attn_server_rank, get_attn_server_group_src_rank
)
from d2.planner.planner import batch_to_items_with_dummy

logger = logging.getLogger(__name__)


######## MISC
def get_device_name() -> str:
    return "cuda"


def get_torch_device() -> any:
    """Return the corresponding torch attribute based on the device type string.
    Returns:
        module: The corresponding torch device namespace, or torch.cuda if not found.
    """
    device_name = get_device_name()
    try:
        return getattr(torch, device_name)
    except AttributeError:
        logger.warning(f"Device namespace '{device_name}' not found in torch, try to load torch.cuda.")
        return torch.cuda


def set_random_seed(seed, set_megatron: bool=True):
    """Set worker side random seed."""
    import random

    import numpy as np
    import torch

    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if get_torch_device().device_count() > 0 and set_megatron:
        from megatron.core import tensor_parallel

        tensor_parallel.model_parallel_cuda_manual_seed(seed)


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
            torch.cuda.set_device(self.device)

    def init_nvshmem(self, buffer_size: int, local_rank: int = None):
        print("====== init_nvshmem ======")
        if self.rank == 0:
            uid = nvshmem_get_unique_id()
            print(f"Init uid = {uid}")
        else:
            uid = nvshmem_alloc_empty_unique_id()
        print(f"Broadcast uid: {uid}")
        torch.distributed.broadcast(uid, src=0)
        print(f"Rank {self.rank} uid = {uid}")
        
        print("FastDispatcherWrapper.init")
        FastDispatcherWrapper.init(
            self.rank, local_rank, self.world_size, buffer_size, uid
        )
        print("====== init_nvshmem done ======")

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
        print(f"[Rank {as_rank}] init nvshmem with uid = {uid}")
        torch.distributed.broadcast(uid, src=as_src_rank, group=group)
        print(f"[Rank {as_rank}] after broadcast uid = {uid}")
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
def create_list(n: int, s: int, min_val: int, t: int) -> list[int] | None:
    """
    Generates a list of n integers that sum to s.

    Each integer in the list must be:
    - in range of [min_val, s).
    - Divisible by t.

    Returns:
        A list of integers meeting the criteria, or None if no solution exists.
    """
    # --- 1. Input Validation ---
    assert n > 0
    assert s % t == 0

    # --- 2. Determine Valid Range for Numbers ---
    if t > 1:
        min_val = math.ceil((min_val + 1) / t) * t
    assert s >= n * min_val

    # --- 3. Construct the List ---
    # Start with a list where every element is the minimum possible value.
    result = [min_val] * n
    remainder = (s - min_val * n) // t
    remain_result = torch.rand((n,))
    remain_result = (remain_result / remain_result.sum() * remainder).to(torch.int)
    # handle rounding error
    if torch.sum(remain_result).item() != remainder:
        diff = remainder - sum(remain_result)
        if diff > 0:
            remain_result[-1] += diff
        else:
            idx = 0
            while remain_result[idx] <= diff: idx += 1
            remain_result[idx] += diff
            assert remain_result[idx] > 0
    remain_result = (remain_result * t).tolist()
    for rid in range(n):
        result[rid] += remain_result[rid]
    assert sum(result) == s

    return result


def create_random_dispatch_from_existing(
    per_rank_shard_lens: list[list[list[int]]], world_size: int, merge_ranks: bool=True,
):
    assert len(per_rank_shard_lens) == world_size
    num_shards = sum(len(doc_shards) for rank_shards in per_rank_shard_lens for doc_shards in rank_shards)
    dsts = torch.concat([
        torch.range(0, world_size - 1, dtype=torch.int32, device="cpu"),
        torch.randint(0, world_size, (num_shards - world_size,), dtype=torch.int32, device="cpu"),
    ], dim=0)
    dsts = dsts[torch.randperm(num_shards, device="cpu")].tolist()
    dsts_iter = iter(dsts)
    outs = []
    for rank in range(world_size):
        rank_outs = []
        for doc_shards in per_rank_shard_lens[rank]:
            doc_outs = []
            for sid, shard_len in enumerate(doc_shards):
                doc_outs.append(
                    ShardInfo(rid=rank, dispatch_rid=next(dsts_iter),
                              logical_sid=sid, shard_len=shard_len)
                )
            rank_outs.append(doc_outs)
        outs.append(rank_outs)
    if merge_ranks:
        outs = [doc_shards for rank_out in outs for doc_shards in rank_out]
    return outs


def create_random_shard_info(
    seed: int, world_size: int, num_doc: int,
    max_num_shard: int, max_shard_len: int=-1, min_shard_len: int=8,
    tot_num_token: int=-1, multiple_of: int=1
):
    set_random_seed(seed, set_megatron=False)
    scheduler_output: list[list[ShardInfo]] = []
    if max_shard_len <= 0:
        assert tot_num_token > 0
        max_shard_len = tot_num_token

    num_shards = torch.randint(1, max_num_shard + 1, (num_doc,)).tolist()
    has_shard_src = [False] * world_size
    has_shard_dst = [False] * world_size
    src_num_token = [0] * world_size
    for doc_id in range(num_doc):
        num_shard = num_shards[doc_id]
        doc_schedule = []
        for shard_id in range(num_shard):
            rid = random.randint(0, world_size - 1)
            d_rid = random.randint(0, world_size - 1)
            has_shard_src[rid] = True
            has_shard_dst[d_rid] = True
            shard_len = random.randint(min_shard_len, max_shard_len)
            src_num_token[rid] += shard_len
            doc_schedule.append(
                ShardInfo(rid=rid, dispatch_rid=d_rid, logical_sid=shard_id, shard_len=shard_len)
            )
        scheduler_output.append(doc_schedule)
    for rank in range(world_size):
        if not has_shard_src[rank]:
            scheduler_output.append([ShardInfo(rid=rank, dispatch_rid=rank, logical_sid=0, shard_len=min_shard_len)])
            has_shard_src[rank] = True
            has_shard_dst[rank] = True
            src_num_token[rank] += min_shard_len
        if not has_shard_dst[rank]:
            scheduler_output.append([ShardInfo(rid=rank, dispatch_rid=rank, logical_sid=0, shard_len=min_shard_len)])
            has_shard_src[rank] = True
            has_shard_dst[rank] = True
            src_num_token[rank] += min_shard_len

    if tot_num_token > 0:
        shards = [[] for _ in range(world_size)]
        for s in scheduler_output:
            for shard in s:
                shards[shard.rid].append(shard)
        for ss in shards:
            num_shards = len(ss)
            shard_lens = create_list(num_shards, tot_num_token, min_shard_len, multiple_of)
            for shard, l in zip(ss, shard_lens):
                shard.shard_len = l
        src_num_token = [tot_num_token] * world_size

    return scheduler_output, src_num_token


def random_shard_info_linear_layout_dp(
    world_size: int, max_num_doc_per_rank: int, tot_num_token: int,
    min_shard_len: int=8, multiple_of: int=1, max_num_shard: int=4, seed: int = None,
    create_backward_redispatch: bool=False,
):
    """
    Create random shard info but guarantee that documents are stored
    in a DP manner on the linear layout.
    For backward redispatch, ideally we should consider that a forward
    shard may be further split into multiple shards during backward,
    but for simplicity we skip this now.
    """
    if seed is not None:
        set_random_seed(seed, set_megatron=False)
    scheduler_output_per_rank: list[list[list[ShardInfo]]] = [[] for _ in range(world_size)]
    has_shard_dst = [False] * world_size
    glob_doc_lens = [[] for _ in range(world_size)]
    for r in range(world_size):
        num_doc = random.randint(1, max_num_doc_per_rank)
        doc_lens = create_list(num_doc, tot_num_token, min_shard_len, multiple_of)
        glob_doc_lens[r] = doc_lens
        for doc_len in doc_lens:
            max_shard = min(max_num_shard, doc_len // min_shard_len)
            num_shard = random.randint(1, max_shard)
            shard_lens = create_list(num_shard, doc_len, min_shard_len, multiple_of)
            shards = [
                ShardInfo(
                    r, dispatch_rid=random.randint(0, world_size - 1),
                    logical_sid=shard_id, shard_len=shard_len
                ) for shard_id, shard_len in enumerate(shard_lens)
            ]
            for s in shards:
                has_shard_dst[s.dispatch_rid] = True
            scheduler_output_per_rank[r].append(shards)

    # guarantee that each rank is at least dst of one shard.
    scheduler_output: list[list[ShardInfo]] = []
    for rank in range(world_size):
        if not has_shard_dst[rank]:
            # 1. find a doc to modify.
            token_updated = False
            for did in range(glob_doc_lens[rank]):
                for shard in scheduler_output_per_rank[rank][did]:
                    if shard.shard_len < 2 * min_shard_len:
                        continue
                    # found the document, modify
                    glob_doc_lens[rank][did] -= min_shard_len
                    shard.shard_len -= min_shard_len
                    token_updated = True
                    break
                if token_updated:
                    break
            # add a new smallest document.
            scheduler_output_per_rank[rank].append([
                ShardInfo(rid=rank, dispatch_rid=rank, logical_sid=0,
                          shard_len=min_shard_len)
            ])
            glob_doc_lens[rank].append(min_shard_len)
        scheduler_output.extend(scheduler_output_per_rank[rank])

    # if need to create a backward redispatch, create a new dispatch_rid.
    if create_backward_redispatch:
        per_rank_shard_lens = [[
                [s.shard_len for s in ds] for ds in so
            ] for so in scheduler_output_per_rank
        ]
        scheduler_output_bwd = create_random_dispatch_from_existing(per_rank_shard_lens, world_size)
        return scheduler_output, glob_doc_lens, scheduler_output_bwd
    else:
        return scheduler_output, glob_doc_lens


def _block_reverse_list(l: list, d: int):
    """
    Blockwise reverse a list:
    return l[-d:0] + l[-2d:-d] + ...
    This is because the backward is the flip of forward in the pp dimension, but
    keep the order in the dp dimension.
    """
    return [item for i in range(len(l), 0, -d) for item in l[max(0, i - d):i]]


def create_pipeline_doclens(
    ref_doc_lens: Optional[list[list[int]]],
    add_dummy: bool,
    is_backward: bool,
    world_size: int,
    total_token_on_rank: int,
    num_docs: int,
    tp_size: int,
    dp_size: int,
    num_batches: int = None,
    use_planner: bool = False,
):
    """
    For a forward tick, its sequence length follows:
        [new_microbatch, last_tick_seq_len[:PP_stage_-1]]

        The new_microbatch is either generated, or a dummy one. (controlled by add_dummy)
        A special case is that, the first tick does not have a previous one.
        In this way, we make all stages' microbatch dummy, except for the first one.

    For a backward tick, its sequence length is the reverse of a forward tick.
    Args:
        ref_shard_lens: None only if this is the first tick in PP. Otherwise, return the shard_lens of last tick.
        add_dummy: add dummy forward microbatches for those pp_ranks == 0 devices
        is_backward: this is a backward seqlens. In this case, it directly flips the seqlens of a corresponding forward
    """
    if is_backward:
        return _block_reverse_list(ref_doc_lens, dp_size)
    # Create new microbatches for the first PP stage
    if add_dummy:
        # Each rank only gets one dummy document, which is of `tp_size`
        # to avoid Sequence Parallel having an issue.
        pp_head_new_doc_len = [[tp_size] for _ in range(dp_size)]
    else:
        assert total_token_on_rank % tp_size == 0, "Sequence Parallel requires total token divisible by tp_size"
        if use_planner == False:
            pp_head_new_doc_len = [
                create_list(random.randint(1, num_docs), total_token_on_rank, tp_size, 1)
                for _ in range(dp_size)
            ]
        else:
            # we need to use GLOBAL_BATCH to get dp number of list.
            #from test_megatron_e2e_pipeline_planner import GLOBAL_BATCH, get_next_batch
            from global_batch_provider import get_next_batch, GLOBAL_BATCH
            print(f"In util.py, before calling get_next_batch: GLOBAL_BATCH is: {GLOBAL_BATCH}")
            if num_batches == None:
                num_batches = dp_size 
            pp_head_new_doc_len = get_next_batch(num_batches)

    #  pp_head_new_doc_len : shape : [dp, num_seqs]  We should sample seq_len here.
    # And add the sampled seq_len to the batch. 
    # Next step is based on the previous batch, move the batch. 
    # Get existing microbatch seqlens
    if ref_doc_lens is not None:
        # Not the first microbatch
        other_pp_doc_len = ref_doc_lens[:-dp_size]
    else:
        dummy_fwd_num = world_size - dp_size
        other_pp_doc_len = [[tp_size] for _ in range(dummy_fwd_num)]
    tick_per_rank_doc_len = pp_head_new_doc_len + other_pp_doc_len
    return tick_per_rank_doc_len


def random_tick_shard_from_doclens(
    per_rank_doc_lens: list[list[int]],
    tp_size: int,   # control the max num shard
    max_cp_degree: int,
):
    world_size = len(per_rank_doc_lens)
    per_rank_shard_lens = []
    for r in range(world_size):
        docs = per_rank_doc_lens[r]
        rank_shards = []
        for doc_len in docs:
            num_shards = min(random.randint(1, max_cp_degree), doc_len // tp_size)
            assert doc_len >= tp_size, f"{r, doc_len, tp_size}"
            rank_shards.append(create_list(num_shards, doc_len, tp_size, 1))
        per_rank_shard_lens.append(rank_shards)
    return create_random_dispatch_from_existing(
        per_rank_shard_lens, world_size,
    )

# This function world size is actually as_world_size.
def create_qkv_dispatch_pipeline_tick(
    world_size: int, total_num_token: int, num_docs: int, max_cp_degree: int,
    hidden_size_q: int, hidden_size_k: int,
    element_size: int, # dtype's size
    softmax_lse_size: int,
    ref_doc_lens: Optional[torch.Tensor],
    add_dummy: bool,
    tp_size: int, dp_size: int,
    num_token_per_rank: int,
    num_batches: int = None,
    use_planner: bool = False
):
    """
    softmax_lse_size (int): size of the softmax_lse tensor when viewed as the dtype,
        should be num_heads_local * fp32.itemsize // element_size.
    """
    create_pp_doclen_kwargs = dict(
        world_size=world_size,
        total_token_on_rank=total_num_token,
        num_docs=num_docs,
        tp_size=tp_size,
        dp_size=dp_size,
    )
    # for perf test. Get doc_lens from GLOBAL_BATCH
    cur_tick_per_rank_doc_lens = create_pipeline_doclens(
        ref_doc_lens, add_dummy, is_backward=False, num_batches = num_batches, use_planner=use_planner,
        **create_pp_doclen_kwargs, 
    )

    if use_planner:
        from d2.planner.planner import Planner
        from types import SimpleNamespace
        pp_size = world_size // dp_size        # in this function, world size is actual as_world_size. pp = world_size // dp_size
        planner = Planner.from_individual_params(tp_size=tp_size,
                                                 pp_size=pp_size,
                                                 dp_size=dp_size,
                                                 world_size=tp_size * pp_size * dp_size,
                                                 hidden_size_q=hidden_size_q,
                                                 hidden_size_k=hidden_size_k)
        # Create a temp_model_config for item initialization.
        temp_model_config = SimpleNamespace(
            hidden_size=hidden_size_q * tp_size,
            num_attention_heads = 1,
            num_key_value_heads = hidden_size_q // hidden_size_k,
            num_hidden_layers = 1
        )
        # Use batch_to_items_with_dummy to handle dummy doc to item.
        items = batch_to_items_with_dummy(batches=cur_tick_per_rank_doc_lens, 
                                          num_tokens_per_rank = num_token_per_rank,
                                          as_world_size=world_size,
                                          model_config=temp_model_config)

        fwd_planner_out = planner.items_to_shardinfo(items)
    else:
        fwd_planner_out = random_tick_shard_from_doclens(
            cur_tick_per_rank_doc_lens, tp_size, max_cp_degree,
        )


    (qkv_linear_to_attn_fa2a, _, out_attn_to_linear_fa2a, _, fwd_attn_metadata,
     ) = from_planner_output(
         world_size, fwd_planner_out, hidden_size_q, hidden_size_k,
         softmax_lse_size, element_size, is_pipeline_tick=True
        )

    bwd_tick_per_rank_doc_lens = create_pipeline_doclens(
        cur_tick_per_rank_doc_lens, add_dummy=False, is_backward=True,
        **create_pp_doclen_kwargs,
    )
    if use_planner:
        items = batch_to_items_with_dummy(bwd_tick_per_rank_doc_lens, temp_model_config)
        bwd_planner_out = planner.items_to_shardinfo(items)
    else:
        bwd_planner_out = random_tick_shard_from_doclens(
            bwd_tick_per_rank_doc_lens, tp_size, max_cp_degree,
        )
    (qkv_resend_and_out_grad_linear_to_attn_fa2a, qkv_grad_attn_to_linear_fa2a,
     bwd_attn_metadata) = backward_from_planner_output(
         world_size, bwd_planner_out, hidden_size_q, hidden_size_k,
         softmax_lse_size, element_size,
     )

    return (fwd_attn_metadata, bwd_attn_metadata,
            qkv_linear_to_attn_fa2a, qkv_grad_attn_to_linear_fa2a,
            out_attn_to_linear_fa2a, qkv_resend_and_out_grad_linear_to_attn_fa2a,
            cur_tick_per_rank_doc_lens,)


######## TODO: deprecate all below
def gen_seq_lens(world_size: int, num_seqs: int, total_len: int) -> torch.Tensor:
    ratio = torch.rand((world_size, num_seqs)) + 0.25 / num_seqs   # Use a min value to guarantee that the sequence is not too short (0 after rounding)
    ratio = ratio / ratio.sum(dim=1, keepdim=True)
    seq_len = (ratio * total_len).round().int()
    seq_len_total = seq_len.sum(dim=1)
    seq_len_total_error = seq_len_total - total_len
    seq_len[:, -1] -= seq_len_total_error
    return seq_len


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
