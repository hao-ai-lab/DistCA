# %%
import torch
import random
import numpy as np
import os
import math
import rich
from dataclasses import dataclass
from typing import Optional
from transformers import AutoConfig
from d2.planner.planner import Planner
from d2.runtime.attn_kernels.ops import FastDispatcherWrapper
from d2.planner.planner import batch_to_items_general, Item
from d2.runtime.attn_kernels.ops import FastDispatcherWrapper

# ------------------------------------
# Global Batch Getter Logic
# ------------------------------------


from typing import Iterable, List, Optional
from d2.simulator.optimizers.samples import sample_wlbllm_docs_upsample, batch_documents, sample_prolong_docs

ITERATION_ID = 0
GLOBAL_BATCH: Optional[Iterable[List[int]]] = None

K = 1024
# TODO(Refactor): Remove this global variable.
iterated_samples = []
modified_batches = []
fa2a_metadata_list = []


def setup_global_batch(
    total_seq_len, 
    up_sample_factor=2,
    elongate_factor=1,
    filter_threshold=64 * 1024,
    filter_ratio=0.90,
    should_add_debug_cases=False,
    change_long_doc_ratio=0.0,
    sample_name='wlbllm',
):
    global GLOBAL_BATCH
    if GLOBAL_BATCH is not None:
        return

    assert elongate_factor > 0, f"elongate_factor: {elongate_factor} must be greater than 0"

    if sample_name == 'wlbllm':
        sample_func = sample_wlbllm_docs_upsample
    elif sample_name == 'prolong':
        sample_func = sample_prolong_docs
    else:
        raise ValueError(f"Invalid sample_name: {sample_name}")

    GLOBAL_BATCH = batch_documents(
        sample_func(
            size=10000,
            filter_threshold=filter_threshold,
            filter_ratio=filter_ratio,
            upsample_long_factor=up_sample_factor,
            elongate_factor=elongate_factor,
            change_long_doc_ratio=change_long_doc_ratio,
        ), max_ctx_length=total_seq_len
    )

    if should_add_debug_cases:
        GLOBAL_BATCH = list(GLOBAL_BATCH)
        manual_case = [
            [total_seq_len],
            # [total_seq_len // 32] * 32
        ] * 16
        GLOBAL_BATCH = manual_case + GLOBAL_BATCH
        GLOBAL_BATCH = iter(GLOBAL_BATCH)
    return


def get_next_batch(dp_size) -> Iterable[List[List[int]]]:
    global GLOBAL_BATCH
    global ITERATION_ID
    global iterated_samples
    # get dp_size number of batches 
    batches = []
    for _ in range(dp_size):    
        batches.append(next(GLOBAL_BATCH))
    ITERATION_ID += 1
    iterated_samples.append(batches)
    return batches


# ------------------------------------
# Parallel Config
# ------------------------------------
@dataclass
class ParallelConfig:
    tensor_model_parallel_size: int = 1
    pipeline_model_parallel_size: int = 1
    virtual_pipeline_model_parallel_size: Optional[int] = None
    context_parallel_size: int = 1
    expert_model_parallel_size: int = 1
    expert_tensor_parallel_size: Optional[int] = None


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
        return torch.cuda

def set_random_seed(seed, set_megatron: bool=True):
    """Set worker side random seed."""
    

    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

# ------------------------------------
# Test Logic
# ------------------------------------
def test(
    num_nodes,
    num_gpus_per_node,
    seed,
    batch_size,
    tp_size,
    world_size,
    total_seq_len,
    num_layers,
    model_path,
    max_sample_id,
    up_sample_factor,
    elongate_factor,
    filter_threshold,
    filter_ratio,
    should_add_debug_cases,
    resend_qkv,
    sample_start_idx,
    change_long_doc_ratio=0,
    sample_name='wlbllm',
):
    if num_layers is not None:
        os.environ["NUM_LAYERS"] = str(num_layers)

    dtype = torch.bfloat16
    element_size = dtype.itemsize

    
    hf_config = AutoConfig.from_pretrained(model_path)
    hidden_size_q = hf_config.hidden_size

    hidden_size_kv = hidden_size_q
    if hasattr(hf_config, "num_key_value_heads"):
        hidden_size_kv = (hidden_size_kv * hf_config.num_key_value_heads //
                          hf_config.num_attention_heads)
    
    set_random_seed(seed, set_megatron=False)
    
    rank = 0
    as_rank = 0
    as_world_size = 8 # as world size

    hidden_size_q_tp = hidden_size_q // tp_size
    hidden_size_k_tp = hidden_size_kv // tp_size

    setup_global_batch(
        total_seq_len,
        up_sample_factor=up_sample_factor,
        elongate_factor=elongate_factor,
        filter_threshold=filter_threshold,
        filter_ratio=filter_ratio,
        should_add_debug_cases=should_add_debug_cases,
        change_long_doc_ratio=change_long_doc_ratio,
        sample_name=sample_name,
    )

    sample_times = []
    for sample_id in range(sample_start_idx, max_sample_id):
        # D2 will get 2 batch each time, one for ping, the other for pong.
        # Suppose we have 
        #   as_world_size = 4
        # Then that means we implicitly have dpcp = 4
        # 1. We get 2 batch, each batch has `total_seq_len`` number of tokens
        # 2. Each GPU should get total_seq_len // as_world_size number of tokens. 
        
        print(f"🟡 [Rank {rank}] hidden_size_q_tp = {hidden_size_q_tp}, hidden_size_k_tp = {hidden_size_k_tp}, element_size = {element_size}")

        dp_size = as_world_size

        model_config = hf_config
        parallel_config = ParallelConfig(
            tensor_model_parallel_size=tp_size,
            pipeline_model_parallel_size=1,
        )

        try:
            _seq_lens: list[list[int]] = get_next_batch(batch_size * 2)
        except StopIteration:
            break
        
        # Rebalance Ping pong
        def balance_ping_pong(seq_lens: list[list[int]]) -> list[list[int]]:
            # [1k x 4 ], [2k x 2 ], [4k] ...
            # taking a list of batch, and interleave them by sorted workload.
            sorted_attn_workload = sorted(seq_lens, key=lambda x: sum(y ** 2 for y in x))
            ping = []
            pong = []
            for batch in sorted_attn_workload:
                if len(ping) < len(pong):
                    ping.append(batch)
                else:
                    pong.append(batch)
            assert len(ping) == len(pong)
            return ping, pong
            
        rich.print(f"🟡 [Rank {rank}] _seq_lens = {_seq_lens}")

        should_d2_balance_ping_pong = os.environ.get("EXPERIMENT_D2_BALANCE_PING_PONG", "0") == "1"
        if should_d2_balance_ping_pong:
            print(f"🟢 [Rank {rank}] Balancing ping pong")
            seq_lens_0, seq_lens_1 = balance_ping_pong(_seq_lens)
        else:
            print(f"🟡 [Rank {rank}] Not Balancing ping pong")
            seq_lens_0, seq_lens_1 = _seq_lens[:batch_size], _seq_lens[batch_size:]
        
        rich.print(f"🟡 [Rank {rank}] seq_lens_0 = {seq_lens_0}")
        rich.print(f"🟡 [Rank {rank}] seq_lens_1 = {seq_lens_1}")

        # num_batched_token_per_as_rank = tokens per as rank = tokens per batch * num batch / (as_world_size = dp_size)
        num_batched_token_per_as_rank = total_seq_len * batch_size // dp_size

        _items_0: list[Item] = batch_to_items_general(seq_lens_0, num_batched_token_per_as_rank, as_world_size, model_config)
        _items_1: list[Item] = batch_to_items_general(seq_lens_1, num_batched_token_per_as_rank, as_world_size, model_config)

        verbose = (rank % 8 == 0)
        required_buffer_size: dict[float, float] = {} # tolerance_factor -> required_buffer_size

        # ------------------------------------
        # Now compute the memory consumption 
        # given tolerance factor
        # ------------------------------------

        # candidate_tolerance_factors = [0.0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        candidate_tolerance_factors = [0.0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

        for tolerance_factor in candidate_tolerance_factors:
            print(f"[Rank {rank}] =========== Tolerance factor = {tolerance_factor} ============ ")
            
            planner = Planner(world_size, parallel_config, model_config=model_config, tolerance_factor=tolerance_factor)
            
            fa2a_metadata_0, as_attn_metadata_0, mlp_shard_len_0 = planner.plan(_items_0, is_resend_qkv=resend_qkv, verbose=verbose)
            fa2a_metadata_1, as_attn_metadata_1, mlp_shard_len_1 = planner.plan(_items_1, is_resend_qkv=resend_qkv, verbose=verbose)
            
            def _check_self_overflow(fa2a_metadata, as_rank_):
                """Return the self-overflow status and the maximum size provisioned."""
                send_sz = [torch.sum(m.fa2a_metadata[1][as_rank_]).item() for m in fa2a_metadata]
                dst_last_offset = [(m.fa2a_metadata[1] + m.fa2a_metadata[2])[as_rank_] for m in fa2a_metadata]
                recv_sz = [torch.sum(m.fa2a_metadata[3][as_rank_]).item() for m in fa2a_metadata]
                src_last_offset = [(m.fa2a_metadata[0] + m.fa2a_metadata[1])[as_rank_] for m in fa2a_metadata]
                max_send_sz = max(send_sz)
                max_recv_sz = max(recv_sz)
                max_dst_last_offset = max(torch.max(o).item() for o in dst_last_offset)
                max_src_last_offset = max(torch.max(o).item() for o in src_last_offset)
                
                if rank % 8 == 0:
                    print(
                        f"🟡 [Rank {rank}]  Overflow check of as_rank_ = {as_rank_}: "
                        f"{max_send_sz / 1024**3:.2f} GB send size, "
                        f"{max_recv_sz / 1024**3:.2f} GB recv size, "
                        f"{max_dst_last_offset / 1024**3:.2f} GB dst last offset, "
                        f"{max_src_last_offset / 1024**3:.2f} GB src last offset. "
                    )

                max_size_provisioned = max(
                    max_send_sz, max_recv_sz, 
                    max_dst_last_offset, max_src_last_offset,
                )
                return max_size_provisioned

            def _check_all_overflow(fa2a_metadata, as_world_size_):
                all_max_size_provisioned = 0
                states = []
                for as_rank_ in range(as_world_size_):
                    max_size_provisioned = _check_self_overflow(fa2a_metadata, as_rank_)
                    all_max_size_provisioned = max(all_max_size_provisioned, max_size_provisioned)
                return all_max_size_provisioned
                
            max_size_provisioned_0 = _check_all_overflow(fa2a_metadata_0, as_world_size)
            max_size_provisioned_1 = _check_all_overflow(fa2a_metadata_1, as_world_size)
            max_size_provisioned = max(max_size_provisioned_0, max_size_provisioned_1) / 1024**3
            required_buffer_size[tolerance_factor] = max_size_provisioned
        sample_times.append(required_buffer_size)
    return sample_times

# %%
test(
    num_nodes=8,
    num_gpus_per_node=8,
    seed=42,
    batch_size=8,
    tp_size=8,
    world_size=64,
    total_seq_len=128*1024,
    num_layers=32,
    model_path="deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
    max_sample_id=10,
    up_sample_factor=1,
    elongate_factor=1,
    filter_threshold=64 * 1024,
    filter_ratio=0.90,
    should_add_debug_cases=False,
    resend_qkv=False,
    sample_start_idx=0,
)
# %%
