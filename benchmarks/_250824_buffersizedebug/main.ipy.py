# %%
from dataclasses import dataclass
from typing import Iterable, List, Optional

from d2.planner.planner import (
    batch_to_items_general,
    Planner,
)

from d2.runtime.inplace_metadata import (
    compute_metadata,
    compute_metadata_kv,
    compute_attn_layout_seqlens,
)
from d2.simulator.optimizers.samples import (
    batch_documents,
    sample_wlbllm_docs_upsample,
)

# %%

GLOBAL_BATCH = None
ITERATION_ID = 0
iterated_samples = []

def setup_global_batch(
    total_seq_len, 
    up_sample_factor=2,
    elongate_factor=1,
    filter_threshold=64 * 1024,
    filter_ratio=0.90,
    should_add_debug_cases=False,
):
    global GLOBAL_BATCH
    if GLOBAL_BATCH is not None:
        return

    GLOBAL_BATCH = batch_documents(
        sample_wlbllm_docs_upsample(
            size=10000,
            filter_threshold=filter_threshold,
            filter_ratio=filter_ratio,
            upsample_long_factor=up_sample_factor,
            elongate_factor=elongate_factor,
        ), max_ctx_length=total_seq_len
    )

    
    if should_add_debug_cases:
        GLOBAL_BATCH = list(GLOBAL_BATCH)
        # DP2 case
        # manual_case = [
        #     [total_seq_len],
        #     [total_seq_len // 2] * 2,
        #     # [total_seq_len],
        #     # [total_seq_len // 8] * 8,
        # ]
        # 🔴 Failed: Cross 3 GPU + non cross for the others
        # manual_case = [
        #     [total_seq_len // 4 * 3 - 512, 512, total_seq_len // 4],
        #     [total_seq_len // 4 * 3 - 512, 512, total_seq_len // 4],
        # ]

        manual_case = [
            [total_seq_len // 4 * 3 - 512, 512, total_seq_len // 4],
            [total_seq_len // 4 * 3 - 512, 512, total_seq_len // 4],
        ]

        # manual_case = [
        #     [2 * K] * (total_seq_len // (2 * K)),
        # ] * 8
        GLOBAL_BATCH = manual_case + GLOBAL_BATCH
        GLOBAL_BATCH = iter(GLOBAL_BATCH)
    # CP debug batch case.  
    # GLOBAL_BATCH = [
    #     [total_seq_len // 4 * 3, total_seq_len // 4],
    # ] * 100
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


# total_seq_len = 1024
total_seq_len = 1024 * 32
# total_seq_len = 16
batch_size = 1
setup_global_batch(total_seq_len=total_seq_len, should_add_debug_cases=True)
_seq_lens = get_next_batch(2 * batch_size)
seq_lens_0: list[list[int]] = _seq_lens[:batch_size]
seq_lens_1: list[list[int]] = _seq_lens[batch_size:]


# %%
from transformers import AutoConfig
model_path = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
hf_config = AutoConfig.from_pretrained(model_path)
# %%

world_size = 8
model_config = hf_config
tp_size = 2
as_world_size = world_size // tp_size
num_batched_token_per_as_rank = total_seq_len * batch_size // as_world_size

# %%
as_world_size, num_batched_token_per_as_rank, total_seq_len, batch_size

# %%
@dataclass
class ParallelConfig:
    tensor_model_parallel_size: int = 1
    pipeline_model_parallel_size: int = 1
    virtual_pipeline_model_parallel_size: Optional[int] = None
    context_parallel_size: int = 1
    expert_model_parallel_size: int = 1
    expert_tensor_parallel_size: Optional[int] = None

parallel_config = ParallelConfig(
    tensor_model_parallel_size=tp_size
)
# %%
from d2.planner.planner import Item

_items_0: list[Item] = batch_to_items_general(seq_lens_0, num_batched_token_per_as_rank, as_world_size, model_config)
_items_1: list[Item] = batch_to_items_general(seq_lens_1, num_batched_token_per_as_rank, as_world_size, model_config)

planner = Planner(world_size, parallel_config, model_config=model_config)

d2_should_replan = True

# %%
seq_lens_0

# %%
import rich
rich.print(_items_0)

# %%




# %%

# %%

# %%


# %%

# hidden_size_q_tp = 4096 // 8
# hidden_size_k_tp = 1024 // 8
hidden_size_q = hf_config.hidden_size
hidden_size_kv = hidden_size_q
if hasattr(hf_config, "num_key_value_heads"):
    hidden_size_kv = (hidden_size_kv * hf_config.num_key_value_heads //
                        hf_config.num_attention_heads)
hidden_size_q_tp = hidden_size_q // tp_size
hidden_size_k_tp = hidden_size_kv // tp_size
element_size = 2

# %%

# hidden_size_q_tp = 4096 // 8
# hidden_size_k_tp = 1024 // 8
# hidden_size_q = hf_config.hidden_size
# hidden_size_kv = hidden_size_q
# if hasattr(hf_config, "num_key_value_heads"):
#     hidden_size_kv = (hidden_size_kv * hf_config.num_key_value_heads //
#                         hf_config.num_attention_heads)
# hidden_size_q_tp = 1
# hidden_size_k_tp = 1
# element_size = 2

# %%
items = _items_0
# def items_to_metadata(items: list[Item]) -> tuple['PingPangSingleStepPackedSeqParams', 'PackedSeqParams']:
from d2.runtime.fast_alltoall_metadata import (
    compute_backward_resend_e2e_metadata,
    compute_e2e_fa2a_metadata,
    compute_fa2a_metadata_from_logical_metadata,
    forward_backward_with_resend_e2e_metadata,
)

# %%
# seq_lens_0
items
# %%

ret = planner.plan_to_raw_qkv_dispatch(items)
(
    mlp_num_seqs,
    mlp_q_dispatch,
    mlp_seq_lens,
    kv_to_q_mapping,
    kv_to_q_rank,
    kv_context_size,
    q_to_num_kv_seq,
    q_to_num_kv_tokens,
) = ret

# %%
mlp_num_seqs
# %%
mlp_seq_lens
# %%
mlp_q_dispatch
# %%
# %%
mlp_seq_lens

# %%

# We probably want to move this function inside the Planner.plan
(
    fwd_metadata_q, bwd_metadata_q, fwd_metadata_kv, bwd_metadata_kv, fa_params, fa2a_metadata
) = compute_e2e_fa2a_metadata(
    mlp_seq_lens,
    mlp_num_seqs,
    mlp_q_dispatch,
    kv_to_q_mapping,
    kv_to_q_rank,
    kv_context_size,
    q_to_num_kv_seq,
    q_to_num_kv_tokens,
    hidden_size_q_tp,
    hidden_size_k_tp,
    element_size,
    # TODO: What is this? How do specify this values?
    softmax_lse_size=0,
)

# %%
mlp_seq_lens
# %%

# %%

# %%

# %%
fwd_metadata_q
rich.print(fwd_metadata_q)
# %%
bwd_metadata_q
rich.print(bwd_metadata_q)
# %%
fwd_metadata_kv
rich.print(fwd_metadata_kv)
# %%
bwd_metadata_kv
rich.print(bwd_metadata_kv)
# %%
fa_params
# %%
(
    qkv_fwd_fa2a_metadata,
    qkv_bwd_fa2a_metadata,
    attn_out_fwd_fa2a_metadata,
    attn_out_bwd_fa2a_metadata,
) = fa2a_metadata
# %%
(
    qkv_fwd_fa2a_metadata__sender_send_offset, 
    qkv_fwd_fa2a_metadata__sender_transfer_sz, 
    qkv_fwd_fa2a_metadata__sender_recv_offset, 
    qkv_fwd_fa2a_metadata__recver_transfer_sz
) = qkv_fwd_fa2a_metadata.fa2a_metadata

# %%
rich.print(_items_0)
# %%
rich.print(f"hidden_size_q_tp = {hidden_size_q_tp}") 
rich.print(f"hidden_size_k_tp = {hidden_size_k_tp}") 
rich.print(f"element_size = {element_size}") 
rich.print(f"element_size_in_bytes = {element_size}")
MB = 1024 * 1024
# %%
rich.print(qkv_fwd_fa2a_metadata__sender_send_offset // MB)
# %%
rich.print(qkv_fwd_fa2a_metadata__sender_transfer_sz // MB)
# %%
rich.print(qkv_fwd_fa2a_metadata__sender_recv_offset // MB)
# %%
rich.print(qkv_fwd_fa2a_metadata__recver_transfer_sz // MB)

# %%
for i in range(len(_items_0)):
    rich.print(_items_0[i])
# %%

# %%

# %%

# %%

# %%
# from d2.runtime.inplace_metadata import PingPangSingleStepPackedSeqParams
# # %%
# def get_single_step_packed_seq_params(
#     fa2a_metadata, attn_metadata, rank: int
# ):
#     (
#         qkv_fwd_fa2a_metadata, qkv_rev_fa2a_metadata,
#         attn_out_fwd_fa2a_metadata, attn_out_rev_fa2a_metadata,
#     ) = fa2a_metadata
#     (
#         cu_seqlens_q, cu_seqlens_kv, max_seqlen_q, max_seqlen_kv,
#     ) = attn_metadata
#     ping_pang_params = PingPangSingleStepPackedSeqParams(
#         qkv_format="thd",
#         cu_seqlens_q=cu_seqlens_q[rank],
#         cu_seqlens_kv=cu_seqlens_kv[rank],
#         max_seqlen_q=max_seqlen_q[rank],
#         max_seqlen_kv=max_seqlen_kv[rank],
#         qkv_fwd_metadata=qkv_fwd_fa2a_metadata.get_slice(rank),
#         qkv_bwd_metadata=qkv_rev_fa2a_metadata.get_slice(rank),
#         attn_out_fwd_metadata=attn_out_fwd_fa2a_metadata.get_slice(rank),
#         attn_out_bwd_metadata=attn_out_rev_fa2a_metadata.get_slice(rank),
#     )
#     return ping_pang_params

# from d2.runtime.inplace_metadata import mlp_layout_packed_params

# # %%
# as_rank = 0
# ping_pang_params = get_single_step_packed_seq_params(
#     fa2a_metadata, fa_params, as_rank
# )

# raw_seq_len = mlp_seq_lens
# mlp_seq_params = mlp_layout_packed_params(raw_seq_len)

# # %%
