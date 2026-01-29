"""
DistCA LLaMA Pre-training Script (Refactored)

A cleaner, more maintainable version of pretrain_llama.py that:
- Uses DistCA worker infrastructure properly
- Supports standard dataset formats (Megatron GPTDataset, HuggingFace)
- Follows best practices from simple_4d.py
- Provides better configurability and logging
"""

import argparse
import logging
import os
import time
from dataclasses import dataclass
from datetime import timedelta
from functools import partial
from pathlib import Path
from typing import Optional, Union

import torch
import torch.cuda.nvtx
from transformers import AutoConfig

# ================================
# Megatron Imports
# ================================
import megatron.core.parallel_state as mpu
from megatron.core import tensor_parallel
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.tensor_parallel.cross_entropy import vocab_parallel_cross_entropy
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.training import get_args, get_tokenizer
from megatron.training.initialize import initialize_megatron
from megatron.training.arguments import core_transformer_config_from_args
from megatron.core.models.gpt import GPTModel
from megatron.core.datasets.gpt_dataset import GPTDataset, GPTDatasetConfig
from megatron.core.datasets.blended_megatron_dataset_builder import BlendedMegatronDatasetBuilder
from megatron.core.enums import ModelType
from megatron.training.training import setup_model_and_optimizer
from megatron.training.utils import get_blend_and_blend_per_split, unwrap_model

# ================================
# DistCA Imports
# ================================
from distca.planner.planner import cp_list_to_mlp_list
from distca.runtime.attn_kernels.ops import (
    nvshmem_get_unique_id,
    nvshmem_alloc_empty_unique_id,
    DispatcherWrapper,
    nvshmem_barrier_all,
)
from distca.runtime.compute_metadata import get_attn_metadata
from distca.runtime.megatron.create_group import (
    get_attn_server_group_gloo,
    get_attn_server_rank,
    get_attn_server_group_src_rank,
    initialize_attention_server_comm,
)
from distca.runtime.megatron.forward_backward_func import (
    forward_backward_pipelining_without_interleaving as forward_backward_func,
)
from distca.runtime.megatron.packed_seq_params import (
    arg_to_cuda,
    PingPangSingleStepPackedSeqParams,
    PingPangPackedSeqParams,
    MLPLayoutPackedSeqParams,
)
from distca.runtime.megatron.ping_pong.transformer_block import PingPongGPTModel
from distca.utils.logging import setup_logging, time_it, setup_log_directories
from distca.utils.megatron_test_utils import (
    gptmodel_forward,
    make_batch_generator,
    init_mcore_model,
)
from distca.utils.test_util import ParallelConfig, create_qkv_dispatch_pipeline_tick
from distca.utils.traceback import enable_clickable_excepthook
from distca.utils.training_utils import setup_global_batch
from distca.utils.worker import set_random_seed

# ================================
# Utilities
# ================================
from utils.cpu_affinity import set_cpu_affinity
from utils.hf_config import get_megatron_args_dict_from_hf_model
from utils.estimate import log_memory_estimate, log_flops_estimate

enable_clickable_excepthook()


# ================================
# Environment Setup
# ================================
rank = int(os.environ.get("RANK", "0"))
world_size = int(os.environ.get("WORLD_SIZE", "1"))
local_rank = int(os.environ.get("LOCAL_RANK", "0"))

# Setup logging
logger = setup_logging(
    rank=rank,
    world_size=world_size,
    level=logging.INFO,
    console_ranks=[0],
)

# Set CPU affinity
set_cpu_affinity(local_rank, ncpu_per_proc=16, logger=logger)

# Setup CUDA
with time_it("set device"):
    torch.cuda.set_device(local_rank)
    torch.set_default_device(torch.device("cuda", local_rank))

# Initialize distributed
with time_it("init_process_group"):
    torch.distributed.init_process_group(
        backend="cpu:gloo,cuda:nccl",
        rank=rank,
        world_size=world_size,
        timeout=timedelta(seconds=60),
    )

logger.info(f"Rank: {rank}, World Size: {world_size}, Local Rank: {local_rank}")


# ================================
# DistCA Configuration
# ================================
@dataclass
class DistCAConfig(TransformerConfig):
    """Extended TransformerConfig with DistCA-specific options."""

    distca_nvshmem_buffer_size_gb: float = 1.0
    distca_quit_if_maybe_oom: bool = False
    distca_debug_nan_detector: bool = False
    distca_use_planner: bool = True

    # Dataset sampling configuration
    distca_sample_name: str = "wlbllm"
    distca_up_sample_factor: int = 4
    distca_elongate_factor: int = 1
    distca_filter_threshold: int = 65536
    distca_filter_ratio: float = 0.50
    distca_change_long_doc_ratio: float = 0.0
    distca_max_total_tokens: Optional[int] = None


def _get_mlp_cu_seqlens_q(packed_seq_params):
    """Extract cu_seqlens_q for MLP layout to mask sequence boundaries."""
    psp = packed_seq_params
    if hasattr(psp, "mlp_packed_seq_params") and psp.mlp_packed_seq_params is not None:
        psp = psp.mlp_packed_seq_params
    if hasattr(psp, "cu_seqlens_q") and psp.cu_seqlens_q is not None:
        return psp.cu_seqlens_q
    if hasattr(packed_seq_params, "mlp_layout_seq_params") and packed_seq_params.mlp_layout_seq_params is not None:
        seq_psp_list = packed_seq_params.mlp_layout_seq_params
        if len(seq_psp_list) == 1:
            return seq_psp_list[0].cu_seqlens_q
        # Concatenate cu_seqlens across ping/pong segments with offsets
        cu_list = []
        offset = None
        for seg in seq_psp_list:
            cu = seg.cu_seqlens_q
            if offset is None:
                cu_list.append(cu)
                offset = cu[-1]
            else:
                cu_list.append(cu[1:] + offset)
                offset = offset + (cu[-1] - cu[0])
        return torch.cat(cu_list, dim=0) if cu_list else None
    return None


def build_next_token_labels(input_ids: torch.Tensor, packed_seq_params=None) -> torch.Tensor:
    """Build next-token prediction labels, masking sequence boundaries with -100."""
    labels = input_ids.clone()
    if labels.numel() > 1:
        labels[:-1] = input_ids[1:]
    labels[-1] = -100

    if packed_seq_params is not None:
        cu_seqlens_q = _get_mlp_cu_seqlens_q(packed_seq_params)
        if cu_seqlens_q is not None and cu_seqlens_q.numel() > 1:
            ends = (cu_seqlens_q[1:] - 1).to(device=labels.device, dtype=torch.long)
            ends = ends[(ends >= 0) & (ends < labels.numel())]
            if ends.numel() > 0:
                labels[ends] = -100
    return labels


def extract_scalar_loss(losses_reduced):
    """Extract scalar loss value from various Megatron loss formats."""
    loss_value = None
    if isinstance(losses_reduced, dict):
        val = losses_reduced.get("loss")
        if val is not None:
            if isinstance(val, (list, tuple)):
                vals = [v.item() if torch.is_tensor(v) else float(v) for v in val]
                if vals:
                    loss_value = sum(vals) / len(vals)
            else:
                loss_value = val.item() if torch.is_tensor(val) else float(val)
    elif torch.is_tensor(losses_reduced):
        loss_value = losses_reduced.item()
    elif isinstance(losses_reduced, (list, tuple)) and len(losses_reduced) > 0:
        vals = []
        for v in losses_reduced:
            if torch.is_tensor(v):
                vals.append(v.item())
            elif isinstance(v, (int, float)):
                vals.append(float(v))
            elif isinstance(v, dict) and "loss" in v:
                lv = v["loss"]
                vals.append(lv.item() if torch.is_tensor(lv) else float(lv))
        if vals:
            loss_value = sum(vals) / len(vals)
    return loss_value


# ================================
# Megatron Initialization
# ================================
def create_megatron_args(config: dict, log_paths: dict) -> list:
    """Create Megatron CLI arguments from config."""
    args = [
        "--seed", str(config.get("seed", 42)),

        # Model Architecture
        "--num-layers", str(config["num_layers"]),
        "--hidden-size", str(config["hidden_size"]),
        "--ffn-hidden-size", str(config["ffn_hidden_size"]),
        "--num-attention-heads", str(config["num_attention_heads"]),
        "--group-query-attention",
        "--num-query-groups", str(config["num_query_groups"]),
        "--max-position-embeddings", str(config["max_position_embeddings"]),
        "--position-embedding-type", str(config["position_embedding_type"]),
        "--rotary-base", str(config["rotary_base"]),
        "--normalization", str(config["normalization"]),
        "--seq-length", str(config["seq_length"]),
        "--vocab-size", str(config["vocab_size"]),

        # Training Hyperparameters
        "--micro-batch-size", str(config.get("micro_batch_size", 1)),
        "--global-batch-size", str(config.get("global_batch_size", 1)),
        "--lr", str(config.get("lr", 1e-5)),
        "--train-iters", str(config.get("train_iters", 100)),
        "--lr-decay-style", "constant",
        "--min-lr", "1e-6",
        "--weight-decay", "0.01",

        # Mixed Precision
        "--bf16" if config.get("use_bf16", True) else "--fp16",
        "--transformer-impl", "transformer_engine",

        # Parallelism
        "--tensor-model-parallel-size", str(config["tp"]),
        "--pipeline-model-parallel-size", str(config["pp"]),
        "--context-parallel-size", str(config.get("cp", 1)),
        "--cp-comm-type", "p2p",
        "--distributed-timeout-minutes", "2",
        "--local-rank", str(local_rank),

        # Data/IO
        "--data-path", str(config.get("data_path", "")),
        "--tokenizer-type", config.get("tokenizer_type", "HuggingFaceTokenizer"),
        "--tokenizer-model", config.get("tokenizer_model", config["model_name"]),
        "--data-cache-path", str(log_paths["data_cache_path"]),
        "--no-create-attention-mask-in-dataloader",
        "--num-workers", "1",
        "--split", "90,5,5",

        # Checkpointing
        "--save", str(log_paths["ckpt_path"]),
        "--load", str(log_paths["ckpt_path"]),
        "--save-interval", str(config.get("save_interval", 0)),

        # Logging
        "--log-interval", "1",
        "--log-params-norm",
        "--log-timers-to-tensorboard",
        "--tensorboard-dir", str(log_paths["tensorboard_path"]),
        "--tensorboard-log-interval", "1",
        "--logging-level", "20",

        # Activation Recomputation
        "--recompute-granularity", "selective",
        "--recompute-modules", "core_attn",
    ]

    # Add optional flags
    if config.get("swiglu", False):
        args.append("--swiglu")
    if config.get("untie_embeddings_and_output_weights", False):
        args.append("--untie-embeddings-and-output-weights")

    # Filter None values
    return [arg for arg in args if arg is not None]


# ================================
# Dataset Providers
# ================================
def is_dataset_built_on_rank() -> bool:
    """DistCA: Build dataset on all ranks for now (TODO: optimize to rank 0 only)."""
    return True


def core_gpt_dataset_config_from_args(args) -> GPTDatasetConfig:
    """Create GPTDatasetConfig from Megatron args."""
    tokenizer = get_tokenizer()
    blend, blend_per_split = get_blend_and_blend_per_split(args)

    return GPTDatasetConfig(
        random_seed=args.seed,
        sequence_length=args.seq_length,
        blend=blend,
        blend_per_split=blend_per_split,
        split=args.split,
        num_dataset_builder_threads=args.num_dataset_builder_threads,
        path_to_cache=args.data_cache_path,
        mmap_bin_files=args.mmap_bin_files,
        tokenizer=tokenizer,
        reset_position_ids=args.reset_position_ids,
        reset_attention_mask=args.reset_attention_mask,
        eod_mask_loss=args.eod_mask_loss,
        create_attention_mask=args.create_attention_mask_in_dataloader,
    )


def train_valid_test_datasets_provider(train_val_test_num_samples):
    """Build train/val/test datasets using Megatron's GPTDataset."""
    args = get_args()
    config = core_gpt_dataset_config_from_args(args)

    logger.info("> building train, validation, and test datasets for GPT ...")
    train_ds, valid_ds, test_ds = BlendedMegatronDatasetBuilder(
        GPTDataset,
        train_val_test_num_samples,
        is_dataset_built_on_rank,
        config,
    ).build()
    logger.info("> finished creating GPT datasets ...")

    return train_ds, valid_ds, test_ds


# ================================
# Model Provider
# ================================
def distca_model_provider(pre_process=True, post_process=True, hf_config=None):
    """Build PingPongGPTModel for DistCA training."""
    args = get_args()
    config = core_transformer_config_from_args(args, config_class=DistCAConfig)

    assert args.transformer_impl == "transformer_engine", "DistCA requires Transformer Engine"
    logger.info(f"Building DistCA model for rank {rank}")

    share_embeddings_and_output_weights = not args.untie_embeddings_and_output_weights

    # Initialize PingPongGPTModel via registry
    parallel_model = init_mcore_model(
        config,
        hf_config,
        pre_process,
        post_process,
        share_embeddings_and_output_weights=share_embeddings_and_output_weights,
        value=False,
        freeze_moe_router=False,
    )
    parallel_model.to("cuda")

    return parallel_model


# ================================
# Microbatch Creation
# ================================
def create_pp_microbatches(
    num_microbatch: int,
    pp_degree: int,
    as_rank: int,
    as_world_size: int,
    total_seq_len: int,
    num_seqs: int,
    max_cp_degree: int,
    hidden_size_q_tp: int,
    hidden_size_k_tp: int,
    element_size: int,
    num_head_in_dtype: int,
    tp_size: int,
    dp_size: int,
    num_token_per_rank: int,
    num_batches: int,
    use_planner: bool = True,
    return_seq_lens: bool = False,
):
    """Create pipeline parallel microbatches with DistCA metadata."""
    tick_per_rank_doc_lens = None
    bwd_metadata = []
    microbatches = []
    all_original_seq_lens = []

    for i in range(num_microbatch + pp_degree - 1):
        add_dummy_forward = i >= num_microbatch

        (
            fa_fwd_params, fa_bwd_params,
            qkv_fwd_fa2a_metadata, qkv_bwd_fa2a_metadata,
            attn_out_fwd_fa2a_metadata, attn_out_qkv_bwd_fa2a_metadata,
            tick_per_rank_doc_lens, original_tick_per_rank_doc_lens,
        ) = create_qkv_dispatch_pipeline_tick(
            as_world_size, total_seq_len, num_seqs, max_cp_degree,
            hidden_size_q_tp, hidden_size_k_tp, element_size, num_head_in_dtype,
            ref_doc_lens=tick_per_rank_doc_lens,
            add_dummy=add_dummy_forward,
            tp_size=tp_size,
            dp_size=dp_size,
            num_token_per_rank=num_token_per_rank,
            num_batches=num_batches,
            use_planner=use_planner,
            return_original_doclen=return_seq_lens,
        )
        all_original_seq_lens.append(original_tick_per_rank_doc_lens)

        # Convert CP layout to MLP layout
        tick_per_rank_doc_lens_after_cp_transfer = cp_list_to_mlp_list(
            tick_per_rank_doc_lens, as_world_size, num_token_per_rank
        )

        this_rank_num_tokens = sum(tick_per_rank_doc_lens_after_cp_transfer[as_rank])
        bwd_packed_seq_params = PackedSeqParams(qkv_format="thd", **fa_bwd_params[as_rank])
        tensor_doc_lens = torch.tensor(tick_per_rank_doc_lens_after_cp_transfer[as_rank], dtype=torch.int32)
        mlp_packed_seq_params = get_attn_metadata(tensor_doc_lens, get_packed_seq_params=True)

        # Create ping-pong params
        ping_pang_params = PingPangSingleStepPackedSeqParams(
            qkv_format="thd",
            **fa_fwd_params[as_rank],
            qkv_fwd_metadata=qkv_fwd_fa2a_metadata.get_slice(as_rank),
            attn_out_fwd_metadata=attn_out_fwd_fa2a_metadata.get_slice(as_rank),
            mlp_packed_seq_params=mlp_packed_seq_params,
        )

        position_ids_local = torch.arange(this_rank_num_tokens)
        microbatch = {
            "position_ids": position_ids_local,
            "packed_seq_params": ping_pang_params,
        }
        microbatches.append(microbatch)

        # Store backward metadata for later ticks
        bwd_metadata.append((
            qkv_bwd_fa2a_metadata.get_slice(as_rank),
            attn_out_qkv_bwd_fa2a_metadata.get_slice(as_rank),
            bwd_packed_seq_params,
        ))

    # Assign backward metadata to correct microbatches
    pp_rank = as_rank // dp_size
    for i, microbatch in enumerate(microbatches):
        bwd_metadata_idx = (i + pp_degree - 1 - pp_rank * 2) % len(bwd_metadata)
        qkv_bwd_metadata, attn_out_bwd_metadata, bwd_packed_seq_params = bwd_metadata[bwd_metadata_idx]
        packed_seq_params = microbatch["packed_seq_params"]
        packed_seq_params.qkv_bwd_metadata = qkv_bwd_metadata
        packed_seq_params.attn_out_bwd_metadata = attn_out_bwd_metadata
        packed_seq_params.bwd_packed_seq_params = bwd_packed_seq_params

    ret = microbatches
    if return_seq_lens:
        ret = (microbatches, all_original_seq_lens)
    return ret


# ================================
# Forward/Backward Step
# ================================
def forward_backward_batch(
    train_module,
    microbatches: list,
    tf_config,
    forward_only: bool = False,
    mode: str = "ping_pong",
    with_dummy: bool = True,
):
    """Execute forward and backward passes for a batch of microbatches."""
    # Move to CUDA
    microbatches = [{k: arg_to_cuda(v) for k, v in mb.items()} for mb in microbatches]

    pp_size = tf_config.pipeline_model_parallel_size
    pp_rank = mpu.get_pipeline_model_parallel_rank()
    n_micro_batch = len(microbatches) - pp_size + 1
    total_seqlen = microbatches[0]['input_ids'].shape[0]

    def forward_step(batch_iter, model):
        """Forward step with loss computation."""
        batch = next(batch_iter)
        torch.cuda.nvtx.range_push("forward_step")

        input_ids = batch['input_ids']
        position_ids = batch['position_ids']
        packed_seq_params = batch['packed_seq_params']
        labels = build_next_token_labels(input_ids, packed_seq_params=packed_seq_params)

        output = gptmodel_forward(
            model, input_ids, None, position_ids,
            tf_config.sequence_parallel, packed_seq_params,
        )

        def loss_func_ce(logits, _labels=labels):
            if isinstance(logits, (tuple, list)):
                logits = logits[0]

            labels_2d = _labels.view(-1, 1)

            # Normalize logits to [seq, batch, vocab/TP]
            if logits.dim() == 2:
                logits = logits.unsqueeze(1)
            elif logits.dim() == 3:
                if logits.shape[1] != 1 and logits.shape[0] == 1:
                    logits = logits.transpose(0, 1)

            # Handle sequence-parallel sharding
            if labels_2d.shape[0] != logits.shape[0]:
                tp_size = mpu.get_tensor_model_parallel_world_size()
                if tf_config.sequence_parallel and tp_size > 1 and labels_2d.shape[0] == logits.shape[0] * tp_size:
                    labels_2d = tensor_parallel.scatter_to_sequence_parallel_region(labels_2d)

            ce = vocab_parallel_cross_entropy(logits.contiguous(), labels_2d.contiguous())
            loss_mask = (labels_2d != -100).float()
            denom = loss_mask.sum().clamp(min=1.0)
            loss = (ce * loss_mask).sum() / denom
            return loss, {"loss": loss}

        torch.cuda.nvtx.range_pop()
        return output, loss_func_ce

    def dummy_backward_step(model, dummy_bwd_iter, skip: bool):
        """Dummy backward step for DistCA attention disaggregation."""
        next_iter_args = next(dummy_bwd_iter)
        if skip:
            return
        unwrap_model(model).dummy_backward(next_iter_args)

    assert len(train_module) == 1, "Only support single module"

    # Shift backward metadata for correct timing
    dummy_bwd_packed_seq_params = [
        microbatch['packed_seq_params'] for microbatch in
        (microbatches[-pp_size + pp_rank + 1:][:pp_size - pp_rank - 1] + microbatches[:pp_rank])
    ]
    dummy_bwd_packed_seq_params = dummy_bwd_packed_seq_params[pp_rank:] + dummy_bwd_packed_seq_params[:pp_rank]

    # Set debug mode
    for module in train_module:
        debug = (mode != "ping_pong")
        debug_fwd_impl = mode if debug else None
        unwrap_model(module).set_debug(debug=debug, debug_fwd_impl=debug_fwd_impl)
        unwrap_model(module).train()

    dummy_bwd_packed_seq_params_iter = iter(dummy_bwd_packed_seq_params)
    batch_generator = make_batch_generator(
        microbatches if with_dummy else microbatches[pp_rank:],
        vpp_size=len(train_module),
    )

    torch.cuda.synchronize()
    nvshmem_barrier_all()

    if with_dummy:
        losses_reduced = forward_backward_func(
            forward_step_func=forward_step,
            data_iterator=batch_generator,
            model=train_module,
            num_microbatches=n_micro_batch,
            seq_length=total_seqlen,
            micro_batch_size=1,
            forward_only=forward_only,
            dummy_bwd_func=partial(
                dummy_backward_step,
                dummy_bwd_iter=dummy_bwd_packed_seq_params_iter,
                skip="orig" in mode,
            ),
        )
    else:
        from megatron.core.pipeline_parallel.schedules import get_forward_backward_func
        losses_reduced = get_forward_backward_func()(
            forward_step_func=forward_step,
            data_iterator=batch_generator,
            model=train_module,
            num_microbatches=n_micro_batch,
            seq_length=total_seqlen,
            micro_batch_size=1,
            forward_only=forward_only,
        )

    # Zero gradients (instead of optimizer step for now)
    for tm in train_module:
        for param in unwrap_model(tm).parameters():
            if hasattr(param, 'main_grad') and param.main_grad is not None:
                param.main_grad.zero_()

    return losses_reduced


# ================================
# Main Training Loop
# ================================
def train(config: dict):
    """Main training function."""
    # Setup logging directories
    log_paths = setup_log_directories(rank=rank, barrier_fn=torch.distributed.barrier)

    # Initialize Megatron parallel groups
    tp = config["tp"]
    pp = config["pp"]
    cp = config.get("cp", 1)
    dp = config["dp"]

    assert tp * pp * cp * dp == world_size, f"tp*pp*cp*dp != world_size: {tp}*{pp}*{cp}*{dp} != {world_size}"

    with time_it("initialize model parallel groups"):
        mpu.initialize_model_parallel(
            tensor_model_parallel_size=tp,
            pipeline_model_parallel_size=pp,
            context_parallel_size=cp,
            distributed_timeout_minutes=2,
        )

    tp_rank = mpu.get_tensor_model_parallel_rank()
    pp_rank = mpu.get_pipeline_model_parallel_rank()
    cp_rank = mpu.get_context_parallel_rank()
    dp_rank = mpu.get_data_parallel_rank()

    logger.info(f"TP: {tp_rank}/{tp}, PP: {pp_rank}/{pp}, CP: {cp_rank}/{cp}, DP: {dp_rank}/{dp}")

    # Initialize NVSHMEM
    with time_it("initialize nvshmem"):
        initialize_attention_server_comm()
        as_group = get_attn_server_group_gloo()
        as_world_size = torch.distributed.get_world_size(group=as_group)
        as_rank = get_attn_server_rank()

        if as_rank == 0:
            uid = nvshmem_get_unique_id()
        else:
            uid = nvshmem_alloc_empty_unique_id()
        torch.distributed.broadcast(uid, src=0)

        buffer_size = int(config.get("nvshmem_buffer_size_gb", 1.0) * 1024 ** 3)
        DispatcherWrapper.init(rank, local_rank, world_size, buffer_size, uid)
        logger.info(f"Successfully initialized NVSHMEM (buffer: {buffer_size / 1024**3:.2f} GB)")

    # Load HuggingFace config
    hf_model_name = config["model_name"]
    try:
        hf_config = AutoConfig.from_pretrained(hf_model_name, local_files_only=True)
    except Exception:
        logger.info(f"Downloading config for {hf_model_name}...")
        hf_config = AutoConfig.from_pretrained(hf_model_name, cache_dir="./models/")

    # Get Megatron config from HF
    all_config_dict = get_megatron_args_dict_from_hf_model(
        hf_model_name,
        seq_length=config["seq_length"],
        num_layers_override=config.get("num_layers_override"),
    )
    model_config_dict = all_config_dict["model_config_dict"]
    model_config_dict.update({
        "tp": tp,
        "pp": pp,
        "cp": cp,
        "micro_batch_size": config.get("micro_batch_size", 1),
        "global_batch_size": config.get("global_batch_size", dp),
        "train_iters": config.get("train_iters", 100),
        "model_name": hf_model_name,
        "data_path": config.get("data_path", ""),
        "tokenizer_type": config.get("tokenizer_type", "HuggingFaceTokenizer"),
        "tokenizer_model": config.get("tokenizer_model", hf_model_name),
    })

    # Create Megatron args
    designated_args = create_megatron_args(model_config_dict, log_paths)

    # Add DistCA arguments
    designated_args.extend([
        "--distca-nvshmem-buffer-size-gb", str(config.get("nvshmem_buffer_size_gb", 1.0)),
    ])
    if config.get("quit_if_maybe_oom", False):
        designated_args.append("--distca-quit-if-maybe-oom")

    def replace_parser_and_parse_args(parser: argparse.ArgumentParser):
        """Hijack parser to use designated args."""
        group = parser.add_argument_group(title='distca')
        group.add_argument("--distca-nvshmem-buffer-size-gb", type=float, default=1.0)
        group.add_argument("--distca-quit-if-maybe-oom", action="store_true", default=False)
        group.add_argument("--distca-use-planner", action="store_true", default=True)

        parser.parse_args = lambda *args, **kwargs: parser.parse_args(designated_args)
        return parser

    # Initialize Megatron
    with time_it("initialize megatron"):
        initialize_megatron(extra_args_provider=replace_parser_and_parse_args)
        args = get_args()

    # Setup model and optimizer
    with time_it("setup model and optimizer"):
        model_type = ModelType.encoder_or_decoder
        model, optimizer, opt_param_scheduler = setup_model_and_optimizer(
            partial(distca_model_provider, hf_config=hf_config),
            model_type,
        )

        train_module = unwrap_model(model)
        device = torch.cuda.current_device()
        for module in train_module:
            unwrap_model(module).init_ping_pong_communication_ctx(device)
        logger.info("Successfully setup model and optimizer")

    # Setup global batch (dataset)
    tokenizer = get_tokenizer() if hasattr(args, 'tokenizer') else None
    setup_global_batch(
        total_seq_len=config["num_tokens"],
        up_sample_factor=config.get("up_sample_factor", 4),
        elongate_factor=config.get("elongate_factor", 1),
        filter_threshold=config.get("filter_threshold", 65536),
        filter_ratio=config.get("filter_ratio", 0.5),
        should_add_debug_cases=config.get("should_add_debug_cases", False),
        change_long_doc_ratio=config.get("change_long_doc_ratio", 0.0),
        sample_name=config.get("sample_name", "wlbllm"),
        tokenizer=tokenizer,
        max_total_tokens=config.get("max_total_tokens"),
    )
    torch.distributed.barrier()

    set_random_seed(config.get("seed", 42), set_megatron=False)

    # Get model hidden sizes
    hidden_size_q = hf_config.hidden_size
    hidden_size_kv = hidden_size_q
    if hasattr(hf_config, "num_key_value_heads"):
        hidden_size_kv = hidden_size_kv * hf_config.num_key_value_heads // hf_config.num_attention_heads

    hidden_size_q_tp = hidden_size_q // tp
    hidden_size_k_tp = hidden_size_kv // tp
    dtype = torch.bfloat16 if config.get("use_bf16", True) else torch.float16
    element_size = dtype.itemsize
    num_head_in_dtype = hf_config.num_attention_heads * torch.float32.itemsize // element_size // tp

    num_tokens = config["num_tokens"]
    num_batches = config.get("num_batches", 1)
    num_microbatch = config.get("num_microbatch", pp)
    num_token_per_rank = num_tokens * num_batches // dp

    # Training loop
    logger.info("Starting training loop...")
    max_iters = config.get("train_iters", 10)

    for iteration in range(max_iters):
        logger.info(f"Iteration {iteration + 1}/{max_iters}")

        # Create microbatches
        microbatches_0, _ = create_pp_microbatches(
            num_microbatch, pp, as_rank, as_world_size,
            num_tokens, config.get("num_seqs", 3), dp,
            hidden_size_q_tp, hidden_size_k_tp, element_size, num_head_in_dtype,
            tp, dp, num_token_per_rank, num_batches,
            use_planner=config.get("use_planner", True),
            return_seq_lens=True,
        )

        microbatches_1, _ = create_pp_microbatches(
            num_microbatch, pp, as_rank, as_world_size,
            num_tokens, config.get("num_seqs", 3), dp,
            hidden_size_q_tp, hidden_size_k_tp, element_size, num_head_in_dtype,
            tp, dp, num_token_per_rank, num_batches,
            use_planner=config.get("use_planner", True),
            return_seq_lens=True,
        )

        set_random_seed(config.get("seed", 42), set_megatron=True)

        # Combine ping-pong microbatches
        microbatches = []
        for mb_0, mb_1 in zip(microbatches_0, microbatches_1):
            mb_0_psp = mb_0["packed_seq_params"]
            mb_1_psp = mb_1["packed_seq_params"]
            mb_0_mlp_psp = mb_0_psp.mlp_packed_seq_params
            mb_1_mlp_psp = mb_1_psp.mlp_packed_seq_params
            mb_0_psp.dispatcher_id = 0
            mb_1_psp.dispatcher_id = 1

            num_tokens_combined = sum(mb["position_ids"].numel() for mb in [mb_0, mb_1])
            ping_pong_params = PingPangPackedSeqParams(
                seq_params=[mb_0_psp, mb_1_psp],
                mlp_layout_seq_params=[mb_0_mlp_psp, mb_1_mlp_psp],
                max_seqlen_q=num_tokens_combined,
                max_seqlen_kv=num_tokens_combined,
            )
            input_ids = torch.randint(10, 1000, (num_tokens_combined,))
            mb = {
                "input_ids": input_ids,
                "position_ids": torch.concat([mb_0["position_ids"], mb_1["position_ids"]]),
                "packed_seq_params": ping_pong_params,
            }
            microbatches.append(mb)

        # Forward/backward pass
        torch.cuda.synchronize()
        torch.distributed.barrier()
        start_time = time.time()

        tf_config = args  # Use args as tf_config for now
        loss_reduced = forward_backward_batch(
            train_module,
            microbatches,
            tf_config,
            forward_only=False,
            mode="ping_pong",
            with_dummy=True,
        )

        torch.cuda.synchronize()
        torch.distributed.barrier()
        duration_ms = (time.time() - start_time) * 1000

        loss_value = extract_scalar_loss(loss_reduced)
        if rank == 0:
            logger.info(f"Iteration {iteration + 1}: loss = {loss_value:.4f}, time = {duration_ms:.2f} ms")

    logger.info("Training complete!")


# ================================
# Entry Point
# ================================
def main():
    parser = argparse.ArgumentParser(description="DistCA LLaMA Pre-training (Refactored)")

    # Model configuration
    parser.add_argument("--model-name", type=str, default="meta-llama/Llama-3.1-8B",
                        help="HuggingFace model name or path")
    parser.add_argument("--num-layers-override", type=int, default=None,
                        help="Override number of layers (for testing)")

    # Training configuration
    parser.add_argument("--train-iters", type=int, default=10,
                        help="Number of training iterations")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")

    # Parallelism configuration
    parser.add_argument("--tp", type=int, default=1,
                        help="Tensor parallel size")
    parser.add_argument("--pp", type=int, default=1,
                        help="Pipeline parallel size")
    parser.add_argument("--cp", type=int, default=1,
                        help="Context parallel size")
    parser.add_argument("--dp", type=int, default=None,
                        help="Data parallel size (auto-computed if not provided)")

    # Data configuration
    parser.add_argument("--seq-length", type=int, default=4096,
                        help="Sequence length")
    parser.add_argument("--num-tokens", type=int, default=1024,
                        help="Number of tokens per batch")
    parser.add_argument("--num-batches", type=int, default=1,
                        help="Number of batches")
    parser.add_argument("--num-microbatch", type=int, default=None,
                        help="Number of microbatches (defaults to PP size)")
    parser.add_argument("--num-seqs", type=int, default=3,
                        help="Number of sequences")

    # Dataset configuration
    parser.add_argument("--sample-name", type=str, default="wlbllm",
                        choices=["wlbllm", "prolong", "bookcorpus", "wikitext", "openwebtext", "c4"],
                        help="Dataset to use (synthetic or real)")
    parser.add_argument("--data-path", type=str, default="",
                        help="Path to dataset for real datasets")
    parser.add_argument("--tokenizer-type", type=str, default="HuggingFaceTokenizer",
                        help="Tokenizer type")
    parser.add_argument("--up-sample-factor", type=int, default=4,
                        help="Dataset up-sampling factor")
    parser.add_argument("--elongate-factor", type=int, default=1,
                        help="Dataset elongate factor")
    parser.add_argument("--filter-threshold", type=int, default=65536,
                        help="Dataset filter threshold")
    parser.add_argument("--filter-ratio", type=float, default=0.5,
                        help="Dataset filter ratio")
    parser.add_argument("--max-total-tokens", type=int, default=None,
                        help="Max total tokens for real datasets")

    # DistCA configuration
    parser.add_argument("--use-planner", action="store_true", default=True,
                        help="Use DistCA planner")
    parser.add_argument("--nvshmem-buffer-size-gb", type=float, default=1.0,
                        help="NVSHMEM buffer size in GB")
    parser.add_argument("--quit-if-maybe-oom", action="store_true",
                        help="Quit if estimated memory exceeds GPU max")

    # Mixed precision
    parser.add_argument("--use-bf16", action="store_true", default=True,
                        help="Use bfloat16 (otherwise fp16)")

    args = parser.parse_args()

    # Compute DP if not provided
    if args.dp is None:
        args.dp = world_size // (args.tp * args.pp * args.cp)

    # Set num_microbatch default to PP size
    if args.num_microbatch is None:
        args.num_microbatch = args.pp

    # Convert to config dict
    config = vars(args)

    logger.info(f"Configuration: {config}")
    train(config)


if __name__ == "__main__":
    main()
