"""
DistCA Test Script
"""

# ================================
# Import torch and megatron
# ================================
import argparse
import os
import logging
import time
from dataclasses import dataclass
from datetime import timedelta
from functools import partial
from pathlib import Path
from typing import Optional, Tuple, List, Union
from pickle import dump
import typing
from typing import Dict

from utils.logging import (
    setup_logging,
    time_it,
    setup_log_directories,
    redirect_external_loggers,
)
from utils.cpu_affinity import set_cpu_affinity
from utils.hf_config import (
    get_megatron_args_dict_from_hf_model,
)
from utils.estimate import log_memory_estimate, log_flops_estimate
from utils.token_monitor import (
    monitor_batch_tokens,
    set_token_monitor_config,
)


if typing.TYPE_CHECKING:
    from transformers import PretrainedConfig

rank = int(os.environ["RANK"])
world_size = int(os.environ["WORLD_SIZE"])
local_rank = int(os.environ["LOCAL_RANK"])

logger = setup_logging(
    rank=rank, world_size=world_size,
    level=logging.DEBUG,
    console_ranks=[0],
    # console_ranks=list(range(0, world_size, 8)),  # Only rank 0 logs to console
)

with time_it("import torch"):
    import torch
    import torch.cuda.nvtx

with time_it("set device"):
    torch.cuda.set_device(local_rank)
    torch.set_default_device(torch.device("cuda", local_rank))

with time_it("init_process_group"):
    torch.distributed.init_process_group(
        backend="cpu:gloo,cuda:nccl", 
        rank=rank, 
        world_size=world_size, 
        timeout = timedelta(seconds=60),
    )

import numpy

# import warnings
# with warnings.catch_warnings():
#     warnings.filterwarnings(
#         "ignore",
#         message=r"Failed to import megatron plugin due to: ImportError\(.*get_tensor_model_parallel_group_if_none.*megatron.core.utils.*\)",
#         category=UserWarning,
#         module=r"modelopt\.torch\.utils\.import_utils"
#     )


# ================================
# Environment Variables
# ================================
env_vars = dict(
    CUDA_DEVICE_MAX_CONNECTIONS=1,
    # TORCH_NCCL_CONNECT_TIMEOUT=60000,
    # TORCH_NCCL_ASYNC_ERROR_HANDLING=1,
)
for k, v in env_vars.items():
    os.environ[k] = str(v)


# ================================
# Logging Setup
# ================================
# Fetch rank early for use in formatter

# Initialize logging with rank-aware formatting
# Only specified ranks will log to console; all ranks get file logging later


# ================================
# Core-binding
# ================================
set_cpu_affinity(local_rank, ncpu_per_proc=16, logger=logger)

logger.info(f"Rank: {rank}, World Size: {world_size}, Local Rank: {local_rank}")


# ================================
# PyTorch Imports and Device Setup
# ================================


# ================================
# Megatron Imports
# ================================
with time_it("import megatron.core"):
    import megatron.core
    from megatron.core import tensor_parallel
    from megatron.core import parallel_state
    mpu = parallel_state

    # Additional Megatron imports (after core is loaded)
    import megatron.core.pipeline_parallel.schedules
    import megatron.core.transformer.transformer_layer
    import megatron.training.training
    from megatron.training.global_vars import get_args, get_wandb_writer
    from megatron.training.arguments import core_transformer_config_from_args
    from megatron.training.yaml_arguments import core_transformer_config_from_yaml
    from megatron.core.transformer.transformer_config import TransformerConfig
    from megatron.core.num_microbatches_calculator import get_num_microbatches
    from megatron.training.training import (
        setup_model_and_optimizer,
        build_train_valid_test_data_iterators,
        train as megatron_original_train,
        get_timers,
    )
    from megatron.core.enums import ModelType
    from megatron.core.models.gpt import GPTModel
    from megatron.training.tokenizer.tokenizer import MegatronTokenizer
    from megatron.core.datasets.utils import Split
    from megatron.core.datasets.blended_megatron_dataset_builder import BlendedMegatronDatasetBuilder
    from megatron.core.datasets.gpt_dataset import MockGPTDataset, GPTDataset, GPTDatasetConfig
    from megatron.training import get_tokenizer
    from megatron.training.utils import (
        get_batch_on_this_cp_rank,
        get_batch_on_this_tp_rank,
        get_blend_and_blend_per_split,
    )
    from megatron.core.rerun_state_machine import RerunDataIterator, get_rerun_state_machine
    from megatron.core.utils import StragglerDetector
    from megatron.training.utils import unwrap_model
    from megatron.core.packed_seq_params import PackedSeqParams



# ====================================
# Initialize Megatron Parallel Groups
# ====================================
tp = 1; pp = 1; cp = 1;

with time_it("initialize model parallel groups"):
    # tp = min(8, world_size);
    mpu.initialize_model_parallel(
        tensor_model_parallel_size=tp, 
        pipeline_model_parallel_size=pp,
        context_parallel_size=cp,
        distributed_timeout_minutes=2,
        # nccl_communicator_config_path=None, # default
        order = "tp-cp-ep-dp-pp", # default
    )

    # get the tp,pp,cp,dp,ep rank of this process
    tp_rank = mpu.get_tensor_model_parallel_rank();   tp_size = mpu.get_tensor_model_parallel_world_size()
    pp_rank = mpu.get_pipeline_model_parallel_rank(); pp_size = mpu.get_pipeline_model_parallel_world_size()
    cp_rank = mpu.get_context_parallel_rank();        cp_size = mpu.get_context_parallel_world_size()
    dp_rank = mpu.get_data_parallel_rank();           dp_size = mpu.get_data_parallel_world_size()
    ep_rank = mpu.get_expert_model_parallel_rank();   ep_size = mpu.get_expert_model_parallel_world_size()
    logger.info(f"TP: {tp_rank} / {tp_size}, PP: {pp_rank} / {pp_size}, CP: {cp_rank} / {cp_size}, DP: {dp_rank} / {dp_size}, EP: {ep_rank} / {ep_size}")

    logger.info(f"TP Ranks: {torch.distributed.get_process_group_ranks(mpu.get_tensor_model_parallel_group())}")
    logger.info(f"PP Ranks: {torch.distributed.get_process_group_ranks(mpu.get_pipeline_model_parallel_group())}")
    logger.info(f"CP Ranks: {torch.distributed.get_process_group_ranks(mpu.get_context_parallel_group())}")
    logger.info(f"DP Ranks: {torch.distributed.get_process_group_ranks(mpu.get_data_parallel_group())}")
    logger.info(f"EP Ranks: {torch.distributed.get_process_group_ranks(mpu.get_expert_model_parallel_group())}")

torch.distributed.barrier()
logger.info(f"Finish initializing megatron parallel groups.")


# ================================
# Initialize NVSHMEM comm group
# ================================

# local_rank
from distca.runtime.attn_kernels.ops import (
    nvshmem_get_unique_id, 
    nvshmem_alloc_empty_unique_id,
    DispatcherWrapper
)
from distca.runtime.megatron.create_group import (
    get_attn_server_group_gloo,
    get_attn_server_rank, 
    get_attn_server_group_src_rank,
    initialize_attention_server_comm
)
from distca.runtime.megatron.ping_pong.transformer_block import PingPongGPTModel
from distca.utils.megatron_test_utils import init_mcore_model
from distca.utils.megatron_test_utils import (
    gptmodel_forward, make_batch_generator, unwrap_model,
)
from distca.runtime.megatron.packed_seq_params import (
    arg_to_cuda, PingPangSingleStepPackedSeqParams, PingPangPackedSeqParams,
)
# Additional imports for create_pp_microbatches
from distca.utils.traceback import enable_clickable_excepthook, enable_trace_calls
enable_clickable_excepthook()

with time_it("initialize nvshmem"):
    initialize_attention_server_comm()
    as_group = get_attn_server_group_gloo()
    as_world_size = torch.distributed.get_world_size(group=as_group)
    as_rank = get_attn_server_rank()
    as_src_rank = get_attn_server_group_src_rank()

    if as_rank == 0:
        uid = nvshmem_get_unique_id()
    else:
        uid = nvshmem_alloc_empty_unique_id()
    torch.distributed.broadcast(uid, src=0)

    buffer_size = 1 * 1024 ** 3 # 1 GB
    DispatcherWrapper.init(
        rank, local_rank, world_size, buffer_size, uid
    )

    logger.info(f"Successfully initialized NVSHMEM comm group")

# ================================
# Setup logging directories
# ================================

log_paths = setup_log_directories(
    rank=rank,
    barrier_fn=torch.distributed.barrier,
)

# Redirect Megatron logs to the same rank log files
redirect_external_loggers(["megatron"], level=logging.INFO)

log_root_dir = log_paths.log_root_dir
data_cache_path = log_paths.data_cache_path
ckpt_path = log_paths.ckpt_path
tensorboard_path = log_paths.tensorboard_path
oom_snapshot_path = log_root_dir / "oom_snapshot"
oom_snapshot_path.mkdir(parents=True, exist_ok=True)
data_path = Path(__file__).parent / 'data_process' / 'code_content_document'
data_path = data_path.resolve().absolute()
logger.info(f"Data path: {data_path}")



# ================================
# Model Configuration from HuggingFace
# ================================

# Only support: Use a HuggingFace model name/path to auto-load config
HF_MODEL_NAME = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
# HF_MODEL_NAME = "meta-llama/Llama-3.1-8B"

# Sequence length for training (can be shorter than model's max)
k = 1024
seq_length = k * 4
# seq_length = k * 128

# Compute the number of layers based on pp
# num_layers_override = None
num_layers_override = 2

# TODO: Overriding logic is a little too awkward. We should find a better way to do this.
logger.info(f"Loading model config from HuggingFace: {HF_MODEL_NAME}")
all_config_dict = get_megatron_args_dict_from_hf_model(
    HF_MODEL_NAME,
    seq_length=seq_length,
    num_layers_override=num_layers_override,
)
hf_config = all_config_dict["hf_config"]
megatron_args = all_config_dict["megatron_args"]
model_config_dict = all_config_dict["model_config_dict"]
logger.info(f"HF config: {type(hf_config) = }, {hf_config = }")
logger.info(f"Megatron args: {type(megatron_args) = }, {megatron_args = }")
logger.info(f"Model config: {type(model_config_dict) = }, {model_config_dict = }")




# ======================================
# Megatron LM Designated CLI Arguments
# ======================================


designated_args = [
    # Minimal required Megatron arguments to pass validation.
    # Organized to match megatron-args.txt ordering.
    "--seed", "42",

    ####################
    # 1. Model Architecture
    ####################
    "--num-layers", str(model_config_dict["num_layers"]),
    "--hidden-size", str(model_config_dict["hidden_size"]),
    "--ffn-hidden-size", str(model_config_dict["ffn_hidden_size"]),
    "--num-attention-heads", str(model_config_dict["num_attention_heads"]),
    "--group-query-attention",
    "--num-query-groups", str(model_config_dict["num_query_groups"]),
    "--max-position-embeddings", str(model_config_dict["max_position_embeddings"]),
    "--position-embedding-type", str(model_config_dict["position_embedding_type"]),
    "--rotary-base", str(model_config_dict["rotary_base"]),
    "--normalization", str(model_config_dict["normalization"]),
    "--swiglu" if model_config_dict["swiglu"] else None,
    "--untie-embeddings-and-output-weights" if model_config_dict["untie_embeddings_and_output_weights"] else None,
    "--seq-length", str(model_config_dict["seq_length"]),
    "--vocab-size", str(model_config_dict["vocab_size"]),
    "--attention-backend", "auto",

    ####################
    # 2. Training Hyperparameters
    ####################
    "--micro-batch-size", "1",
    "--lr", "1.0e-5",
    # "--train-samples", "100000",
    # "--train-iters", "100000",
    "--train-iters", "2",
    "--lr-warmup-init", "1e-5",
    "--lr-decay-iters", "1000000",
    "--lr-decay-style", "constant",
    # "--lr-warmup-iters", "1000",
    # "--lr-warmup-fraction", "0.0",
    "--min-lr", "1e-6",
    # "--min-lr-ratio", None,
    # "--warmup-style", "constant",
    "--weight-decay", "0.01",
    "--weight-decay-incr-style", "constant",
    # "--use-checkpoint-opt-param-scheduler",
    "--lr-wsd-decay-style", "linear",

    ####################
    # 3. Dropout & Regularization
    ####################
    # "--no-masked-softmax-fusion",
    # "--no-bias-gelu-fusion",
    # "--no-bias-swiglu-fusion",
    # "--no-bias-dropout-fusion",
    # "--no-rope-fusion",

    ####################
    # 4. Mixed Precision
    ####################
    # "--bf16",  # REQUIRED for flash attention to work!
    "--fp16",  # REQUIRED for flash attention to work!
    "--transformer-impl", "transformer_engine",

    ####################
    # 5. Parallelism & Distributed Training
    ####################
    "--tensor-model-parallel-size", str(tp),
    "--pipeline-model-parallel-size", str(pp),
    "--context-parallel-size", str(cp),
    "--cp-comm-type", "p2p",  # async
    # "--distributed-backend", "nccl",
    "--distributed-timeout-minutes", "1",
    "--local-rank", str(local_rank),

    ####################
    # 6. Data/IO
    ####################
    "--data-path", str(data_path),
    # "--mock-data",
    # "--tokenizer-type", "NullTokenizer",
    "--tokenizer-type", "HuggingFaceTokenizer",
    "--tokenizer-model", "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
    # Note: --vocab-size is now provided by model_arch_args from HF config
    "--data-cache-path", str(data_cache_path),
    "--tiktoken-pattern", "v2",
    # "--no-create-attention-mask-in-dataloader",
    # "--no-mmap-bin-files",
    "--num-workers", "1",
    "--split", "90,5,5",
    # --data-path $DATA_PATH
    # --vocab-file $VOCAB_FILE
    # --merge-file $MERGE_FILE

    ####################
    # 7. Checkpointing & Saving
    ####################
    "--save", str(ckpt_path),
    "--load", str(ckpt_path),
    "--save-interval", "0",

    ####################
    # 8. Logging & Monitoring
    ####################
    # "--eval-interval", "10",
    # "--log-progress",
    "--log-interval", "1",
    "--log-params-norm",
    "--log-timers-to-tensorboard",
    "--log-memory-to-tensorboard",
    "--log-throughput",
    "--tensorboard-dir", str(tensorboard_path),  # Required for wandb logging to work!
    "--tensorboard-log-interval", "1",
    "--wandb-project", "distca",
    "--wandb-exp-name", "test-megatron-init",
    "--logging-level", "20",  # Megatron logging level: INFO = 20
    # "--record-memory-history",
    # "--profile",
    "--profile-step-start", "2",
    "--profile-step-end", "4",
    # "--profile-ranks",

    ####################
    # 9. Advanced Features & Extensions
    ####################
    # MoE related
    # fp8 related

    ####################
    # 10. Special Modes/Debugging
    ####################
    # "--no-check-for-nan-in-loss-and-grad",
    "--check-for-spiky-loss",
    "--check-for-large-grads",

    ####################
    # 11. Miscellaneous
    ####################
    # Activation Recomputation
    # "--recompute-granularity", "selective",
    "--recompute-granularity", "selective",
    "--recompute-modules", "core_attn", 
    # "mlp",

    # Cuda Graphs
    # "--enable-cuda-graph", "True",

    # Network/Communication Overlap
    # "--overlap-grad-reduce",
    # "--overlap-param-gather",
    # "--overlap-param-gather-with-optimizer-step",
    # "--tp-comm-overlap",
    "--distca-quit-if-maybe-oom",
]

# Filter out None values (from conditional args like --swiglu)
designated_args = [arg for arg in designated_args if arg is not None]


# ================================
# DistCA Configuration
# ================================
@dataclass
class DistCAConfig(TransformerConfig):
    """Configuration object for DistCA."""

    distca_nvshmem_buffer_size_gb: float = 1.0
    """Set the NVSHMEM buffer size (in GB). 
    TODO: By default, we want to make it configurable by the planner."""


    distca_quit_if_maybe_oom: bool = False
    """If True, the program will quit if the estimated memory can probably exceeds the GPU max memory."""

    pass

def replace_parser_and_parse_args(parser: argparse.ArgumentParser):    
    """Hijack the parser, and use the `designated_args` as the arguments. Then we also inject some of our own arguments."""

    # Define the extra arguments for DistCA
    group = parser.add_argument_group(title='distca')
    
    group.add_argument("--distca-nvshmem-buffer-size-gb", type=int, default=2, 
    help="Set the NVSHMEM buffer size (in GB)")

    group.add_argument("--distca-quit-if-maybe-oom", action="store_true", default=False, 
    help="If True, the program will quit if the estimated memory can probably exceeds the GPU max memory.")

    old_parser_parse_args = parser.parse_args
    old_parser_parse_known_args = parser.parse_known_args
    parser.parse_args = lambda *args, **kwargs: old_parser_parse_args(designated_args)
    parser.parse_known_args = lambda *args, **kwargs: old_parser_parse_known_args(designated_args)

    return parser

with time_it("initialize megatron"):
    from megatron.training.initialize import initialize_megatron    
    initialize_megatron(
        extra_args_provider=replace_parser_and_parse_args, # default
        args_defaults={}, # default
        get_embedding_ranks=None, # default
        get_position_embedding_ranks=None, # default
    )
    logger.info(f"Successfully initialized megatron")



checkpointing_context = {}

args = get_args()
if args.yaml_cfg is not None:
    assert False, "YAML config is not supported yet becuase inside the core_transformer_config_from_yaml, you can only use TransformerConfig. We need to extend the config class to support DistCA configs."
    config = core_transformer_config_from_yaml(args, "language_model")
else:
    config = core_transformer_config_from_args(args, config_class=DistCAConfig)
assert isinstance(config, TransformerConfig)


# ================================
# Report theoretical memory and FLOPS estimates
# ================================
num_microbatches = get_num_microbatches()

# Log memory estimates
memory_estimate = log_memory_estimate(
    args, 
    num_microbatches=num_microbatches, 
    verbose=True, 
    logger=logger
)

if memory_estimate.maybe_oom() and args.distca_quit_if_maybe_oom:
    logger.error(f"Estimated memory ({memory_estimate.total_gb:.2f} GB) can probably exceeds GPU max memory ({memory_estimate.gpu_max_memory_gb:.2f} GB). Training will likely OOM!")
    raise RuntimeError(f"Estimated memory {memory_estimate.total_gb:.2f} GB can probably exceeds GPU max memory {memory_estimate.gpu_max_memory_gb:.2f} GB. Training will likely OOM! Quit the program.")

# Log FLOPS estimates
flops_estimate = log_flops_estimate(
    args,
    num_microbatches=num_microbatches,
    micro_batch_size=args.micro_batch_size,
    dp_world_size=mpu.get_data_parallel_world_size(),
    tp=tp,
    pp=pp,
    cp=cp,
    logger=logger,
)


# ================================
# Set pytorch JIT layer fusion options and warmup JIT functions.
# ================================
# with time_it("set_jit_fusion_options()"):
#     # 01:08:00 [Rank 0] INFO set_jit_fusion_options() took 5.3942 seconds
#     from megatron.training.initialize import set_jit_fusion_options
#     set_jit_fusion_options()


def monitor_oom():
    torch.cuda.memory._record_memory_history(True,
        # keep 100,000 alloc/free events from before the snapshot
        trace_alloc_max_entries=100000,

        # record stack information for the trace events
        trace_alloc_record_context=True)

    def oom_observer(device, alloc, device_alloc, device_free):
        # snapshot right after an OOM happened
        print('saving allocated state during OOM')
        snapshot = torch.cuda.memory._snapshot()
        snapshot_path = oom_snapshot_path / f"oom_rank-{torch.distributed.get_rank()}.pkl"
        with open(snapshot_path, 'wb') as f:
            dump(snapshot, f)

    torch._C._cuda_attach_out_of_memory_observer(oom_observer)
    return

# ================================
# Model, optimizer, and learning rate.
# ================================

def model_provider(pre_process=True, post_process=True) -> Union[GPTModel, 'megatron.legacy.model.GPTModel']:
    # TODO: Simplified model provider for DistCA.
    """Builds the model.

    If you set the use_legacy_models to True, it will return the legacy GPT model and if not the mcore GPT model.

    Args:
        pre_process (bool, optional): Set to true if you need to compute embedings. Defaults to True.
        post_process (bool, optional): Set to true if you need to want to compute output logits/loss. Defaults to True.


    Returns:
        Union[GPTModel, megatron.legacy.model.GPTModel]: The returned model
    """
    args = get_args()
    use_te = args.transformer_impl == "transformer_engine"
    assert use_te, "Transformer Engine is required for DistCA."

    if args.record_memory_history:
        monitor_oom()
    logger.info(f"Building model for rank {torch.distributed.get_rank()}")

    
    from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_with_transformer_engine_spec
    transformer_layer_spec = get_gpt_layer_with_transformer_engine_spec(
        args.num_experts, args.moe_grouped_gemm,
        args.qk_layernorm, args.multi_latent_attention, args.moe_use_legacy_grouped_gemm)

    model = GPTModel(
        config=config,
        transformer_layer_spec=transformer_layer_spec,
        vocab_size=args.padded_vocab_size,
        max_sequence_length=args.max_position_embeddings,
        pre_process=pre_process,
        post_process=post_process,
        fp16_lm_cross_entropy=args.fp16_lm_cross_entropy,
        parallel_output=True,
        share_embeddings_and_output_weights=not args.untie_embeddings_and_output_weights,
        position_embedding_type=args.position_embedding_type,
        rotary_percent=args.rotary_percent,
        rotary_base=args.rotary_base,
        rope_scaling=args.use_rope_scaling,
        mtp_block_spec=None,
    )

    return model



def distca_model_provider(
    pre_process=True, post_process=True, 
    hf_config_: 'PretrainedConfig'= hf_config, # bound argument
) -> Union['PingPongGPTModel']:
    """Builds the DistCA model using init_mcore_model (similar to _build_model_optimizer approach).
    
    This function is parallel to model_provider but uses init_mcore_model which automatically
    creates a PingPongGPTModel through the model registry system.
    
    Args:
        pre_process (bool, optional): Set to true if you need to compute embeddings. Defaults to True.
        post_process (bool, optional): Set to true if you need to compute output logits/loss. Defaults to True.
    
    Returns:
        PingPongGPTModel: The returned model
    """
    args = get_args()
    use_te = args.transformer_impl == "transformer_engine"
    assert use_te, "Transformer Engine is required for DistCA."
    
    if args.record_memory_history:
        monitor_oom()
    logger.info(f"Building DistCA model for rank {torch.distributed.get_rank()}")
    
    # Get share_embeddings_and_output_weights from args (mirrors model_provider)
    share_embeddings_and_output_weights = not args.untie_embeddings_and_output_weights
    freeze_moe_router=False # TODO: Find this configuration in the args
    
    # Initialize model using init_mcore_model (this returns PingPongGPTModel via registry)
    parallel_model = init_mcore_model(
        config,  # tf_config (TransformerConfig) - already available as 'config'
        hf_config_,
        pre_process,
        post_process,
        share_embeddings_and_output_weights=share_embeddings_and_output_weights,
        value=False,
        freeze_moe_router=freeze_moe_router,
    )
    parallel_model.to("cuda")
    
    return parallel_model


with time_it("setup_model_and_optimizer()"):
    model_type = ModelType.encoder_or_decoder
    model, optimizer, opt_param_scheduler = setup_model_and_optimizer(
        # model_provider,
        distca_model_provider,
        model_type, checkpointing_context=checkpointing_context
    )
    
    

    train_module = unwrap_model(model)
    device = torch.cuda.current_device()
    for module in train_module:
        unwrap_model(module).init_ping_pong_communication_ctx(device)
    logger.info(f"Successfully setup model and optimizer for rank {rank}")

    logger.info('after model, optimizer, and learning rate scheduler are built')

    logger.debug(f"Model: {model}")
    logger.debug(f"Optimizer: {optimizer}")
    logger.debug(f"Learning Rate Scheduler: {opt_param_scheduler}")

# ================================
# DataLoader Setup
# ================================


def __original_is_dataset_built_on_rank():
    return (
        mpu.is_pipeline_first_stage() or mpu.is_pipeline_last_stage()
    ) and mpu.get_tensor_model_parallel_rank() == 0


def is_dataset_built_on_rank() -> bool:
    """DistCA: Only build the dataset on Rank 0. 
    Rank 0 will be the only rank that will build the dataset and plan.
    """
    return mpu.get_data_parallel_rank() == 0    


def core_gpt_dataset_config_from_args(args):
    """Adapt from pretrain_gpt.py"""
    tokenizer = get_tokenizer()

    # Sometimes --data-path is too long, instead we parse it from a file.
    blend: Optional[Tuple[List[str], Optional[List[float]]]]
    blend_per_split: Optional[List[Optional[Tuple[List[str], Optional[List[float]]]]]]
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
        s3_cache_path=args.s3_cache_path,
    )


def train_valid_test_datasets_provider(train_val_test_num_samples):
    """Build the train test and validation datasets.
    Only build the dataset on the first rank. 
    
    This way, we will only return the dataset on the first rank, 
    and then build the dataset on the other ranks.

    Args:
        train_val_test_num_samples : A list containing the number of samples in train test and validation.
    """
    args = get_args()

    config = core_gpt_dataset_config_from_args(args)
    logger.info(f"> GPTDatasetConfig: {config}")

    dataset_type = GPTDataset # NOTE: Add DistCAMockGPTDataset and stuff

    logger.info("> building train, validation, and test datasets for GPT ...")

    train_ds, valid_ds, test_ds = BlendedMegatronDatasetBuilder(
        dataset_type,
        train_val_test_num_samples,
        is_dataset_built_on_rank,
        config
    ).build()

    logger.info("> finished creating GPT datasets ...")

    return train_ds, valid_ds, test_ds


def cyclic_iter(iter):
    while True:
        for x in iter:
            yield x

old_args_dataloader_type = args.dataloader_type
args.dataloader_type = "external" # TODO: Set to "single" for the normal data parallel data loader.

with time_it("setup data iterators"):    
    train_valid_test_datasets_provider.is_distributed = True

    # Build loaders.
    # NOTE: (Hack) This hack will make sure the dataset only builds on Rank 0
    # such that all of the decisions are made on Rank 0.
    # This will be useful when we get to `get_batch()` function.
    # Expected Call stack:
    # - build_train_valid_test_data_iterators
    #   |- build_train_valid_test_data_loaders
    #      |- build_pretraining_data_loader
    #        (if normal path is chosen, then you will get a torch.utils.data.DataLoader back)
    #        |- torch.utils.data.DataLoader
    #           |- MegatronPretrainingSampler (contains in the torch DataLoader)
    #        (if "external" is specified, then you only get back the GPTDataset dataset object)
    from megatron.training.training import build_train_valid_test_data_loaders
    train_dataloader, valid_dataloader, test_dataloader = \
        build_train_valid_test_data_loaders(
            train_valid_test_datasets_provider)

    # Build iterators.
    dl_type = args.dataloader_type
    assert dl_type in ['single', 'cyclic', 'external'], f"Expected dl_type is one of 'single', 'cyclic', or 'external', but got {dl_type}"

    def _get_iterator(dataloader_type, dataloader, is_train=False):
        """Return dataset iterator."""

        # Build the MegatronPretrainingSampler and torch dataloader here.
        if dl_type == 'external':
            dataset = dataloader
            assert isinstance(dataset, GPTDataset), f"Expected GPTDataset, but got {type(dataset)}"
            # Actually build the MegatronPretrainingSampler and torch dataloader here.
            from megatron.legacy.data.data_samplers import MegatronPretrainingSampler
            consumed_samples = 0
            if is_train:
                consumed_samples = args.consumed_train_samples
            
            micro_batch_size = args.micro_batch_size
            batch_sampler = MegatronPretrainingSampler(
                total_samples=len(dataset),
                consumed_samples=consumed_samples,
                micro_batch_size=micro_batch_size,
                # data_parallel_rank=mpu.get_data_parallel_rank(),
                # data_parallel_size=mpu.get_data_parallel_world_size(),
                data_parallel_rank=0,
                data_parallel_size=1
            )
            dataloader = torch.utils.data.DataLoader(
                dataset,
                batch_sampler=batch_sampler,
                num_workers=args.num_workers,
                pin_memory=True,
                persistent_workers=True if args.num_workers > 0 else False,
            )
            pass
    
        # Return the RerunDataIterator here.
        if dataloader_type == "single":
            return RerunDataIterator(iter(dataloader))
        elif dataloader_type == "cyclic":
            return RerunDataIterator(iter(cyclic_iter(dataloader)))
        elif dataloader_type == "external":
            # External dataloader is passed through. User is expected to define how to iterate.
            if isinstance(dataloader, list):
                return [RerunDataIterator(iter(d)) for d in dataloader]
            else:
                return RerunDataIterator(iter(dataloader))
        else:
            raise RuntimeError("unexpected dataloader type")

    if train_dataloader is not None:
        train_data_iterator = _get_iterator(dl_type, train_dataloader, is_train=True)
    else:
        train_data_iterator = None

    if valid_dataloader is not None:
        valid_data_iterator = _get_iterator(dl_type, valid_dataloader)
    else:
        valid_data_iterator = None

    if test_dataloader is not None:
        test_data_iterator = _get_iterator(dl_type, test_dataloader)
    else:
        test_data_iterator = None

    print(f"train_data_iterator: {train_data_iterator}")
    print(f"valid_data_iterator: {valid_data_iterator}")
    print(f"test_data_iterator: {test_data_iterator}")

    # item = next(train_data_iterator)
    # print(f"item: {item}")
    args.dataloader_type = old_args_dataloader_type

logger.info('done with setup ...')
# exit(0)

# ==========================================================
# Monkey Patch training functions with NVTX markers
# ==========================================================
def monkey_patch_nvtx_markers():
    # Patch forward_step and backward_step with nvtx markers
    old_forward_step = megatron.core.pipeline_parallel.schedules.forward_step
    old_backward_step = megatron.core.pipeline_parallel.schedules.backward_step


    def forward_step_with_nvtx(*args, **kwargs):
        with torch.cuda.nvtx.range("forward_step"):
            return old_forward_step(*args, **kwargs)

    def backward_step_with_nvtx(*args, **kwargs):
        # with torch.autograd.profiler.emit_nvtx(record_shapes=True):
        with torch.cuda.nvtx.range("backward_step"):
            return old_backward_step(*args, **kwargs)

    megatron.core.pipeline_parallel.schedules.forward_step = forward_step_with_nvtx
    megatron.core.pipeline_parallel.schedules.backward_step = backward_step_with_nvtx



    # Patch the functions in forward_backward_func
    old_forward_backward_pipelining_with_interleaving = megatron.core.pipeline_parallel.schedules.forward_backward_pipelining_with_interleaving
    old_forward_backward_pipelining_without_interleaving = megatron.core.pipeline_parallel.schedules.forward_backward_pipelining_without_interleaving
    old_forward_backward_no_pipelining = megatron.core.pipeline_parallel.schedules.forward_backward_no_pipelining

    def forward_backward_pipelining_with_interleaving_with_nvtx(*args, **kwargs):
        with torch.cuda.nvtx.range("forward_backward_pipelining_with_interleaving"):
            return old_forward_backward_pipelining_with_interleaving(*args, **kwargs)
    def forward_backward_pipelining_without_interleaving_with_nvtx(*args, **kwargs):
        with torch.cuda.nvtx.range("forward_backward_pipelining_without_interleaving"):
            return old_forward_backward_pipelining_without_interleaving(*args, **kwargs)
    def forward_backward_no_pipelining_with_nvtx(*args, **kwargs):
        with torch.cuda.nvtx.range("forward_backward_no_pipelining"):
            return old_forward_backward_no_pipelining(*args, **kwargs)
    megatron.core.pipeline_parallel.schedules.forward_backward_pipelining_with_interleaving = forward_backward_pipelining_with_interleaving_with_nvtx
    megatron.core.pipeline_parallel.schedules.forward_backward_pipelining_without_interleaving = forward_backward_pipelining_without_interleaving_with_nvtx
    megatron.core.pipeline_parallel.schedules.forward_backward_no_pipelining = forward_backward_no_pipelining_with_nvtx


    # Patch optimizer.step with nvtx markers
    old_optimizer_step = optimizer.step
    def optimizer_step_with_nvtx(*args, **kwargs):
        with torch.cuda.nvtx.range("optimizer_step"):
            return old_optimizer_step(*args, **kwargs)
    optimizer.step = optimizer_step_with_nvtx


    # Patch a train_step
    old_train_step = megatron.training.training.train_step
    def train_step_with_nvtx(*args, **kwargs):
        logger.debug(f"Start train_step")
        with torch.cuda.nvtx.range("train_step"):
            return old_train_step(*args, **kwargs)
    megatron.training.training.train_step = train_step_with_nvtx


    # Patch each layer such that I will know when the forward function gets called
    old_transformer_layer_forward = megatron.core.transformer.transformer_layer.TransformerLayer.forward
    def transformer_layer_forward_with_nvtx(self, *args, **kwargs):
        if self.layer_number in [0, 1]:
            logger.debug(f"Start transformer_layer_forward[{self.layer_number}]")
        with torch.cuda.nvtx.range(f"transformer_layer_forward[{self.layer_number}]"):
            r = old_transformer_layer_forward(self, *args, **kwargs)
        if self.layer_number == model_config_dict["num_layers"] - 1:
            logger.debug(f"End transformer_layer_forward[{self.layer_number}]")
        return r
    megatron.core.transformer.transformer_layer.TransformerLayer.forward = transformer_layer_forward_with_nvtx

    logger.info(f"model: {model}")
    return

monkey_patch_nvtx_markers()


# ================================
# Get batch functions
# ================================
stimer = StragglerDetector()

# Configure token monitoring (optional - defaults are sensible)
set_token_monitor_config(enabled=True, max_tokens_to_decode=200, max_samples_to_log=2)

def __original_get_batch(data_iterator):
    """Generate a batch."""
    # TODO: this is pretty hacky, find a better way
    if (not mpu.is_pipeline_first_stage()) and (not mpu.is_pipeline_last_stage()):
        return None, None, None, None, None

    # get batches based on the TP rank you are on
    batch = get_batch_on_this_tp_rank(data_iterator)
    batch = get_batch_on_this_cp_rank(batch)
    return batch.values()


def distca_get_batch(data_iterator) -> Dict[str, torch.Tensor]:
    """
    DistCA-specific get_batch function.
    This function will:
    1. Fetch dp_size number of batches from data_iterator.
    2. Send each batch to the corresponding dp_rank's tp_rank=0.
    3. Broadcast within each tp_group (similar to get_batch_on_this_tp_rank).
    4. Return the batch.
    """
    # TODO: this is pretty hacky, find a better way
    if (not mpu.is_pipeline_first_stage()) and (not mpu.is_pipeline_last_stage()):
        return None, None, None, None, None

    args = get_args()
    
    # Step 1: Rank 0 fetches dp_size batches from data_iterator
    if rank == 0:
        # Fetch dp_size number of batches
        batches_for_all_dp = []
        for _ in range(dp_size):
            if data_iterator is not None:
                data = next(data_iterator)
                batch_dict = {
                    'tokens': data["tokens"].cuda(non_blocking=True),
                    'labels': data["labels"].cuda(non_blocking=True),
                    'loss_mask': data["loss_mask"].cuda(non_blocking=True),
                    'attention_mask': None if "attention_mask" not in data else data["attention_mask"].cuda(non_blocking=True),
                    'position_ids': data["position_ids"].cuda(non_blocking=True)
                }
                batches_for_all_dp.append(batch_dict)
            else:
                batches_for_all_dp.append(None)
    
    # Step 2: Rank 0 sends each batch to the corresponding dp_rank's tp_rank=0
    # Each dp_rank receives its batch (on tp_rank=0)
    if tp_rank == 0:
        # Allocate tensors for receiving
        tokens = torch.empty((args.micro_batch_size, args.seq_length), dtype=torch.int64, device=torch.cuda.current_device())
        labels = torch.empty((args.micro_batch_size, args.seq_length), dtype=torch.int64, device=torch.cuda.current_device())
        loss_mask = torch.empty((args.micro_batch_size, args.seq_length), dtype=torch.float32, device=torch.cuda.current_device())
        if args.create_attention_mask_in_dataloader:
            attention_mask = torch.empty((args.micro_batch_size, 1, args.seq_length, args.seq_length), dtype=torch.bool, device=torch.cuda.current_device())
        else:
            attention_mask = None
        position_ids = torch.empty((args.micro_batch_size, args.seq_length), dtype=torch.int64, device=torch.cuda.current_device())
        
        if rank == 0:
            # Rank 0 is the source, it already has the batch for dp_rank=0
            batch = batches_for_all_dp[0]
            tokens = batch['tokens']
            labels = batch['labels']
            loss_mask = batch['loss_mask']
            attention_mask = batch['attention_mask']
            position_ids = batch['position_ids']
            
            # Send batches to other dp_ranks (at their tp_rank=0)
            dp_group_ranks = torch.distributed.get_process_group_ranks(mpu.get_data_parallel_group())
            for i in range(1, dp_size):
                target_dp_rank = i
                # Find the global rank for dp_rank=i, tp_rank=0
                target_global_rank = dp_group_ranks[target_dp_rank]
                
                batch = batches_for_all_dp[i]
                torch.distributed.send(batch['tokens'], dst=target_global_rank)
                torch.distributed.send(batch['labels'], dst=target_global_rank)
                torch.distributed.send(batch['loss_mask'], dst=target_global_rank)
                
                # Send flag to indicate if attention_mask is None
                has_attention_mask = torch.tensor([1 if batch['attention_mask'] is not None else 0], 
                                                   dtype=torch.int64, device=torch.cuda.current_device())
                torch.distributed.send(has_attention_mask, dst=target_global_rank)
                # Only send attention_mask if it's not None
                if batch['attention_mask'] is not None:
                    torch.distributed.send(batch['attention_mask'], dst=target_global_rank)
                
                torch.distributed.send(batch['position_ids'], dst=target_global_rank)
        else:
            # Other dp_ranks (with tp_rank=0) receive from rank 0
            torch.distributed.recv(tokens, src=0)
            torch.distributed.recv(labels, src=0)
            torch.distributed.recv(loss_mask, src=0)
            
            # Receive flag to check if attention_mask is None
            has_attention_mask = torch.tensor([0], dtype=torch.int64, device=torch.cuda.current_device())
            torch.distributed.recv(has_attention_mask, src=0)
            # Only receive attention_mask if it's not None
            if has_attention_mask.item() == 1:
                torch.distributed.recv(attention_mask, src=0)
            else:
                attention_mask = None
            
            torch.distributed.recv(position_ids, src=0)
        
        batch = {
            'tokens': tokens,
            'labels': labels,
            'loss_mask': loss_mask,
            'attention_mask': attention_mask,
            'position_ids': position_ids
        }
    else:
        # tp_rank != 0: allocate empty tensors
        tokens = torch.empty((args.micro_batch_size, args.seq_length), dtype=torch.int64, device=torch.cuda.current_device())
        labels = torch.empty((args.micro_batch_size, args.seq_length), dtype=torch.int64, device=torch.cuda.current_device())
        loss_mask = torch.empty((args.micro_batch_size, args.seq_length), dtype=torch.float32, device=torch.cuda.current_device())
        if args.create_attention_mask_in_dataloader:
            attention_mask = torch.empty((args.micro_batch_size, 1, args.seq_length, args.seq_length), dtype=torch.bool, device=torch.cuda.current_device())
        else:
            attention_mask = None
        position_ids = torch.empty((args.micro_batch_size, args.seq_length), dtype=torch.int64, device=torch.cuda.current_device())
        
        batch = {
            'tokens': tokens,
            'labels': labels,
            'loss_mask': loss_mask,
            'attention_mask': attention_mask,
            'position_ids': position_ids
        }
    
    # Step 3: Broadcast within each tp_group (similar to get_batch_on_this_tp_rank)
    def _broadcast(item):
        if item is not None:
            torch.distributed.broadcast(item, mpu.get_tensor_model_parallel_src_rank(), group=mpu.get_tensor_model_parallel_group())
    
    # Handle different pipeline stages
    if args.pipeline_model_parallel_size == 1:
        _broadcast(batch['tokens'])
        _broadcast(batch['labels'])
        _broadcast(batch['loss_mask'])
        _broadcast(batch['attention_mask'])
        _broadcast(batch['position_ids'])
    elif mpu.is_pipeline_first_stage():
        _broadcast(batch['tokens'])
        _broadcast(batch['attention_mask'])
        _broadcast(batch['position_ids'])
        # First stage doesn't need labels and loss_mask
        if tp_rank != 0:
            batch['labels'] = None
            batch['loss_mask'] = None
    elif mpu.is_pipeline_last_stage():
        # Multi-Token Prediction (MTP) layers need tokens and position_ids
        if args.mtp_num_layers is not None:
            _broadcast(batch['tokens'])
            _broadcast(batch['position_ids'])
        else:
            if tp_rank != 0:
                batch['tokens'] = None
                batch['position_ids'] = None
        
        _broadcast(batch['labels'])
        _broadcast(batch['loss_mask'])
        _broadcast(batch['attention_mask'])
    
    return batch
    
# define spiky loss as a loss that's 10x the max loss observed
SPIKY_LOSS_FACTOR = 10

# ================================
# Define loss function
# ================================


def loss_func(loss_mask: torch.Tensor, output_tensor: torch.Tensor):
    """Loss function.

    Args:
        loss_mask (torch.Tensor): Used to mask out some portions of the loss
        output_tensor (torch.Tensor): The tensor with the losses

    Returns:
        the loss scalar for this micro-batch
        the number of non-padded tokens in this microbatch
        a dict containing reporting metrics on the loss and number of tokens across
            the data parallel ranks
    """
    args = get_args()

    losses = output_tensor.float()
    loss_mask = loss_mask.view(-1).float()
    total_tokens = loss_mask.sum()
    loss = torch.cat([torch.sum(losses.view(-1) * loss_mask).view(1), total_tokens.view(1)])

    if args.context_parallel_size > 1:
        torch.distributed.all_reduce(loss, group=mpu.get_context_parallel_group())

    # Check individual rank losses are not NaN prior to DP all-reduce.
    rerun_state_machine = get_rerun_state_machine()
    if args.check_for_nan_in_loss_and_grad:
        rerun_state_machine.validate_result(
            result=loss[0],
            rejection_func=torch.isnan,
            message="found NaN in local forward loss calculation",
            tolerance=0.0,        # forward pass calculations are determinisic
            fatal=True,
        )
        rerun_state_machine.validate_result(
            result=loss[0],
            rejection_func=torch.isinf,
            message="found Inf in local forward loss calculation",
            tolerance=0.0,        # forward pass calculations are determinisic
            fatal=True,
        )
    # Check for spiky loss
    if args.check_for_spiky_loss:
        rerun_state_machine.validate_result(
            result=loss[0],
            rejection_func=partial(
                rerun_state_machine.is_unexpectedly_large,
                threshold=SPIKY_LOSS_FACTOR,
                context="loss",
            ),
            message="Spiky loss",
            tolerance=0.0,        # forward pass calculations are determinisic
            fatal=False,
        )
    # Reduce loss for logging.
    reporting_loss = loss.clone().detach()
    torch.distributed.all_reduce(reporting_loss, group=mpu.get_data_parallel_group())

    # loss[0] is a view of loss, so it has ._base not None, which triggers assert error
    # in core/pipeline_parallel/schedule.py::deallocate_output_tensor, calling .clone()
    # on loss[0] fixes this
    local_num_tokens = loss[1].clone().detach().to(torch.int)
    return (
        loss[0].clone(),
        local_num_tokens,
        {'lm loss': (reporting_loss[0], reporting_loss[1])},
    )


# ================================
# Define forward step function
# forward_step(data_iterator, model)
# ================================


def original_forward_step(data_iterator, model: GPTModel):
    """Forward training step.

    Args:
        data_iterator : Input data iterator
        model (GPTModel): The GPT Model
    """
    args = get_args()
    timers = get_timers()

    # Get the batch.
    timers('batch-generator', log_level=2).start()
    global stimer
    with stimer(bdata=True):
        batch = distca_get_batch(data_iterator)
        tokens = batch['tokens']
        labels = batch['labels']
        loss_mask = batch['loss_mask']
        attention_mask = batch['attention_mask']
        position_ids = batch['position_ids']

        logger.debug(f"tokens: {tokens.shape}")
        logger.debug(f"labels: {labels.shape}")
        logger.debug(f"loss_mask: {loss_mask.shape}")
        logger.debug(f"attention_mask: {attention_mask.shape}")
        logger.debug(f"position_ids: {position_ids.shape}")

        # batch = __get_batch_original(data_iterator)
        # tokens, labels, loss_mask, attention_mask, position_ids = batch
    
    # Monitor tokens - log decoded text for debugging
    # Only log on first pipeline stage and TP rank 0 to avoid duplicate logs
    # if tokens is not None and mpu.is_pipeline_first_stage() and mpu.get_tensor_model_parallel_rank() == 0:
    tokenizer = get_tokenizer()
    monitor_batch_tokens(
        tokens, 
        tokenizer,
        loss_mask=loss_mask,
        # attention_mask=attention_mask,
        position_ids=position_ids,
        logger=logger,
    )

    timers('batch-generator').stop()

    with stimer:
        if args.use_legacy_models:
            output_tensor = model(tokens, position_ids, attention_mask,
                                labels=labels)
        else:
            output_tensor = model(tokens, position_ids, attention_mask,
                                labels=labels, loss_mask=loss_mask)

    return output_tensor, partial(loss_func, loss_mask)



def distca_forward_step(data_iterator, model: 'PingPongGPTModel'):
    """Forward training step.
    Correspond to `MegatronE2eWorker.forward_step()`

    Args:
        data_iterator : Input data iterator
        model (PingPongGPTModel): The PingPongGPTModel

    Call stack
    ==========
    - PingPongGPTModel.forward(input_ids, *args, **kwargs) 
        that calls `super().forward(input_ids, *args, **kwargs)`
    -> megatron.core.models.gpt.gpt_model.GPTModel.forward(input_ids, *args, **kwargs) 
        note that the GPTModel.forward has a strict signature about what to pass in (see below)
    -> PingPongTransformerBlockInterface.ping_pong_forward()
    
    
    Signatures
    ==========
    ```python
    > megatron.core.models.gpt.gpt_model.GPTModel
    GPTModel.forward(
        self,
        input_ids: Tensor,
        position_ids: Tensor,
        attention_mask: Tensor,
        decoder_input: Tensor = None,
        labels: Tensor = None,
        inference_context: BaseInferenceContext = None,
        packed_seq_params: PackedSeqParams = None,
        extra_block_kwargs: dict = None,
        runtime_gather_output: Optional[bool] = None,
        *,
        inference_params: Optional[BaseInferenceContext] = None,
        loss_mask: Optional[Tensor] = None,
    ) -> Tensor:...


    > distca.runtime.megatron.ping_pong.transformer_block.PingPongTransformerBlockInterface
    PingPongTransformerBlockInterface.ping_pong_forward(
        self,
        hidden_states: Union[Tensor, WrappedTensor],
        attention_mask: Optional[Tensor],
        context: Optional[Tensor] = None,
        context_mask: Optional[Tensor] = None,
        rotary_pos_emb: Optional[Tensor] = None,
        rotary_pos_cos: Optional[Tensor] = None,
        rotary_pos_sin: Optional[Tensor] = None,
        attention_bias: Optional[Tensor] = None,
        inference_context: Optional[BaseInferenceContext] = None,
        packed_seq_params: Optional[PingPangPackedSeqParams] = None,
        sequence_len_offset: Optional[Tensor] = None,
        *,
        inference_params: Optional[BaseInferenceContext] = None,
    ):
    ```
    """
    raise NotImplementedError("DistCA forward step is not implemented yet")

    args = get_args()
    timers = get_timers()

    # Get the batch.
    timers('batch-generator', log_level=2).start()
    global stimer
    with stimer(bdata=True):
        batch = distca_get_batch(data_iterator)
        tokens = batch['tokens']
        labels = batch['labels']
        loss_mask = batch['loss_mask']
        attention_mask = batch['attention_mask']
        position_ids = batch['position_ids']

        logger.debug(f"tokens: {tokens.shape}")
        logger.debug(f"labels: {labels.shape}")
        logger.debug(f"loss_mask: {loss_mask.shape}")
        logger.debug(f"attention_mask: {attention_mask.shape}")
        logger.debug(f"position_ids: {position_ids.shape}")

        # batch = __get_batch_original(data_iterator)
        # tokens, labels, loss_mask, attention_mask, position_ids = batch
        timers('batch-generator').stop()
    
        # Monitor tokens - log decoded text for debugging
        # Only log on first pipeline stage and TP rank 0 to avoid duplicate logs
        # if tokens is not None and mpu.is_pipeline_first_stage() and mpu.get_tensor_model_parallel_rank() == 0:
        tokenizer = get_tokenizer()
        monitor_batch_tokens(
            tokens, 
            tokenizer,
            loss_mask=loss_mask,
            # attention_mask=attention_mask,
            position_ids=position_ids,
            logger=logger,
        )


    # Build the DistCA Specific metadata
    from distca.runtime.megatron.packed_seq_params import PingPangPackedSeqParams
    ping_pong_params = PingPangPackedSeqParams(
        seq_params = [mb_0_psp, mb_1_psp],
        mlp_layout_seq_params = [mb_0_mlp_psp, mb_1_mlp_psp],
        max_seqlen_q = max(mb_0_mlp_psp.max_seqlen_q, mb_1_mlp_psp.max_seqlen_q),
        max_seqlen_kv = max(mb_0_mlp_psp.max_seqlen_kv, mb_1_mlp_psp.max_seqlen_kv),
    )


    def gptmodel_forward(
        model, input_ids, attention_mask, position_ids, sequence_parallel, packed_seq_params,
        labels=None, logits_processor=None, logits_processor_kwargs=None,
    ):
        """Migrated from distca.utils.megatron_test_utils. 
        Entry point to call the model.forward()"""
        pre_process = unwrap_model(model).pre_process
        if pre_process:
            input_ids = input_ids.unsqueeze(0)
        post_process = unwrap_model(model).post_process
        output_orig = model(
            input_ids=input_ids,
            labels=labels,
            attention_mask=None,
            position_ids=position_ids,
            packed_seq_params=packed_seq_params,
        )
        if post_process and logits_processor is not None:
            # FIXME(yonghao): add logits processor logic, if needed
            pass
        return output_orig

    
    # TODO: <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< DistCA patch
    output_tensor = gptmodel_forward(
        model, tokens, attention_mask, position_ids, 
        args.sequence_parallel,
        ping_pong_params,
        labels=labels, loss_mask=loss_mask,
    )


    # TODO: ================================
    with stimer:
        if args.use_legacy_models:
            output_tensor = model(tokens, position_ids, attention_mask,
                                labels=labels)
        else:
            output_tensor = model(tokens, position_ids, attention_mask,
                                labels=labels, loss_mask=loss_mask)
    # TODO: >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> megatron original code

    return output_tensor, partial(loss_func, loss_mask)

process_non_loss_data_func = None
non_loss_data_func = None

# ================================
# Main training function
# ================================

memory_estimate = log_memory_estimate(args, num_microbatches=num_microbatches, verbose=True, logger=logger)
logger.info(f"Memory estimate: {memory_estimate}")

# ================================
# Simple forward_backward_batch demonstration
# ================================
# This replaces the distca_train call with a simpler approach similar to
# test_megatron_e2e_pipeline_with_cp.py

def forward_backward_batch(
    microbatches: list[dict],
    model,
    forward_only: bool = False,
    mode: str = "ping_pong",
    with_dummy: bool = True,
):
    """
    Simple forward_backward_batch function that:
    1. Takes pre-created microbatches
    2. Defines a forward_step function
    3. Handles dummy backward batches
    4. Uses forward_backward_func with dummy_bwd_func
    
    This is a simplified version of the pattern from test_megatron_e2e_pipeline_with_cp.py
    """
    # Move microbatches to CUDA
    microbatches = [{
        k: arg_to_cuda(v) for k, v in microbatch.items()
    } for microbatch in microbatches]
    
    # Get pipeline parallel info
    pp_size = mpu.get_pipeline_model_parallel_world_size()
    pp_rank = mpu.get_pipeline_model_parallel_rank()
    n_micro_batch = len(microbatches) - pp_size + 1
    total_seqlen = microbatches[0]['input_ids'].shape[0] if 'input_ids' in microbatches[0] else None
    
    # Define forward_step function
    def forward_step(batch_iter, model):
        batch = next(batch_iter)
        
        input_ids = batch['input_ids']
        position_ids = batch['position_ids']
        attention_mask = None
        packed_seq_params = batch.get('packed_seq_params', None)
        
        sequence_parallel = get_args().sequence_parallel
        logger.debug(f"forward_step: {input_ids.shape=}, {sequence_parallel = }")
        
        # Use gptmodel_forward utility function
        with torch.cuda.nvtx.range("forward_step"):
            output = gptmodel_forward(
                model, input_ids, attention_mask, position_ids, sequence_parallel,
                packed_seq_params, labels=input_ids.unsqueeze(0) if 'input_ids' in batch else None,
            )
        # Mock the loss_mask: set to ones with same number of tokens as input_ids
        # (Reasonable stand-in for a mask meaning all tokens contribute to loss)
        loss_mask = torch.ones_like(input_ids, dtype=torch.float32)
        return output, partial(loss_func, loss_mask)
    
    # Define dummy backward step function
    def dummy_backward_step(model, dummy_bwd_iter, skip: bool):
        try:
            next_iter_args = next(dummy_bwd_iter)
            if skip:
                return
            unwrap_model(model).dummy_backward(next_iter_args)
        except StopIteration:
            # No more dummy backward batches
            pass
    
    # Prepare model for training
    # Note: model might be a list in case of virtual pipeline parallelism
    if isinstance(model, list):
        train_module = model
    else:
        train_module = [model]
    
    assert len(train_module) == 1, "only support one module"
    
    # Prepare dummy backward metadata
    # Shift bwd metadata since the order it runs is different from the
    # corresponding dummy forward's.
    dummy_bwd_packed_seq_params = [
        microbatch.get('packed_seq_params') for microbatch in
        (microbatches[-pp_size + pp_rank + 1:][:pp_size - pp_rank - 1] + microbatches[:pp_rank])
    ]
    dummy_bwd_packed_seq_params = dummy_bwd_packed_seq_params[pp_rank:] + dummy_bwd_packed_seq_params[:pp_rank]
    
    assert mode in ["ping_pong", "orig_reimpl", "single_sided"]
    
    # Set debug mode for modules
    for module in train_module:
        debug = (mode != "ping_pong")
        debug_fwd_impl = mode if debug else None
        unwrap_model(module).set_debug(debug=debug, debug_fwd_impl=debug_fwd_impl)
        unwrap_model(module).train()
    
    # Create batch generator and dummy backward iterator
    dummy_bwd_packed_seq_params_iter = iter(dummy_bwd_packed_seq_params)
    batch_generator = make_batch_generator(
        microbatches if with_dummy else microbatches[pp_rank:],
        vpp_size=len(train_module)
    )
    
    # Synchronize before starting
    torch.cuda.synchronize()
    from distca.runtime.attn_kernels.ops import nvshmem_barrier_all
    nvshmem_barrier_all()
    
    # Call forward_backward_func
    from distca.runtime.megatron.forward_backward_func import forward_backward_pipelining_without_interleaving
    if with_dummy:
        losses_reduced = forward_backward_pipelining_without_interleaving(
            forward_step_func=forward_step,
            data_iterator=batch_generator,
            model=train_module,
            num_microbatches=n_micro_batch,
            seq_length=total_seqlen,  # no use, since variable_seq_lengths=True
            micro_batch_size=1,  # no use when input_shapes was set
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
    
    for tm in train_module:
        for param in unwrap_model(tm).parameters():
            param.main_grad.zero_()
    return losses_reduced


# ================================
# create_pp_microbatches function
# ================================
# Copied from test_megatron_e2e_pipeline_with_cp.py
from distca.runtime.compute_metadata import get_attn_metadata
from distca.planner.planner import cp_list_to_mlp_list
from distca.utils.test_util import create_qkv_dispatch_pipeline_tick

def create_pp_microbatches(
    num_microbatch: int, pp_degree: int, as_rank: int,
    as_world_size: int, total_seq_len: int, num_seqs: int,
    max_cp_degree: int, hidden_size_q_tp: int,
    hidden_size_k_tp: int, element_size: int,
    num_head_in_dtype: int, tp_size: int, dp_size: int,
    num_token_per_rank: int,
    num_batches: int = None,
    use_planner: bool = False,
    return_seq_lens: bool = False
):
    # print("Create pp microbatches")
    tick_per_rank_doc_lens = None
    bwd_metadata = []
    microbatches = []

    all_original_seq_lens = []
    for i in range(num_microbatch + pp_degree - 1):
        # For the last few ticks (drain-out ticks)
        # add a dummy forward microbatch at PP rank 0.
        add_dummy_forward = i >= num_microbatch
        (
            fa_fwd_params, 
            fa_bwd_params,
            qkv_fwd_fa2a_metadata, 
            qkv_bwd_fa2a_metadata,
            attn_out_fwd_fa2a_metadata, 
            attn_out_qkv_bwd_fa2a_metadata,
            tick_per_rank_doc_lens, 
            original_tick_per_rank_doc_lens,
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
        
        # For MLP-CP, we need to transfer List[List[int]] from CP layout back to DP, so each rank knows its number of tokens.
        #   Example1 DP case:
        # tick_per_rank_doc_lens cp list: List[List[int]] = [[8], [8], [8], [8], [256, 256],[128, 384],[512], [10, 502] ]
        # tick_per_rank_doc_lens mlp list : [[8], [8], [8], [8], [256, 256],[128, 384],[512], [10, 502] ]
        #   Example2 CP case:
        # tick_per_rank_doc_lens cp list: List[List[int]] = [[8], [8], [8], [8], [256, 768],[512, 10, 502] ]
        # tick_per_rank_doc_lens mlp list: [[8], [8], [8], [8], [256, 128, 128], [256, 256], [512], [10, 502]]
        tick_per_rank_doc_lens_after_cp_transfer = cp_list_to_mlp_list(tick_per_rank_doc_lens, as_world_size, num_token_per_rank)
        this_rank_num_tokens = sum(tick_per_rank_doc_lens_after_cp_transfer[as_rank])
        bwd_packed_seq_params = PackedSeqParams(qkv_format="thd", **fa_bwd_params[as_rank])
        tensor_doc_lens = torch.tensor(tick_per_rank_doc_lens_after_cp_transfer[as_rank], dtype=torch.int32)
        mlp_packed_seq_params = get_attn_metadata(tensor_doc_lens, get_packed_seq_params=True)

        # Create packed_params. Note that we do not add backward params here.
        ping_pang_params = PingPangSingleStepPackedSeqParams(
            qkv_format="thd",
            **fa_fwd_params[as_rank],
            qkv_fwd_metadata=qkv_fwd_fa2a_metadata.get_slice(as_rank),
            attn_out_fwd_metadata=attn_out_fwd_fa2a_metadata.get_slice(as_rank),
            mlp_packed_seq_params=mlp_packed_seq_params,
        )

        # NOTE: we init input_ids at the end after creating dispatching strategy
        # and seq lens of each iteration. This is to ensure that each
        # rank has the same randomly initialized strategy.
        position_ids_local = torch.arange(this_rank_num_tokens)
        microbatch = {
            "position_ids": position_ids_local,
            "packed_seq_params": ping_pang_params,
        }
        microbatches.append(microbatch)

        # store the corresponding bwd metadata (for later ticks)
        bwd_metadata.append(
            (
                qkv_bwd_fa2a_metadata.get_slice(as_rank), 
                attn_out_qkv_bwd_fa2a_metadata.get_slice(as_rank), 
                bwd_packed_seq_params
            )
        )

    start_time = time.time()
    # put bwd metadata to the corresponding side
    for i, microbatch in enumerate(microbatches):
        # When mb_i is computed on pp_rank at forward tick t, assume the backward right after this forward is at tick t'.
        # mb_i requires another (pp_degree - 1 - pp_rank) forward steps to start mb_i backward,
        # and another (pp_degree - 1 - pp_rank) backward steps to compute mb_i backward on this pp_rank.
        # Since in 1F1B, every forward follows a backward, so backward tick
        # t' + 2 * (pp_degree - 1 - pp_rank) is the one to compute mb_i on this pp_rank.
        # Note that at the end of PP warmup, forward tick is pp_degree - 1, while backward tick
        # is 0. Therefore, t - t' = pp_degree - 1, and thus
        # t' + 2 * (pp_degree - 1 - pp_rank) == t + pp_degree - 1 - pp_rank * 2
        bwd_metadata_idx = (i + pp_degree - 1 - pp_rank * 2) % len(bwd_metadata)
        qkv_bwd_metadata, attn_out_bwd_metadata, bwd_packed_seq_params = bwd_metadata[bwd_metadata_idx]
        packed_seq_params = microbatch["packed_seq_params"]
        packed_seq_params.qkv_bwd_metadata = qkv_bwd_metadata
        packed_seq_params.attn_out_bwd_metadata = attn_out_bwd_metadata
        packed_seq_params.bwd_packed_seq_params = bwd_packed_seq_params
    end_time = time.time()
    ret = microbatches
    if return_seq_lens:
        ret = (microbatches, all_original_seq_lens)
    return ret


# ================================
# Example usage: Replace distca_train with forward_backward_batch
# ================================

# ================================
# Training Loop using forward_backward_batch
# ================================
# This replaces distca_train with a simpler approach using forward_backward_batch
from distca.runtime.megatron.packed_seq_params import PingPangPackedSeqParams

# Check if we should use forward_backward_batch instead of distca_train
logger.info("Using forward_backward_batch instead of distca_train")

# Calculate attention server parameters
# as_world_size and as_rank are already defined above (lines 240-241)

# Get model dimensions from config
hidden_size_q = config.hidden_size
hidden_size_kv = hidden_size_q
if hasattr(hf_config, "num_key_value_heads"):
    hidden_size_kv = (hidden_size_q * hf_config.num_key_value_heads //
                        hf_config.num_attention_heads)

hidden_size_q_tp = hidden_size_q // tp_size
hidden_size_k_tp = hidden_size_kv // tp_size
element_size = args.params_dtype.itemsize
num_head_in_dtype = (hf_config.num_attention_heads *
                        torch.float32.itemsize // element_size // tp_size)

# Calculate token distribution
num_token_per_rank = args.seq_length * args.micro_batch_size // dp_size
total_seq_len = args.seq_length
num_seqs = args.micro_batch_size
num_microbatch = num_microbatches
max_cp_degree = cp_size if cp_size > 1 else dp_size
num_batches = None  # Use MLP-DP by default

logger.info(f"Creating microbatches: num_microbatch={num_microbatch}, pp_size={pp_size}, "
            f"as_rank={as_rank}, as_world_size={as_world_size}")

# Create microbatches for ping-pong (two dispatchers)
microbatches_0, tick_per_rank_doc_lens_0 = create_pp_microbatches(
    num_microbatch, pp_size, as_rank,
    as_world_size, total_seq_len, num_seqs, max_cp_degree,
    hidden_size_q_tp, hidden_size_k_tp, element_size, num_head_in_dtype,
    tp_size, dp_size,
    num_token_per_rank, num_batches, use_planner=False,
    return_seq_lens=True,
)

microbatches_1, tick_per_rank_doc_lens_1 = create_pp_microbatches(
    num_microbatch, pp_size, as_rank,
    as_world_size, total_seq_len, num_seqs, max_cp_degree,
    hidden_size_q_tp, hidden_size_k_tp, element_size, num_head_in_dtype,
    tp_size, dp_size,
    num_token_per_rank, num_batches, use_planner=False,
    return_seq_lens=True,
)

# Combine microbatches for ping-pong (similar to test_megatron_e2e_pipeline_with_cp.py)
combined_microbatches = []
for mb_0, mb_1 in zip(microbatches_0, microbatches_1):
    mb_0_psp = mb_0["packed_seq_params"]
    mb_1_psp = mb_1["packed_seq_params"]
    mb_0_mlp_psp = mb_0_psp.mlp_packed_seq_params
    mb_1_mlp_psp = mb_1_psp.mlp_packed_seq_params
    mb_0_psp.dispatcher_id = 0
    mb_1_psp.dispatcher_id = 1
    ping_pong_params = PingPangPackedSeqParams(
        seq_params=[mb_0_psp, mb_1_psp],
        mlp_layout_seq_params=[mb_0_mlp_psp, mb_1_mlp_psp],
        max_seqlen_q=max(mb_0_mlp_psp.max_seqlen_q, mb_1_mlp_psp.max_seqlen_q),
        max_seqlen_kv=max(mb_0_mlp_psp.max_seqlen_kv, mb_1_mlp_psp.max_seqlen_kv),
    )
    num_tokens = sum(mb["position_ids"].numel() for mb in [mb_0, mb_1])
    input_ids = torch.randint(10, 1000, (num_tokens,), device=torch.cuda.current_device())
    combined_mb = {
        "input_ids": input_ids,
        "position_ids": torch.concat([mb_0["position_ids"], mb_1["position_ids"]]),
        "packed_seq_params": ping_pong_params,
    }
    combined_microbatches.append(combined_mb)

logger.info(f"Created {len(combined_microbatches)} combined microbatches")

# Training loop
iteration = 0
max_iterations = args.train_iters if hasattr(args, 'train_iters') else 1

for iteration in range(max_iterations):
    logger.info(f"Starting iteration {iteration}")
    
    # Run forward_backward_batch
    losses_reduced = forward_backward_batch(
        microbatches=combined_microbatches,
        model=model,
        forward_only=False,
        mode="ping_pong",
        with_dummy=True,
    )
    
    # Reduce losses across data parallel groups
    print(f"losses_reduced: {losses_reduced}")
    if losses_reduced:
        # losses_reduced is a list of dicts, each dict has 'lm loss': (loss_tensor, count_tensor)
        # losses_reduced: [{'lm loss': (tensor(103096.1875, device='cuda:0'), tensor(8192., device='cuda:0'))}]
        numerator = 0
        denominator = 0
        for loss_dict in losses_reduced:
            if 'lm loss' in loss_dict:
                val = loss_dict['lm loss']
                if isinstance(val, (tuple, list)) and len(val) == 2:
                    numerator += val[0]  # sum of losses
                    denominator += val[1]  # sum of token counts
        if denominator > 0:
            reduced_loss = numerator / denominator
            logger.info(f"Iteration {iteration}, reduced loss: {reduced_loss.item()}")
        else:
            logger.warning(f"Iteration {iteration}, no valid loss found")
    
    # Optimizer step
    optimizer.step()
    
    # TODO: Some logic including increment the opt_param_scheduler step should be included here.

    # Zero gradients (handled by optimizer, but can be explicit)
    optimizer.zero_grad()
    
    logger.info(f"Completed iteration {iteration}")

num_floating_point_operations_so_far = 0  # Calculate if needed
logger.info(f"Training completed. Total iterations: {iteration + 1}")


# ================================
# Finish Training
# ================================
wandb_writer = get_wandb_writer()
if wandb_writer:
    wandb_writer.finish()


# ================================
# Test Individual components
# ================================
# from test_vocab import test_vocab
# test_vocab()

logger.info(f"Rank {rank} is exiting")
torch.distributed.destroy_process_group()

