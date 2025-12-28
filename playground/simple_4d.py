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
from utils.nan_detector import (
    register_nan_detection_hooks,
    remove_hooks,
)


if typing.TYPE_CHECKING:
    from transformers import PretrainedConfig

rank = int(os.environ["RANK"])
world_size = int(os.environ["WORLD_SIZE"])
local_rank = int(os.environ["LOCAL_RANK"])
micro_batch_size = int(os.environ["MICRO_BATCH_SIZE"])
global_batch_size = int(os.environ["GLOBAL_BATCH_SIZE"])
seq_length = int(os.environ["SEQ_LENGTH"])

logger = setup_logging(
    rank=rank, world_size=world_size,
    level=logging.DEBUG,
    console_ranks=[0],
    # console_ranks=list(range(0, world_size, 8)),  # Only rank 0 logs to console
)

# Create a dedicated logger for batch_to_items_with_dummy
# This allows independent control via standard logging configuration:
#   logging.getLogger(__name__ + ".batch_to_items_with_dummy").setLevel(logging.DEBUG)
# The logger name forms a hierarchy: __main__ -> __main__.batch_to_items_with_dummy
logger_batch_to_items = logging.getLogger(__name__ + ".batch_to_items_with_dummy")
# By default, inherit level from parent (logger), but can be overridden
# The level will be set later based on args.distca_debug_batch_to_items if needed

# Update logger formatters to include function:lineno
# This adds funcName:lineno to all log messages for better debugging
# Also add a filter to optionally filter by category (e.g., 'batch_to_items')
class CategoryFilter(logging.Filter):
    """Filter logs by category if specified in extra dict."""
    def __init__(self, allowed_categories=None):
        super().__init__()
        self.allowed_categories = allowed_categories or set()
    
    def filter(self, record):
        # If no category filtering is set, allow all
        if not self.allowed_categories:
            return True
        # Check if record has a category in extra
        category = getattr(record, 'category', None)
        return category in self.allowed_categories if category else True

# Update all handlers to include funcName:lineno in format
for handler in logger.handlers:
    formatter = handler.formatter
    if formatter:
        # Check if it's a RankFormatter (has rank attribute) or standard Formatter
        if hasattr(formatter, 'fmt') or hasattr(formatter, '_fmt'):
            # Get the current format string
            current_fmt = getattr(formatter, 'fmt', None) or getattr(formatter, '_fmt', None)
            if current_fmt and '%(funcName)s' not in current_fmt:
                # Insert funcName:lineno before the message
                new_fmt = current_fmt.replace('%(message)s', '%(funcName)s:%(lineno)d %(message)s')
                # Preserve rank formatting if it's a RankFormatter
                if hasattr(formatter, 'rank'):
                    from utils.logging import RankFormatter
                    handler.setFormatter(RankFormatter(fmt=new_fmt, datefmt=formatter.datefmt, 
                                                       rank=formatter.rank, use_color=getattr(formatter, 'use_color', True)))
                else:
                    handler.setFormatter(logging.Formatter(new_fmt, formatter.datefmt))

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
tp = int(os.environ["TP"]); 
pp = int(os.environ["PP"]); 
cp = int(os.environ["CP"]);
dp = int(os.environ["DP"]);
# ensure that tp * pp * cp * dp == world_size
assert tp * pp * cp * dp == world_size, f"{tp = }, {pp = }, {cp = }, {dp = }: tp * pp * cp * dp = {tp * pp * cp * dp} != world_size: {world_size}"

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

with time_it("setup logging directories"):
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
    "--micro-batch-size", str(micro_batch_size),
    "--global-batch-size", str(global_batch_size),
    "--lr", "1.0e-5",
    # "--train-samples", "100000",
    # "--train-iters", "100000",
    "--train-iters", "10",
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
    "--no-create-attention-mask-in-dataloader",
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
    
    
    
    # DistCA arguments
    "--distca-quit-if-maybe-oom",
    # "--distca-debug-nan-detector",
    # "--distca-debug-batch-to-items",
]

# Filter out None values (from conditional args like --swiglu)
designated_args = [arg for arg in designated_args if arg is not None]


# ================================
# DistCA Configuration
# ================================
with time_it("initialize megatron"):

    @dataclass
    class DistCAConfig(TransformerConfig):
        """Configuration object for DistCA."""

        distca_nvshmem_buffer_size_gb: float = 1.0
        """Set the NVSHMEM buffer size (in GB). 
        TODO: By default, we want to make it configurable by the planner."""


        distca_quit_if_maybe_oom: bool = False
        """If True, the program will quit if the estimated memory can probably exceeds the GPU max memory."""


        distca_debug_nan_detector: bool = False
        """If True, the program will debug the nan detector."""

        distca_debug_batch_to_items: bool = False
        """If True, enable detailed debug logging in batch_to_items_with_dummy function.
        This sets the logger level for __main__.batch_to_items_with_dummy to DEBUG.
        Alternatively, use standard logging configuration:
        logging.getLogger('__main__.batch_to_items_with_dummy').setLevel(logging.DEBUG)
        """

        pass

    def replace_parser_and_parse_args(parser: argparse.ArgumentParser):    
        """Hijack the parser, and use the `designated_args` as the arguments. Then we also inject some of our own arguments."""

        # Define the extra arguments for DistCA
        group = parser.add_argument_group(title='distca')
        
        group.add_argument("--distca-nvshmem-buffer-size-gb", type=int, default=2, 
        help="Set the NVSHMEM buffer size (in GB)")

        group.add_argument("--distca-quit-if-maybe-oom", action="store_true", default=False, 
        help="If True, the program will quit if the estimated memory can probably exceeds the GPU max memory.")

        group.add_argument("--distca-debug-batch-to-items", action="store_true", default=False,
        help="If True, enable detailed debug logging in batch_to_items_with_dummy function.")

        old_parser_parse_args = parser.parse_args
        old_parser_parse_known_args = parser.parse_known_args
        parser.parse_args = lambda *args, **kwargs: old_parser_parse_args(designated_args)
        parser.parse_known_args = lambda *args, **kwargs: old_parser_parse_known_args(designated_args)

        return parser

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
    
    # Configure logger for batch_to_items_with_dummy based on flag
    # This provides backward compatibility with --distca-debug-batch-to-items
    # Standard way: logging.getLogger(__name__ + ".batch_to_items_with_dummy").setLevel(logging.DEBUG)
    if hasattr(args, 'distca_debug_batch_to_items') and args.distca_debug_batch_to_items:
        logger_batch_to_items.setLevel(logging.DEBUG)
        logger.info(f"Enabled debug logging for batch_to_items_with_dummy (logger: {logger_batch_to_items.name})")
    else:
        # Inherit from parent logger (default behavior)
        logger_batch_to_items.setLevel(logging.NOTSET)  # NOTSET means inherit from parent


# ================================
# Report theoretical memory and FLOPS estimates
# ================================
with time_it("report theoretical memory and FLOPS estimates"):
    num_microbatches = get_num_microbatches()
    logger.info(f"Num microbatches: {num_microbatches}")
    logger.info(f"micro batch size: {args.micro_batch_size}")

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
                # micro_batch_size=micro_batch_size,
                micro_batch_size=1,
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
# define spiky loss as a loss that's 10x the max loss observed
SPIKY_LOSS_FACTOR = 10

stimer = StragglerDetector()

# Configure token monitoring (optional - defaults are sensible)
set_token_monitor_config(enabled=True, max_tokens_to_decode=200, max_samples_to_log=2)


# ================================
# Define loss function
# ================================
def loss_func(loss_mask: torch.Tensor, output_tensor: torch.Tensor):
    """Loss function.

    When using loss function, remember to bind the `loss_mask` in advanced.
    ```python
    loss_func_ = partial(loss_func, loss_mask=loss_mask)
    ```

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

    # if args.context_parallel_size > 1:
    #     torch.distributed.all_reduce(loss, group=mpu.get_context_parallel_group())

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
process_non_loss_data_func = None
non_loss_data_func = None

# ================================
# Main training function
# ================================

memory_estimate = log_memory_estimate(args, num_microbatches=num_microbatches, verbose=True, logger=logger)
logger.info(f"Memory estimate: {memory_estimate}")

# =============================================
# Simple forward_backward_batch demonstration
# =============================================
# This replaces the distca_train call with a simpler approach similar to
# test_megatron_e2e_pipeline_with_cp.py

# Define forward_step function
args = get_args()
sequence_parallel = args.sequence_parallel

def forward_step(batch_iter, model):
    batch = next(batch_iter)

    input_ids = batch['input_ids']
    position_ids = batch['position_ids']
    loss_mask = batch['loss_mask']
    labels = batch['labels']
    attention_mask = None
    packed_seq_params = batch['packed_seq_params']
    
    # Use gptmodel_forward utility function
    with torch.cuda.nvtx.range("forward_step"):
        output = gptmodel_forward(
            model, input_ids, attention_mask, position_ids, sequence_parallel,
            packed_seq_params, labels=labels,
        )
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
        k: arg_to_cuda(v) 
        for k, v in microbatch.items()
    } for microbatch in microbatches]
    # logger.debug(f"forward_backward_batch: {microbatches = }")
    
    # Get pipeline parallel info
    pp_size = mpu.get_pipeline_model_parallel_world_size()
    pp_rank = mpu.get_pipeline_model_parallel_rank()
    n_micro_batch = len(microbatches) - pp_size + 1
    total_seqlen = (
        microbatches[0]['input_ids'].shape[0] 
        if 'input_ids' in microbatches[0] 
        else None
    )
    
    # Prepare model for training
    # Note: model might be a list in case of virtual pipeline parallelism
    train_module = model if isinstance(model, list) else [model]
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
        assert debug == False, "debug mode is not supported"
        debug_fwd_impl = mode if debug else None
        unwrap_model(module).set_debug(debug=debug, debug_fwd_impl=debug_fwd_impl)
        unwrap_model(module).train()
    
    # Create batch generator and dummy backward iterator
    dummy_bwd_packed_seq_params_iter = iter(dummy_bwd_packed_seq_params)
    batch_generator = make_batch_generator(
        microbatches if with_dummy else microbatches[pp_rank:],
        vpp_size=len(train_module)
    )
    
    # TODO: Maybe remove for performance purpose.
    # (Benchmark use) Synchronize before starting 
    torch.cuda.synchronize()
    from distca.runtime.attn_kernels.ops import nvshmem_barrier_all
    nvshmem_barrier_all()
    
    # Call forward_backward_func
    from distca.runtime.megatron.forward_backward_func import forward_backward_pipelining_without_interleaving
    kwargs = dict(
        forward_step_func=forward_step,
        data_iterator=batch_generator,
        model=train_module,
        num_microbatches=n_micro_batch,
        seq_length=total_seqlen,  # no use, since variable_seq_lengths=True
        micro_batch_size=1,  # no use when input_shapes was set
        forward_only=forward_only,
    )
    if with_dummy:
        dummy_bwd_func=partial(
            dummy_backward_step, dummy_bwd_iter=dummy_bwd_packed_seq_params_iter,
            skip="orig" in mode,
        )
        losses_reduced = forward_backward_pipelining_without_interleaving(
            forward_step_func=forward_step,
            data_iterator=batch_generator,
            model=train_module,
            num_microbatches=n_micro_batch,
            seq_length=total_seqlen,  # no use, since variable_seq_lengths=True
            micro_batch_size=1,  # no use when input_shapes was set
            forward_only=forward_only,
            dummy_bwd_func=dummy_bwd_func,
        )
    else:
        from megatron.core.pipeline_parallel.schedules import get_forward_backward_func
        func = get_forward_backward_func()
        losses_reduced = func(**kwargs)
    
    # Optimizer step
    for tm in train_module:
        for param in unwrap_model(tm).parameters():
            param.main_grad.zero_()
    
    return losses_reduced


# ==========================================================
# Plan and define the microbatch creation function
# ==========================================================

# Copied from test_megatron_e2e_pipeline_with_cp.py
from distca.runtime.compute_metadata import get_attn_metadata
from distca.planner.planner import cp_list_to_mlp_list
# from distca.utils.test_util import create_qkv_dispatch_pipeline_tick
from distca.planner.planner import Planner, Item
from types import SimpleNamespace

@dataclass
class MicrobatchCreateConfig:
    pp_size: int
    tp_size: int    
    dp_size: int
    as_rank: int
    as_world_size: int

    hidden_size_q_tp: int
    hidden_size_k_tp: int
    element_size: int
    num_head_in_dtype: int
    
    # num_token_per_rank = num_tokens * num_batches // dpcp_size
    # represent the the number of batch per tick we need to get.
    num_token_per_rank: int
    # context length after document packing of one batch
    total_seq_len: int
    num_seqs: int # deprecate

    # num_microbatch = pp micro batches
    num_microbatch: int
    # num_batches = dp micro batch number
    num_batches: int
    softmax_lse_size: float
    # softmax_lse_size = num_heads_local * fp32.itemsize // element_size

with time_it("create_pp_microbatches"):
    from distca.runtime.megatron.packed_seq_params import PingPangPackedSeqParams

    # Get model dimensions from config
    hidden_size_q = config.hidden_size
    hidden_size_kv = hidden_size_q
    if hasattr(hf_config, "num_key_value_heads"):
        hidden_size_kv = (hidden_size_q * hf_config.num_key_value_heads //
                            hf_config.num_attention_heads)

    hidden_size_q_tp = hidden_size_q // tp_size
    hidden_size_k_tp = hidden_size_kv // tp_size
    element_size = args.params_dtype.itemsize
    num_head_in_dtype = (hf_config.num_attention_heads * torch.float32.itemsize // element_size // tp_size)

    # Calculate token distribution
    num_token_per_rank = args.seq_length * num_microbatches // dp_size
    total_seq_len = args.seq_length
    num_seqs = args.micro_batch_size
    max_cp_degree = cp_size if cp_size > 1 else dp_size
    # TODO: Properly check what is the affect of num_batches 
    # num_batches = None  # Use MLP-DP by default
    num_batches = global_batch_size // micro_batch_size

    softmax_lse_size = (hf_config.num_attention_heads // tp_size) * torch.float32.itemsize // element_size

    mbc_config = MicrobatchCreateConfig(
        pp_size=pp_size,
        tp_size=tp_size,
        dp_size=dp_size,
        as_rank=as_rank,
        as_world_size=as_world_size,
        total_seq_len=total_seq_len,
        num_seqs=num_seqs,
        hidden_size_q_tp=hidden_size_q_tp,
        hidden_size_k_tp=hidden_size_k_tp,
        element_size=element_size,
        num_head_in_dtype=num_head_in_dtype,
        num_token_per_rank=num_token_per_rank,
        num_microbatch=num_microbatches,
        num_batches=num_batches,
        softmax_lse_size=softmax_lse_size,
    )

logger.info(f"[mbc_config] {mbc_config}")

# TODO: Need to implement this function to fetch the data iterator.
# Temporary storage for the batches.
batch_registry = []
doclength_registry = []
tokenizer = get_tokenizer()

def clear_registry():
    batch_registry.clear()
    doclength_registry.clear()

def get_registry():
    return batch_registry, doclength_registry

def get_doc_lengths_from_item(item: dict, tokenizer) -> list[int]:
    """
    Get document lengths within a batch item.
    
    Documents are separated by EOD (end-of-document) tokens. This function
    finds document boundaries and calculates per-document token counts.
    
    Args:
        item: Dictionary containing 'tokens' tensor with shape [1, seq_length]
        tokenizer: Tokenizer with an 'eod' property that returns the EOD token ID
        
    Returns:
        List of document lengths (number of tokens per document)
    """
    # TODO: Support multiple sequences.

    # Get the tokens from the item.
    assert 'tokens' in item, f"item: {item} does not contain 'tokens'"
    tokens_tensor = item['tokens']
    
    # Extract tokens from item - shape is [1, seq_length]
    # assert it only has 1 sequence.
    assert tokens_tensor.shape[0] == 1, f"tokens_tensor: {tokens_tensor.shape} should have 1 sequence"

    if tokens_tensor.dim() == 2:
        # Extract the first (and only) sequence
        tokens = tokens_tensor[0].cpu().tolist()
    else:
        # Already 1D
        tokens = tokens_tensor.cpu().tolist()
    
    # Get EOD token ID from tokenizer
    eod_token_id = tokenizer.eod
    
    # Find EOD positions
    eod_positions = [i for i, tok in enumerate(tokens) if tok == eod_token_id]
    
    if not eod_positions:
        # No EOD found - treat entire sequence as one document
        return [len(tokens)]
    
    # Calculate tokens per document
    # Documents are: [0:eod_0+1], [eod_0+1:eod_1+1], ...
    doc_lengths = []
    prev_end = 0
    for eod_pos in eod_positions:
        doc_length = eod_pos - prev_end + 1  # Include the EOD token
        doc_lengths.append(doc_length)
        prev_end = eod_pos + 1
    
    # If there are tokens after the last EOD, count them as a partial document
    if prev_end < len(tokens):
        doc_lengths.append(len(tokens) - prev_end)
    
    return doc_lengths



def modify_train_data_iterator(
    mbc_config: MicrobatchCreateConfig,
    data_iterator: 'Iterator[dict]',
):
    """
    Return a modified data iterator that yields items with `num_token_per_rank` tokens.
    Dictionary item has keys (tokens, labels, loss_mask, position_ids).
    """
    num_token_per_rank = mbc_config.num_token_per_rank
    while True:
        item = next(data_iterator)
        # Reorganize the item such that each tensor will have `num_token_per_rank` tokens.
        
        # Only assume the batch has keys (tokens, labels, loss_mask, position_ids)
        original_tokens = item['tokens']
        original_labels = item['labels']
        original_loss_mask = item['loss_mask']
        original_position_ids = item['position_ids']
        
        # Reorganize the item such that each tensor will have `num_token_per_rank` tokens.
        splitted_tokens = torch.split(original_tokens, num_token_per_rank, dim=1)
        splitted_labels = torch.split(original_labels, num_token_per_rank, dim=1)
        splitted_loss_mask = torch.split(original_loss_mask, num_token_per_rank, dim=1)
        splitted_position_ids = torch.split(original_position_ids, num_token_per_rank, dim=1)

        # Now we loop over the items in each split, and yield the item.
        num_splits = len(splitted_tokens)
        for i in range(num_splits):
            new_item = {
                'tokens': splitted_tokens[i],
                'labels': splitted_labels[i],
                'loss_mask': splitted_loss_mask[i],
                'position_ids': splitted_position_ids[i],
            }
            yield new_item

    pass

train_data_iterator__modified = modify_train_data_iterator(mbc_config, train_data_iterator)

def get_next_batch(num_batches: int) -> list[list[int]]:
    """
    Get the next batch from the global batch.
    """
    global mbc_config

    this_batch: list[dict] = []
    doc_lengths: list[list[int]] = []
    for _ in range(num_batches):
        item = next(train_data_iterator__modified)
        # logger.debug(f'item: {item}')
        doc_len: list[int] = get_doc_lengths_from_item(item, tokenizer)
        # logger.debug(f'doc_len: {doc_len}')
        this_batch.append(item)
        doc_lengths.append(doc_len)

    batch_registry.append(this_batch)
    doclength_registry.append(doc_lengths)
    logger.info(f"[get_next_batch] [as_rank = {mbc_config.as_rank}] doc_lengths: {doc_lengths}, num_batches: {num_batches}, total_seq_len: {mbc_config.total_seq_len}")
    # return [[seq_length] for _ in range(num_batches)]
    return doc_lengths


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
    tp_size: int,
    dp_size: int,
    num_batches: int = None,
) -> tuple[list[list[int]], list[bool]]: # list of batch[int], list of dummy mask
    """
    Create `num_batches` batch for a microbatch (one pp-tick).
    - Take a batch from get_next_batch (or random) and also handle some padding logic.

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
        result = _block_reverse_list(ref_doc_lens, dp_size)
        # For backward, we don't have dummy markers from forward, so create all False
        # (backward batches are typically not dummy)
        dummy_mask = [False] * len(result)
        return result, dummy_mask
    
    dummy_length = total_token_on_rank // dp_size
    # Create new microbatches for the first PP stage
    dummy_mask = []
    if add_dummy:
        # Each rank only gets one dummy document, which is of `tp_size`
        # to avoid Sequence Parallel having an issue.

        # FIXME: using total_token_on_rank for cudagraph to work out-of-box, can be tp_size when cudagraph is disabled.
        pp_head_new_doc_len = [[dummy_length] for _ in range(dp_size)]  
        dummy_mask = [True] * dp_size
    else:
        assert total_token_on_rank % tp_size == 0, "Sequence Parallel requires total token divisible by tp_size"
        num_batches = num_batches or dp_size 
        pp_head_new_doc_len = get_next_batch(num_batches)
        dummy_mask = [False] * num_batches

    # pp_head_new_doc_len : shape : [dp, num_seqs]  
    # We should sample seq_len here.
    # And add the sampled seq_len to the batch. 
    # Next step is based on the previous batch, move the batch. 
    # Get existing microbatch seqlens
    if ref_doc_lens is not None:
        if num_batches == None:
            num_batches = dp_size
        # Not the first microbatch
        if ref_doc_lens[-1] == [tp_size]:
            other_pp_doc_len = ref_doc_lens[:-dp_size]
        else:
            other_pp_doc_len = ref_doc_lens[:-num_batches]
    else:
        dummy_fwd_num = world_size - dp_size
        # FIXME: using total_token_on_rank for cudagraph to work out-of-box, 
        # can be tp_size when cudagraph is disabled.
        other_pp_doc_len = [[dummy_length] for _ in range(dummy_fwd_num)]  
    tick_per_rank_doc_len = pp_head_new_doc_len + other_pp_doc_len
    dummy_mask += [False] * len(other_pp_doc_len)

    return tick_per_rank_doc_len, dummy_mask


# Can handle MLP DPCP and dummy doc.
# Copied from distca.planner.planner for debugging and adjustment
# from distca.planner.planner import (
#     batch_to_items_with_dummy,
# )

def batch_to_items_with_dummy(
    batches: List[List[int]], 
    num_tokens_per_rank: int, 
    as_world_size: int, 
    model_config: dict,
    dummy_mask: List[bool],
):

    items = []
    seqid = 0
    rank_budgets = [num_tokens_per_rank] * as_world_size
    current_rank_idx = 0

    assert len(batches) == len(dummy_mask), \
        f"Length mismatch: batches={len(batches)}, dummy_mask={len(dummy_mask)}"
    
    logger_batch_to_items.debug(f"START: batches={batches}, num_tokens_per_rank={num_tokens_per_rank}, as_world_size={as_world_size}")
    logger_batch_to_items.debug(f"dummy_mask={dummy_mask}")
    logger_batch_to_items.debug(f"INITIAL STATE: rank_budgets={rank_budgets}, current_rank_idx={current_rank_idx}, seqid={seqid}")

    for batch_idx, batch in enumerate(batches):
        logger_batch_to_items.debug(f"\nProcessing batch[{batch_idx}]={batch}")
        logger_batch_to_items.debug(f"BEFORE batch: rank_budgets={rank_budgets}, current_rank_idx={current_rank_idx}, seqid={seqid}")
        
        # Use explicit dummy_mask instead of inference
        is_dummy = dummy_mask[batch_idx]
        
        if is_dummy:
            logger_batch_to_items.debug(f"DUMMY BATCH detected: len={len(batch)}, doc_len={batch[0]}")
            logger_batch_to_items.debug(f"BEFORE finding rank: current_rank_idx={current_rank_idx}, rank_budgets={rank_budgets}")
            
            while current_rank_idx < as_world_size and rank_budgets[current_rank_idx] == 0:
                old_idx = current_rank_idx
                current_rank_idx += 1
                logger_batch_to_items.debug(f"Skipped rank {old_idx} (budget=0), moved to rank_idx={current_rank_idx}")
            
            logger_batch_to_items.debug(f"Selected rank_idx={current_rank_idx} for dummy doc, rank_budget={rank_budgets[current_rank_idx]}")
            assert rank_budgets[current_rank_idx] == num_tokens_per_rank, "dummy doc should put on a empty rank"
            # dummy doc, this rank only put this dummy item.
            doc_len = batch[0]
            item_dict = {'q': doc_len, 'kv': doc_len}
            new_item = Item(model_config, doc_len, seqid, current_rank_idx, current_rank_idx,
                            item_dict, is_original=True)
            items.append(new_item)
            logger_batch_to_items.debug(f"Created dummy item: seqid={seqid}, rank={current_rank_idx}, doc_len={doc_len}, items_count={len(items)}")
            logger_batch_to_items.debug(f"BEFORE increment: current_rank_idx={current_rank_idx}, rank_budgets={rank_budgets}")
            current_rank_idx += 1
            seqid += 1
            logger_batch_to_items.debug(f"AFTER dummy processing: current_rank_idx={current_rank_idx}, seqid={seqid}, rank_budgets={rank_budgets} (NOTE: budget NOT decremented!)")
        else:
            logger_batch_to_items.debug(f"REGULAR BATCH: len={len(batch)}, docs={batch}")
            for doc_idx, doc_length in enumerate(batch):
                doc_len = doc_length
                remaining_len = doc_len
                head_prefix_len = 0
                tail_suffix_start_pos = doc_len
                is_original_flag = True

                logger_batch_to_items.debug(f"Processing doc[{doc_idx}]: doc_len={doc_len}, remaining_len={remaining_len}")
                logger_batch_to_items.debug(f"BEFORE doc loop: current_rank_idx={current_rank_idx}, rank_budgets={rank_budgets}, seqid={seqid}")

                while remaining_len > 0:
                    logger_batch_to_items.debug(f"WHILE remaining_len={remaining_len} > 0")
                    logger_batch_to_items.debug(f"BEFORE finding available rank: current_rank_idx={current_rank_idx}, rank_budgets={rank_budgets}")
                    
                    while current_rank_idx < as_world_size and rank_budgets[current_rank_idx] == 0:
                        old_idx = current_rank_idx
                        current_rank_idx += 1
                        logger_batch_to_items.debug(f"Skipped rank {old_idx} (budget=0), moved to rank_idx={current_rank_idx}")
                    
                    logger_batch_to_items.debug(f"AFTER finding rank: current_rank_idx={current_rank_idx}, as_world_size={as_world_size}")
                    
                    if current_rank_idx >= as_world_size:
                        total_capacity = as_world_size * num_tokens_per_rank
                        total_allocated = total_capacity - sum(rank_budgets)
                        total_remaining = sum(rank_budgets)
                        logger_batch_to_items.debug(f"ERROR: current_rank_idx={current_rank_idx} >= as_world_size={as_world_size}")
                        logger_batch_to_items.debug(f"ERROR: total_remaining={total_remaining}, rank_budgets={rank_budgets}")
                        raise ValueError(
                            f"Not enough space to allocate document. \n"
                            f"seqid={seqid}, original_doc_len={doc_len}, remaining_len={remaining_len}. \n"
                            f"Configuration: as_world_size={as_world_size}, num_tokens_per_rank={num_tokens_per_rank}, total_capacity={total_capacity}. \n"
                            f"Current state: total_allocated={total_allocated}, total_remaining={total_remaining}, rank_budgets={rank_budgets}. \n"
                            f"Current batch being processed: {batch}. \n"
                            f"All batches: {batches}. \n"
                            f"Current rank index: {current_rank_idx}. \n"
                        )
                    
                    chunk_size = min(remaining_len, rank_budgets[current_rank_idx])
                    logger_batch_to_items.debug(f"Allocating chunk: chunk_size={chunk_size} (min of remaining_len={remaining_len}, rank_budget={rank_budgets[current_rank_idx]})")
                    
                    if remaining_len == doc_len and chunk_size == doc_len:
                        item_dict = {'q': doc_len, 'kv': doc_len}
                        new_item = Item(model_config, doc_len, seqid, current_rank_idx, current_rank_idx,
                                        item_dict, is_original=True)
                        items.append(new_item)
                        logger_batch_to_items.debug(f"Created single-chunk item: seqid={seqid}, rank={current_rank_idx}, doc_len={doc_len}, items_count={len(items)}")
                    else:
                        q_len_head = chunk_size // 2
                        q_len_tail = chunk_size - q_len_head
                        
                        new_head_kv = head_prefix_len + q_len_head
                        head_item = {'q': q_len_head, 'kv': new_head_kv}
                        tail_item = {'q': q_len_tail, 'kv': tail_suffix_start_pos}

                        new_item = Item(model_config, doc_len, seqid, current_rank_idx, current_rank_idx,
                                        head_item, tail_item, is_original=is_original_flag)
                        items.append(new_item)
                        logger_batch_to_items.debug(f"Created split-chunk item: seqid={seqid}, rank={current_rank_idx}, chunk_size={chunk_size}, head={head_item}, tail={tail_item}, items_count={len(items)}")

                        head_prefix_len = new_head_kv
                        tail_suffix_start_pos -= q_len_tail
                        is_original_flag = False
                        logger_batch_to_items.debug(f"Updated split state: head_prefix_len={head_prefix_len}, tail_suffix_start_pos={tail_suffix_start_pos}")

                    old_budget = rank_budgets[current_rank_idx]
                    rank_budgets[current_rank_idx] -= chunk_size
                    remaining_len -= chunk_size
                    logger_batch_to_items.debug(f"Updated budgets: rank[{current_rank_idx}] {old_budget} -> {rank_budgets[current_rank_idx]}, remaining_len: {remaining_len + chunk_size} -> {remaining_len}")

                seqid += 1
                logger_batch_to_items.debug(f"Finished doc[{doc_idx}]: seqid incremented to {seqid}")
        
        logger_batch_to_items.debug(f"AFTER batch[{batch_idx}]: rank_budgets={rank_budgets}, current_rank_idx={current_rank_idx}, seqid={seqid}, items_count={len(items)}")
    
    logger_batch_to_items.debug(f"\nFINAL: items_count={len(items)}, rank_budgets={rank_budgets}, current_rank_idx={current_rank_idx}, seqid={seqid}")
    return items


# In this function, world size is actually as_world_size.
def create_qkv_dispatch_pipeline_tick(
    mbc_config: MicrobatchCreateConfig,
    tolerance_factor: float = 0.05,
    ref_doc_lens: Optional[torch.Tensor] = None,
    add_dummy: bool = None,
):
    """
    softmax_lse_size (int): size of the softmax_lse tensor when viewed as the dtype,
        should be num_heads_local * fp32.itemsize // element_size.
    """
    # from distca.planner.planner import (
    #     batch_to_items_with_dummy,
    # )
    from distca.runtime.compute_metadata import (
        from_planner_output, backward_from_planner_output
    )
    
    create_pp_doclen_kwargs = dict(
        world_size=mbc_config.as_world_size,
        total_token_on_rank=mbc_config.total_seq_len,
        tp_size=mbc_config.tp_size,
        dp_size=mbc_config.dp_size,
    )

    # Get the new sequence lens from get_next_batch.
    cur_tick_per_rank_doc_lens, cur_tick_dummy_mask = create_pipeline_doclens(
        ref_doc_lens, 
        add_dummy, 
        is_backward=False, 
        world_size=mbc_config.as_world_size,
        total_token_on_rank=mbc_config.total_seq_len,
        tp_size=mbc_config.tp_size,
        dp_size=mbc_config.dp_size,
        num_batches = mbc_config.num_batches, 
    )
    assert isinstance(cur_tick_per_rank_doc_lens, list), f"cur_tick_per_rank_doc_lens: {cur_tick_per_rank_doc_lens} is not a list"
    assert len(cur_tick_per_rank_doc_lens) == len(cur_tick_dummy_mask), \
        f"Length mismatch: batches={len(cur_tick_per_rank_doc_lens)}, dummy_mask={len(cur_tick_dummy_mask)}"


    world_size = mbc_config.tp_size * mbc_config.pp_size * mbc_config.dp_size
    planner = Planner.from_individual_params(
        tp_size=mbc_config.tp_size,
        pp_size=mbc_config.pp_size,
        dp_size=mbc_config.dp_size,
        world_size=world_size,
        hidden_size_q=mbc_config.hidden_size_q_tp,
        hidden_size_k=mbc_config.hidden_size_k_tp
    )
    planner.tolerance_factor = tolerance_factor
    
    # Create a temp_model_config for item initialization.
    temp_model_config = SimpleNamespace(
        hidden_size=mbc_config.hidden_size_q_tp * mbc_config.tp_size,
        num_attention_heads = 1,
        num_key_value_heads = mbc_config.hidden_size_q_tp // mbc_config.hidden_size_k_tp,
        num_hidden_layers = 1
    )
    # Use batch_to_items_with_dummy to handle dummy doc to item.
    items = batch_to_items_with_dummy(
        batches=cur_tick_per_rank_doc_lens, 
        num_tokens_per_rank = mbc_config.num_token_per_rank,
        as_world_size=mbc_config.as_world_size,
        model_config=temp_model_config,
        dummy_mask=cur_tick_dummy_mask)
    
    fwd_planner_out = planner.items_to_shardinfo(items)
    
    (
        qkv_linear_to_attn_fa2a, _, 
        out_attn_to_linear_fa2a, _, 
        fwd_attn_metadata
    ) = from_planner_output(
        mbc_config.as_world_size, 
        fwd_planner_out, 
        mbc_config.hidden_size_q_tp, 
        mbc_config.hidden_size_k_tp,
        mbc_config.softmax_lse_size, 
        mbc_config.element_size, 
        is_pipeline_tick=True
    )
    
    # CP flip logic.
    if len(cur_tick_per_rank_doc_lens) < mbc_config.as_world_size:
        bwd_tick_per_rank_doc_lens = _block_reverse_list(cur_tick_per_rank_doc_lens, mbc_config.num_batches)
        # Reverse dummy_mask in the same way as batches
        bwd_tick_dummy_mask = _block_reverse_list(cur_tick_dummy_mask, mbc_config.num_batches)
    else:
        # None CP flip logic.
        bwd_tick_per_rank_doc_lens, bwd_tick_dummy_mask = create_pipeline_doclens(
            cur_tick_per_rank_doc_lens, add_dummy=False, is_backward=True,
            **create_pp_doclen_kwargs,
        )
    
    items = batch_to_items_with_dummy(
        batches=bwd_tick_per_rank_doc_lens, 
        num_tokens_per_rank=mbc_config.num_token_per_rank,
        as_world_size=mbc_config.as_world_size,
        model_config=temp_model_config,
        dummy_mask=bwd_tick_dummy_mask,
    )
    bwd_planner_out = planner.items_to_shardinfo(items)
    
    (
        qkv_resend_and_out_grad_linear_to_attn_fa2a, 
        qkv_grad_attn_to_linear_fa2a,
        bwd_attn_metadata
    ) = backward_from_planner_output(
         mbc_config.as_world_size, bwd_planner_out, mbc_config.hidden_size_q_tp, mbc_config.hidden_size_k_tp,
         mbc_config.softmax_lse_size, mbc_config.element_size,
     )

    return (fwd_attn_metadata, bwd_attn_metadata,
            qkv_linear_to_attn_fa2a, qkv_grad_attn_to_linear_fa2a,
            out_attn_to_linear_fa2a, qkv_resend_and_out_grad_linear_to_attn_fa2a,
            cur_tick_per_rank_doc_lens,)



def create_pp_microbatches(
    mbc_config: MicrobatchCreateConfig,
) -> 'List[MicroBatchItem]':
    # MicroBatchItem = Dict[str, Thing]
    #     "tokens": torch.Tensor,
    #     "labels": torch.Tensor,
    #     "loss_mask": torch.Tensor,
    #     "position_ids": torch.Tensor,
    #     "packed_seq_params": PingPangSingleStepPackedSeqParams,

    num_microbatch = mbc_config.num_microbatch
    pp_degree = mbc_config.pp_size # this is just pp-size, not pp-rank.
    as_rank = mbc_config.as_rank
    as_world_size = mbc_config.as_world_size
    num_token_per_rank = mbc_config.num_token_per_rank
    num_batches = mbc_config.num_batches

    tick_per_rank_doc_lens = None
    bwd_metadata = []
    microbatches = []

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
        ) = create_qkv_dispatch_pipeline_tick(
            mbc_config=mbc_config,
            ref_doc_lens=tick_per_rank_doc_lens,
            add_dummy=add_dummy_forward,
        )
        
        # For MLP-CP, we need to transfer List[List[int]] from CP layout back to DP, 
        # so each rank knows its number of tokens.
        #
        # Example 1 DP case:
        # tick_per_rank_doc_lens cp list: List[List[int]] 
        #     = [[8], [8], [8], [8], [256, 256],[128, 384],[512], [10, 502] ]
        # tick_per_rank_doc_lens mlp list 
        #     = [[8], [8], [8], [8], [256, 256],[128, 384],[512], [10, 502] ]
        #
        # Example 2 CP case:
        # tick_per_rank_doc_lens cp list: List[List[int]] 
        #     = [[8], [8], [8], [8], [256, 768],[512, 10, 502] ]
        # tick_per_rank_doc_lens mlp list 
        #     = [[8], [8], [8], [8], [256, 128, 128], [256, 256], [512], [10, 502]]
        #
        tick_per_rank_doc_lens_after_cp_transfer = cp_list_to_mlp_list(
            tick_per_rank_doc_lens, as_world_size, num_token_per_rank
        )
        this_rank_num_tokens = sum(tick_per_rank_doc_lens_after_cp_transfer[as_rank])
        logger.info(f"[create_pp_microbatches] [as_rank = {as_rank}] this_rank_num_tokens: {this_rank_num_tokens}, tick_per_rank_doc_lens_after_cp_transfer: {tick_per_rank_doc_lens_after_cp_transfer}, num_token_per_rank: {num_token_per_rank}")
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
        batch_registry, doclength_registry = get_registry()
        # assert len(batch_registry) == 1, f"batch_registry: {len(batch_registry)}"
        # tokens, labels, loss_mask, position_ids

        # Concatenate the different tokens, labels, loss_mask, and position_ids
        tokens = torch.cat([b['tokens'] for b in batch_registry[0]], dim=0)
        labels = torch.cat([b['labels'] for b in batch_registry[0]], dim=0)
        loss_mask = torch.cat([b['loss_mask'] for b in batch_registry[0]], dim=0)
        position_ids = torch.cat([b['position_ids'] for b in batch_registry[0]], dim=0)

        logger.debug(f'create_pp_microbatches[{i}] batch_registry: {len(batch_registry)}, doclength_registry: {len(doclength_registry)}')

        # position_ids_local = torch.arange(this_rank_num_tokens)
        microbatch = {
            "tokens": tokens,
            "labels": labels,
            "loss_mask": loss_mask,
            "position_ids": position_ids,
            "packed_seq_params": ping_pang_params,
        }
        # logger.debug(f"{microbatch = }")
        microbatches.append(microbatch)

        clear_registry()

        # store the corresponding bwd metadata (for later ticks)
        bwd_metadata.append(
            (
                qkv_bwd_fa2a_metadata.get_slice(as_rank), 
                attn_out_qkv_bwd_fa2a_metadata.get_slice(as_rank), 
                bwd_packed_seq_params
            )
        )

    # put bwd metadata to the corresponding side
    for i, microbatch in enumerate(microbatches):
        # 
        # When mb_i is computed on pp_rank at forward tick t, 
        # assume the backward right after this forward is at tick t'.
        # mb_i requires another (pp_degree - 1 - pp_rank) forward steps to start mb_i backward,
        # and another (pp_degree - 1 - pp_rank) 
        # backward steps to compute mb_i backward on this pp_rank.
        #
        # Since in 1F1B, every forward follows a backward, so backward tick
        # t' + 2 * (pp_degree - 1 - pp_rank) is the one to compute mb_i on this pp_rank.
        # Note that at the end of PP warmup, forward tick is pp_degree - 1, 
        # while backward tick is 0. 
        #
        # Therefore, t - t' = pp_degree - 1, and thus
        # t' + 2 * (pp_degree - 1 - pp_rank) == t + pp_degree - 1 - pp_rank * 2
        bwd_metadata_idx = (i + pp_degree - 1 - pp_rank * 2) % len(bwd_metadata)
        qkv_bwd_metadata, attn_out_bwd_metadata, bwd_packed_seq_params = bwd_metadata[bwd_metadata_idx]
        packed_seq_params = microbatch["packed_seq_params"]
        packed_seq_params.qkv_bwd_metadata = qkv_bwd_metadata
        packed_seq_params.attn_out_bwd_metadata = attn_out_bwd_metadata
        packed_seq_params.bwd_packed_seq_params = bwd_packed_seq_params

    return microbatches


def create_combined_ping_pong_microbatches(
    mbc_config: MicrobatchCreateConfig,
) -> list:
    """
    Create combined microbatches for ping-pong (two dispatchers).
    Returns a list of combined microbatches.
    """
    from distca.runtime.megatron.packed_seq_params import PingPangPackedSeqParams
    
    # Create microbatches for ping-pong (two dispatchers)
    microbatches_0 = create_pp_microbatches(mbc_config)
    # assert len(microbatches_0) == mbc_config.num_microbatch + mbc_config.pp_size - 1, f"microbatches_0: {len(microbatches_0)}"
    microbatches_1 = create_pp_microbatches(mbc_config)
    # assert len(microbatches_1) == mbc_config.num_microbatch + mbc_config.pp_size - 1, f"microbatches_1: {len(microbatches_1)}"

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
        
        # TODO: Need to implement the input_ids as the token ids from the data loader.
        # # For ping-pong mode, input_ids should be 1D (num_tokens,) not 2D (1, num_tokens)
        # # because gptmodel_forward will unsqueeze(0) if pre_process is True, and ping-pong
        # # needs to split hidden_states along batch dimension into 2 parts for the 2 dispatchers.
        # # The actual splitting happens based on packed_seq_params which contains metadata
        # # for both dispatchers (mb_0 and mb_1).
        
        # # This is the reference shape of the input_ids, labels, loss_mask, and position_ids tensors
        # input_ids = torch.randint(10, 1000, (num_tokens,), device=torch.cuda.current_device())
        # labels = torch.randint(10, 1000, (num_tokens,), device=torch.cuda.current_device())
        # loss_mask = torch.randint(0, 2, (num_tokens,), device=torch.cuda.current_device())
        # position_ids = torch.concat([mb_0["position_ids"], mb_1["position_ids"]])
        # labels = labels.unsqueeze(0)
        
        dim = 1
        input_ids = torch.cat([mb_0["tokens"], mb_1["tokens"]], dim=dim).reshape(-1)
        labels = torch.cat([mb_0["labels"], mb_1["labels"]], dim=dim).reshape(-1).unsqueeze(0)
        loss_mask = torch.cat([mb_0["loss_mask"], mb_1["loss_mask"]], dim=dim).reshape(-1)
        position_ids = torch.cat([mb_0["position_ids"], mb_1["position_ids"]], dim=dim).reshape(-1)
        logger.info(f"{input_ids.shape = }, {labels.shape = }, {loss_mask.shape = }, {position_ids.shape = }")

        combined_mb = dict(
            input_ids=input_ids,
            labels=labels,
            loss_mask=loss_mask,
            position_ids=position_ids,
            packed_seq_params=ping_pong_params,
        )
        combined_microbatches.append(combined_mb)
    
    return combined_microbatches


logger.info(f"Creating microbatches: num_microbatch={num_microbatches}, pp_size={pp_size}, "
            f"as_rank={as_rank}, as_world_size={as_world_size}")

# combined_microbatches = create_combined_ping_pong_microbatches(mbc_config)

# logger.info(f"Created {len(combined_microbatches)} combined microbatches")

# ===============================================
# Training Loop using forward_backward_batch
# ===============================================

iteration = 0
max_iterations = args.train_iters if hasattr(args, 'train_iters') else 1

# Register NaN detection hooks (uncomment to enable)
if args.distca_debug_nan_detector:
    nan_hooks, nan_detected = register_nan_detection_hooks(
        model, 
        logger=logger, 
        raise_on_nan=True,  # Set to True to stop on first NaN
        check_inputs=True,    # Set to True to also check inputs
        log_all_modules=True,
        log_tensor_stats=True,
        tensor_stats_elements=10,
    )

for iteration in range(max_iterations):
    logger.info(f"Starting iteration {iteration}")

    if rank == 0:
        # Create combined microbatches for this iteration
        combined_microbatches = create_combined_ping_pong_microbatches(mbc_config)
    torch.distributed.barrier()

    # Broadcast the microbatches to the different ranks after planning.
    if rank == 0:
        pass
    else:
        pass
    
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
    
    # TODO: Some logic including increment the opt_param_scheduler 
    # step should be included here.

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

