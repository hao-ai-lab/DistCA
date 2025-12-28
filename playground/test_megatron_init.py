import argparse
import os
import logging

from datetime import timedelta

from utils.logging import (
    setup_logging,
    log_tensor_stats,
    log_module,
    time_it,
)
from utils.cpu_affinity import set_cpu_affinity
from utils.hf_config import (
    get_megatron_args_dict_from_hf_model,
    get_known_model_config,
    MegatronModelArgs,
    KNOWN_MODEL_CONFIGS,
)

# --------------------------------
# Environment Variables
# --------------------------------
"""
[rank0]:   File "/mnt/weka/home/hao.zhang/jd/distca/playground/test_megatron_init.py", line 427, in <module>
[rank0]:     initialize_megatron(
[rank0]:   File "/mnt/weka/home/hao.zhang/jd/Megatron-LM/megatron/training/initialize.py", line 86, in initialize_megatron
[rank0]:     validate_args(args, args_defaults)
[rank0]:   File "/mnt/weka/home/hao.zhang/jd/Megatron-LM/megatron/training/arguments.py", line 779, in validate_args
[rank0]:     assert os.environ.get('CUDA_DEVICE_MAX_CONNECTIONS') == "1", \
[rank0]:            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
[rank0]: AssertionError: Using tensor model parallelism or context parallelism require setting the environment variable CUDA_DEVICE_MAX_CONNECTIONS to 1
"""
env_vars = dict(
    CUDA_DEVICE_MAX_CONNECTIONS=1,
    # TORCH_NCCL_CONNECT_TIMEOUT=60000,
    # TORCH_NCCL_ASYNC_ERROR_HANDLING=1,
)
for k, v in env_vars.items():
    os.environ[k] = str(v)


# --------------------------------
# Logging
# --------------------------------
# Fetch rank early for use in formatter
rank = int(os.environ["RANK"])
world_size = int(os.environ["WORLD_SIZE"])
local_rank = int(os.environ["LOCAL_RANK"])

# Initialize logging with rank-aware formatting
# Only specified ranks will log to console; all ranks get file logging later
logger = setup_logging(
    rank=rank, world_size=world_size,
    level=logging.DEBUG,
    console_ranks=list(range(0, world_size, 8)),  # Only rank 0 logs to console
)


with time_it("import torch"):
    import torch


# --------------------------------
# Core-binding
# --------------------------------
with time_it("core-binding"):
    set_cpu_affinity(local_rank, ncpu_per_proc=16, logger=logger)



logger.info(f"Rank: {rank}, World Size: {world_size}, Local Rank: {local_rank}")

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

with time_it("import megatron.core"):
    import megatron.core
with time_it("import megatron.core.tensor_parallel"):
    from megatron.core import tensor_parallel
with time_it("import megatron.core.parallel_state"):
    from megatron.core import parallel_state




# Use parallel_state to initialize the model parallel groups
with time_it("initialize model parallel groups"):
    # 2 gpus: tp = 2
    # 4 gpus: tp = 2, pp = 2
    # 8 gpus: tp = 2, pp = 2, cp = 2
    # 16 gpus: tp = 4, pp = 2, cp = 2
    # 32 gpus: tp = 8, pp = 2, cp = 2
    # more gpus: tp = 8, pp = 2, cp = N where N is the number of gpus divided by 16

    # # tp = 1; pp = 1; cp = 1;
    # if world_size == 1:
    #     tp = 1; pp = 1; cp = 1;
    # elif world_size == 2:
    #     tp = 2; pp = 1; cp = 1;
    # elif world_size == 4:
    #     tp = 2; pp = 2; cp = 1;
    #     # tp = 4; pp = 1; cp = 1;
    # elif world_size == 8:
    #     tp = 2; pp = 2; cp = 2;
    #     # tp = 8; pp = 1; cp = 1;
    # elif world_size == 16:
    #     tp = 2; pp = 2; cp = 2;
    # elif world_size == 32:
    #     tp = 4; pp = 2; cp = 2;
    # else:
    #     tp = 8; pp = 2; cp = 2;
    
    tp = 1; pp = 1; cp = 1;
    # tp = min(8, world_size);
    # pp = world_size // tp;
    cp = 1;
    # tp = 1; pp = 2; cp = 1;

    parallel_state.initialize_model_parallel(
        tensor_model_parallel_size=tp, 
        pipeline_model_parallel_size=pp,
        context_parallel_size=cp,
        distributed_timeout_minutes=2,
        # nccl_communicator_config_path=None, # default
        order = "tp-cp-ep-dp-pp", # default
    )

    # get the tp,pp,cp,dp,ep rank of this process
    tp_rank = parallel_state.get_tensor_model_parallel_rank()
    tp_size = parallel_state.get_tensor_model_parallel_world_size()
    pp_rank = parallel_state.get_pipeline_model_parallel_rank()
    pp_size = parallel_state.get_pipeline_model_parallel_world_size()
    cp_rank = parallel_state.get_context_parallel_rank()
    cp_size = parallel_state.get_context_parallel_world_size()
    dp_rank = parallel_state.get_data_parallel_rank()
    dp_size = parallel_state.get_data_parallel_world_size()
    ep_rank = parallel_state.get_expert_model_parallel_rank()
    ep_size = parallel_state.get_expert_model_parallel_world_size()
    logger.info(f"TP: {tp_rank} / {tp_size}, PP: {pp_rank} / {pp_size}, CP: {cp_rank} / {cp_size}, DP: {dp_rank} / {dp_size}, EP: {ep_rank} / {ep_size}")

    # Print all of the tp/pp/cp/dp ranks
    def print_ranks(group):
        ranks = torch.distributed.get_process_group_ranks(group)
        return ranks
    logger.info(f"TP Ranks: {print_ranks(parallel_state.get_tensor_model_parallel_group())}")
    logger.info(f"PP Ranks: {print_ranks(parallel_state.get_pipeline_model_parallel_group())}")
    logger.info(f"CP Ranks: {print_ranks(parallel_state.get_context_parallel_group())}")
    logger.info(f"DP Ranks: {print_ranks(parallel_state.get_data_parallel_group())}")
    logger.info(f"EP Ranks: {print_ranks(parallel_state.get_expert_model_parallel_group())}")


tp_groups = parallel_state.get_tensor_model_parallel_group()
if isinstance(tp_groups, list):
    # If multiple groups (for multi-TP), allreduce on each group separately
    for group in tp_groups:
        t = torch.tensor([tp_rank], device="cuda")
        torch.distributed.all_reduce(t, group=group)
        logger.info(f"Allreduced value in tp_group {group}: {t.item()}")
else:
    # Single group (typical for Megatron-LM)
    t = torch.tensor([tp_rank], device="cuda")
    torch.distributed.all_reduce(t, group=tp_groups)
    logger.info(f"Allreduced value in tp_group: {t.item()}")


torch.distributed.barrier()
logger.info(f"Finish initializing megatron parallel groups.")

from utils.logging import setup_log_directories, redirect_external_loggers
from pathlib import Path

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

data_path = Path(__file__).parent / 'data_process' / 'code_content_document'
data_path = data_path.resolve().absolute()
logger.info(f"Data path: {data_path}")



# --------------------------------
# Model Configuration from HuggingFace
# --------------------------------
# You can either:
# 1. Use a HuggingFace model name/path to auto-load config
# 2. Use a known model config key (works offline)

# Option 1: Load from HuggingFace (requires network or cached model)
# HF_MODEL_NAME = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
# HF_MODEL_NAME = "meta-llama/Llama-3.1-8B"

# Option 2: Use known config (works offline)
HF_MODEL_NAME = None  # Set to None to use KNOWN_MODEL_KEY
KNOWN_MODEL_KEY = "llama-8b"  # Options: "llama-7b", "llama-8b", "llama-70b", "deepseek-llama-8b"

# Sequence length for training (can be shorter than model's max)
k = 1024
seq_length = k * 4
# seq_length = k * 128

# Compute the number of layers based on pp
# num_layers_override = None
num_layers_override = 2

# Get model architecture config dict from HF config or known config
if HF_MODEL_NAME is not None:
    logger.info(f"Loading model config from HuggingFace: {HF_MODEL_NAME}")
    model_config_dict = get_megatron_args_dict_from_hf_model(
        HF_MODEL_NAME,
        seq_length=seq_length,
        num_layers_override=num_layers_override,
    )
else:
    logger.info(f"Using known model config: {KNOWN_MODEL_KEY}")
    # Get config object for reference (returns a copy with overrides applied)
    model_config = get_known_model_config(
        KNOWN_MODEL_KEY,
        seq_length=seq_length,
        num_layers_override=num_layers_override,
    )
    model_config_dict = {
        "num_layers": model_config.num_layers,
        "hidden_size": model_config.hidden_size,
        "num_attention_heads": model_config.num_attention_heads,
        "num_query_groups": model_config.num_query_groups,
        "ffn_hidden_size": model_config.ffn_hidden_size,
        "seq_length": seq_length,
        "max_position_embeddings": seq_length,
        "vocab_size": model_config.vocab_size,
        "normalization": model_config.normalization,
        "position_embedding_type": model_config.position_embedding_type,
        "rotary_base": model_config.rotary_base,
        "swiglu": model_config.swiglu,
        "untie_embeddings_and_output_weights": model_config.untie_embeddings_and_output_weights,
    }

logger.info(f"Model config: {model_config_dict}")


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
    "--train-iters", "1",
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
    "--bf16",  # REQUIRED for flash attention to work!

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
    # "--recompute-modules", "mlp",
    "--recompute-granularity", "selective",
    "--recompute-modules", "core_attn", #"mlp",

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


# -----------------------------------------
# Get the proper config object
# -----------------------------------------
from megatron.core.transformer.transformer_config import TransformerConfig
from dataclasses import dataclass
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

from megatron.training.global_vars import get_args
from megatron.training.arguments import core_transformer_config_from_args
from megatron.training.yaml_arguments import core_transformer_config_from_yaml
from megatron.core.transformer.transformer_config import TransformerConfig

args = get_args()
if args.yaml_cfg is not None:
    assert False, "YAML config is not supported yet becuase inside the core_transformer_config_from_yaml, you can only use TransformerConfig. We need to extend the config class to support DistCA configs."
    config = core_transformer_config_from_yaml(args, "language_model")
else:
    config = core_transformer_config_from_args(args, config_class=DistCAConfig)
assert isinstance(config, TransformerConfig)


# -----------------------------------------
# Report theoretical memory and FLOPS estimates
# -----------------------------------------
from megatron.core.num_microbatches_calculator import get_num_microbatches
from utils.estimate import log_memory_estimate, log_flops_estimate


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
    dp_world_size=parallel_state.get_data_parallel_world_size(),
    tp=tp,
    pp=pp,
    cp=cp,
    logger=logger,
)


# -----------------------------------------
# Set pytorch JIT layer fusion options and warmup JIT functions.
# -----------------------------------------
# with time_it("set_jit_fusion_options()"):
#     # 01:08:00 [Rank 0] INFO set_jit_fusion_options() took 5.3942 seconds
#     from megatron.training.initialize import set_jit_fusion_options
#     set_jit_fusion_options()


# -----------------------------------------
# Model, optimizer, and learning rate.
# -----------------------------------------
from megatron.training.training import setup_model_and_optimizer
from megatron.core.enums import ModelType
from typing import Union
from megatron.core.models.gpt import GPTModel
def model_provider(pre_process=True, post_process=True) -> Union[GPTModel, 'megatron.legacy.model.GPTModel']:
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

    if args.record_memory_history:
        torch.cuda.memory._record_memory_history(True,
            # keep 100,000 alloc/free events from before the snapshot
            trace_alloc_max_entries=100000,

            # record stack information for the trace events
            trace_alloc_record_context=True)

        def oom_observer(device, alloc, device_alloc, device_free):
            # snapshot right after an OOM happened
            print('saving allocated state during OOM')
            snapshot = torch.cuda.memory._snapshot()
            from pickle import dump
            dump(snapshot, open(f"oom_rank-{torch.distributed.get_rank()}_{args.memory_snapshot_path}", 'wb'))

        torch._C._cuda_attach_out_of_memory_observer(oom_observer)
    logger.info(f"Building model for rank {torch.distributed.get_rank()}")


    if args.num_experts:
        from megatron.core.models.gpt.gpt_layer_specs import get_gpt_decoder_block_spec
        # Define the decoder block spec
        transformer_layer_spec = get_gpt_decoder_block_spec(config, use_transformer_engine=use_te, normalization=args.normalization)
    elif args.heterogeneous_layers_config_path is not None:
        from megatron.core.models.gpt.gpt_layer_specs import get_gpt_heterogeneous_layer_spec
        transformer_layer_spec = get_gpt_heterogeneous_layer_spec(config, use_te)
    else:
        # Define the decoder layer spec
        if use_te:
            from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_with_transformer_engine_spec
            transformer_layer_spec = get_gpt_layer_with_transformer_engine_spec(
                args.num_experts, args.moe_grouped_gemm,
                args.qk_layernorm, args.multi_latent_attention, args.moe_use_legacy_grouped_gemm)
        else:
            from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_local_spec
            transformer_layer_spec = get_gpt_layer_local_spec(
                args.num_experts, args.moe_grouped_gemm,
                args.qk_layernorm, args.multi_latent_attention, args.moe_use_legacy_grouped_gemm,
                normalization=args.normalization)
    
    mtp_block_spec = None
    if args.mtp_num_layers is not None:
        from megatron.core.models.gpt.gpt_layer_specs import get_gpt_mtp_block_spec
        mtp_block_spec = get_gpt_mtp_block_spec(config, transformer_layer_spec, use_transformer_engine=use_te)

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
        mtp_block_spec=mtp_block_spec,
    )

    return model

with time_it("setup_model_and_optimizer()"):
    model_type = ModelType.encoder_or_decoder
    model, optimizer, opt_param_scheduler = setup_model_and_optimizer(
        model_provider, model_type, checkpointing_context=checkpointing_context
    )
    logger.info(f"Successfully setup model and optimizer for rank {torch.distributed.get_rank()}")

logger.info('after model, optimizer, and learning rate '
            'scheduler are built')

logger.debug(f"Model: {model}")
logger.debug(f"Optimizer: {optimizer}")
logger.debug(f"Learning Rate Scheduler: {opt_param_scheduler}")

# --------------------------------
# Initialize DataLoader (?)
# --------------------------------
import numpy
from typing import Optional, Tuple, List
from megatron.training.tokenizer.tokenizer import MegatronTokenizer
from megatron.core.datasets.utils import Split
from megatron.training.training import build_train_valid_test_data_iterators
from megatron.core.datasets.blended_megatron_dataset_builder import BlendedMegatronDatasetBuilder
from megatron.core.datasets.gpt_dataset import MockGPTDataset, GPTDataset
from megatron.training import get_tokenizer
from typing import Optional, Tuple, List
from megatron.training.utils import (
    get_batch_on_this_cp_rank,
    get_batch_on_this_tp_rank,
    get_blend_and_blend_per_split,
)
from megatron.core.datasets.gpt_dataset import GPTDatasetConfig

mpu = parallel_state

def is_dataset_built_on_rank():
    return (
        mpu.is_pipeline_first_stage() or mpu.is_pipeline_last_stage()
    ) and mpu.get_tensor_model_parallel_rank() == 0


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



class DistCAMockGPTLowLevelDataset:
    """The mock GPT low level dataset

    This class is meant to generate tokenized data in the classic "Megatron-LM" GPT style. Notably,
    we add the end of document token to each element indexed in __getitem__

    Args:
        tokenizer (MegatronTokenizer): The tokenizer the special token information of which we use
            to augment the mock data.
    """

    seed: int = 0
    """The hard-coded random seed to use to set the NumPy RNG"""

    size: int = 100000
    """The hard-coded number of samples to generate"""

    max_sequence_length: int = 512
    """The hard-coded max sequence length to generate"""

    def __init__(self, tokenizer: MegatronTokenizer) -> None:
        self.tokenizer = tokenizer
        rng = numpy.random.default_rng(seed=self.seed)
        self.sequence_lengths = rng.integers(
            low=1, high=self.max_sequence_length, 
            size=self.size, dtype=numpy.int32
        )

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, idx: int) -> numpy.number:
        length = self.sequence_lengths[idx]
        sample = numpy.int64(
            numpy.concatenate([numpy.arange(length - 1) + 1, [self.tokenizer.eod]])
        )
        return sample

    def get(self, idx: int, offset: int = 0, length: Optional[int] = None) -> numpy.ndarray:
        """This function is an abstraction over __getitem__ with support for slicing

        Args:
            idx (int): The index into the dataset

            offset (int): The integer token offset in the sequence

            length (Optional[int]): The number of tokens to grab from the sequence

        Returns:
            numpy.ndarray: The sequence tokens at the index
        """
        if length is None:
            length = self.sequence_lengths[idx] - offset
        return self[idx][offset : offset + length]


class DistCAMockGPTDataset(GPTDataset):
    """The mock GPT dataset

    Args:
        indexed_dataset (MockGPTLowLevelDataset): The MockGPTLowLevelDataset around which to build
            the MockGPTDataset

        dataset_path (Optional[str]): This argument is of no consequence for the MockGPTDataset

        indices (numpy.ndarray): The set of the dataset indices to expose

        num_samples (int): The number of samples to draw from the dataset

        index_split (Split): The indices Split

        config (GPTDatasetConfig): The config
    """

    def __init__(
        self,
        dataset: DistCAMockGPTLowLevelDataset,
        dataset_path: Optional[str],
        indices: numpy.ndarray,
        num_samples: int,
        index_split: Split,
        config: GPTDatasetConfig,
    ) -> None:
        assert config.mock

        super().__init__(dataset, dataset_path, indices, num_samples, index_split, config)

    @staticmethod
    def numel_low_level_dataset(low_level_dataset: DistCAMockGPTLowLevelDataset) -> int:
        """Abstract method implementation

        Args:
            low_level_dataset (DistCAMockGPTLowLevelDataset): The underlying DistCAMockGPTLowLevelDataset

        Returns:
            int: The number of unique elements in the underlying DistCAMockGPTLowLevelDataset
        """
        return len(low_level_dataset)

    @staticmethod
    def build_low_level_dataset(
        dataset_path: Optional[str], config: GPTDatasetConfig
    ) -> DistCAMockGPTLowLevelDataset:
        """Abstract method implementation

        Args:
            dataset_path (Optional[str]): This argument is of no consequence for the
                DistCAMockGPTLowLevelDataset

            config (GPTDatasetConfig): The config

        Returns:
            DistCAMockGPTLowLevelDataset: The underlying DistCAMockGPTLowLevelDataset
        """
        return DistCAMockGPTLowLevelDataset(config.tokenizer)


def train_valid_test_datasets_provider(train_val_test_num_samples):
    """Build the train test and validation datasets.

    Args:
        train_val_test_num_samples : A list containing the number of samples in train test and validation.
    """
    args = get_args()

    config = core_gpt_dataset_config_from_args(args)
    logger.info(f"> GPTDatasetConfig: {config}")

    if args.mock_data:
        dataset_type = DistCAMockGPTDataset
    else:
        dataset_type = GPTDataset

    logger.info("> building train, validation, and test datasets for GPT ...")

    train_ds, valid_ds, test_ds = BlendedMegatronDatasetBuilder(
        dataset_type,
        train_val_test_num_samples,
        is_dataset_built_on_rank,
        config
    ).build()

    logger.info("> finished creating GPT datasets ...")

    return train_ds, valid_ds, test_ds

train_valid_test_datasets_provider.is_distributed = True
train_data_iterator, valid_data_iterator, test_data_iterator \
    = build_train_valid_test_data_iterators(
        train_valid_test_datasets_provider)


logger.debug(f"train_data_iterator: {train_data_iterator}")
logger.debug(f"valid_data_iterator: {valid_data_iterator}")
logger.debug(f"test_data_iterator: {test_data_iterator}")

if rank == 0:
    # train_data_iterator: RerunDataIterator
    from megatron.core.rerun_state_machine import RerunDataIterator
    # assert isinstance(train_data_iterator, RerunDataIterator)
    # logger.debug(f"train_data_iterator.iterable: {train_data_iterator.iterable}")
    # logger.debug(f"train_data_iterator.saved_microbatches: {train_data_iterator.saved_microbatches}")

    # for i, item in enumerate(train_data_iterator.iterable):
    #     logger.debug(f"train_data_iterator item {i}: {item}")
    #     if i > 5:
    #         break

logger.info('done with setup ...')



# -----------------------------------------
# Monkey Patch some training functions
# -----------------------------------------

# Patch forward_step and backward_step with nvtx markers
import torch.cuda.nvtx
import megatron.core.pipeline_parallel.schedules
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
import megatron.core.pipeline_parallel.schedules 
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
import megatron.training.training
old_train_step = megatron.training.training.train_step
def train_step_with_nvtx(*args, **kwargs):
    logger.debug(f"Start train_step")
    with torch.cuda.nvtx.range("train_step"):
        return old_train_step(*args, **kwargs)
megatron.training.training.train_step = train_step_with_nvtx


# Patch each layer such that I will know when the forward function gets called
import megatron.core.transformer.transformer_layer
old_transformer_layer_forward = megatron.core.transformer.transformer_layer.TransformerLayer.forward
def transformer_layer_forward_with_nvtx(self, *args, **kwargs):
    # logger.debug(f"Start transformer_layer_forward[{self.layer_number}]")
    with torch.cuda.nvtx.range(f"transformer_layer_forward[{self.layer_number}]"):
        r = old_transformer_layer_forward(self, *args, **kwargs)
    # logger.debug(f"End transformer_layer_forward[{self.layer_number}]")
    return r
megatron.core.transformer.transformer_layer.TransformerLayer.forward = transformer_layer_forward_with_nvtx

logger.info(f"model: {model}")


# -----------------------------------------
# Start training
# -----------------------------------------
from megatron.training.training import train

def get_batch(data_iterator):
    """Generate a batch."""

    # TODO: this is pretty hacky, find a better way
    if (not mpu.is_pipeline_first_stage()) and (not mpu.is_pipeline_last_stage()):
        return None, None, None, None, None

    # get batches based on the TP rank you are on
    # logger.debug(f"Get batch on this TP rank")
    batch = get_batch_on_this_tp_rank(data_iterator)
    # logger.debug(f"Finished get batch on this TP rank.")
    # logger.debug(f"Batch: {batch}")

    # slice batch along sequence dimension for context parallelism
    # logger.debug(f"Slice batch along sequence dimension for context parallelism")
    batch = get_batch_on_this_cp_rank(batch)
    # logger.debug(f"Finished slice batch along sequence dimension for context parallelism.")
    # logger.debug(f"Batch: {batch}")

    # logger.debug(f"Return batch")
    return batch.values()


# define spiky loss as a loss that's 10x the max loss observed
SPIKY_LOSS_FACTOR = 10

from functools import partial
from megatron.core.rerun_state_machine import get_rerun_state_machine


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


from megatron.training.training import get_timers
from megatron.core.utils import StragglerDetector

stimer = StragglerDetector()

# Token monitoring - import from utils
from utils.token_monitor import (
    monitor_batch_tokens,
    set_token_monitor_config,
    get_token_monitor_config,
)

# Configure token monitoring (optional - defaults are sensible)
set_token_monitor_config(enabled=True, max_tokens_to_decode=200, max_samples_to_log=2)


def forward_step(data_iterator, model: GPTModel):
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
        tokens, labels, loss_mask, attention_mask, position_ids = get_batch(
            data_iterator)
    
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




process_non_loss_data_func = None
non_loss_data_func = None


# %%
# # Add additional safe guard to ensure everyone is here.
# # Then do a broadcast and an allreduce to ensure everyone is here.
# torch.distributed.barrier()

# logger.info(f"Rank {rank} starting safe guard...")
# t = torch.tensor([rank], device="cuda")
# torch.distributed.all_reduce(t)
# logger.info(f"Allreduced value: {t.item()}")

# logger.info(f"Rank {rank} preparing to broadcast or receive value...")
# t = torch.tensor([0], device="cuda")
# if rank == 0:
#     t.fill_(42)
# torch.distributed.broadcast(t, src=0)
# logger.info(f"Rank {rank} received broadcast value: {t.item()}")

# torch.distributed.barrier()

# logger.info(f"Rank {rank} finished safe guard. Now starting training...")


memory_estimate = log_memory_estimate(
    args, 
    num_microbatches=num_microbatches, 
    verbose=True, 
    logger=logger
)
logger.info(f"Memory estimate: {memory_estimate}")


# %%
iteration = 0
iteration, num_floating_point_operations_so_far = train(
    forward_step,
    model, optimizer, opt_param_scheduler,
    train_data_iterator, valid_data_iterator,
    process_non_loss_data_func, config, checkpointing_context,
    non_loss_data_func
)



# %%
# -----------------------------------------
# Finish training
# -----------------------------------------

from megatron.training.global_vars import get_wandb_writer
wandb_writer = get_wandb_writer()
if wandb_writer:
    wandb_writer.finish()




# -----------------------------------------
# Test Individual components
# -----------------------------------------
# from test_vocab import test_vocab
# test_vocab()


logger.info(f"Rank {rank} is exiting")
torch.distributed.destroy_process_group()