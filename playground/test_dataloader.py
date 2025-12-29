"""
Minimal Dataloader Test Script
Extracted from test_distca.py
"""

# ================================
# Import torch and megatron
# ================================
import contextlib


with contextlib.nullcontext():
    import os
    import argparse
    import logging
    from pathlib import Path
    from typing import Optional, Tuple, List

    from distca.utils.logging import (
        setup_logging,
        time_it,
        setup_log_directories,
        redirect_external_loggers,
    )
    from utils.cpu_affinity import set_cpu_affinity
    from utils.hf_config import (
        get_megatron_args_dict_from_hf_model,
    )


rank = int(os.environ["RANK"])
world_size = int(os.environ["WORLD_SIZE"])
local_rank = int(os.environ["LOCAL_RANK"])
logger = setup_logging(
    rank=rank, world_size=world_size,
    level=logging.DEBUG,
    console_ranks=[0],
)

with time_it("import torch"):
    import torch

with time_it("set device"):
    torch.cuda.set_device(local_rank)
    torch.set_default_device(torch.device("cuda", local_rank))

with time_it("init_process_group"):
    from datetime import timedelta
    torch.distributed.init_process_group(
        backend="cpu:gloo,cuda:nccl", 
        rank=rank, 
        world_size=world_size, 
        timeout=timedelta(seconds=60),
    )

# ================================
# Megatron Imports
# ================================
with time_it("import megatron.core"):
    import megatron.core
    from megatron.core import parallel_state
    mpu = parallel_state

    from megatron.training.global_vars import get_args
    from megatron.training.yaml_arguments import core_transformer_config_from_yaml
    from megatron.training.arguments import core_transformer_config_from_args
    from megatron.core.transformer.transformer_config import TransformerConfig
    from megatron.core.datasets.utils import Split
    from megatron.core.datasets.blended_megatron_dataset_builder import BlendedMegatronDatasetBuilder
    from megatron.core.datasets.gpt_dataset import GPTDataset, GPTDatasetConfig
    from megatron.training import get_tokenizer
    from megatron.training.utils import (
        get_blend_and_blend_per_split,
    )
    from megatron.core.rerun_state_machine import RerunDataIterator
    from megatron.training.training import build_train_valid_test_data_loaders

# ====================================
# Initialize Megatron Parallel Groups
# ====================================
tp = int(os.environ["TP"])
pp = int(os.environ["PP"])
cp = int(os.environ["CP"])
dp = int(os.environ["DP"])
assert tp * pp * cp * dp == world_size, f"{tp = }, {pp = }, {cp = }, {dp = }: tp * pp * cp * dp = {tp * pp * cp * dp} != world_size: {world_size}"

with time_it("initialize model parallel groups"):
    mpu.initialize_model_parallel(
        tensor_model_parallel_size=tp, 
        pipeline_model_parallel_size=pp,
        context_parallel_size=cp,
        distributed_timeout_minutes=2,
        order="tp-cp-ep-dp-pp",
    )

    tp_rank = mpu.get_tensor_model_parallel_rank()
    pp_rank = mpu.get_pipeline_model_parallel_rank()
    cp_rank = mpu.get_context_parallel_rank()
    dp_rank = mpu.get_data_parallel_rank()
    logger.info(f"TP: {tp_rank} / {tp}, PP: {pp_rank} / {pp}, CP: {cp_rank} / {cp}, DP: {dp_rank} / {dp}")

torch.distributed.barrier()
logger.info(f"Finish initializing megatron parallel groups.")


# ================================
# Setup logging directories
# ================================
log_paths = setup_log_directories(
    rank=rank,
    barrier_fn=torch.distributed.barrier,
)

redirect_external_loggers(["megatron"], level=logging.INFO)

log_root_dir = log_paths.log_root_dir
data_cache_path = log_paths.data_cache_path
data_path = Path(__file__).parent / 'data_process' / 'code_content_document'
data_path = data_path.resolve().absolute()
logger.info(f"Data path: {data_path}")

# ================================
# Model Configuration from HuggingFace
# ================================
with time_it("load model config from HuggingFace"):
    HF_MODEL_NAME = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
    k = 1024
    # seq_length = k * 4
    seq_length = 8 * k
    num_layers_override = 1

    logger.info(f"Loading model config from HuggingFace: {HF_MODEL_NAME}")
    all_config_dict = get_megatron_args_dict_from_hf_model(
        HF_MODEL_NAME,
        seq_length=seq_length,
        num_layers_override=num_layers_override,
    )
    hf_config = all_config_dict["hf_config"]
    megatron_args = all_config_dict["megatron_args"]
    model_config_dict = all_config_dict["model_config_dict"]

# ======================================
# Megatron LM Designated CLI Arguments
# ======================================
designated_args = [
    "--seed", "42",
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
    
    "--micro-batch-size", "4",
    "--global-batch-size", "16",

    "--lr", "1.0e-5",
    "--train-iters", "2",
    "--lr-warmup-init", "1e-5",
    "--lr-decay-iters", "1000000",
    "--lr-decay-style", "constant",
    "--min-lr", "1e-6",
    "--weight-decay", "0.01",
    "--weight-decay-incr-style", "constant",
    "--lr-wsd-decay-style", "linear",
    "--fp16",
    "--transformer-impl", "transformer_engine",
    "--tensor-model-parallel-size", str(tp),
    "--pipeline-model-parallel-size", str(pp),
    "--context-parallel-size", str(cp),
    "--cp-comm-type", "p2p",
    "--distributed-timeout-minutes", "1",
    "--local-rank", str(local_rank),
    "--data-path", str(data_path),
    "--tokenizer-type", "HuggingFaceTokenizer",
    "--tokenizer-model", "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
    "--data-cache-path", str(data_cache_path),
    "--tiktoken-pattern", "v2",
    "--num-workers", "1",
    "--split", "90,5,5",
    "--log-interval", "1",
    "--logging-level", "20",
    "--no-create-attention-mask-in-dataloader",
]

designated_args = [arg for arg in designated_args if arg is not None]

# ================================
# Initialize Megatron
# ================================

with time_it("initialize megatron"):
    def replace_parser_and_parse_args(parser: argparse.ArgumentParser):
        old_parser_parse_args = parser.parse_args
        old_parser_parse_known_args = parser.parse_known_args
        parser.parse_args = lambda *args, **kwargs: old_parser_parse_args(designated_args)
        parser.parse_known_args = lambda *args, **kwargs: old_parser_parse_known_args(designated_args)
        return parser

    from megatron.training.initialize import initialize_megatron
    initialize_megatron(
        extra_args_provider=replace_parser_and_parse_args,
        args_defaults={},
        get_embedding_ranks=None,
        get_position_embedding_ranks=None,
    )
    logger.info(f"Successfully initialized megatron")

args = get_args()
# logger.info(f"Args: {args}")
# Set default iteration if not present (needed for build_train_valid_test_data_loaders)
if not hasattr(args, 'iteration'):
    args.iteration = 0
config = core_transformer_config_from_args(args, config_class=TransformerConfig)

# ================================
# DataLoader Setup
# ================================
def is_dataset_built_on_rank() -> bool:
    """DistCA: Only build the dataset on Rank 0."""
    return mpu.get_data_parallel_rank() == 0

def core_gpt_dataset_config_from_args(args):
    """Adapt from pretrain_gpt.py"""
    tokenizer = get_tokenizer()

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
    """Build the train test and validation datasets."""
    args = get_args()

    config = core_gpt_dataset_config_from_args(args)
    logger.info(f"> GPTDatasetConfig: {config}")

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

def cyclic_iter(iter):
    while True:
        for x in iter:
            yield x

old_args_dataloader_type = args.dataloader_type
args.dataloader_type = "external"

with time_it("setup data iterators"):
    from megatron.legacy.data.data_samplers import MegatronPretrainingSampler

    train_valid_test_datasets_provider.is_distributed = True

    train_dataloader, valid_dataloader, test_dataloader = \
        build_train_valid_test_data_loaders(
            train_valid_test_datasets_provider)

    dl_type = args.dataloader_type
    assert dl_type in ['single', 'cyclic', 'external'], f"Expected dl_type is one of 'single', 'cyclic', or 'external', but got {dl_type}"

    def _get_iterator(dataloader_type, dataloader, is_train=False):
        """Return dataset iterator."""
        if dl_type == 'external':
            dataset = dataloader
            assert isinstance(dataset, GPTDataset), f"Expected GPTDataset, but got {type(dataset)}"
            consumed_samples = 0
            if is_train:
                consumed_samples = args.consumed_train_samples
            
            micro_batch_size = args.micro_batch_size
            batch_sampler = MegatronPretrainingSampler(
                total_samples=len(dataset),
                consumed_samples=consumed_samples,
                micro_batch_size=micro_batch_size,
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
    
        if dataloader_type == "single":
            return RerunDataIterator(iter(dataloader))
        elif dataloader_type == "cyclic":
            return RerunDataIterator(iter(cyclic_iter(dataloader)))
        elif dataloader_type == "external":
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

    logger.info(f"train_data_iterator: {train_data_iterator}")
    logger.info(f"valid_data_iterator: {valid_data_iterator}")
    logger.info(f"test_data_iterator: {test_data_iterator}")

    args.dataloader_type = old_args_dataloader_type

logger.info('done with setup ...')

# ================================
# Test the dataloader
# ================================

logger.info("Testing dataloader iteration...")
logger.info(f"micro_batch_size: {args.micro_batch_size}")
logger.info(f"global_batch_size: {args.global_batch_size}")


pp_batch_size: int = args.micro_batch_size # PP mb
dp_batch_grp_size: int = args.global_batch_size // pp_batch_size # DP mb



# Get tokenizer for token monitoring
tokenizer = get_tokenizer()

def simple_token_monitor(tokens: torch.Tensor, tokenizer, logger: logging.Logger):
    """Simple token monitor: shows text and tokens per document."""
    batch_size = tokens.shape[0] if len(tokens.shape) > 1 else 1
    eod_token_id = tokenizer.eod
    
    for sample_idx in range(batch_size):
        sample_tokens = tokens[sample_idx] if len(tokens.shape) > 1 else tokens
        token_list = sample_tokens.tolist()
        
        # Find document boundaries (EOD positions)
        eod_positions = [i for i, tok in enumerate(token_list) if tok == eod_token_id]
        
        # Calculate tokens per document
        tokens_per_doc = []
        prev_end = 0
        for eod_pos in eod_positions:
            doc_length = eod_pos - prev_end + 1  # Include EOD token
            tokens_per_doc.append(doc_length)
            prev_end = eod_pos + 1
        
        # If there are tokens after the last EOD, count them as a partial document
        if prev_end < len(token_list):
            tokens_per_doc.append(len(token_list) - prev_end)
        
        # Calculate document start/end positions
        doc_starts = [0] + [pos + 1 for pos in eod_positions if pos + 1 < len(token_list)]
        doc_ends = [pos + 1 for pos in eod_positions]
        if prev_end < len(token_list):
            doc_ends.append(len(token_list))
        if not eod_positions:
            doc_ends = [len(token_list)]
        
        logger.info(f"  Sample {sample_idx}: {len(tokens_per_doc)} documents; Token count sum: {sum(tokens_per_doc)}")
        logger.info(f"    Tokens per doc: {tokens_per_doc}; ")
        
        # # Show text and tokens for each document
        # for doc_idx, start_pos in enumerate(doc_starts):
        #     end_pos = doc_ends[doc_idx] if doc_idx < len(doc_ends) else len(token_list)
        #     doc_tokens = token_list[start_pos:end_pos]
        #     doc_token_count = len(doc_tokens)
            
        #     try:
        #         doc_text = tokenizer.detokenize(doc_tokens)
        #         # Truncate long text for display
        #         # if len(doc_text) > 200:
        #             # doc_text = doc_text[:200] + "..."
        #         logger.info(f"    Doc {doc_idx}: {doc_token_count} tokens")
        #         # logger.info(f"      Text: {repr(doc_text)}")
        #     except Exception as e:
        #         logger.info(f"    Doc {doc_idx}: {doc_token_count} tokens (decode failed: {e})")

for grp_idx in range(dp_batch_grp_size):
    logger.info("-" * 60)
    logger.info(f"Group {grp_idx} of {dp_batch_grp_size}")
    

    # dict_keys(['tokens', 'labels', 'loss_mask', 'position_ids'])
    if rank == 0:
        assert train_data_iterator is not None
        batch = next(train_data_iterator)
        assert isinstance(batch, dict)

        tokens = batch.get('tokens') 
        labels = batch.get('labels')
        loss_mask = batch.get('loss_mask')

        logger.info(f"tokens: {tokens}")
        logger.info(f"labels: {labels}")
        logger.info(f"loss_mask: {batch.get('loss_mask')}")

        # Extract tensors for simple token monitoring
        assert tokens is not None
        
        # Use simple token monitor
        simple_token_monitor(tokens, tokenizer, logger)
        pass
    else:
        pass

logger.info(f"Rank {rank} is exiting")
torch.distributed.barrier()
torch.distributed.destroy_process_group()

