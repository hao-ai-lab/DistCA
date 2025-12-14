
"""
Combined Megatron E2E Test (D2 + Baseline)

This script combines both D2 and baseline approaches for testing:
- Baseline mode: Uses simple batch generation and normal forward function
- D2 mode: Uses balanced flops planning and ping-pang parameters

Usage:
```bash
bash test_e2e_combined.multi.sh <rzv_endpoint> <n_nodes>
```
"""
import torch.distributed

import time
start_time__ = time.time()

import psutil, os
rank = int(os.environ.get("RANK", os.environ.get("SLURM_PROCID","0")))
local = int(os.environ.get("LOCAL_RANK", os.environ.get("SLURM_LOCALID","0")))
p = psutil.Process(os.getpid())
p.cpu_affinity([local * 16, local * 16 + 1])  # pin to core based on local rank
print(f"[{rank}] allowed CPUs:", p.cpu_affinity())

# ----------------
# Taskset confirm
# ----------------
import check_cpu_binding
aff, mems = check_cpu_binding.check_cpu_binding()
print(f"CPUS={aff} MEMS={mems}")


# ----------------
# Main Imports
# ----------------
from torch.profiler import profile, record_function, ProfilerActivity
import d2.planner.wlb_planner
import d2.mem
import math
import argparse
import os
import gc
import pytz
import json
import time
import rich
import signal
import traceback
import sys
from contextlib import contextmanager
import numpy as np

from megatron.core import mpu
from megatron.core.optimizer import get_megatron_optimizer
from megatron.core.pipeline_parallel.schedules import get_forward_backward_func
from megatron.core.packed_seq_params import PackedSeqParams
from omegaconf import OmegaConf
import torch
from transformers import AutoConfig, AutoTokenizer, AutoProcessor

from d2.runtime.attn_kernels.ops import DispatcherWrapper
from d2.runtime.megatron.packed_seq_params import arg_to_cuda, PingPangPackedSeqParams
from d2.runtime.compute_metadata import get_attn_metadata
from d2.runtime.megatron.ops.stream_sync_fn import TickSync

from test_util import MegatronBaseWorker, ParallelConfig, init_worker_torch_distributed, set_random_seed
from test_pingpong_layer import get_single_step_packed_seq_params
from megatron_test_utils import (
    get_megatron_optimizer_param_scheduler, get_model, get_torch_device, gptmodel_forward,
    hf_to_mcore_config, init_mcore_model, init_megatron_optim_config,
    make_batch_generator, print_model_size, update_model_config, unwrap_model,
)

from d2.planner.planner import (
    batch_to_items_general,
    Planner,
    Item,
)


from d2.utils.traceback import enable_clickable_excepthook, enable_trace_calls
enable_clickable_excepthook()


def timeout_handler(signum, frame):
    raise TimeoutError("forward_backward_batch operation timed out after 5 minutes")

# from d2.utils.torch_profiler import ProfilerCtx


def debug_print(*args, **kwargs):
    if os.getenv("D2_DEBUG_PRINT", "0") == "1":
        if torch.distributed.is_initialized():
            rank = torch.distributed.get_rank()
            print(f"[Rank {rank}]", *args, **kwargs)
    return

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


def log_memory_usage(message: str, force:bool = False):
    d2.mem.log_memory_usage(message, force=force)

@contextmanager
def log_memory_usage_context():
    old_env_var = os.environ.get("EXPERIMENT_LOG_MEMORY_USAGE", "0")
    os.environ["EXPERIMENT_LOG_MEMORY_USAGE"] = "1"
    yield
    os.environ["EXPERIMENT_LOG_MEMORY_USAGE"] = old_env_var


def dump_tensor(tensor, name: str, msg:str=None):
    tensor_dump_dir = os.environ.get("TENSOR_DUMP_DIR", None)
    tensor_dump_suffix = os.environ.get("TENSOR_DUMP_SUFFIX", None)
    if not torch.isfinite(tensor).all():
        print(f"🔴 {msg}: Non-finite values detected.")
    else:
        print(f"🟢 {msg}: No non-finite values detected.")
        pass
    # if tensor_dump_dir is not None and tensor_dump_suffix is not None:
    #     torch.save(tensor.cpu(), os.path.join(tensor_dump_dir, f"{name}.{tensor_dump_suffix}.pt"))
    #     print(f"🟡 Dumped tensor to {os.path.join(tensor_dump_dir, f"{name}.{tensor_dump_suffix}.pt")}")
    return

class MegatronE2eWorker(MegatronBaseWorker):
    def __init__(self, rank: int, world_size: int):
        super().__init__(rank, world_size)
        self.dtype = torch.bfloat16
        self.enable_gradient_checkpointing = False
        self.gradient_checkpointing_kwargs = {}

    def init_comm(self, *args, **kwargs):
        super().init_comm(*args, **kwargs)

    def set_config(self, dtype=torch.bfloat16, enable_gradient_checkpointing=False, gradient_checkpointing_kwargs={}):
        self.dtype = dtype
        self.enable_gradient_checkpointing = enable_gradient_checkpointing
        self.gradient_checkpointing_kwargs = gradient_checkpointing_kwargs

    def init(self, model_path, seed=42):
        """
        tp_comm_overlap_cfg:
        tp_comm_overlap_ag: true
        tp_comm_overlap_rs: true
        tp_comm_bulk_wgrad: true
        tp_comm_bulk_dgrad: true
        ub_tp_comm_overlap: true
        """
        set_random_seed(seed)
        self.model_path = model_path
        override_model_config = OmegaConf.create()
        override_transformer_config = OmegaConf.create({
            "apply_rope_fusion": True,
            # bias-act fusion
            "bias_activation_fusion": True,
            # no layer norm so no need for that fusion
            # attention is in FA so no masked_softmax fusion
            # bias-drop_out-add fusion
            "bias_dropout_fusion": True,
            "tp_comm_overlap": False,

        })
        # A default optim config
        optim_config = OmegaConf.create({
            "clip_grad": 1.0,
            "lr": 1e-5,
            "lr_warmup_init": 1e-5,
            "lr_decay_steps": 1000000,
            "lr_decay_style": 'constant',
            "lr_warmup_steps": 1000,
            "lr_warmup_steps_ratio": 0.0,
            "min_lr": 1e-6,
            "min_lr_ratio": None,
            "total_training_steps": -1,
            "warmup_style": "constant",
            "weight_decay": 0.01,
            "weight_decay_incr_style": "constant",
            "use_checkpoint_opt_param_scheduler": False,
            "lr_wsd_decay_style": "linear",

        })
        self._build_model_optimizer(model_path, optim_config, override_model_config, override_transformer_config)

        assert self.device is not None
        for module in self.train_module:
            unwrap_model(module).init_ping_pong_communication_ctx(self.device)

    def _init_hf_config_and_tf_config(
        self,
        model_path,
        dtype,
        override_model_config,
        override_transformer_config,
        trust_remote_code=True,
    ):

        # Step 1: initialize the tokenizer
        self.local_path = model_path
        self.tokenizer = AutoTokenizer.from_pretrained(self.local_path, trust_remote_code=trust_remote_code)
        self.processor = AutoProcessor.from_pretrained(self.local_path, trust_remote_code=trust_remote_code)

        # Step 2: get the hf
        hf_config = AutoConfig.from_pretrained(self.local_path, trust_remote_code=trust_remote_code)

        # Step 3: override the hf config
        override_config_kwargs = {
            "bos_token_id": self.tokenizer.bos_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
            "pad_token_id": self.tokenizer.pad_token_id,
        }
        override_config_kwargs.update(override_model_config.get("model_config", {}))
        self.share_embeddings_and_output_weights = getattr(hf_config, "tie_word_embeddings", False)
        update_model_config(hf_config, override_config_kwargs=override_config_kwargs)
        self.architectures = getattr(hf_config, "architectures", None)
        if self.rank == 0:
            print(f"Model config after override: {hf_config}")
        tf_config = hf_to_mcore_config(hf_config, dtype, **override_transformer_config)

        def add_optimization_config_to_tf_config(tf_config):
            # add optimization config to tf_config, e.g. checkpointing
            if self.enable_gradient_checkpointing:
                gradient_checkpointing_cfg = dict(self.gradient_checkpointing_kwargs)
                tf_config.recompute_method = gradient_checkpointing_cfg.get("activations_checkpoint_method", "full")
                tf_config.recompute_granularity = gradient_checkpointing_cfg.get(
                    "activations_checkpoint_granularity", "full"
                )
                tf_config.recompute_num_layers = gradient_checkpointing_cfg.get("activations_checkpoint_num_layers", -1)
                # tf_config.distribute_saved_activations = gradient_checkpointing_cfg.get("activations_checkpoint_distribute_saved_activations", None)
                tf_config.recompute_modules = gradient_checkpointing_cfg.get("activations_checkpoint_recompute_modules", None)

                print(f"🟡 [Rank {self.rank}] Adding selective checkpoint: {gradient_checkpointing_cfg}")

        add_optimization_config_to_tf_config(tf_config)

        if self.rank == 0:
            print(f"TF config: {tf_config}")
        self.hf_config = hf_config
        self.tf_config = tf_config

    def forward_backward_batch(self, microbatches: list[dict], forward_only: bool=False, normal_forward_fn: bool=False):
        # TODO: for PP, since backward has a different attention layout dispatching order,
        # we should modify the forward_backward_func here.

        microbatches = [{
            # TODO: If we have gradient accumulation, then need to take all microbatches
            k: arg_to_cuda(v) 
            for k, v in mb.items()
        }for mb in microbatches]

        for module in self.train_module:
            unwrap_model(module).set_debug(normal_forward_fn)
        assert len(self.train_module) == 1, "only support one module"

        forward_backward_func = get_forward_backward_func()
        n_micro_batch = len(microbatches)
        # thd layout
        total_seqlen = microbatches[0]['input_ids'].shape[0]

        from megatron.core.tensor_parallel.cross_entropy import vocab_parallel_cross_entropy
        def loss_func(logits):
            loss = logits.sum()  # no gradient, but can trigger backward
            # Print the memory usage here
            log_memory_usage("loss_func")
            return loss, {'loss': loss}

        def forward_step(batch_iter, model):
            batch = next(batch_iter)
            input_ids = batch['input_ids']
            position_ids = batch['position_ids']
            attention_mask = None
            packed_seq_params = batch['packed_seq_params']
            # returns "hidden_states" if not model.post_process (not the last layer)
            # returns "logits" when label is None.
            output = gptmodel_forward(
                model, input_ids, attention_mask, position_ids, self.tf_config.sequence_parallel, packed_seq_params
            )
            # Build next-token labels (shifted by 1, ignore the last token)
            labels = input_ids.clone()
            labels[:-1] = input_ids[1:]
            labels[-1] = -100

            def loss_func_ce(logits, _labels=labels):
                dump_tensor(logits, f"nonfinite_output.rank{self.rank}", msg="before loss_func_ce")
                ce = vocab_parallel_cross_entropy(logits.contiguous(), _labels)
                dump_tensor(ce, f"nonfinite_ce.rank{self.rank}", msg="after loss_func_ce")
                loss_mask = (_labels != -100).float()
                denom = loss_mask.sum().clamp(min=1.0)
                loss = (ce * loss_mask).sum() / denom
                log_memory_usage("loss_func")
                # Dump final scalar loss
                # dump_tensor(loss, f"loss.rank{self.rank}", msg="after normalized loss")
                print(f"🟡 [Rank {self.rank}] loss = {loss}")
                return loss, {'loss': loss}
            return output, loss_func_ce

        batch_generator = make_batch_generator(microbatches, vpp_size=len(self.train_module))
        if mpu.get_pipeline_model_parallel_world_size() > 1:
            losses_reduced = forward_backward_func(
                forward_step_func=forward_step,
                data_iterator=batch_generator,
                model=self.train_module,
                num_microbatches=n_micro_batch,
                seq_length=total_seqlen,  # no use, since variable_seq_lengths=True
                micro_batch_size=1,  # no use when input_shapes was set
                forward_only=forward_only,
            )
        else:
            losses_reduced = forward_backward_func(
                forward_step_func=forward_step,
                data_iterator=batch_generator,
                model=self.train_module,
                num_microbatches=n_micro_batch,
                seq_length=total_seqlen,  # in use for pp = 1
                micro_batch_size=1,  # in use for pp = 1
                forward_only=forward_only,
            )

        # Optional distributed barrier before optimizer step
        if os.getenv("EXPERIMENT_BARRIER_BEFORE_OPTIMIZER_STEP", "0") == "1":
            if torch.distributed.is_initialized():
                torch.cuda.synchronize()
                torch.distributed.barrier()
                log_memory_usage("barrier_before_optimizer_step")
        
        with torch.cuda.nvtx.range("optimizer_step"):
            # Enhanced optimizer step timing
            torch.cuda.synchronize()
            if torch.distributed.is_initialized():
                torch.distributed.barrier()
            opt_start_time = time.time()
            log_memory_usage("optimizer_step:(start)")
            
            if os.getenv("EXPERIMENT_SKIP_OPTIMIZER_STEP", "0") == "1":
                # when testing numerical correctness, instead of running optimizer step, reset grads.
                update_successful, grad_norm, num_zeros_in_grad = True, 0.0, 0
                for tm in self.train_module:
                    for param in unwrap_model(tm).parameters():
                        param.main_grad.zero_()
            else:
                update_successful, grad_norm, num_zeros_in_grad = self.optimizer.step()
                
            torch.cuda.synchronize()
            if torch.distributed.is_initialized():
                torch.distributed.barrier()
            opt_end_time = time.time()
            
            current_rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
            opt_duration_ms = (opt_end_time - opt_start_time) * 1000
            print(f"🚀 OptimizerStep [Rank {current_rank}]: duration={opt_duration_ms:.3f}ms")
            print(f"🚀 OptimizerStep [Rank {current_rank}]: grad_norm={grad_norm}")
            print(f"🚀 OptimizerStep [Rank {current_rank}]: update_successful={update_successful}")
            
            log_memory_usage("optimizer_step:(end)")
            if os.getenv("EXPERIMENT_BARRIER_BEFORE_OPTIMIZER_STEP", "0") == "1":
                if torch.distributed.is_initialized():
                    torch.cuda.synchronize()
                    torch.distributed.barrier()
                    log_memory_usage("barrier_after_optimizer_step")

        return losses_reduced, grad_norm

    def _build_model_optimizer(self,
        model_path, optim_config, override_model_config, override_transformer_config
    ):

        self._init_hf_config_and_tf_config(
            model_path,
            self.dtype,
            override_model_config,
            override_transformer_config,
            True, # trust_remote_code
        )

        def make_model(wrap_with_ddp=False):
            def megatron_actor_model_provider(pre_process, post_process):

                parallel_model = init_mcore_model(
                    self.tf_config,
                    self.hf_config,
                    pre_process,
                    post_process,
                    share_embeddings_and_output_weights=self.share_embeddings_and_output_weights,
                    value=False,
                    freeze_moe_router=override_model_config.get("moe_config", {}).get("freeze_moe_router", False),
                )
                parallel_model.to("cuda")
                return parallel_model

            override_ddp_config = OmegaConf.to_container(
                OmegaConf.create(), resolve=True
            )
            return get_model(
                megatron_actor_model_provider,
                wrap_with_ddp=wrap_with_ddp,
                use_distributed_optimizer=True,
                override_ddp_config=override_ddp_config,
            )

        train_module = make_model(wrap_with_ddp=True)
        # load_megatron_gptmodel_weights

        if self.rank == 0:
            print_model_size(train_module[0])

        optim_config_megatron = init_megatron_optim_config(optim_config)
        optimizer = get_megatron_optimizer(
            model_chunks=train_module, config=optim_config_megatron)
        optimizer_scheduler = get_megatron_optimizer_param_scheduler(
            optimizer=optimizer, config=optim_config
        )


        self.train_module = train_module
        self.optimizer = optimizer
        print(f"🟡 [Rank {self.rank}] optimizer = {optimizer}")
        self.optimizer_scheduler = optimizer_scheduler
        self.hf_config = self.hf_config
        self.optim_config = optim_config


def init_megatron_e2e_test(
    hidden_size_q: int, hidden_size_kv: int, num_tokens: int,
    world_size: int, max_cp_degree: int, tp_size: int,
    dtype, worker_cls=MegatronE2eWorker
):
    token_bytes_q = hidden_size_q * dtype.itemsize // tp_size
    token_bytes_kv = hidden_size_kv * dtype.itemsize // tp_size
    max_tokens_query = num_tokens * (world_size // tp_size)
    max_tokens_key_value = num_tokens * (world_size // tp_size)
    buffer_size = (
        token_bytes_q * max_tokens_query +
        token_bytes_kv * max_tokens_key_value * max_cp_degree * 2
    ) * 1.5
    buffer_size = int(buffer_size)

    EXPERIMENT_NVSHMEM_BUFFER_SIZE_GB = os.environ.get("EXPERIMENT_NVSHMEM_BUFFER_SIZE_GB", "-1")
    try:
        EXPERIMENT_NVSHMEM_BUFFER_SIZE_GB = float(EXPERIMENT_NVSHMEM_BUFFER_SIZE_GB)
        if EXPERIMENT_NVSHMEM_BUFFER_SIZE_GB > 0:
            EXPERIMENT_NVSHMEM_BUFFER_SIZE_GB *= (1024 ** 3)
            EXPERIMENT_NVSHMEM_BUFFER_SIZE_GB = int(EXPERIMENT_NVSHMEM_BUFFER_SIZE_GB)
            buffer_size = EXPERIMENT_NVSHMEM_BUFFER_SIZE_GB
            pass
    except:
        pass

    buffer_size_gb = buffer_size // 1024 / 1024 / 1024
    print(f"🟡 buffer_size = {buffer_size_gb} GB")
    parallel_config = ParallelConfig(
        tensor_model_parallel_size=tp_size
    )
    log_memory_usage("init_worker_torch_distributed", force=True)
    worker = init_worker_torch_distributed(
        world_size, buffer_size, worker_cls, parallel_config
    )
    print("Communication groups initialized")

    log_memory_usage("comm group initialized", force=True)

    
    # # FIXME: We don't have to do the hack like that...
    # # # If the buffer size is < 2GB, 
    # # update the real buffer size to 2GB,
    # # but keep the nominal buffer size to the original value.
    # if buffer_size < 2 * 1024 ** 3:
    #     FastDispatcherWrapper.update_buffer_size(2 * 1024 ** 3)
    #     # now the real buffer size is 2GB, but the nominal buffer size is the original value.
    #     FastDispatcherWrapper.update_buffer_size(buffer_size)
    #     print(f"🟡 [Rank {worker.rank}] Updated real buffer size to 2GB, but keep the nominal buffer size to {buffer_size / 1024**3} GB")
    #     # now the nominal buffer size remains the original value.
    #     # this will help us do replanning with the original buffer size.

    log_memory_usage("buffer initialized", force=True)
    # exit(0)
    return worker

def init_wlbllm_e2e_test(
    hidden_size_q: int, hidden_size_kv: int, num_tokens: int,
    world_size: int, max_cp_degree: int, tp_size: int,
    dtype, worker_cls=MegatronE2eWorker
):
    token_bytes_q = hidden_size_q * dtype.itemsize // tp_size
    token_bytes_kv = hidden_size_kv * dtype.itemsize // tp_size
    max_tokens_query = num_tokens * (world_size // tp_size)
    max_tokens_key_value = num_tokens * (world_size // tp_size)
    buffer_size = (
        token_bytes_q * max_tokens_query +
        token_bytes_kv * max_tokens_key_value * max_cp_degree * 2
    )
    parallel_config = ParallelConfig(
        tensor_model_parallel_size=tp_size,
        context_parallel_size=max_cp_degree,
    )
    # TODO(HACK): We bypass some logic in init_worker_torch_distributed.
    # worker = init_worker_torch_distributed(
    #     world_size, buffer_size, worker_cls, parallel_config, skip_nvshmem_init=True
    # )
    assert world_size == int(os.environ.get("WORLD_SIZE"))
    rank = int(os.environ.get("RANK"))
    local_rank = int(os.environ.get("LOCAL_RANK"))
    torch.cuda.set_device(local_rank)
    worker = worker_cls(
        rank, world_size
    )
    assert parallel_config is not None
    # worker.init_comm(buffer_size, parallel_config, local_rank)
    # -- if use the original one, it will hit the nvshmem init logic, 
    # which needs to assert that other group initialization is not called.
    log_memory_usage("init_worker_torch_distributed", force=True)
    worker.init_torch_distributed()
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
        # order="tp-cp-ep-dp-pp",
    )
    print("Communication groups initialized")

    log_memory_usage("comm_group_init finished", force=True)
    return worker


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
    
    # if should_add_debug_cases:
    #     GLOBAL_BATCH = list(GLOBAL_BATCH)
    #     manual_case = [
    #         [total_seq_len // 4 * 3 - 512, 512, total_seq_len // 4],
    #     ] * 16
    #     GLOBAL_BATCH = manual_case + GLOBAL_BATCH
    #     GLOBAL_BATCH = iter(GLOBAL_BATCH)

    if should_add_debug_cases:
        GLOBAL_BATCH = list(GLOBAL_BATCH)
        # manual_case = [
        #     [total_seq_len],
        #     # [total_seq_len // 32] * 32
        # ] * 128
        manual_case = [
            [total_seq_len // 2, ] + [total_seq_len // 32] * 16,
        ] + [
            # [total_seq_len // 2] * 2,
            # [total_seq_len // 4] * 4,
            [total_seq_len // 32] * 32,
        ] * 128
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


# ========== D2 Specific Functions ==========

# from transformer_engine.pytorch.attention.dot_product_attention.backends import get_attention_duration
import traceback
try:
    import wlbllm
    import wlbllm.utils
    import wlbllm.registry
    import wlbllm.megatron_patch.dot_product_attention
    import wlbllm.megatron_patch.backends
    import wlbllm.fastmemcpy.fast_memcpy
    from wlbllm.fastmemcpy.fast_memcpy import prepare_metadata as wlb_memcpy_prepare_metadata
except ImportError as e:
    traceback.print_exc()
    print(f"🟡 ImportError: {e}")
    print("""⚠️ WLBLLM is not installed. This only affects if you're testing WLBLLM tests. To install:

    cd d2/baseline/wlbllm_original
    pip install -e .
    """)
    exit(1)




def test(args):
    global start_time__
    num_nodes = args.num_nodes
    num_gpus_per_node = args.num_gpus_per_node
    seed = args.seed
    batch_size = args.batch_size
    num_tokens = args.num_tokens
    cp_degree = max_cp_degree = args.cp_degree
    tp_size = args.tp_size
    world_size = num_nodes * num_gpus_per_node
    total_seq_len = args.num_tokens
    num_layers = args.num_layers
    model_path = args.model_path
    max_sample_id = args.max_sample_id
    up_sample_factor = args.up_sample_factor
    replan_iter = args.replan_iter
    elongate_factor = args.elongate_factor
    filter_threshold = args.filter_threshold
    filter_ratio = args.filter_ratio
    should_add_debug_cases = args.should_add_debug_cases
    resend_qkv = args.should_resend_qkv
    sample_start_idx = args.sample_start_idx
    alpha_factor = args.alpha_factor
    if num_layers is not None:
        os.environ["NUM_LAYERS"] = str(num_layers)


    mode = args.mode
    output_dir = args.output_dir
    dtype = torch.bfloat16
    element_size = dtype.itemsize

    # Set forward function mode based on test mode
    normal_forward_fn = (mode in ["baseline", "wlbllm"])
    # TODO: (Refactor) If WLBLLM is set, we must inform the transformer_engine to use the WLBLLM function. 
    os.environ["WLBLLM_MODE"] = "1" if mode == "wlbllm" else "0"
    
    # Setup unified attention timing collection for both WLBLLM and D2 modes
    if os.getenv("UNIFIED_RECORD_ATTENTION_TIMES", "0") == "1":
        setup_unified_attention_timing_patch()
        print(f"🟡 Unified attention timing collection setup. This may impact the performance, but recording the attention timing.")
    
    # Setup unified all-to-all timing collection for both WLBLLM and D2 modes
    if os.getenv("UNIFIED_RECORD_A2A_TIMES", "0") == "1":
        setup_unified_a2a_timing_patch()
        print(f"🟡 Unified all-to-all timing collection setup. This may impact the performance, but recording the all-to-all timing.")
    
    # Setup tick operations timing collection
    if os.getenv("UNIFIED_RECORD_TICK_TIMES", "0") == "1":
        from d2.runtime.megatron.ping_pong.tick_ops import setup_tick_timing
        setup_tick_timing()
        print(f"🟡 Unified tick operations timing collection setup. This may impact the performance, but recording the tick operations timing.")
    
    # Setup TickSync blocking detection
    if os.getenv("D2_TICKSYNC_BLOCKING_DETECTION", "0") == "1":
        threshold_ms = float(os.getenv("D2_TICKSYNC_THRESHOLD_MS", "1.0"))
        TickSync.enable_blocking_detection(enabled=True, threshold_ms=threshold_ms)
        print(f"🟡 TickSync blocking detection enabled with threshold {threshold_ms}ms")
    
    memory_log_output_dir = os.path.join(output_dir, "mem-log")
    enable_memory_usage_logging(memory_log_output_dir)

    log_memory_usage("enter test", force=True)
    
    def write_status_log(message):
        # get the caller's file and line number
        import traceback
        stack = traceback.extract_stack()
        caller_file = stack[-2].filename
        caller_line = stack[-2].lineno

        status_log_file = os.path.join(output_dir, "status.log")
        elapsed_time = time.time() - start_time__
        message = f"🕛 [T{elapsed_time:.2f}] ({caller_file}:{caller_line}) {message}"
        with open(status_log_file, "a") as f:
            f.write(message + "\n")
        print(message)
        return

    def write_loss_log(loss_value, sample_id=None, repeat_idx=None):
        # get the caller's file and line number
        import traceback
        stack = traceback.extract_stack()
        caller_file = stack[-2].filename
        caller_line = stack[-2].lineno
        
        loss_log_file = os.path.join(output_dir, "loss.log")
        elapsed_time = time.time() - start_time__
        sid = "NA" if sample_id is None else sample_id
        rep = "NA" if repeat_idx is None else repeat_idx
        try:
            loss_float = float(loss_value)
        except Exception:
            # best effort conversion
            loss_float = loss_value.item() if torch.is_tensor(loss_value) else float("nan")
        message = f"📉 [T{elapsed_time:.2f}] ({caller_file}:{caller_line}) sample_id={sid} repeat={rep} loss={loss_float:.6f}"
        with open(loss_log_file, "a") as f:
            f.write(message + "\n")
        print(message)
        return
    


    if mode == "wlbllm":
        import wlbllm.megatron_patch.dot_product_attention
        wlbllm.megatron_patch.dot_product_attention.monkey_patch()
        import wlbllm.megatron_patch.backends
        wlbllm.megatron_patch.backends.monkey_patch()
        pass
    
    # Check world size
    if mode == "wlbllm":
        # assert cp_degree * tp_size == world_size, f"WLBLLM world size ({world_size}) = num_nodes ({args.num_nodes}) * num_gpus_per_node ({args.num_gpus_per_node}) must be divisible by cp_degree ({cp_degree}) * tp_size ({tp_size})"
        print(f"🟡 Running WLBLLM config: cp_degree={cp_degree}, tp_size={tp_size}, world_size={world_size}")
    elif mode == "d2":
        print(f"🟡 Running D2 config: tp_size={tp_size}, world_size={world_size}")
    else:
        pass
        
    write_status_log(f"Pass world size check")
    
    print(f"🟡 setup_global_batch (mode={mode}): ")
    print(f"  - total_seq_len = {total_seq_len}")

    hf_config = AutoConfig.from_pretrained(model_path)
    hidden_size_q = hf_config.hidden_size

    hidden_size_kv = hidden_size_q
    if hasattr(hf_config, "num_key_value_heads"):
        hidden_size_kv = (hidden_size_kv * hf_config.num_key_value_heads //
                          hf_config.num_attention_heads)

    # TODO(HACK): WLBLLM and Megatron have different comm group initialization process.
    # This is a code divergence. We need to consolidate the comm group.
    if mode == "wlbllm":
        worker: MegatronE2eWorker = init_wlbllm_e2e_test(
            hidden_size_q, hidden_size_kv, num_tokens,
            world_size, max_cp_degree * 1, tp_size,
            dtype, MegatronE2eWorker
        )
    else:
        worker: MegatronE2eWorker = init_megatron_e2e_test(
            hidden_size_q, hidden_size_kv, num_tokens,
            world_size, max_cp_degree * 1, tp_size,
            dtype, MegatronE2eWorker
        )

    if mode == "d2":
        print(f"🟡 [Rank {worker.rank}] {worker.as_rank = } {worker.as_world_size = }")


    write_status_log(f"Finish init worker")
    log_memory_usage("init worker object done", force=True)

    enable_gradient_checkpointing = False
    gradient_checkpointing_kwargs = {}
    if os.environ.get("EXPERIMENT_ADD_SELECTIVE_CKPT", "0") == "1":
        enable_gradient_checkpointing = True
        gradient_checkpointing_kwargs = dict(
                activations_checkpoint_method="mlp",
                activations_checkpoint_granularity="selective",
                activations_checkpoint_num_layers=None, # num-layers
                activations_checkpoint_recompute_modules = ["mlp"],
            )
        # print(f"🟡 [Rank {worker.rank}] Adding selective checkpoint: {gradient_checkpointing_kwargs}")
    worker.set_config(
        dtype=dtype,
        enable_gradient_checkpointing=enable_gradient_checkpointing,
        gradient_checkpointing_kwargs=gradient_checkpointing_kwargs
    )
    worker.init(model_path, seed=seed)
    print(f"🟡 [Rank {worker.rank}] init done")
    log_memory_usage("init done", force=True)
    write_status_log(f"Finish worker.init()")
    # set again to potentially adapt to the ray launch case.
    set_random_seed(seed, set_megatron=False)

    # parallel_config = worker.parallel_config

    # torch.cuda.memory.set_per_process_memory_fraction(0.85)
    # print("🟡 [Rank {worker.rank}] torch.cuda.memory.set_per_process_memory_fraction to 0.85")

    if mode == "wlbllm":
        rank = torch.distributed.get_rank()
        as_rank = mpu.get_context_parallel_rank()
        as_world_size = mpu.get_context_parallel_world_size() * mpu.get_data_parallel_world_size()
        pass
    else:
        rank = worker.rank
        as_rank = worker.as_rank
        as_world_size = worker.as_world_size

    hidden_size_q_tp = hidden_size_q // tp_size
    hidden_size_k_tp = hidden_size_kv // tp_size

    # TODO(Refactor): Properly refactor this into a function and we call it multiple times

    setup_global_batch(
        total_seq_len,
        up_sample_factor=up_sample_factor,
        elongate_factor=elongate_factor,
        filter_threshold=filter_threshold,
        filter_ratio=filter_ratio,
        should_add_debug_cases=should_add_debug_cases,
        change_long_doc_ratio=args.change_long_doc_ratio,
        sample_name=args.sample_name,
    )



    
    # Test out the distributed all gather latency as a simple test.
    # if False: 
    if True: 
        dist_all_gather_func = torch.distributed.all_gather_into_tensor
        
        print("🟡 Start testing all gather")
        # Create a group containing ranks 0, 2, 4, ..., 14
        odd_ranks = list(range(1, 16, 2))  # [1, 3, 5, 7, 9, 11, 13, 15]
        even_ranks = list(range(0, 16, 2))  # [0, 2, 4, 6, 8, 10, 12, 14]
        even_group = torch.distributed.new_group(even_ranks)
        odd_group = torch.distributed.new_group(odd_ranks)
        if rank % 2 == 0:
            group = even_group
        else:
            group = odd_group
        output_size = 125301120
        input_size = output_size // 8
        output_tensor = torch.randn(output_size, device="cuda")
        input_tensor = torch.randn(input_size, device="cuda")

        for i in range(10):
            torch.distributed.barrier()
            torch.cuda.synchronize()

            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            start_event.record()
            with torch.cuda.nvtx.range(f"all_gather.{mode}.{i}"):
                dist_all_gather_func(
                    output_tensor,
                    input_tensor,
                    group=group,
                    async_op=False,
                )
            end_event.record()
            torch.distributed.barrier()
            torch.cuda.synchronize()
            elapsed_time = start_event.elapsed_time(end_event)
            print(f"🟡 All gather latency: {elapsed_time}ms")
        
        # return

    sample_times = []
    for sample_id in range(sample_start_idx, max_sample_id):
        # Set current sample ID for unified timing collection
        if os.getenv("UNIFIED_RECORD_ATTENTION_TIMES", "0") == "1":
            set_unified_current_sample_id(sample_id)
        if os.getenv("UNIFIED_RECORD_A2A_TIMES", "0") == "1":
            set_unified_current_a2a_sample_id(sample_id)
        if os.getenv("UNIFIED_RECORD_TICK_TIMES", "0") == "1":
            from d2.runtime.megatron.ping_pong.tick_ops import set_current_tick_sample_id
            set_current_tick_sample_id(sample_id)
        if mode == "baseline":
            try:
                # TOOD: This should be batch_size * 2
                _seq_lens: list[list[int]] = get_next_batch(as_world_size * 2)
            except StopIteration:
                break
            print(f"🟡 sample_id={sample_id}: {_seq_lens}")
            # TODO: Adding proper support for context parallel in megatron.
            # Baseline mode: Use simple batch generation
            # seq_lens = _seq_lens[2 * as_rank] + _seq_lens[2 * as_rank + 1]
            seq_lens = _seq_lens[as_rank] + _seq_lens[as_rank + as_world_size]

            total_seq_len_x2 = total_seq_len * 2
            input_ids = torch.randint(100, 10000, (as_world_size, total_seq_len_x2))
            input_ids_local = input_ids[as_rank]
            
            # Use normal packed seq params for baseline
            seq_lens_local = torch.tensor(seq_lens, dtype=torch.int32)
            packed_seq_params = get_attn_metadata(seq_lens_local, get_packed_seq_params=True)
            
            position_ids = torch.arange(total_seq_len, dtype=torch.int64).repeat(as_world_size, 2)
            position_ids_local = position_ids[as_rank]

            microbatch = {
                "input_ids": input_ids_local,
                "position_ids": position_ids_local,
                "packed_seq_params": packed_seq_params,
            }
            assert isinstance(microbatch["packed_seq_params"], PackedSeqParams)

        elif mode == "wlbllm":
            cp_rank = mpu.get_context_parallel_rank()
            cp_size = mpu.get_context_parallel_world_size()
            dp_rank = mpu.get_data_parallel_rank()
            dp_size = mpu.get_data_parallel_world_size()
            cp_group = mpu.get_context_parallel_group()

            rank = torch.distributed.get_rank()
            device = torch.cuda.current_device()

            try:
                _seq_lens: list[list[int]] = get_next_batch(batch_size * 2)
            except StopIteration:
                break
            print(f"🟡 sample_id={sample_id}: {_seq_lens}")
            # TODO: Adding proper support for context parallel in megatron.
            # Baseline mode: Use simple batch generation
            # seq_lens = _seq_lens[2 * as_rank] + _seq_lens[2 * as_rank + 1]
            # seq_lens = _seq_lens[as_rank] + _seq_lens[as_rank + as_world_size]

            print(f"🟡 [Rank {rank}] cp_rank={cp_rank}, cp_size={cp_size}, dp_rank={dp_rank}, dp_size={dp_size}, as_rank={as_rank}, as_world_size={as_world_size}, device={device}")
            # test an all reduce to ensure things are doing good
            # exit(0)

            print(f"🟡 _seq_lens={_seq_lens}")

            def flatten(a):
                return [y for x in a for y in x]

            
            alpha_factor = args.alpha_factor # at max tolerate 2x memory imbalance. This number can go infinite...
            seq_lens, new_batch = d2.planner.wlb_planner.balance_data_for_wlbllm(
                dp_size, dp_rank, total_seq_len, batch_size, _seq_lens, 
                ENABLE_BALANCED_FLOS_NO_DEFER=True,
                model_config=hf_config, # TODO: (Refactor) This is a hack to pass the model config to the WLBLLM planner.
                Lmax=int(total_seq_len * 2 * batch_size // dp_size * alpha_factor),
            )
            
            # TODO: Estimate and calculate flops imbalance and print it here...
            print(f"🟡 [Rank {rank}] WLBLLM Reordered Batch: new_batch={new_batch}")
            # also calculate the flops 
            def calc_relative_flops(batches: list[list[int]]):
                flops_per_batch = [0] * len(batches)
                for batch_idx, batch in enumerate(batches):
                    for seq_len in batch:
                        flops_per_batch[batch_idx] += seq_len ** 2
                # Now find the min flops per gpu, and everyone divide the min flops per gpu
                min_flops_per_gpu = min(flops_per_batch)
                flops_per_batch = [flops / min_flops_per_gpu for flops in flops_per_batch]
                return flops_per_batch
            
            relative_flops_per_batch = calc_relative_flops(new_batch)
            print(f"🟡 [Rank {rank}] Relative Flops per batch: {relative_flops_per_batch}")
            print(f"🟡 [Rank {rank}] Taking seq_lens={seq_lens}")


            doc_lens = flatten(seq_lens)
            if len(doc_lens) < dp_size:
                # Pad the doc_lens to dp_size
                doc_lens += [512] * (dp_size - len(doc_lens))
                pass
            if sum(doc_lens) % (cp_size * 2 * 8) != 0:
                # TODO(HACK): This is a hack to ensure the doc_lens is divisible by cp_size*2.
                sum_of_doc_lens = sum(doc_lens)
                doc_lens[-1] += (cp_size * 2 * 8) - sum_of_doc_lens % (cp_size * 2 * 8)
                # assert doc_lens[-1] > 0
                pass
            assert sum(doc_lens) % (cp_size * 2 * 8) == 0, f"sum(doc_lens)={sum(doc_lens)} must be divisible by {cp_size * 2 * 8}"
            assert sum(doc_lens) % (cp_size * 2) == 0, f"sum(doc_lens)={sum(doc_lens)} must be divisible by {cp_size * 2}"
            
            rank = torch.distributed.get_rank()
            
            print(f"🟡 [Rank {rank}] doc_lens={doc_lens}")
            assert cp_size == cp_degree

            # local_context_length = total_seq_len * 2
            local_context_length = sum(doc_lens) // cp_size
            context_length = local_context_length * cp_size

            # cp_group = d2.runtime.megatron_patch.create_group.get_attn_server_group()
            # debug_print(f"cp_size", cp_size)
            debug_print(f"local_context_length", local_context_length)
            debug_print(f"context_length", context_length)
            doc_shards = wlbllm.utils.compute_per_doc_cp_shard_doc_len(
                doc_lens, context_length, cp_size
            )
            chunk_size = context_length // (2 * cp_size)
            with torch.no_grad():
                wlb_memcpy_args = wlb_memcpy_prepare_metadata(doc_shards, device, cp_size, chunk_size)
            # debug_print(f"doc_shards", doc_shards)
            (
                cu_seqlens_q_list, cu_seqlens_k_list, 
                max_seqlen_q_list, max_seqlen_k_list, 
                kv_idx_list,
            ) = wlbllm.utils.compute_per_doc_metadate_combined__metadata_only(    
                context_length, 
                doc_lens, 
                doc_shards,
                cp_size, 
                cp_rank, 
                device=torch.cuda.current_device()
            )
            

            input_ids = torch.randint(100, 10000, (as_world_size, local_context_length))
            input_ids_local = input_ids[cp_rank]
            position_ids = torch.arange(total_seq_len, dtype=torch.int64).repeat(as_world_size, 2)
            position_ids_local = position_ids[cp_rank]
            # debug_print(f"input_ids_local", input_ids_local.shape)
            # debug_print(f"position_ids_local", position_ids_local.shape)

            packed_seq_params = PackedSeqParams(
                qkv_format="thd",
                # TODO(HACK): These variables are not used in the WLBLLM functions.
                # If anywhere we used them, we will fail.
                # See PerDocCPAttention in the wlbllm/per_doc_cp_attn.py
                cu_seqlens_q=cu_seqlens_q_list[-1],
                cu_seqlens_kv=cu_seqlens_k_list[-1],
                max_seqlen_q=max_seqlen_q_list[-1],
                max_seqlen_kv=max_seqlen_k_list[-1],
            )

            microbatch = {
                "input_ids": input_ids_local,
                "position_ids": position_ids_local,
                "packed_seq_params": packed_seq_params,
            }
            assert isinstance(microbatch["packed_seq_params"], PackedSeqParams)

            
            # Now save some context for the use of WLBLLM function
            if rank % 8 == 0:
                rich.print(f"[Rank {rank}] [sample_id = {sample_id}] doc_lens", doc_lens)
                rich.print(f"[Rank {rank}] [sample_id = {sample_id}] doc_shards", doc_shards)
                rich.print(f"[Rank {rank}] [sample_id = {sample_id}] cu_seqlens_q_list", cu_seqlens_q_list)
                rich.print(f"[Rank {rank}] [sample_id = {sample_id}] cu_seqlens_k_list", cu_seqlens_k_list)
                rich.print(f"[Rank {rank}] [sample_id = {sample_id}] max_seqlen_q_list", max_seqlen_q_list)
                rich.print(f"[Rank {rank}] [sample_id = {sample_id}] max_seqlen_k_list", max_seqlen_k_list)
                rich.print(f"[Rank {rank}] [sample_id = {sample_id}] kv_idx_list", kv_idx_list)
                rich.print(f"[Rank {rank}] [sample_id = {sample_id}] input_ids_local", input_ids_local.shape)
                rich.print(f"[Rank {rank}] [sample_id = {sample_id}] position_ids_local", position_ids_local.shape)


            # Create for wlbllm
            # cp_stream = torch.cuda.Stream()
            cp_stream = torch.cuda.current_stream()

            wlbllm.registry.clear()
            wlbllm.registry.set("doc_lens", doc_lens)
            wlbllm.registry.set("doc_shards", doc_shards)
            wlbllm.registry.set("kv_idx_list", kv_idx_list)
            wlbllm.registry.set("cp_group", cp_group)
            wlbllm.registry.set("cp_stream", cp_stream)
            wlbllm.registry.set("cu_seqlens_q_list", cu_seqlens_q_list)
            wlbllm.registry.set("cu_seqlens_kv_list", cu_seqlens_k_list)
            wlbllm.registry.set("max_seqlen_q_list", max_seqlen_q_list)
            wlbllm.registry.set("max_seqlen_kv_list", max_seqlen_k_list)
            wlbllm.registry.set("global_tensor_length", context_length)
            # wlbllm.registry.set("global_tensor_length", (total_seq_len * cp_size * 2))
            wlbllm.registry.set("memcpy_args", wlb_memcpy_args)
            # Set current sample ID for attention timing
            wlbllm.registry.set("current_sample_id", sample_id)
            # Initialize timing for this sample if enabled
            if os.getenv("WLBLLM_RECORD_ATTENTION_TIMES", "0") == "1":
                wlbllm.registry.init_attention_timing_for_sample(sample_id)

        
        elif mode == "d2":
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
            def balance_ping_pong(seq_lens: list[list[int]]) -> tuple[list[list[int]], list[list[int]]]:
                def batch_flops(batch):
                    return sum(y ** 2 // 2 for y in batch)

                assert len(seq_lens) % 2 == 0, f"ping pong should have even number of batches, but got {len(seq_lens)} batches, seq_lens={seq_lens}"
                sorted_batches = sorted(seq_lens, key=batch_flops, reverse=True)
                ping, pong = [], []
                ping_flops, pong_flops = 0, 0
                avg_num_batches = len(seq_lens) // 2

                for batch in sorted_batches:
                    if (ping_flops <= pong_flops and len(ping) < avg_num_batches) or len(pong) >= avg_num_batches:
                        ping.append(batch)
                        ping_flops += batch_flops(batch)
                    else:
                        pong.append(batch)
                        pong_flops += batch_flops(batch)

                assert len(ping) == len(pong) == avg_num_batches, f"ping batches={ping}, pong batches={pong}"
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

            if rank % 8 == 0:
                rich.print(f"🟡 [Rank {rank}] _items_0 = {_items_0}")
                rich.print(f"🟡 [Rank {rank}] _items_1 = {_items_1}")

            
            # Try different tolerance factors and see which one fits the buffer size.
            # This will sacrifice performance for safety.
            
            # TODO: Pass a knob as a tradeoff of network and latency balance.
            verbose = (rank % 8 == 0)
            did_pass_overflow_check = False
            required_buffer_size: list[float] = []
            can_pass_tolerance_factor: list[bool] = []

            MIN_TOLERANCE_FACTOR = 0.05
            try:
                MIN_TOLERANCE_FACTOR = os.environ.get("MIN_TOLERANCE_FACTOR", "0.05")
                MIN_TOLERANCE_FACTOR = float(MIN_TOLERANCE_FACTOR)
            except ValueError:
                pass
            print(f"🟡 [Rank {rank}] MIN_TOLERANCE_FACTOR = {MIN_TOLERANCE_FACTOR}")

            candidate_tolerance_factors = [0.0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

            # FIXME: (Only for ablation) Tune the candidate tolerance factor.

            for tolerance_factor in candidate_tolerance_factors:
                if tolerance_factor < MIN_TOLERANCE_FACTOR:
                    continue

                print(f"[Rank {rank}] =========== Tolerance factor = {tolerance_factor} ============ ")
                
                planner = Planner(world_size, parallel_config, model_config=model_config, tolerance_factor=tolerance_factor)
                
                fa2a_metadata_0, as_attn_metadata_0, mlp_shard_len_0 = planner.plan(_items_0, is_resend_qkv=resend_qkv, verbose=verbose)
                fa2a_metadata_1, as_attn_metadata_1, mlp_shard_len_1 = planner.plan(_items_1, is_resend_qkv=resend_qkv, verbose=verbose)


                if verbose:
                    def print_2d_tensor(name: str, tensor):
                        print(f"🟡 [Rank {rank}] {name} = ")
                        for row in tensor.tolist():
                            print(f"    {row}")
                    
                    def exclude_self_and_sum(t):
                        for i in range(len(t)):
                            t[i][i] = 0
                        return t.sum(dim=1)
                        
                    def inspect_network_metadata(metadata, is_ping, sample_id, tolerance_factor, output_dir, rank):
                        qkv_fwd_metadata__send_transfer_sz_mb = metadata[0].fa2a_metadata[1] // 1024 // 1024
                        qkv_fwd_metadata__recv_transfer_sz_mb = metadata[0].fa2a_metadata[3] // 1024 // 1024
                        attn_out_fwd_metadata__send_transfer_sz_mb = metadata[1].fa2a_metadata[1] // 1024 // 1024
                        attn_out_fwd_metadata__recv_transfer_sz_mb = metadata[1].fa2a_metadata[3] // 1024 // 1024
                                
                        # Print qkv_fwd_metadata
                        print_2d_tensor("qkv_fwd_metadata.send_transfer_sz_mb", qkv_fwd_metadata__send_transfer_sz_mb)
                        print_2d_tensor("qkv_fwd_metadata.recv_transfer_sz_mb", qkv_fwd_metadata__recv_transfer_sz_mb)
                        
                        # Print attn_out_fwd_metadata  
                        print_2d_tensor("attn_out_fwd_metadata.send_transfer_sz_mb", attn_out_fwd_metadata__send_transfer_sz_mb)
                        print_2d_tensor("attn_out_fwd_metadata.recv_transfer_sz_mb", attn_out_fwd_metadata__recv_transfer_sz_mb)


                        # Calculate send size from me to others by subtracting diagonal (self-send) from total send
                        qkv_fwd_metadata__send_transfer_sz_mb_to_others = exclude_self_and_sum(qkv_fwd_metadata__send_transfer_sz_mb)
                        qkv_fwd_metadata__recv_transfer_sz_mb_to_others = exclude_self_and_sum(qkv_fwd_metadata__recv_transfer_sz_mb)
                        
                        print_2d_tensor("qkv_fwd_metadata.send_transfer_sz_mb_to_others", qkv_fwd_metadata__send_transfer_sz_mb_to_others)
                        print_2d_tensor("qkv_fwd_metadata.recv_transfer_sz_mb_to_others", qkv_fwd_metadata__recv_transfer_sz_mb_to_others)

                        attn_out_fwd_metadata__send_transfer_sz_mb_to_others = exclude_self_and_sum(attn_out_fwd_metadata__send_transfer_sz_mb)
                        attn_out_fwd_metadata__recv_transfer_sz_mb_to_others = exclude_self_and_sum(attn_out_fwd_metadata__recv_transfer_sz_mb)

                        print_2d_tensor("attn_out_fwd_metadata.send_transfer_sz_mb_to_others", attn_out_fwd_metadata__send_transfer_sz_mb_to_others)
                        print_2d_tensor("attn_out_fwd_metadata.recv_transfer_sz_mb_to_others", attn_out_fwd_metadata__recv_transfer_sz_mb_to_others)
                        
                        # Expected send-recv time
                        bandwidth_mb = 40 # MB/ms
                        send_time_ms = qkv_fwd_metadata__send_transfer_sz_mb_to_others / bandwidth_mb
                        recv_time_ms = qkv_fwd_metadata__recv_transfer_sz_mb_to_others / bandwidth_mb
                        print_2d_tensor("send_time_ms", send_time_ms)
                        print_2d_tensor("recv_time_ms", recv_time_ms)

                        max_comm_budget_all_rank = (
                              qkv_fwd_metadata__send_transfer_sz_mb_to_others 
                            + qkv_fwd_metadata__recv_transfer_sz_mb_to_others 
                            + attn_out_fwd_metadata__send_transfer_sz_mb_to_others 
                            + attn_out_fwd_metadata__recv_transfer_sz_mb_to_others
                        ).max().item()

                        if rank == 0:
                            network_inspect_file = os.path.join(output_dir, "network_inspect.jsonl")
                            with open(network_inspect_file, "a") as f:
                                f.write(json.dumps({
                                    "sample_id": sample_id,
                                    "is_ping": is_ping,
                                    "tolerance_factor": tolerance_factor,
                                    "qkv_fwd_metadata__send_transfer_sz_mb": qkv_fwd_metadata__send_transfer_sz_mb.tolist(),
                                    "qkv_fwd_metadata__recv_transfer_sz_mb": qkv_fwd_metadata__recv_transfer_sz_mb.tolist(),
                                    "attn_out_fwd_metadata__send_transfer_sz_mb": attn_out_fwd_metadata__send_transfer_sz_mb.tolist(),
                                    "attn_out_fwd_metadata__recv_transfer_sz_mb": attn_out_fwd_metadata__recv_transfer_sz_mb.tolist(),

                                    "qkv_fwd_metadata__send_transfer_sz_mb_to_others": qkv_fwd_metadata__send_transfer_sz_mb_to_others.tolist(),
                                    "qkv_fwd_metadata__recv_transfer_sz_mb_from_others": qkv_fwd_metadata__recv_transfer_sz_mb_to_others.tolist(),

                                    "max_comm_budget_all_rank": max_comm_budget_all_rank,
                                    "bandwidth_mb": bandwidth_mb,
                                    "send_time_ms": send_time_ms.tolist(),
                                    "recv_time_ms": recv_time_ms.tolist(),
                                }) + "\n")

                            network_inspect_summary_file = os.path.join(output_dir, "network_inspect.summary.jsonl")
                            with open(network_inspect_summary_file, "a") as f:
                                f.write(json.dumps({
                                    "sample_id": sample_id,
                                    "is_ping": is_ping,
                                    "tolerance_factor": tolerance_factor,
                                    "qkv_fwd_send_mb": qkv_fwd_metadata__send_transfer_sz_mb_to_others.tolist(),
                                    "qkv_fwd_recv_mb": qkv_fwd_metadata__recv_transfer_sz_mb_to_others.tolist(),

                                    "max_comm_budget_all_rank_mb": max_comm_budget_all_rank,
                                    "send_time_ms": send_time_ms.tolist(),
                                    "recv_time_ms": recv_time_ms.tolist(),
                                }) + "\n")

                    # Inspect both metadata sets
                    inspect_network_metadata(fa2a_metadata_0, True, sample_id, tolerance_factor, output_dir, rank)
                    inspect_network_metadata(fa2a_metadata_1, False, sample_id, tolerance_factor, output_dir, rank)
                    
                    

                # Check size:
                buffer_size = DispatcherWrapper.instance[0].buffer_size
                
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
                            f"{max_src_last_offset / 1024**3:.2f} GB src last offset, "
                            f"{buffer_size / 1024**3:.2f} GB buffer size"
                        )

                    max_size_provisioned = max(
                        max_send_sz, max_recv_sz, 
                        max_dst_last_offset, max_src_last_offset,
                    )
                    if not (buffer_size >= max_size_provisioned):
                        return False, max_size_provisioned
                    return True, max_size_provisioned

                def _check_all_overflow(fa2a_metadata, as_world_size_):
                    all_max_size_provisioned = 0
                    states = []
                    for as_rank_ in range(as_world_size_):
                        state, max_size_provisioned = _check_self_overflow(fa2a_metadata, as_rank_)
                        all_max_size_provisioned = max(all_max_size_provisioned, max_size_provisioned)
                        states.append(state)
                    all_state = all(states)
                    return all_state, all_max_size_provisioned
                    
                check_0, max_size_provisioned_0 = _check_all_overflow(fa2a_metadata_0, as_world_size)
                check_1, max_size_provisioned_1 = _check_all_overflow(fa2a_metadata_1, as_world_size)
                max_size_provisioned = max(max_size_provisioned_0, max_size_provisioned_1) / 1024**3
                required_buffer_size.append(max_size_provisioned)
                
                
                can_pass_tolerance_factor.append(check_0 and check_1)
                if not (check_0 and check_1):
                    print(f"⚠️ [Rank {rank}] Tolerance factor = {tolerance_factor}: Overflow check failed for fa2a_metadata_0 or fa2a_metadata_1 with tolerance_factor {tolerance_factor} and buffer_size {buffer_size / 1024**3} GB. Retry...")
                else:
                    did_pass_overflow_check = True
                    break

                    
            
            if not did_pass_overflow_check:
                print(f"🔴 [Rank {rank}] Inspected required_buffer_size = {required_buffer_size}")
                print(f"🔴 [Rank {rank}] Specified buffer_size = {buffer_size / 1024**3} GB")
                recommended_buffer_size = math.ceil(max_size_provisioned) + 0.5
                print(f"🔴 [Rank {rank}] Force update buffer_size to = {recommended_buffer_size} GB")
                buffer_size = int(recommended_buffer_size * 1024**3) # bytes


                DispatcherWrapper.update_buffer_size(buffer_size)


                rich.print(f"🟡 [Rank {rank}] Successfully force updated buffer_size to = {buffer_size / 1024**3} GB")
                buffer_size = DispatcherWrapper.instance[0].buffer_size

            rich.print(f"🟡 [Rank {rank}] Overflow check passed for fa2a_metadata_0 and fa2a_metadata_1 with tolerance_factor {tolerance_factor} and buffer_size {buffer_size / 1024**3} GB")

            # params for ping-pong batch0
            ping_pang_params_0 = get_single_step_packed_seq_params(
                fa2a_metadata_0, as_attn_metadata_0, as_rank, resend_qkv=resend_qkv
            )
            # params for ping-pong batch1
            ping_pang_params_1 = get_single_step_packed_seq_params(
                fa2a_metadata_1, as_attn_metadata_1, as_rank, resend_qkv=resend_qkv
            )

            mlp_seq_params_0 = get_attn_metadata(mlp_shard_len_0[as_rank], get_packed_seq_params=True)
            mlp_seq_params_1 = get_attn_metadata(mlp_shard_len_1[as_rank], get_packed_seq_params=True)

            # if rank % 8 == 0:
            #     rich.print(f"🟡 [Rank {rank}] all_metadata[0] -> qkv_fwd_fa2a_metadata =", fa2a_metadata_0[0].fa2a_metadata.__better_print__())
            #     rich.print(f"🟡 [Rank {rank}] all_metadata[0] -> qkv_rev_fa2a_metadata =", fa2a_metadata_0[1].fa2a_metadata.__better_print__())
            #     rich.print(f"🟡 [Rank {rank}] all_metadata[1] -> qkv_fwd_fa2a_metadata =", fa2a_metadata_1[0].fa2a_metadata.__better_print__())
            #     rich.print(f"🟡 [Rank {rank}] all_metadata[1] -> qkv_rev_fa2a_metadata =", fa2a_metadata_1[1].fa2a_metadata.__better_print__())

            def debug_set_metadata_transfer_size_to_0(ping_pang_params: 'PingPangSingleStepPackedSeqParams'):
                for param in [
                    ping_pang_params.qkv_fwd_metadata,
                    ping_pang_params.qkv_bwd_metadata,
                    ping_pang_params.attn_out_fwd_metadata,
                    ping_pang_params.attn_out_bwd_metadata,
                ]:
                    param.fa2a_metadata[1][:] = 1
                    param.fa2a_metadata[3][:] = 1
                    param.my_rank_send_sz = 1
                return
            
            
            if os.environ.get("EXPERIMENT_DEBUG_SET_METADATA_TRANSFER_SIZE_TO_0", "0") == "1":
                print(f"🟡 [Rank {rank}] Debug set metadata transfer size to 0")
                debug_set_metadata_transfer_size_to_0(ping_pang_params_0)
                debug_set_metadata_transfer_size_to_0(ping_pang_params_1)


            if rank % 8 == 0:
                rich.print(f"🟡 [Rank {rank}] ping_pang_params_0.qkv_fwd_metadata =", ping_pang_params_0.qkv_fwd_metadata.__better_print__())
                rich.print(f"🟡 [Rank {rank}] ping_pang_params_1.qkv_fwd_metadata =", ping_pang_params_1.qkv_fwd_metadata.__better_print__())
                rich.print(f"🟡 [Rank {rank}] mlp_seq_params_0 =", mlp_seq_params_0)
                rich.print(f"🟡 [Rank {rank}] mlp_seq_params_1 =", mlp_seq_params_1)

                # Adding backward metadata
                rich.print(f"🟡 [Rank {rank}] ping_pang_params_0.qkv_bwd_metadata =", ping_pang_params_0.qkv_bwd_metadata.__better_print__())
                rich.print(f"🟡 [Rank {rank}] ping_pang_params_1.qkv_bwd_metadata =", ping_pang_params_1.qkv_bwd_metadata.__better_print__())

            packed_seq_params = PingPangPackedSeqParams(
                seq_params=[ping_pang_params_0, ping_pang_params_1],
                mlp_layout_seq_params=[mlp_seq_params_0, mlp_seq_params_1],
                # max_seqlen_q=torch.tensor([total_seq_len * 2], dtype=torch.int32)[0],
                # max_seqlen_kv=torch.tensor([total_seq_len_including_cp * 2], dtype=torch.int32)[0],
                
                # TODO:(Question) Not sure if the values are correct here??
                max_seqlen_q=torch.tensor([total_seq_len * 2], dtype=torch.int32)[0],
                max_seqlen_kv=torch.tensor([total_seq_len * 2], dtype=torch.int32)[0],
                qkv_format="thd",
            )

            input_ids_local = torch.randint(0, 100, (1, num_batched_token_per_as_rank * 2))[0]
            position_ids_local = torch.arange(num_batched_token_per_as_rank, dtype=torch.int64).repeat(1, 2)[0]

            if rank % 8 == 0:
                rich.print(f"🟡 [Rank {rank}] [{sample_id = }] input_ids_local.shape =", input_ids_local.shape)
                rich.print(f"🟡 [Rank {rank}] [{sample_id = }] position_ids_local.shape =", position_ids_local.shape)


            microbatch = {
                "input_ids": input_ids_local,
                "position_ids": position_ids_local,
                "packed_seq_params": packed_seq_params,
            }
            pass

        else:
            raise ValueError(f"Unknown mode: {mode}")

        microbatches = [microbatch]

        if sample_id == 0:
            log_memory_usage("warmup start")
            write_status_log(f"Warmup start")
            with log_memory_usage_context():
                # Warmup
                warmup_times = 5
                try:
                    warmup_times = int(os.environ.get("EXPERIMENT_WARMUP_TIMES", 5))
                except:
                    pass

                warmup_timeout_sec = 240
                try:
                    warmup_timeout_sec = int(os.environ.get("EXPERIMENT_WARMUP_TIMEOUT_SEC", 240))
                except:
                    pass

                # Test passing the nvshmem init
                try:
                    signal.signal(signal.SIGALRM, timeout_handler)
                    signal.alarm(warmup_timeout_sec)  # 60 seconds = 1 minute

                    should_dump_traceback = os.environ.get("EXPERIMENT_SHOULD_DUMP_TRACEBACK", "0") == "1"
                    if should_dump_traceback:
                        print("Start profiling with stack trace...")
                        with torch.profiler.profile(
                            activities=[
                                torch.profiler.ProfilerActivity.CPU,
                                torch.profiler.ProfilerActivity.CUDA,
                            ],
                            record_shapes=True,
                            with_stack=True,
                        ) as prof:
                            ref = worker.forward_backward_batch(
                                microbatches=microbatches,
                                normal_forward_fn=normal_forward_fn,
                                forward_only=False,
                            )
                        signal.alarm(0)
                        print("End profiling with stack trace. Now dumping stack trace to trace.json...")
                        if rank == 0:
                            prof.export_chrome_trace(os.path.join(output_dir, "trace.json"))
                    else:
                        ref = worker.forward_backward_batch(
                            microbatches=microbatches,
                            normal_forward_fn=normal_forward_fn,
                            forward_only=False,
                        )
                        signal.alarm(0)
                except TimeoutError as e:
                    print(f"🔴 Timeout {warmup_timeout_sec} seconds at the first warmup forward_backward function. It may suggest our all2all kernel failed, or just warmup did not completed.")
                    sys.exit(1)

            
            write_status_log(f"Finish warmup first time.")
            for warmup_idx in range(max(warmup_times - 1, 0)):
                ref = worker.forward_backward_batch(
                    microbatches=microbatches,
                    normal_forward_fn=normal_forward_fn,
                    forward_only=False,
                )
                
            time.sleep(1)
            torch.cuda.synchronize()
            torch.distributed.barrier()
            if rank == 0:
                print("=" * 20 + "warmup done")
            log_memory_usage("warmup done")
            write_status_log(f"Finish warmup.")
        
        
        # # --------------
        # # Profiling Run
        # # --------------
        # Useless - doesn't dump as much information as nsys profile as we want.
        # EXPERIMENT_PROFILE_RUN = os.environ.get("EXPERIMENT_PROFILE_RUN", "0")
        # try:
        #     EXPERIMENT_PROFILE_RUN = int(EXPERIMENT_PROFILE_RUN)
        # except:
        #     EXPERIMENT_PROFILE_RUN = 0
        #     pass
        # EXPERIMENT_PROFILE_RUN = 1

        
        # print(f"[Rank {rank}] ⚪ Reaching profiling run.")
        # if EXPERIMENT_PROFILE_RUN > 0:
        #     print(f"[Rank {rank}] ⚪ Running profiling run...")
        #     profile_output_dir = os.path.join(output_dir, "profile_runs")
        #     profile_chrom_trace = os.path.join(profile_output_dir, f"prof_trace.sid{sample_id}.r{rank}.json")
        #     os.makedirs(profile_output_dir, exist_ok=True)
            
        #     with ProfilerCtx(profile_output_dir, chrome_name=profile_chrom_trace) as prof:
        #         for run_id in range(EXPERIMENT_PROFILE_RUN):
        #             print(f"[Rank {rank}] ⚪ Running profiling run (repeat={run_id})...")
        #             torch.cuda.synchronize()  
        #             torch.distributed.barrier()

        #             ref = worker.forward_backward_batch(
        #                 microbatches=microbatches,
        #                 normal_forward_fn=normal_forward_fn,
        #                 forward_only=False,
        #             )
        #             torch.cuda.synchronize()  
        #             torch.distributed.barrier()
        #             print(f"[Rank {rank}] ⚪ Finish profiling run (repeat={run_id}).")
        #             prof.step()
        #             print(f"[Rank {rank}] ⚪ Finish step in profiling run (repeat={run_id})...")

        # exit(1)   

            
        # --------------
        # Real Experiment
        # --------------
        N = 3
        try:
            N = int(os.environ.get("EXPERIMENT_REPEAT_TIMES", 3))
        except:
            N = 3

        torch.cuda.synchronize()
        torch.distributed.barrier()
        
        # Calculate the average duration of the forward_backward_batch
        iteration_times = []
        start_time = time.time()
        torch.cuda.nvtx.range_push(f"sample_{sample_id}(repeat={N})")
        
        
        should_log_memory_during_real_experiment = (
            os.environ.get("EXPERIMENT_SHOULD_LOG_MEMORY_DURING_REAL_EXPERIMENT", "0") == "1"
        )
        if should_log_memory_during_real_experiment:
            log_memory_usage_ctx = log_memory_usage_context()
            log_memory_usage_ctx.__enter__()
            pass
        
        for repeat_idx in range(N):
            # start_event = torch.cuda.Event(enable_timing=True)
            # end_event = torch.cuda.Event(enable_timing=True)pr:
            write_status_log(f"Start Forward_backward_batch (sample_id={sample_id},repeat={repeat_idx})")
            torch.cuda.synchronize()
            torch.distributed.barrier()
            # start_event.record()
            start_it_time = time.time()
            log_memory_usage(f"forward_backward_batch:start(sample_id={sample_id},repeat={repeat_idx})")
            losses_reduced, grad_norm = worker.forward_backward_batch(
                microbatches=microbatches,
                normal_forward_fn=normal_forward_fn,
                forward_only=False,
            )
            # Try to extract a scalar loss for logging
            try:
                loss_value = None
                if isinstance(losses_reduced, dict):
                    val = losses_reduced.get('loss', None)
                    if val is not None:
                        if isinstance(val, (list, tuple)):
                            vals = []
                            for v in val:
                                if torch.is_tensor(v):
                                    vals.append(v.item())
                                else:
                                    vals.append(float(v))
                            if len(vals) > 0:
                                loss_value = sum(vals) / len(vals)
                        else:
                            loss_value = val.item() if torch.is_tensor(val) else float(val)
                elif torch.is_tensor(losses_reduced):
                    loss_value = losses_reduced.item()
                elif isinstance(losses_reduced, (list, tuple)) and len(losses_reduced) > 0:
                    # Handle list of tensors or floats
                    vals = []
                    for v in losses_reduced:
                        if torch.is_tensor(v):
                            vals.append(v.item())
                        elif isinstance(v, (int, float)):
                            vals.append(float(v))
                        elif isinstance(v, dict) and 'loss' in v:
                            lv = v['loss']
                            vals.append(lv.item() if torch.is_tensor(lv) else float(lv))
                    if len(vals) > 0:
                        loss_value = sum(vals) / len(vals)
                if loss_value is not None:
                    write_status_log(f"Loss (sample_id={sample_id},repeat={repeat_idx}) = {loss_value:.6f}")
                    write_loss_log(loss_value, sample_id=sample_id, repeat_idx=repeat_idx)
            except Exception as _:
                # Best-effort logging; ignore extraction failures
                pass
            torch.cuda.synchronize()
            torch.distributed.barrier()
            # end_event.record()
            end_it_time = time.time()
            log_memory_usage(f"forward_backward_batch:done(sample_id={sample_id},repeat={repeat_idx})")
            iteration_time = end_it_time - start_it_time
            write_status_log(f"Finish Forward_backward_batch (sample_id={sample_id},repeat={repeat_idx})")
            iteration_times.append(iteration_time)
        torch.cuda.nvtx.range_pop()
        
        torch.cuda.synchronize()
        torch.distributed.barrier()
        end_time = time.time()
        duration = end_time - start_time
        # duration_ms = duration * 1000
        # avg_duration_ms = duration_ms / N
        avg_duration_ms = 0
        if iteration_times:
            avg_duration_ms = sum(iteration_times) / len(iteration_times) * 1000
        sample_times.append(avg_duration_ms)
        if rank == 0:
            rich.print(f"[Sample ID=({sample_id})] Mode={mode} forward_backward_batch: avg_time_per_iteration = {avg_duration_ms:.2f} ms")
        device = torch.cuda.current_device()
        
        if rank % 8 == 0:
            (
                allocated_cur, 
                allocated_peak, 
                total_alloc
            ) = d2.mem.get_torch_cuda_memory_usage(device)
            pynvml_gpu_memory_usage = d2.mem.get_pynvml_gpu_memory_usage(device)
            rich.print(f"Ⓜ️Ⓜ️ [Sample ID=({sample_id})] Memory usage: allocated_cur: {(allocated_cur/1024):.2f} GB, allocated_peak: {(allocated_peak/1024):.2f} GB, total_alloc: {(total_alloc/1024):.2f} GB, pynvml_gpu_memory_usage: {(pynvml_gpu_memory_usage/1024):.2f} GB")
            

        time.sleep(2) # to ensure the profile sees a better profiling result
        torch.cuda.synchronize()
        torch.distributed.barrier()

        # Log attention timing data for this iteration if unified timing enabled
        if os.getenv("UNIFIED_RECORD_ATTENTION_TIMES", "0") == "1":
            # Synchronize and collect timing data from CUDA events
            sync_and_collect_timing()
            
            timing_data = get_unified_attention_times().get(sample_id, {"forward_times": [], "backward_times": []})
            forward_times = timing_data["forward_times"]
            backward_times = timing_data["backward_times"]
            
            # Calculate medians
            forward_median = np.median(forward_times) if forward_times else 0.0
            backward_median = np.median(backward_times) if backward_times else 0.0
            
            # Create attn_time directory structure
            attn_time_dir = os.path.join(output_dir, "attn_time")
            os.makedirs(attn_time_dir, exist_ok=True)
            
            # Log to per-rank JSONL file
            attn_time_file = os.path.join(attn_time_dir, f"attn_time.rank{rank}.jsonl")
            iteration_data = {
                "sample_id": sample_id,
                "mode": mode,
                "forward_times": forward_times,
                "backward_times": backward_times,
                "forward_median_ms": forward_median,
                "backward_median_ms": backward_median,
                "forward_count": len(forward_times),
                "backward_count": len(backward_times)
            }
            
            with open(attn_time_file, 'a') as f:
                f.write(json.dumps(iteration_data) + '\n')
            
            # Print median times for this iteration
            if rank % 8 == 0:  # Only print from a subset of ranks to avoid spam
                rich.print(f"🕒 [Sample {sample_id}] {mode.upper()} Attention timing - Forward median: {forward_median:.2f} ms ({len(forward_times)} measurements), Backward median: {backward_median:.2f} ms ({len(backward_times)} measurements)")

        # Log all-to-all timing data for this iteration if unified timing enabled
        if os.getenv("UNIFIED_RECORD_A2A_TIMES", "0") == "1":
            # Synchronize and collect timing data from CUDA events
            sync_and_collect_a2a_timing()
            
            timing_data = get_unified_a2a_times().get(sample_id, {"a2a_forward": []})
            a2a_times = timing_data["a2a_forward"]
            
            # Calculate median
            a2a_median = np.median(a2a_times) if a2a_times else 0.0
            
            # Create a2a_time directory structure
            a2a_time_dir = os.path.join(output_dir, "a2a_time")
            os.makedirs(a2a_time_dir, exist_ok=True)
            
            # Log to per-rank JSONL file
            a2a_time_file = os.path.join(a2a_time_dir, f"a2a_time.rank{rank}.jsonl")
            iteration_data = {
                "sample_id": sample_id,
                "mode": mode,
                "a2a_forward_times": a2a_times,
                "a2a_forward_median_ms": a2a_median,
                "a2a_forward_count": len(a2a_times)
            }
            
            with open(a2a_time_file, 'a') as f:
                f.write(json.dumps(iteration_data) + '\n')
            
            # Print median times for this iteration
            if rank % 8 == 0:  # Only print from a subset of ranks to avoid spam
                rich.print(f"🕒 [Sample {sample_id}] {mode.upper()} All-to-All timing - Median: {a2a_median:.2f} ms ({len(a2a_times)} measurements)")

        # Log TickSync blocking events for this iteration if enabled
        if os.getenv("D2_TICKSYNC_BLOCKING_DETECTION", "0") == "1":
            # Create ticksync_blocking directory structure    
            ticksync_blocking_dir = os.path.join(output_dir, "ticksync_blocking")
            os.makedirs(ticksync_blocking_dir, exist_ok=True)
            ticksync_blocking_file = os.path.join(ticksync_blocking_dir, f"ticksync_blocking.rank{rank}.jsonl")

            # Process pending CUDA events to measure actual GPU timing
            TickSync.process_pending_events()
            blocking_events = TickSync.get_blocking_events()
            
            # Log each blocking event separately to per-rank JSONL file
            for event in blocking_events:
                iteration_data = {
                    "sample_id": sample_id,
                    "mode": mode,
                    "blocking_event": event
                }
                with open(ticksync_blocking_file, 'a') as f:
                    f.write(json.dumps(iteration_data) + '\n')
            
            # Print blocking events for this iteration
            # if rank % 8 == 0:  # Only print from a subset of ranks to avoid spam
            #     rich.print(f"⚠️  [Sample {sample_id}] {mode.upper()} TickSync Blocking - {len(blocking_events)} events detected")
            #     for event in blocking_events[-3:]:  # Show last 3 events
            #         rich.print(f"    {event['layer_info']} {event['operation_info']} ({event['phase']}): {event['wait_time_ms']:.2f}ms > {event['threshold_ms']}ms")
            
            # Clear events after logging
            TickSync.clear_blocking_events()

        # Log tick operations timing data for this iteration if enabled
        if os.getenv("UNIFIED_RECORD_TICK_TIMES", "0") == "1":
            # Synchronize and collect timing data from CUDA events
            from d2.runtime.megatron.ping_pong.tick_ops import sync_and_collect_tick_timing, get_tick_times
            sync_and_collect_tick_timing()
            
            timing_data = get_tick_times().get(sample_id, {
                "forward_pre_core_attn": [], 
                "forward_post_core_attn": [],
            })
            pre_attn_times = timing_data["forward_pre_core_attn"]
            post_attn_times = timing_data["forward_post_core_attn"]
            
            # Calculate medians
            pre_attn_median = np.median([t["duration_ms"] for t in pre_attn_times]) if pre_attn_times else 0.0
            post_attn_median = np.median([t["duration_ms"] for t in post_attn_times]) if post_attn_times else 0.0
            
            # Create tick_time directory structure
            tick_time_dir = os.path.join(output_dir, "tick_time")
            os.makedirs(tick_time_dir, exist_ok=True)
            
            # Log to per-rank JSONL file
            tick_time_file = os.path.join(tick_time_dir, f"tick_time.rank{rank}.jsonl")
            iteration_data = {
                "sample_id": sample_id,
                "mode": mode,
                "forward_pre_core_attn_times": pre_attn_times,
                "forward_post_core_attn_times": post_attn_times,
                "forward_pre_core_attn_median_ms": pre_attn_median,
                "forward_post_core_attn_median_ms": post_attn_median,
                "forward_pre_core_attn_count": len(pre_attn_times),
                "forward_post_core_attn_count": len(post_attn_times)
            }
            
            
            with open(tick_time_file, 'a') as f:
                f.write(json.dumps(iteration_data) + '\n')
            
            # Print median times for this iteration
            if rank % 8 == 0:  # Only print from a subset of ranks to avoid spam
                timing_msg = f"🕒 [Sample {sample_id}] {mode.upper()} Tick Operations timing - Forward Pre-attn median: {pre_attn_median:.2f} ms ({len(pre_attn_times)} measurements), Forward Post-attn median: {post_attn_median:.2f} ms ({len(post_attn_times)} measurements)"
                rich.print(timing_msg)

        # Write to the benchmark jsonl log
        if rank == 0:
            # benchmark_data
            items = {
                "sample_id": sample_id,
                "duration_ms": avg_duration_ms,
                "samples": iterated_samples[-1],
            }
            # if os.environ.get("EXPERIMENT_ENABLE_BENCHMARK_SAVING", "1") == "1":
            output_file = os.path.join(output_dir, "benchmark.raw.jsonl")
            with open(output_file, 'a') as f:
                f.write(json.dumps(items))
                f.write('\n')

    torch.cuda.synchronize()
    torch.distributed.barrier()
    print("=" * 20 + "forward_backward_batch attention server, done")

    # Collect attention start and end events and calculate the average duration.
    from datetime import datetime
    pst = pytz.timezone('US/Pacific')
    timestamp = datetime.now(pst).strftime("%Y-%m-%d %H:%M:%S PST")
    now_ts = datetime.now(pst).strftime("%Y%m%d_%H%M%S")
    

    if rank % 8 == 0:

        summary_log_file = os.path.join(output_dir, "summary.log")
        with open(summary_log_file, "w") as f:
            f.write("===============Summary Log===============\n")

        def log_to_console_and_file(*args, **kwargs):
            rich.print(*args, **kwargs)
            if rank == 0:
                with open(summary_log_file, "a") as f:
                    print(*args, **kwargs, file=f)

        
        log_to_console_and_file(f"🟢 Test {__file__} passed")
        
        config = dict(
            mode=mode, 
            nodes=num_nodes,
            num_gpus_per_node=num_gpus_per_node,
            tp_size=tp_size, dp_size=dp_size, cp_size=cp_degree, 
            num_tokens=num_tokens, model_path=model_path, num_layers=num_layers, 
            max_sample_id=max_sample_id, up_sample_factor=up_sample_factor, filter_threshold=filter_threshold, filter_ratio=filter_ratio, 
            replan_iter=replan_iter, elongate_factor=elongate_factor,
        )
        log_to_console_and_file(f"🟢 Test Config: {config}")
        log_to_console_and_file(f"🟢 Test DateTime: ", timestamp)
        
        # Prepare benchmark data
        benchmark_data = {
            "test_file": __file__,
            "args": str(args),
            "timestamp": timestamp,
            "config": config,
            "samples": [],
            # "modified_batches": modified_batches,
            # "fa2a_metadata_list": fa2a_metadata_list,
        }
        
        for idx in range(len(sample_times)):
            samples = iterated_samples[idx]
            duration = sample_times[idx]
            log_to_console_and_file(f"🟢 Sample {idx}: duration: {duration:.2f} ms, samples = {samples}")
            benchmark_data["samples"].append({
                "sample_id": idx,
                "samples": samples,
                "duration_ms": duration
            })
        
        
    if rank == 0:
        # Write benchmark results to file
        # TODO: Legacy behavior. Can delete this later... just treat this as a log file.
        file_dir = os.path.dirname(os.path.abspath(__file__))
        benchmark_dir = os.path.join(file_dir, "..", "benchmarks", "_250809_e2e_benchmark", "data")
        os.makedirs(benchmark_dir, exist_ok=True)
        benchmark_file = os.path.join(benchmark_dir, f"benchmark.{now_ts}.{mode}.json")
        with open(benchmark_file, 'w') as f:
            json.dump(benchmark_data, f, indent=2)

        # Save another copy of the benchmark data to the output directory
        output_file = os.path.join(output_dir, "benchmark.json")
        with open(output_file, 'w') as f:
            json.dump(benchmark_data, f, indent=2)
        
        rich.print(f"🟢 Benchmark results saved to: {output_file}")

    # for idx, (sample, duration) in enumerate(zip(iterated_samples, sample_times)):
    #     rich.print(f"🟢 Sample {idx}: {sample}, duration: {duration} ms")

    # Report memory usage
    # save_memory_usage_to_file(memory_usage_output_dir)
    
    # Save attention timing data if unified timing was enabled
    if os.getenv("UNIFIED_RECORD_ATTENTION_TIMES", "0") == "1":
        # Final synchronization to collect any remaining timing data
        sync_and_collect_timing()
        
        if rank == 0:
            all_attention_times = get_unified_attention_times()
            attention_timing_file = os.path.join(output_dir, "attention_timing.json")
            with open(attention_timing_file, 'w') as f:
                json.dump(all_attention_times, f, indent=2)
            rich.print(f"🟢 {mode.upper()} Attention timing data saved to: {attention_timing_file}")
            
            # Also print summary with medians
            
            rich.print(f"🟢 ===== {mode.upper()} Attention Timing Summary =====")
            for sample_id, timing_data in all_attention_times.items():
                fwd_times = timing_data["forward_times"]
                bwd_times = timing_data["backward_times"]
                if fwd_times:
                    avg_fwd = sum(fwd_times) / len(fwd_times)
                    median_fwd = np.mean(fwd_times)
                    rich.print(f"Sample {sample_id}: Forward - Avg: {avg_fwd:.2f} ms, Mean: {median_fwd:.2f} ms ({len(fwd_times)} measurements)")
                if bwd_times:
                    avg_bwd = sum(bwd_times) / len(bwd_times)
                    median_bwd = np.mean(bwd_times)
                    rich.print(f"Sample {sample_id}: Backward - Avg: {avg_bwd:.2f} ms, Mean: {median_bwd:.2f} ms ({len(bwd_times)} measurements)")
            rich.print(f"🟢 Individual rank attention timing logs saved to: {os.path.join(output_dir, 'attn_time')}")
    
    # Save all-to-all timing data if unified timing was enabled
    if os.getenv("UNIFIED_RECORD_A2A_TIMES", "0") == "1":
        # Final synchronization to collect any remaining timing data
        sync_and_collect_a2a_timing()
        
        if rank == 0:
            all_a2a_times = get_unified_a2a_times()
            a2a_timing_file = os.path.join(output_dir, "a2a_timing.json")
            with open(a2a_timing_file, 'w') as f:
                json.dump(all_a2a_times, f, indent=2)
            rich.print(f"🟢 {mode.upper()} All-to-All timing data saved to: {a2a_timing_file}")
            
            # Also print summary with medians
            rich.print(f"🟢 ===== {mode.upper()} All-to-All Timing Summary =====")
            for sample_id, timing_data in all_a2a_times.items():
                a2a_times = timing_data["a2a_forward"]
                if a2a_times:
                    avg_a2a = sum(a2a_times) / len(a2a_times)
                    median_a2a = np.median(a2a_times)
                    rich.print(f"Sample {sample_id}: All-to-All Forward - Avg: {avg_a2a:.2f} ms, Median: {median_a2a:.2f} ms ({len(a2a_times)} measurements)")
            rich.print(f"🟢 Individual rank all-to-all timing logs saved to: {os.path.join(output_dir, 'a2a_time')}")
    
    # Save tick operations timing data if enabled
    if os.getenv("UNIFIED_RECORD_TICK_TIMES", "0") == "1":
        # Final synchronization to collect any remaining timing data
        from d2.runtime.megatron.ping_pong.tick_ops import sync_and_collect_tick_timing, get_tick_times
        sync_and_collect_tick_timing()
        
        if rank == 0:
            all_tick_times = get_tick_times()
            tick_timing_file = os.path.join(output_dir, "tick_timing.json")
            with open(tick_timing_file, 'w') as f:
                json.dump(all_tick_times, f, indent=2)
            rich.print(f"🟢 {mode.upper()} Tick operations timing data saved to: {tick_timing_file}")
            
            # Also print summary with medians
            rich.print(f"🟢 ===== {mode.upper()} Tick Operations Timing Summary =====")
            for sample_id, timing_data in all_tick_times.items():
                pre_attn_times = timing_data["forward_pre_core_attn"]
                post_attn_times = timing_data["forward_post_core_attn"]
                
                if pre_attn_times:
                    avg_pre = sum(t["duration_ms"] for t in pre_attn_times) / len(pre_attn_times)
                    median_pre = np.median([t["duration_ms"] for t in pre_attn_times])
                    rich.print(f"Sample {sample_id}: Forward Pre-Core Attn - Avg: {avg_pre:.2f} ms, Median: {median_pre:.2f} ms ({len(pre_attn_times)} measurements)")
                if post_attn_times:
                    avg_post = sum(t["duration_ms"] for t in post_attn_times) / len(post_attn_times)
                    median_post = np.median([t["duration_ms"] for t in post_attn_times])
                    rich.print(f"Sample {sample_id}: Forward Post-Core Attn - Avg: {avg_post:.2f} ms, Median: {median_post:.2f} ms ({len(post_attn_times)} measurements)")
                
                        
            rich.print(f"🟢 Individual rank tick operations timing logs saved to: {os.path.join(output_dir, 'tick_time')}")
    
    # Save TickSync blocking data if enabled
    if os.getenv("D2_TICKSYNC_BLOCKING_DETECTION", "0") == "1":
        # Process any remaining pending events
        TickSync.process_pending_events()
        # Final collection of any remaining blocking events
        final_blocking_events = TickSync.get_blocking_events()
        
        if rank == 0:
            # Create summary of all blocking events
            ticksync_summary_file = os.path.join(output_dir, "ticksync_blocking_summary.json")
            summary_data = {
                "total_blocking_events": len(final_blocking_events),
                "blocking_events": final_blocking_events,
                "threshold_ms": float(os.getenv("D2_TICKSYNC_THRESHOLD_MS", "1.0"))
            }
            
            with open(ticksync_summary_file, 'w') as f:
                json.dump(summary_data, f, indent=2)
            rich.print(f"🟢 {mode.upper()} TickSync blocking summary saved to: {ticksync_summary_file}")
            
            # Print summary
            if final_blocking_events:
                rich.print(f"🟢 ===== {mode.upper()} TickSync Blocking Summary =====")
                rich.print(f"Total blocking events: {len(final_blocking_events)}")
                
                # Group by layer and operation
                layer_ops = {}
                for event in final_blocking_events:
                    key = f"{event['layer_info']} {event['operation_info']}"
                    if key not in layer_ops:
                        layer_ops[key] = []
                    layer_ops[key].append(event['wait_time_ms'])
                
                for key, times in layer_ops.items():
                    avg_time = sum(times) / len(times)
                    max_time = max(times)
                    rich.print(f"{key}: {len(times)} events, avg: {avg_time:.2f}ms, max: {max_time:.2f}ms")
            else:
                rich.print(f"✅ No TickSync blocking events detected")
            
            rich.print(f"🟢 Individual rank TickSync blocking logs saved to: {os.path.join(output_dir, 'ticksync_blocking')}")
    
    # Cleanup and exit
    rich.print(f"❄️ [Rank {rank}] Finished test and exit.")        
    write_status_log(f"Finish test and exit.")
    # if False: # Only use it when force exit
    if args.force_exit: 
        print(f"[Rank {rank}] Starting aggressive cleanup process...")
        os._exit(0)



def save_memory_usage_to_file(memory_usage_dir: str):
    os.makedirs(memory_usage_dir, exist_ok=True)
    
    rank = torch.distributed.get_rank()
    memory_usage: list[dict] = d2.mem.get_memory_usage()
    memory_usage_output_file = os.path.join(memory_usage_dir, f"mem.rank{rank}.jsonl")
    with open(memory_usage_output_file, 'w') as f:
        for memory_usage_item in memory_usage:
            f.write(json.dumps(memory_usage_item) + '\n')
    rich.print(f"🟢 Memory usage saved to: {memory_usage_output_file}")
    return


def enable_memory_usage_logging(memory_usage_dir: str):
    os.makedirs(memory_usage_dir, exist_ok=True)
    rank = os.environ.get("RANK", os.environ.get("SLURM_PROCID", "0"))
    memory_usage_log_file = os.path.join(memory_usage_dir, f"mem.rank{rank}.log.jsonl")
    with open(memory_usage_log_file, 'w') as f:
        pass
    d2.mem.set_memory_usage_log_file(memory_usage_log_file)
    pass

# Unified Attention Timing Collection (works for both WLBLLM and D2)
_unified_attention_times = {}
_current_sample_id = None
_pending_events = []  # List of (sample_id, phase, start_event, end_event) tuples

def setup_unified_attention_timing_patch():
    """Setup monkey patching for unified attention timing collection (WLBLLM + D2)."""
    import flash_attn.flash_attn_interface as flash_attn_interface
    
    # Store original functions
    original_forward = flash_attn_interface._wrapped_flash_attn_varlen_forward
    original_backward = flash_attn_interface._wrapped_flash_attn_varlen_backward
    
    def timed_forward(*args, **kwargs):
        if _current_sample_id is not None:
            # Create CUDA events for timing
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            
            # Record start event
            start_event.record()
        
        result = original_forward(*args, **kwargs)
        
        if _current_sample_id is not None:
            # Record end event
            end_event.record()
            
            # Store events for later synchronization
            _pending_events.append((_current_sample_id, "forward", start_event, end_event))
        
        return result
    
    def timed_backward(*args, **kwargs):
        if _current_sample_id is not None:
            # Create CUDA events for timing
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            
            # Record start event
            start_event.record()
        
        result = original_backward(*args, **kwargs)
        
        if _current_sample_id is not None:
            # Record end event
            end_event.record()
            
            # Store events for later synchronization
            _pending_events.append((_current_sample_id, "backward", start_event, end_event))
        
        return result
    
    # Apply monkey patches
    flash_attn_interface._wrapped_flash_attn_varlen_forward = timed_forward
    flash_attn_interface._wrapped_flash_attn_varlen_backward = timed_backward
    
    print("🟢 Unified attention timing patch applied successfully (works for both WLBLLM and D2)")

def set_unified_current_sample_id(sample_id):
    """Set the current sample ID for unified attention timing."""
    global _current_sample_id
    _current_sample_id = sample_id

def sync_and_collect_timing():
    """Synchronize all pending events and collect timing data."""
    global _pending_events, _unified_attention_times
    
    if not _pending_events:
        return
    
    # Synchronize all events
    torch.cuda.synchronize()
    
    # Process all pending events
    for sample_id, phase, start_event, end_event in _pending_events:
        # Calculate duration in milliseconds
        duration_ms = start_event.elapsed_time(end_event)
        
        # Initialize sample data if needed
        if sample_id not in _unified_attention_times:
            _unified_attention_times[sample_id] = {"forward_times": [], "backward_times": []}
        
        # Store timing data
        if phase == "forward":
            _unified_attention_times[sample_id]["forward_times"].append(duration_ms)
        elif phase == "backward":
            _unified_attention_times[sample_id]["backward_times"].append(duration_ms)
    
    # Clear pending events
    _pending_events.clear()

def get_unified_attention_times():
    """Get all unified attention timing data."""
    return _unified_attention_times.copy()

def clear_unified_attention_times():
    """Clear unified attention timing data."""
    global _unified_attention_times, _pending_events
    _unified_attention_times.clear()
    _pending_events.clear()

# Unified All-to-All Timing Collection (works for both WLBLLM and D2)
_a2a_attention_times = {}
_current_a2a_sample_id = None
_pending_a2a_events = []  # List of (sample_id, operation, start_event, end_event) tuples

def setup_unified_a2a_timing_patch():
    """Setup monkey patching for unified all-to-all timing collection."""
    from d2.runtime.attn_kernels.ops import _ops_fast_a2a_wrapper, DispatcherWrapper
    
    # Store original function
    original_wrapper = _ops_fast_a2a_wrapper
    
    def timed_wrapper(*args):
        if _current_a2a_sample_id is not None:
            # Get the communication stream
            comm_stream = DispatcherWrapper.comm_stream
            if comm_stream is None:
                comm_stream = torch.cuda.current_stream()
            
            # Create CUDA events for timing
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            
            # Record start event on the communication stream
            start_event.record(comm_stream)
        
        # Call original function
        result = original_wrapper(*args)
        
        if _current_a2a_sample_id is not None:
            # Record end event on the communication stream
            end_event.record(comm_stream)
            
            # Store events for later synchronization
            _pending_a2a_events.append((_current_a2a_sample_id, "a2a_forward", start_event, end_event))
        
        return result
    
    # Apply monkey patch
    import d2.runtime.attn_kernels.ops as ops_module
    ops_module._ops_fast_a2a_wrapper = timed_wrapper
    
    print("🟢 Unified all-to-all timing patch applied successfully")

def set_unified_current_a2a_sample_id(sample_id):
    """Set the current sample ID for unified all-to-all timing."""
    global _current_a2a_sample_id
    _current_a2a_sample_id = sample_id

def sync_and_collect_a2a_timing():
    """Synchronize all pending all-to-all events and collect timing data."""
    global _pending_a2a_events, _a2a_attention_times
    
    if not _pending_a2a_events:
        return {}
    
    # Get the communication stream and synchronize
    from d2.runtime.attn_kernels.ops import DispatcherWrapper
    comm_stream = DispatcherWrapper.comm_stream
    if comm_stream is not None:
        comm_stream.synchronize()
    else:
        torch.cuda.synchronize()
    
    # Process all pending events
    for sample_id, operation, start_event, end_event in _pending_a2a_events:
        # Calculate duration in milliseconds
        duration_ms = start_event.elapsed_time(end_event)
        
        # Initialize sample data if needed
        if sample_id not in _a2a_attention_times:
            _a2a_attention_times[sample_id] = {"a2a_forward": []}
        
        # Store timing data
        if operation in _a2a_attention_times[sample_id]:
            _a2a_attention_times[sample_id][operation].append(duration_ms)
    
    # Clear pending events
    _pending_a2a_events.clear()
    return _a2a_attention_times.copy()

def get_unified_a2a_times():
    """Get all unified all-to-all timing data."""
    return _a2a_attention_times.copy()

def clear_unified_a2a_times():
    """Clear unified all-to-all timing data."""
    global _a2a_attention_times, _pending_a2a_events
    _a2a_attention_times.clear()
    _pending_a2a_events.clear()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, choices=["baseline", "d2", "wlbllm"], default="baseline", 
                        help="Test mode: 'baseline' for simple batch generation, 'd2' for balanced flops planning, 'wlbllm' for wlbllm")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-tokens", type=int, default=1024)
    parser.add_argument("--cp-degree", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-nodes", type=int, default=1)
    parser.add_argument("--num-gpus-per-node", type=int, default=2)
    parser.add_argument("--tp-size", type=int, default=1)
    parser.add_argument("--num-layers", type=int, default=None)
    parser.add_argument("--model-path", type=str, default="deepseek-ai/DeepSeek-R1-Distill-Llama-8B")
    parser.add_argument("--max-sample-id", type=int, default=10)
    parser.add_argument("--up-sample-factor", type=int, default=2)
    parser.add_argument("--replan-iter", type=int, default=1)
    parser.add_argument("--elongate-factor", type=int, default=1)
    parser.add_argument("--filter-threshold", type=int, default=64 * 1024)
    parser.add_argument("--filter-ratio", type=float, default=0.90)
    parser.add_argument("--force-exit", action="store_true")
    parser.add_argument("--should-add-debug-cases", action="store_true")
    parser.add_argument("--should-profile-memory", type=str, default=None)
    parser.add_argument("--should-resend-qkv", action="store_true", help="Whether to resend qkv in the backward pass")
    parser.add_argument("--output-dir", type=str, default=None)   
    parser.add_argument("--sample-start-idx", type=int, default=0, help="Start index of the sample ids to sample") 
    parser.add_argument("--change-long-doc-ratio", type=float, default=0.0, help="Ratio of long docs to change")
    parser.add_argument("--sample-name", type=str, default="wlbllm", help="Name of the sample to use", choices=["wlbllm", "prolong"])
    parser.add_argument("--alpha-factor", type=float, default=1.0, help="Alpha factor for memory imbalance")
    
    args = parser.parse_args()
    print(f"🟡 Args: {args}")

    # "D2_SKIP_FLOAT_CONVERSION"
    os.environ["D2_SKIP_FLOAT_CONVERSION"] = "1"

    if args.output_dir is None:
        args.output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
        pass
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    
    should_profile_memory = args.should_profile_memory
    if should_profile_memory:
        torch.cuda.memory._record_memory_history()
        mem_snapshots_dir = os.path.join(args.output_dir, "mem_snapshots")
        os.makedirs(mem_snapshots_dir, exist_ok=True)
        print(f"🟡 Will save mem snapshots to: {mem_snapshots_dir}")
        pass

    memory_usage_output_dir = os.path.join(args.output_dir, "mem")
    memory_log_output_dir = os.path.join(args.output_dir, "mem-log")
    os.makedirs(memory_usage_output_dir, exist_ok=True)
    os.makedirs(memory_log_output_dir, exist_ok=True)

    start_time = time.time()
    
    if should_profile_memory:
        with torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            record_shapes=True,
            profile_memory=True,
            with_stack=True,
        ) as prof:
            try:
                test(args)
            except Exception as e:
                import traceback
                traceback.print_exc()
            finally:
                save_memory_usage_to_file(memory_usage_output_dir)
    else:
        try:
            test(args)
        except Exception as e:
            import traceback
            traceback.print_exc()
            raise e
        finally:
            save_memory_usage_to_file(memory_usage_output_dir)
    log_memory_usage("test:end", force=True)
    
    end_time = time.time()
    elapsed_time = end_time - start_time
    rich.print(f"🕛 Test elapsed time: {elapsed_time:.2f} seconds")
    
    if should_profile_memory:
        mode = args.mode
        batch_size = args.batch_size
        num_tokens = args.num_tokens
        cp_degree = args.cp_degree
        tp_size = args.tp_size
        num_layers = args.num_layers

        rank = torch.distributed.get_rank()
        mem_snapshot_output_path = os.path.join(mem_snapshots_dir, f"memory_profile.rank{rank}.pickle")
        memory_timeline_output_path = os.path.join(mem_snapshots_dir, f"memory_profile.rank{rank}.html")
        memory_timeline_output_raw = os.path.join(mem_snapshots_dir, f"memory_profile.rank{rank}.json.gz")
        print(f"🟡 Will save mem snapshot to: {mem_snapshot_output_path}")
        print(f"🟡 Will save mem timeline to: {memory_timeline_output_path}")
        # if rank % 8 == 0:
        if rank == 0:
            print("Dumping memory snapshot")
            torch.cuda.memory._dump_snapshot(mem_snapshot_output_path)
            prof.export_memory_timeline(memory_timeline_output_path, device=torch.cuda.current_device())
            prof.export_memory_timeline(memory_timeline_output_raw, device=torch.cuda.current_device())
            print("Memory snapshot dumped")

