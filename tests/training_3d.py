

"""
Combined Megatron E2E Test (D2)

This script runs the D2 attention server pipeline with balanced flops planning
and ping-pang parameters.

Dataset Options:
- Synthetic sequence distributions: 'wlbllm', 'prolong' (default)
- Real datasets with tokens: 'bookcorpus', 'wikitext', 'openwebtext', 'c4'

Usage:
```bash
# With synthetic data (default)
bash test_e2e_combined.multi.sh <rzv_endpoint> <n_nodes>

# With real dataset (e.g., bookcorpus)
python training_3d.py --sample-name bookcorpus ...
```
"""
import traceback
import time
import psutil, os
from dataclasses import dataclass
start_time__ = time.time()

rank = int(os.environ.get("RANK", os.environ.get("SLURM_PROCID","0")))
local = int(os.environ.get("LOCAL_RANK", os.environ.get("SLURM_LOCALID","0")))
p = psutil.Process(os.getpid())
p.cpu_affinity([local * 16, local * 16 + 1])  # pin to core based on local rank
print(f"[{rank}] allowed CPUs:", p.cpu_affinity())


def extract_scalar_loss(losses_reduced):
    loss_value = None
    if isinstance(losses_reduced, dict):
        val = losses_reduced.get("loss")
        if val is not None:
            if isinstance(val, (list, tuple)):
                vals = []
                for v in val:
                    if torch.is_tensor(v):
                        vals.append(v.item())
                    else:
                        vals.append(float(v))
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

# ----------------
# Taskset confirm
# ----------------
import check_cpu_binding
aff, mems = check_cpu_binding.check_cpu_binding()
print(f"CPUS={aff} MEMS={mems}")


# ----------------
# Main Imports
# ----------------
import d2.planner.wlb_planner
import d2.mem
import math
import argparse
import os
import gc
import pytz
import json
import time
import traceback
import sys
from contextlib import contextmanager
from collections import defaultdict
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

from wandb_driver import WandbDriver


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


def _write_status_log(output_dir: str, message: str):
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


def _write_loss_log(
    output_dir: str,
    loss_value,
    sample_id: int | None = None,
    repeat_idx: int | None = None,
):
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
        loss_float = loss_value.item() if torch.is_tensor(loss_value) else float("nan")
    message = (
        f"📉 [T{elapsed_time:.2f}] ({caller_file}:{caller_line}) "
        f"sample_id={sid} repeat={rep} loss={loss_float:.6f}"
    )
    with open(loss_log_file, "a") as f:
        f.write(message + "\n")
    print(message)
    return


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

        # In forward-only mode (e.g., validation), skip the optimizer step entirely.
        if forward_only:
            grad_norm = None
            return losses_reduced, grad_norm

        with torch.cuda.nvtx.range("optimizer_step"):
            # torch.cuda.synchronize()
            log_memory_usage("optimizer_step:(start)")
            if os.getenv("EXPERIMENT_SKIP_OPTIMIZER_STEP", "0") == "1":
                # when testing numerical correctness, instead of running optimizer step, reset grads.
                update_successful, grad_norm, num_zeros_in_grad = True, 0.0, 0
                for tm in self.train_module:
                    for param in unwrap_model(tm).parameters():
                        param.main_grad.zero_()
            else:
                update_successful, grad_norm, num_zeros_in_grad = self.optimizer.step()
            # torch.cuda.synchronize()
            log_memory_usage("optimizer_step:(end)")
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

    log_memory_usage("buffer initialized", force=True)
    return worker


from typing import List, Optional, Dict, Tuple

# Import data loading and batch utilities
from training_utils import (
    setup_global_batch,
    get_next_batch,
    build_sequence_records,
    build_rank_shards,
)

K = 1024
# TODO(Refactor): Remove this global variable.
iterated_samples = []
modified_batches = []
fa2a_metadata_list = []


# ========== D2 Specific Functions ==========

def main(args):
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
    output_dir = args.output_dir
    val_every_n_steps = getattr(args, "val_every_n_steps", 1)
    
    # Wandb configuration (support both CLI args and env vars)
    enable_wandb = args.enable_wandb or os.environ.get("ENABLE_WANDB", "0") == "1"
    wandb_project = args.wandb_project or os.environ.get("WANDB_PROJECT", "d2-training")
    wandb_run_name = args.wandb_run_name or os.environ.get("WANDB_RUN_NAME", None)
    allow_all_ranks_loss = os.environ.get("ALLOW_ALL_RANKS_LOSS", "0") == "1"
    print(f"🟡 allow_all_ranks_loss = {allow_all_ranks_loss}")

    normal_forward_fn = False # becuase D2 mode doesn't use normal forward

    if num_layers is not None:
        os.environ["NUM_LAYERS"] = str(num_layers)

    mode = args.mode
    assert mode == "d2", f"Mode {mode} is not supported for D2"
    
    dtype = torch.bfloat16
    element_size = dtype.itemsize

    def write_status_log(message: str):
        _write_status_log(output_dir, message)
        return

    def write_loss_log(loss_value, sample_id: int | None = None):
        _write_loss_log(output_dir, loss_value, sample_id=sample_id)
        return

    # Set forward function mode based on test mode
    log_memory_usage("enter test", force=True)
    
    # Check world size
    print(f"🟡 Running D2 config: tp_size={tp_size}, world_size={world_size}")
        
    write_status_log(f"Pass world size check")
    
    print(f"🟡 setup_global_batch (mode={mode}): ")
    print(f"  - total_seq_len = {total_seq_len}")

    hf_config = AutoConfig.from_pretrained(model_path)
    hidden_size_q = hf_config.hidden_size

    hidden_size_kv = hidden_size_q
    if hasattr(hf_config, "num_key_value_heads"):
        hidden_size_kv = (hidden_size_kv * hf_config.num_key_value_heads //
                          hf_config.num_attention_heads)

    worker: MegatronE2eWorker = init_megatron_e2e_test(
        hidden_size_q, hidden_size_kv, num_tokens,
        world_size, max_cp_degree * 1, tp_size,
        dtype, MegatronE2eWorker
    )
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

    rank = worker.rank
    as_rank = worker.as_rank
    as_world_size = worker.as_world_size
    
    # Initialize wandb driver
    wandb_driver = WandbDriver()
    wandb_driver.initialize(
        enable_wandb=enable_wandb,
        rank=rank,
        project=wandb_project,
        run_name=wandb_run_name,
        allow_all_ranks=allow_all_ranks_loss,
        config={
            "num_nodes": num_nodes,
            "num_gpus_per_node": num_gpus_per_node,
            "world_size": world_size,
            "tp_size": tp_size,
            "cp_degree": cp_degree,
            "batch_size": batch_size,
            "num_tokens": num_tokens,
            "num_layers": num_layers,
            "model_path": model_path,
            "seed": seed,
            "sample_name": args.sample_name,
        }
    )

    hidden_size_q_tp = hidden_size_q // tp_size
    hidden_size_k_tp = hidden_size_kv // tp_size

    if rank == 0:
        setup_global_batch(
            total_seq_len,
            up_sample_factor=up_sample_factor,
            elongate_factor=elongate_factor,
            filter_threshold=filter_threshold,
            filter_ratio=filter_ratio,
            should_add_debug_cases=should_add_debug_cases,
            change_long_doc_ratio=args.change_long_doc_ratio,
            sample_name=args.sample_name,
            tokenizer=worker.tokenizer,  # Pass the tokenizer to load real datasets
            max_total_tokens=args.max_total_tokens,
        )
    torch.distributed.barrier()


    print(f"🟡 [Rank {rank}] hidden_size_q_tp = {hidden_size_q_tp}, hidden_size_k_tp = {hidden_size_k_tp}, element_size = {element_size}")
    sample_times = []
    sample_losses = []
    val_sample_times = []
    val_sample_losses = []
    total_tokens_consumed = 0  # cumulative token counter across all iterations
    for sample_id in range(sample_start_idx, max_sample_id):

        # D2 will get 2 batch each time, one for ping, the other for pong.
        # Suppose we have 
        #   as_world_size = 4
        # Then that means we implicitly have dpcp = 4
        # 1. We get 2 batch, each batch has `total_seq_len`` number of tokens
        # 2. Each GPU should get total_seq_len // as_world_size number of tokens. 
        

        dp_size = as_world_size

        model_config = hf_config
        parallel_config = ParallelConfig(
            tensor_model_parallel_size=tp_size,
            pipeline_model_parallel_size=1,
        )

        if rank == 0:
            try:
                _seq_lens: list[list[int]]
                _batch_tokens: list[Optional[List[torch.Tensor]]]
                _seq_lens, _batch_tokens = get_next_batch(batch_size * 2, iterated_samples)
            except StopIteration:
                _seq_lens, _batch_tokens = None, None
        else:
            _seq_lens, _batch_tokens = None, None

        payload = [_seq_lens, _batch_tokens]
        torch.distributed.broadcast_object_list(payload, src=0)
        _seq_lens, _batch_tokens = payload
        if _seq_lens is None:
            print(f"🟡 [Rank {rank}] StopIteration at sample_id={sample_id}: Ran out of batches. Total batches consumed: {len(iterated_samples)}")
            break
        # Count tokens for this iteration and update cumulative counter
        tokens_this_iter = sum(sum(batch) for batch in _seq_lens)
        total_tokens_consumed += tokens_this_iter
        
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

            
        print(f"🟡 [Rank {rank}] _seq_lens = {_seq_lens}")

        should_d2_balance_ping_pong = os.environ.get("EXPERIMENT_D2_BALANCE_PING_PONG", "0") == "1"
        if should_d2_balance_ping_pong:
            print(f"🟢 [Rank {rank}] Balancing ping pong")
            seq_lens_0, seq_lens_1 = balance_ping_pong(_seq_lens)
        else:
            print(f"🟡 [Rank {rank}] Not Balancing ping pong")
        seq_lens_0, seq_lens_1 = _seq_lens[:batch_size], _seq_lens[batch_size:]
        batch_tokens_ping = _batch_tokens[:batch_size]
        batch_tokens_pong = _batch_tokens[batch_size:]
        
        print(f"🟡 [Rank {rank}] seq_lens_0 = {seq_lens_0}")
        print(f"🟡 [Rank {rank}] seq_lens_1 = {seq_lens_1}")

        # num_batched_token_per_as_rank = tokens per as rank = tokens per batch * num batch / (as_world_size = dp_size)
        num_batched_token_per_as_rank = total_seq_len * batch_size // dp_size

        _items_0: list[Item] = batch_to_items_general(seq_lens_0, num_batched_token_per_as_rank, as_world_size, model_config)
        _items_1: list[Item] = batch_to_items_general(seq_lens_1, num_batched_token_per_as_rank, as_world_size, model_config)

        
        tolerance_factor = 0.05
        print(f"[Rank {rank}] Using tolerance factor = {tolerance_factor}")

        planner = Planner(
            world_size,
            parallel_config,
            model_config=model_config,
            tolerance_factor=tolerance_factor,
        )

        verbose = False
        fa2a_metadata_0, as_attn_metadata_0, mlp_shard_len_0 = planner.plan(
            _items_0, is_resend_qkv=resend_qkv, verbose=verbose
        )
        fa2a_metadata_1, as_attn_metadata_1, mlp_shard_len_1 = planner.plan(
            _items_1, is_resend_qkv=resend_qkv, verbose=verbose
        )


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

        doc_records_ping = build_sequence_records(seq_lens_0, batch_tokens_ping)
        doc_records_pong = build_sequence_records(seq_lens_1, batch_tokens_pong)

        rank_tokens_ping, rank_positions_ping = build_rank_shards(
            _items_0, doc_records_ping, as_world_size
        )
        rank_tokens_pong, rank_positions_pong = build_rank_shards(
            _items_1, doc_records_pong, as_world_size
        )

        input_ids_ping = rank_tokens_ping[as_rank].to(dtype=torch.long)
        input_ids_pong = rank_tokens_pong[as_rank].to(dtype=torch.long)
        position_ids_ping = rank_positions_ping[as_rank].to(dtype=torch.long)
        position_ids_pong = rank_positions_pong[as_rank].to(dtype=torch.long)

        expected_ping = int(torch.sum(mlp_shard_len_0[as_rank]).item())
        expected_pong = int(torch.sum(mlp_shard_len_1[as_rank]).item())
        if input_ids_ping.numel() != expected_ping:
            raise RuntimeError(
                f"Rank {as_rank} ping tokens mismatch: expected {expected_ping}, got {input_ids_ping.numel()}."
            )
        if input_ids_pong.numel() != expected_pong:
            raise RuntimeError(
                f"Rank {as_rank} pong tokens mismatch: expected {expected_pong}, got {input_ids_pong.numel()}."
            )

        input_ids_local = torch.cat([input_ids_ping, input_ids_pong], dim=0)
        position_ids_local = torch.cat([position_ids_ping, position_ids_pong], dim=0)

        if rank % 8 == 0:
            print(f"🟡 [Rank {rank}] [{sample_id = }] input_ids_local.shape =", input_ids_local.shape)
            print(f"🟡 [Rank {rank}] [{sample_id = }] position_ids_local.shape =", position_ids_local.shape)
            if _batch_tokens is not None and _batch_tokens[0] is not None:
                print(f"🟢 [Rank {rank}] Using REAL tokens from dataset")
            else:
                print(f"🟡 [Rank {rank}] Using RANDOM tokens (fallback)")


        microbatch = {
            "input_ids": input_ids_local,
            "position_ids": position_ids_local,
            "packed_seq_params": packed_seq_params,
        }

        microbatches = [microbatch]

        # --------------
        # Real Experiment
        # --------------

        torch.cuda.synchronize()
        torch.distributed.barrier()
        
        # Calculate the duration of the forward_backward_batch
        torch.cuda.nvtx.range_push(f"sample_{sample_id}")

        should_log_memory_during_real_experiment = (
            os.environ.get("EXPERIMENT_SHOULD_LOG_MEMORY_DURING_REAL_EXPERIMENT", "0") == "1"
        )
        log_memory_usage_ctx = None
        if should_log_memory_during_real_experiment:
            log_memory_usage_ctx = log_memory_usage_context()
            log_memory_usage_ctx.__enter__()

        write_status_log(f"Start Forward_backward_batch (sample_id={sample_id})")
        torch.cuda.synchronize()
        torch.distributed.barrier()
        start_it_time = time.time()
        log_memory_usage(f"forward_backward_batch:start(sample_id={sample_id})")
        losses_reduced, grad_norm = worker.forward_backward_batch(
            microbatches=microbatches,
            normal_forward_fn=normal_forward_fn,
            forward_only=False,
        )
        loss_value = extract_scalar_loss(losses_reduced)
        if loss_value is not None:
            write_status_log(f"Loss (sample_id={sample_id}) = {loss_value:.6f}")
            write_loss_log(loss_value, sample_id=sample_id)
        torch.cuda.synchronize()
        torch.distributed.barrier()
        end_it_time = time.time()
        log_memory_usage(f"forward_backward_batch:done(sample_id={sample_id})")
        iteration_time = end_it_time - start_it_time
        write_status_log(f"Finish Forward_backward_batch (sample_id={sample_id})")
        if log_memory_usage_ctx is not None:
            log_memory_usage_ctx.__exit__(None, None, None)
        torch.cuda.nvtx.range_pop()
        
        torch.cuda.synchronize()
        torch.distributed.barrier()
        avg_duration_ms = iteration_time * 1000
        sample_times.append(avg_duration_ms)
        sample_losses.append(loss_value)

        # -----------------
        # Validation step
        # -----------------
        val_loss_value = None
        if val_every_n_steps > 0 and ((sample_id + 1) % val_every_n_steps == 0):
            write_status_log(f"Start Validation (sample_id={sample_id})")
            torch.cuda.synchronize()
            torch.distributed.barrier()
            val_start_it_time = time.time()
            log_memory_usage(f"validation:start(sample_id={sample_id})")
            torch.cuda.nvtx.range_push(f"val_sample_{sample_id}")
            val_losses_reduced, _ = worker.forward_backward_batch(
                microbatches=microbatches,
                normal_forward_fn=normal_forward_fn,
                forward_only=True,
            )
            val_loss_value = extract_scalar_loss(val_losses_reduced)
            if val_loss_value is not None:
                write_status_log(f"Validation Loss (sample_id={sample_id}) = {val_loss_value:.6f}")
                # Use repeat_idx=-1 to distinguish validation entries in loss.log
                _write_loss_log(output_dir, val_loss_value, sample_id=sample_id, repeat_idx=-1)
            torch.cuda.synchronize()
            torch.distributed.barrier()
            val_end_it_time = time.time()
            log_memory_usage(f"validation:done(sample_id={sample_id})")
            torch.cuda.nvtx.range_pop()

            val_iteration_time = val_end_it_time - val_start_it_time
            val_avg_duration_ms = val_iteration_time * 1000
            val_sample_times.append(val_avg_duration_ms)
            val_sample_losses.append(val_loss_value)
            write_status_log(f"Finish Validation (sample_id={sample_id})")
        
        # Print loss from all ranks if enabled
        wandb_driver.print_loss(
            sample_id=sample_id,
            loss=loss_value,
            rank=rank,
            allow_all_ranks=allow_all_ranks_loss,
        )
        
        # Log to wandb if enabled (only on rank 0). Attach validation loss if available.
        extra_metrics = {
            "tokens_this_iter": tokens_this_iter,
            "tokens_consumed": total_tokens_consumed,
        }
        if val_loss_value is not None:
            extra_metrics["val_loss"] = float(val_loss_value)

        wandb_driver.log(
            sample_id=sample_id,
            duration_ms=avg_duration_ms,
            iteration_time_ms=iteration_time * 1000,
            loss=loss_value,
            rank=rank,
            **extra_metrics,
        )
        
        if rank == 0:
            print(f"[Sample ID=({sample_id})] Mode={mode} forward_backward_batch: avg_time_per_iteration = {avg_duration_ms:.2f} ms")
        device = torch.cuda.current_device()
        
        if rank % 8 == 0:
            (
                allocated_cur, 
                allocated_peak, 
                total_alloc
            ) = d2.mem.get_torch_cuda_memory_usage(device)
            pynvml_gpu_memory_usage = d2.mem.get_pynvml_gpu_memory_usage(device)
            print(f"Ⓜ️Ⓜ️ [Sample ID=({sample_id})] Memory usage: allocated_cur: {(allocated_cur/1024):.2f} GB, allocated_peak: {(allocated_peak/1024):.2f} GB, total_alloc: {(total_alloc/1024):.2f} GB, pynvml_gpu_memory_usage: {(pynvml_gpu_memory_usage/1024):.2f} GB")
            
            # Log memory usage to wandb with separate step
            wandb_driver.log_memory(
                allocated_cur_gb=allocated_cur,
                allocated_peak_gb=allocated_peak,
                total_alloc_gb=total_alloc,
                pynvml_gpu_memory_usage_gb=pynvml_gpu_memory_usage,
                rank=rank,
            )
            

        time.sleep(2) # to ensure the profile sees a better profiling result
        torch.cuda.synchronize()
        torch.distributed.barrier()

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

    # ------------------------------------------------------------------
    # Collect attention start/end timestamps and emit summary logs
    # ------------------------------------------------------------------
    from datetime import datetime
    pst = pytz.timezone('US/Pacific')
    timestamp = datetime.now(pst).strftime("%Y-%m-%d %H:%M:%S PST")
    now_ts = datetime.now(pst).strftime("%Y%m%d_%H%M%S")
    

    if rank % 8 == 0:

        summary_log_file = os.path.join(output_dir, "summary.log")
        with open(summary_log_file, "w") as f:
            f.write("===============Summary Log===============\n")

        def log_to_console_and_file(*args, **kwargs):
            print(*args, **kwargs)
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
            loss = sample_losses[idx] if idx < len(sample_losses) else None
            # log_to_console_and_file(f"🟢 Sample {idx}: duration: {duration:.2f} ms, samples = {samples}")
            if loss is not None:
                log_to_console_and_file(f"🟢 Sample {idx}: duration: {duration:.2f} ms, loss: {loss:.6f}")
            else:
                log_to_console_and_file(f"🟢 Sample {idx}: duration: {duration:.2f} ms, loss: N/A")
            benchmark_data["samples"].append({
                "sample_id": idx,
                "samples": samples,
                "duration_ms": duration,
                "loss": float(loss) if loss is not None else None
            })
    # ------------------------------------------------------------------
    # Write benchmark results to file
    # ------------------------------------------------------------------
    if rank == 0:
        file_dir = os.path.dirname(os.path.abspath(__file__))
        # Save another copy of the benchmark data to the output directory
        output_file = os.path.join(output_dir, "benchmark.json")
        with open(output_file, 'w') as f:
            json.dump(benchmark_data, f, indent=2)
        
        print(f"🟢 Benchmark results saved to: {output_file}")
    
    # ------------------------------------------------------------------
    # Cleanup and exit
    # ------------------------------------------------------------------
    
    # Finish wandb run if enabled
    wandb_driver.finish(rank=rank)

    # ------------------------------------------------------------------
    # Save final checkpoint (per-rank sharded weights) if requested
    # ------------------------------------------------------------------
    ckpt_root = os.environ.get("CKPT_DIR", "/mnt/sharefs/users/yonghao.zhuang/d2-logs/ckpts")
    if ckpt_root:
        try:
            # Each run gets its own subdirectory named after the output_dir basename
            run_name = os.path.basename(os.path.abspath(output_dir.rstrip("/")))
            ckpt_dir = os.path.join(ckpt_root, run_name)
            if rank == 0:
                os.makedirs(ckpt_dir, exist_ok=True)
            torch.distributed.barrier()

            # Save this rank's model shard and minimal metadata
            model_to_save = unwrap_model(worker.train_module[0])
            ckpt_path = os.path.join(ckpt_dir, f"rank{rank}_final.pt")
            ckpt_obj = {
                "model_state_dict": model_to_save.state_dict(),
                "optimizer_state_dict": worker.optimizer.state_dict() if hasattr(worker, "optimizer") else None,
                "hf_config": getattr(worker, "hf_config", None),
                "tf_config": getattr(worker, "tf_config", None),
                "args": vars(args),
                "total_tokens_consumed": total_tokens_consumed,
                "max_sample_id": max_sample_id,
            }
            torch.save(ckpt_obj, ckpt_path)
            if rank == 0:
                write_status_log(f"Saved final checkpoints to {ckpt_dir}")
                print(f"🟢 [Rank {rank}] Saved final checkpoint shard to: {ckpt_path}")
        except Exception as e:
            if rank == 0:
                write_status_log(f"Failed to save checkpoint to CKPT_DIR={ckpt_root}: {e}")
                print(f"⚠️ [Rank {rank}] Failed to save checkpoint: {e}")

    
    print(f"❄️ [Rank {rank}] Finished test and exit.")        
    write_status_log(f"Finish test and exit.")
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, choices=["d2"], default="d2", 
                        help="Test mode: currently only supports 'd2' for balanced flops planning")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument(
        "--val-every-n-steps",
        type=int,
        default=1,
        help="Evaluate validation loss every N training steps (set to 0 to disable validation).",
    )
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
    parser.add_argument("--should-resend-qkv", action="store_true", help="Whether to resend qkv in the backward pass")
    parser.add_argument("--output-dir", type=str, default=None)   
    parser.add_argument("--sample-start-idx", type=int, default=0, help="Start index of the sample ids to sample") 
    parser.add_argument("--change-long-doc-ratio", type=float, default=0.0, help="Ratio of long docs to change")
    parser.add_argument("--sample-name", type=str, default="wlbllm", 
                        help="Name of the sample/dataset to use. Use 'bookcorpus', 'wikitext', 'openwebtext', or 'c4' for real datasets with actual tokens.", 
                        choices=["wlbllm", "prolong", "bookcorpus", "wikitext", "openwebtext", "c4"])
    parser.add_argument("--alpha-factor", type=float, default=1.0, help="Alpha factor for memory imbalance")
    parser.add_argument(
        "--max-total-tokens",
        type=int,
        default=None,
        help=(
            "Optional global token budget for the data loader. If set, the real-data loader "
            "in training_utils will stop tokenizing once this many tokens have been produced "
            "from the underlying dataset, even if max-sample-id has not been reached."
        ),
    )
    parser.add_argument("--enable-wandb", action="store_true", help="Enable Weights & Biases logging (or set ENABLE_WANDB=1)")
    parser.add_argument("--wandb-project", type=str, default="d2-training", help="Wandb project name (or set WANDB_PROJECT env var)")
    parser.add_argument("--wandb-run-name", type=str, default=None, help="Wandb run name (or set WANDB_RUN_NAME env var). Set WANDB_API_KEY for authentication.")
    parser.add_argument("--allow-all-ranks-loss", action="store_true", help="Allow all ranks to output loss values (or set ALLOW_ALL_RANKS_LOSS=1)")
    
    args = parser.parse_args()
    print(f"🟡 Args: {args}")

    # "D2_SKIP_FLOAT_CONVERSION"
    os.environ["D2_SKIP_FLOAT_CONVERSION"] = "1"

    if args.output_dir is None:
        args.output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
        pass
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    
    start_time = time.time()
    
    try:
        main(args)
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise e
    log_memory_usage("test:end", force=True)
    
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"🕛 Test elapsed time: {elapsed_time:.2f} seconds")

