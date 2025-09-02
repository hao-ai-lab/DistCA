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
def timeout_handler(signum, frame):
    raise TimeoutError("forward_backward_batch operation timed out after 5 minutes")


from megatron.core import mpu
from megatron.core.optimizer import get_megatron_optimizer
from megatron.core.pipeline_parallel.schedules import get_forward_backward_func
from megatron.core.packed_seq_params import PackedSeqParams
from omegaconf import OmegaConf
import torch
from transformers import AutoConfig, AutoTokenizer, AutoProcessor

from d2.runtime.attn_kernels.ops import FastDispatcherWrapper
from d2.runtime.megatron_patch.packed_seq_params import arg_to_cuda, PingPangPackedSeqParams
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

def debug_print(*args, **kwargs):
    if os.getenv("D2_DEBUG_PRINT", "0") == "1":
        if torch.distributed.is_initialized():
            rank = torch.distributed.get_rank()
            rich.print(f"[Rank {rank}]", *args, **kwargs)
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

        add_optimization_config_to_tf_config(tf_config)

        if self.rank == 0:
            print(f"TF config: {tf_config}")
        self.hf_config = hf_config
        self.tf_config = tf_config

    def forward_backward_batch(self, microbatches: list[dict], forward_only: bool=False, normal_forward_fn: bool=False):
        # TODO: for PP, since backward has a different attention layout dispatching order,
        # we should modify the forward_backward_func here.

        microbatches = [{
            k: arg_to_cuda(v) for k, v in microbatches[0].items()
        }]
        for module in self.train_module:
            unwrap_model(module).set_debug(normal_forward_fn)
        assert len(self.train_module) == 1, "only support one module"

        forward_backward_func = get_forward_backward_func()
        n_micro_batch = len(microbatches)
        # thd layout
        total_seqlen = microbatches[0]['input_ids'].shape[0]

        def loss_func(logits):
            loss = logits.sum()  # no gradient, but can trigger backward
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
            return output, loss_func

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

        with torch.cuda.nvtx.range("optimizer_step"):
            update_successful, grad_norm, num_zeros_in_grad = self.optimizer.step()
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
    worker = init_worker_torch_distributed(
        world_size, buffer_size, worker_cls, parallel_config
    )
    print("Communication groups initialized")
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
    return worker


from typing import Iterable, List, Optional
from d2.simulator.optimizers.samples import sample_wlbllm_docs_upsample, batch_documents

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


# ========== D2 Specific Functions ==========

# from transformer_engine.pytorch.attention.dot_product_attention.backends import get_attention_duration
try:
    import wlbllm
    import wlbllm.utils
    import wlbllm.registry
except ImportError:
    print("""⚠️ WLBLLM is not installed. This only affects if you're testing WLBLLM tests. To install:

    cd d2/baseline/wlbllm_original
    pip install -e .
    """)
    pass


def test(args):
    seed = args.seed
    batch_size = args.batch_size
    num_tokens = args.num_tokens
    cp_degree = max_cp_degree = args.cp_degree
    num_seqs = args.num_seqs
    tp_size = args.tp_size
    world_size = args.num_nodes * args.num_gpus_per_node
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
    if num_layers is not None:
        os.environ["NUM_LAYERS"] = str(num_layers)

    mode = args.mode
    output_dir = args.output_dir

    # Set forward function mode based on test mode
    normal_forward_fn = (mode in ["baseline", "wlbllm"])
    # TODO: (Refactor) If WLBLLM is set, we must inform the transformer_engine to use the WLBLLM function. 
    os.environ["WLBLLM_MODE"] = "1" if mode == "wlbllm" else "0"


    if mode == "wlbllm":
        import wlbllm.megatron_patch.dot_product_attention
        wlbllm.megatron_patch.dot_product_attention.monkey_patch()
        import wlbllm.megatron_patch.backends
        wlbllm.megatron_patch.backends.monkey_patch()
        pass

    dtype = torch.bfloat16
    element_size = dtype.itemsize

    
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

    worker.set_config(dtype=dtype)
    worker.init(model_path, seed=seed)
    rich.print(f"🟡 [Rank {worker.rank}] init done")
    # set again to potentially adapt to the ray launch case.
    set_random_seed(seed, set_megatron=False)

    # parallel_config = worker.parallel_config

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
    )

    max_sample_id = max_sample_id
    sample_times = []
    for sample_id in range(max_sample_id):
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
            
            # Balance the data here for WLBLLM.
            # TODO: This only works for DP+CP.
            # ENABLE_BALANCED_FLOS_NO_DEFER = False
            ENABLE_BALANCED_FLOS_NO_DEFER = True
            if ENABLE_BALANCED_FLOS_NO_DEFER and dp_size > 1:
                # how many tokens per dp replicate (with cp) can hold?
                # max_seq_len_without_cp * cp_size * 2 (ping pong)
                Lmax = total_seq_len * 2 * batch_size // dp_size
                # Lmax = total_seq_len * 2 * batch_size // cp_size
                rich.print(f"🟡 [Rank {rank}] Lmax = {Lmax}")

                all_docs = flatten(_seq_lens)
                all_docs.sort(reverse=True)
                new_batch = []
                for r in range(dp_size):
                    new_batch.append([])
                
                def get_workload(micro_batch: list[int]) -> int:
                    # TODO: Fix this get_workload function to calculate the `breakpoint` of a model.
                    a = [ i / (64 * K) for i in micro_batch]
                    return sum(i ** 2 + i for i in a)

                def get_length(micro_batch: list[int]) -> int:
                    return sum(micro_batch)

                # Step 1: Pack the docs into the new batch.
                remained_docs = []
                for doc in all_docs:
                    workloads = [get_workload(batch) for batch in new_batch]
                    lengths = [get_length(batch) for batch in new_batch]
                    min_workload_idx = workloads.index(min(workloads))
                    min_length_idx = lengths.index(min(lengths))
                    
                    if lengths[min_workload_idx] + doc <= Lmax:
                        new_batch[min_workload_idx].append(doc)
                    else:
                        if lengths[min_length_idx] + doc <= Lmax:
                            new_batch[min_length_idx].append(doc)
                        else:
                            remained_docs.append(doc)
                    pass

                # Step 2: Pack the remained docs, by workload.
                for doc in remained_docs:
                    workloads = [get_workload(batch) for batch in new_batch]
                    lengths = [get_length(batch) for batch in new_batch]
                    min_workload_idx = workloads.index(min(workloads))
                    new_batch[min_workload_idx].append(doc)

                if rank == 0:
                    rich.print(f"🟡 [Rank {rank}] WLBLLM Reordered Batch: new_batch = {new_batch}")

                modified_batches.append(new_batch)

                
                # print(f"🟡 [Rank {rank}] Before - assigning seq_lens={_seq_lens}")
                # print(f"🟡 new_batch={new_batch}")
                seq_lens = [new_batch[dp_rank]]
                # print(f"🟡 [Rank {rank}] Taking seq_lens={seq_lens}")

            else:
                seq_lens = _seq_lens[
                    dp_rank * cp_size * 2: 
                    (dp_rank + 1) * cp_size * 2
                ]


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
            
            debug_print(f"doc_lens", doc_lens)
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
            
            if rank % 8 == 0:
                # debug_print(f"doc_lens", doc_lens)
                # debug_print(f"doc_shards", doc_shards)
                debug_print(f"cu_seqlens_q_list", cu_seqlens_q_list)
                debug_print(f"cu_seqlens_k_list", cu_seqlens_k_list)
            #     debug_print(f"max_seqlen_q_list", max_seqlen_q_list)
            #     debug_print(f"max_seqlen_k_list", max_seqlen_k_list)
            #     debug_print(f"kv_idx_list", kv_idx_list)
            # exit(0)

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
            wlbllm.registry.set("global_tensor_length", (total_seq_len * cp_size * 2))

        
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
            seq_lens_0: list[list[int]] = _seq_lens[:batch_size]
            seq_lens_1: list[list[int]] = _seq_lens[batch_size:]
            rich.print(f"🟡 [Rank {rank}] _seq_lens = {_seq_lens}")

            # num_batched_token_per_as_rank = tokens per as rank = tokens per batch * num batch / (as_world_size = dp_size)
            num_batched_token_per_as_rank = total_seq_len * batch_size // dp_size

            _items_0: list[Item] = batch_to_items_general(seq_lens_0, num_batched_token_per_as_rank, as_world_size, model_config)
            _items_1: list[Item] = batch_to_items_general(seq_lens_1, num_batched_token_per_as_rank, as_world_size, model_config)

            planner = Planner(world_size, parallel_config, model_config=model_config)
            
            fa2a_metadata_0, as_attn_metadata_0, mlp_shard_len_0 = planner.plan(_items_0)
            fa2a_metadata_1, as_attn_metadata_1, mlp_shard_len_1 = planner.plan(_items_1)


            if rank % 8 == 0:
                qkv_fwd_fa2a_metadata__send_transfer_sz_mb = fa2a_metadata_0[0].fa2a_metadata[1] // 1024 // 1024
                qkv_fwd_fa2a_metadata__recv_transfer_sz_mb = fa2a_metadata_0[0].fa2a_metadata[3] // 1024 // 1024
                attn_out_fwd_fa2a_metadata__send_transfer_sz_mb = fa2a_metadata_0[1].fa2a_metadata[1] // 1024 // 1024
                attn_out_fwd_fa2a_metadata__recv_transfer_sz_mb = fa2a_metadata_0[1].fa2a_metadata[3] // 1024 // 1024
                        
                # Print qkv_fwd_fa2a_metadata
                rich.print(f"🟡 [Rank {rank}] qkv_fwd_fa2a_metadata.send_transfer_sz_mb = ", qkv_fwd_fa2a_metadata__send_transfer_sz_mb)
                rich.print(f"🟡 [Rank {rank}] qkv_fwd_fa2a_metadata.recv_transfer_sz_mb = ", qkv_fwd_fa2a_metadata__recv_transfer_sz_mb)
                
                # Print attn_out_fwd_fa2a_metadata
                rich.print(f"🟡 [Rank {rank}] attn_out_fwd_fa2a_metadata.send_transfer_sz_mb = ", attn_out_fwd_fa2a_metadata__send_transfer_sz_mb)
                rich.print(f"🟡 [Rank {rank}] attn_out_fwd_fa2a_metadata.recv_transfer_sz_mb = ", attn_out_fwd_fa2a_metadata__recv_transfer_sz_mb)
            # Check size:
            def _check_overflow(fa2a_metadata):
                send_sz = [torch.sum(m.fa2a_metadata[1][as_rank]).item() for m in fa2a_metadata]
                # send_sz + sender_recv_offset = sender_recv_last_token
                send_last_offset = [(m.fa2a_metadata[1] + m.fa2a_metadata[2])[as_rank] for m in fa2a_metadata]
                recv_sz = [torch.sum(m.fa2a_metadata[3][as_rank]).item() for m in fa2a_metadata]
                max_send_sz = max(send_sz)
                max_recv_sz = max(recv_sz)
                buffer_size = FastDispatcherWrapper.instance[0].buffer_size
                assert buffer_size >= max_send_sz and buffer_size >= max_recv_sz, f"{buffer_size / 1024**3} GB buffer, {
                    [s / 1024**3 for s in send_sz]} GB send sizes, {
                    [sz / 1024**3 for sz in recv_sz]} GB recv sizes"
                assert max(torch.max(o).item() for o in send_last_offset) <= buffer_size, f"{buffer_size / 1024**3} GB buffer, {[o / 1024**3 for o in send_last_offset]} GB send last offsets"
                
                
                rich.print(f"🟡 [Rank {rank}] Overflow check passed: {max_send_sz} GB, {max_recv_sz} GB recv size, {max(torch.max(o).item() for o in send_last_offset)} GB send last offset, {buffer_size / 1024**3} GB buffer size")
            
            _check_overflow(fa2a_metadata_0)
            rich.print("🟡 [Rank {rank}] Overflow check passed for fa2a_metadata_0")
            _check_overflow(fa2a_metadata_1)
            rich.print("🟡 [Rank {rank}] Overflow check passed for fa2a_metadata_1")


            # params for ping-pong batch0
            ping_pang_params_0 = get_single_step_packed_seq_params(
                fa2a_metadata_0, as_attn_metadata_0, as_rank
            )
            # params for ping-pong batch1
            ping_pang_params_1 = get_single_step_packed_seq_params(
                fa2a_metadata_1, as_attn_metadata_1, as_rank
            )

            mlp_seq_params_0 = get_attn_metadata(mlp_shard_len_0[as_rank], get_packed_seq_params=True)
            mlp_seq_params_1 = get_attn_metadata(mlp_shard_len_1[as_rank], get_packed_seq_params=True)
            
            # old approach to generate metadata.
            # ping_pang_params_0, mlp_seq_params_0 = items_to_metadata(_items_0)
            # ping_pang_params_1, mlp_seq_params_1 = items_to_metadata(_items_1)

            if rank % 8 == 0:
                rich.print(f"🟡 [Rank {rank}] ping_pang_params_0.qkv_fwd_metadata =", ping_pang_params_0.qkv_fwd_metadata.__better_print__())
                rich.print(f"🟡 [Rank {rank}] ping_pang_params_1.qkv_fwd_metadata =", ping_pang_params_1.qkv_fwd_metadata.__better_print__())
                rich.print(f"🟡 [Rank {rank}] mlp_seq_params_0 =", mlp_seq_params_0)
                rich.print(f"🟡 [Rank {rank}] mlp_seq_params_1 =", mlp_seq_params_1)

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
                rich.print(f"🟡 [Rank {rank}] input_ids_local.shape =", input_ids_local.shape)
                rich.print(f"🟡 [Rank {rank}] position_ids_local.shape =", position_ids_local.shape)

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
            # Warmup
            warmup_times = 5
            try:
                warmup_times = int(os.environ.get("EXPERIMENT_WARMUP_TIMES", 5))
            except:
                pass

            warmup_timeout_sec = 60
            try:
                warmup_timeout_sec = int(os.environ.get("EXPERIMENT_WARMUP_TIMEOUT_SEC", 60))
            except:
                warmup_timeout_sec = 60

            # Test passing the nvshmem init
            try:
                signal.signal(signal.SIGALRM, timeout_handler)
                signal.alarm(warmup_timeout_sec)  # 60 seconds = 1 minute
                ref = worker.forward_backward_batch(
                    microbatches=microbatches,
                    normal_forward_fn=normal_forward_fn,
                    forward_only=False,
                )
                signal.alarm(0)
            except TimeoutError as e:
                print("🔴 Timeout at the first warmup forward_backward function. It may suggest our all2all kernel failed.")
                sys.exit(1)

            
            for warmup_idx in range(warmup_times):
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
        
        # Real Experiment
        N = 3
        
        try:
            N = int(os.environ.get("EXPERIMENT_REPEAT_TIMES", 3))
        except:
            N = 3

        torch.cuda.synchronize()
        
        # Calculate the average duration of the forward_backward_batch
        start_time = time.time()
        torch.cuda.nvtx.range_push(f"sample_{sample_id}(repeat={N})")
        for _ in range(N):
            ref = worker.forward_backward_batch(
                microbatches=microbatches,
                normal_forward_fn=normal_forward_fn,
                forward_only=False,
            )
        torch.cuda.nvtx.range_pop()
        
        torch.cuda.synchronize()
        torch.distributed.barrier()
        end_time = time.time()
        duration = end_time - start_time
        duration_ms = duration * 1000
        avg_duration_ms = duration_ms / N
        sample_times.append(avg_duration_ms)
        if rank == 0:
            if mode == "baseline":
                rich.print(f"[Sample ID=({sample_id})] seq_lens = {seq_lens}")
            rich.print(f"[Sample ID=({sample_id})] Mode={mode} forward_backward_batch: avg_time_per_iteration = {avg_duration_ms:.2f} ms")
            

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
    
    file_dir = os.path.dirname(os.path.abspath(__file__))
    benchmark_dir = os.path.join(file_dir, "..", "benchmarks", "_250809_e2e_benchmark", "data")
    os.makedirs(benchmark_dir, exist_ok=True)
    benchmark_file = os.path.join(benchmark_dir, f"benchmark.{now_ts}.{mode}.json")
    
    # rank = torch.distributed.get_rank()
    # with open(attn_output_file, "w") as f:
    #     attention_durations = get_attention_duration()
    #     json.dump(attention_durations, f)
    #     formatted_durations = [f"{duration:.2f}" for duration in attention_durations]
    #     rich.print(f"🟢 Attention durations: {formatted_durations}")

    if rank % 8 == 0:

        summary_log_file = os.path.join(output_dir, "summary.log")
        with open(summary_log_file, "w") as f:
            f.write("Summary Log\n===============\n")

        def log_to_console_and_file(*args, **kwargs):
            rich.print(*args, **kwargs)
            with open(summary_log_file, "a") as f:
                rich.print(*args, **kwargs, file=f)

        
        log_to_console_and_file(f"🟢 Test {__file__} passed")
        
        config = dict(
            mode=mode, tp_size=tp_size, dp_size=dp_size, cp_size=cp_degree, 
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
            # total_flops_factor = 
            log_to_console_and_file(f"🟢 Sample {idx}: duration: {duration:.2f} ms, samples = {samples}")
            benchmark_data["samples"].append({
                "sample_id": idx,
                "samples": samples,
                "duration_ms": duration
            })
        
        # Write benchmark results to file
        # TODO: Make the output directory configurable.
        
    if rank == 0:
        with open(benchmark_file, 'w') as f:
            json.dump(benchmark_data, f, indent=2)

        # Save another copy of the benchmark data to the output directory
        output_file = os.path.join(output_dir, "benchmark.json")
        with open(output_file, 'w') as f:
            json.dump(benchmark_data, f, indent=2)
        
        rich.print(f"🟢 Benchmark results saved to: {output_file}")

        # for idx, (sample, duration) in enumerate(zip(iterated_samples, sample_times)):
        #     rich.print(f"🟢 Sample {idx}: {sample}, duration: {duration} ms")

    
    # Cleanup and exit
    rich.print(f"❄️ [Rank {rank}] Finished test and exit.")
    
    
    # if False: # Only use it when force exit
    if args.force_exit: 
        print(f"[Rank {rank}] Starting aggressive cleanup process...")
        os._exit(0)


from torch.profiler import profile, record_function, ProfilerActivity

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, choices=["baseline", "d2", "wlbllm"], default="baseline", 
                        help="Test mode: 'baseline' for simple batch generation, 'd2' for balanced flops planning, 'wlbllm' for wlbllm")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-tokens", type=int, default=1024)
    parser.add_argument("--cp-degree", type=int, default=2)
    parser.add_argument("--num-seqs", type=int, default=3)
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
    parser.add_argument("--output-dir", type=str, default=None)
    
    args = parser.parse_args()

    if args.output_dir is None:
        args.output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
        pass
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
        

    should_profile_memory = args.should_profile_memory
    if should_profile_memory:
        torch.cuda.memory._record_memory_history()
        pass

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
                pass
    else:
        test(args)
    
    if should_profile_memory:
        mode = args.mode
        batch_size = args.batch_size
        num_tokens = args.num_tokens
        cp_degree = args.cp_degree
        tp_size = args.tp_size
        num_layers = args.num_layers

        rank = torch.distributed.get_rank()
        if rank % 8 == 0:
            print("Dumping memory snapshot")

            mem_snapshots_dir = os.path.join(args.output_dir, "mem_snapshots")
            os.makedirs(mem_snapshots_dir, exist_ok=True)
            
            stem = f"{now_ts}.mem_snapshot.{mode}.rank{rank}.batch{batch_size}.tokens{num_tokens}.cp{cp_degree}.tp{tp_size}.layers{num_layers}"
            mem_snapshot_output_path = os.path.join(mem_snapshots_dir, f"{stem}.pickle")
            memory_timeline_output_path = os.path.join(mem_snapshots_dir, f"{stem}.html")


            now_ts = get_current_timestamp()
            torch.cuda.memory._dump_snapshot(mem_snapshot_output_path)
            prof.export_memory_timeline(memory_timeline_output_path, device=torch.cuda.current_device())
            print("Memory snapshot dumped")

