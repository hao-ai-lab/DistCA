"""
Debug example:
NVTE_ALLOW_NONDETERMINISTIC_ALGO=0 torchrun --nnodes 1 --nproc_per_node 2 test_megatron_e2e_pipeline.py --num-gpus-per-node 2 --pp-size 2 --num-microbatch 2
"""


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



import argparse
from functools import partial
import os
import time
import json
import traceback
from typing import Any

from transformers import AutoTokenizer, AutoProcessor


from contextlib import nullcontext


import megatron.core.parallel_state as mpu
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.pipeline_parallel.schedules import get_forward_backward_func
import torch
from transformers import AutoConfig

from d2.runtime.compute_metadata import get_attn_metadata
from d2.runtime.megatron.packed_seq_params import arg_to_cuda, PingPangSingleStepPackedSeqParams, PingPangPackedSeqParams
from d2.runtime.megatron.forward_backward_func import forward_backward_pipelining_without_interleaving as forward_backward_func

from test_util import ParallelConfig, init_worker_torch_distributed, create_qkv_dispatch_pipeline_tick
from test_megatron_e2e import MegatronE2eWorker as BaseMegatronE2eWorker, set_random_seed
from megatron_test_utils import (
    gptmodel_forward, make_batch_generator, unwrap_model,
    hf_to_mcore_config, make_batch_generator, update_model_config,
)
from typing import Optional

import d2.planner.wlb_planner
import wlbllm
import wlbllm.utils
import wlbllm.registry
import wlbllm.megatron_patch
import wlbllm.megatron_patch.dot_product_attention
import wlbllm.megatron_patch.backends
import wlbllm.megatron_patch.pp_schedules
from dataclasses import dataclass

from d2.utils.traceback import enable_clickable_excepthook, clickable_excepthook
from d2.mem import set_memory_usage_log_file, log_memory_usage, log_memory_usage_context, enable_memory_usage_logging

enable_clickable_excepthook()

@dataclass
class WLBPackedSeqParams():
    qkv_format: str = "thd"
    cu_seqlens_q: 'Tensor' = None
    cu_seqlens_kv: 'Tensor' = None
    cu_seqlens_q_padded: 'Tensor' = None
    cu_seqlens_kv_padded: 'Tensor' = None
    max_seqlen_q: 'Tensor' = None
    max_seqlen_kv: 'Tensor' = None
    wlb_metadata: dict = None
    pass


class MegatronE2eWorker(BaseMegatronE2eWorker):
    def __init__(self, rank: int, world_size: int):
        super().__init__(rank, world_size)
        local_rank = int(os.getenv("LOCAL_RANK"))
        torch.cuda.set_device(local_rank)
        torch.set_default_device(torch.device("cuda", local_rank))

    def init_comm(self, parallel_config: ParallelConfig):
        # Init megatron communication.
        self.init_torch_distributed()
        # NOTE: do not set to local_rank here because the cuda visible device is set by ray.
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
        )
        self.as_world_size = None
        self.as_rank = None

    def forward_backward_batch(self, microbatches: list[dict], forward_only: bool=False,
                               mode: str="ping_pong", with_dummy: bool=True):

        microbatches = [{
            k: arg_to_cuda(v) for k, v in microbatch.items()
        } for microbatch in microbatches]
        if "orig" in mode:
            for mb in microbatches:
                psp = mb["packed_seq_params"]
                if isinstance(psp, PingPangSingleStepPackedSeqParams):
                    mb["packed_seq_params"] = mb["packed_seq_params"].mlp_packed_seq_params
                else:
                    assert isinstance(psp, (PackedSeqParams, WLBPackedSeqParams))

        # forward_backward_func = get_forward_backward_func()
        pp_size = self.tf_config.pipeline_model_parallel_size
        pp_rank = mpu.get_pipeline_model_parallel_rank()
        # n_micro_batch = len(microbatches) - pp_size + 1
        n_micro_batch = len(microbatches)
        print(f"🟡 n_micro_batch: {n_micro_batch}")
        # thd layout
        # TODO: (FIXME) This is a hack to get the total sequence length, 
        # but wlbllm has variable sequence lengths.
        total_seqlen = microbatches[0]['input_ids'].shape[0]

        def loss_func(output):
            # NOTE: this is a dummy loss function.
            loss = output.mean()
            return loss, {'loss': loss}

        def forward_step(batch_iter, model):
            print(f"🟡 forward_step")
            # import traceback
            # traceback.print_stack()

            batch = next(batch_iter)
            torch.cuda.nvtx.range_push("forward_step")
            input_ids = batch['input_ids']
            position_ids = batch['position_ids']
            attention_mask = None
            packed_seq_params = batch['packed_seq_params']
            # returns "hidden_states" if not model.post_process (not the last layer)
            # returns "logits" when label is None.
            output = gptmodel_forward(
                model, input_ids, attention_mask, position_ids, self.tf_config.sequence_parallel,
                packed_seq_params, 
                # returns "hidden_states" if not model.post_process (not the last layer)
                # returns "logits" when label is None.
                labels=None, 
                # labels=input_ids.unsqueeze(0),
            )
            # print(f"🟡 output = {output}")
            # print(f"🟡 output.shape = {output.shape if output is not None else None}")
            torch.cuda.nvtx.range_pop()
            return output, loss_func

        def dummy_backward_step(model, dummy_bwd_iter, skip: bool):
            next_iter_args = next(dummy_bwd_iter)
            if skip:
                return
            unwrap_model(model).dummy_backward(next_iter_args)

        assert len(self.train_module) == 1, "only support one module"

        # shift bwd metadata since the order it runs is different from the
        # corresponding dummy forward's.
        dummy_bwd_packed_seq_params = [
            microbatch['packed_seq_params'] for microbatch in
            (microbatches[-pp_size + pp_rank + 1:][:pp_size - pp_rank - 1] + microbatches[:pp_rank])
        ]
        dummy_bwd_packed_seq_params = dummy_bwd_packed_seq_params[pp_rank:] + dummy_bwd_packed_seq_params[:pp_rank]

        assert mode in ["ping_pong", "orig_reimpl", "single_sided"]

        for module in self.train_module:
            debug = (mode != "ping_pong")
            debug_fwd_impl = mode if debug else None
            unwrap_model(module).set_debug(debug=debug, debug_fwd_impl=debug_fwd_impl)
            unwrap_model(module).train()
        assert len(self.train_module) == 1, "only support one module"

        dummy_bwd_packed_seq_params_iter = iter(dummy_bwd_packed_seq_params)
        # batch_generator = make_batch_generator(
        #     microbatches if with_dummy else microbatches[pp_rank:],
        #     vpp_size=len(self.train_module)
        # )
        batch_generator = make_batch_generator(
            microbatches, 
            vpp_size=len(self.train_module)
        )
        # if mpu.get_pipeline_model_parallel_world_size() > 1:

        torch.cuda.synchronize()
        from d2.runtime.attn_kernels.ops import nvshmem_barrier_all
        nvshmem_barrier_all()
        if with_dummy:
            raise NotImplementedError("Dummy backward is not implemented for WLBLLM")
            
        # Set the sequence lengths for all microbatches
        seq_lengths = [mb['input_ids'].shape[0] for mb in microbatches]
        print(f"🟡 Setting microbatch sequence lengths: {seq_lengths}")
        from wlbllm.megatron_patch.pp_schedules import set_microbatch_seq_lengths
        set_microbatch_seq_lengths(seq_lengths)
        
        # orig_fwd_backward_func = get_forward_backward_func()
        import wlbllm.megatron_patch.pp_schedules
        orig_fwd_backward_func = wlbllm.megatron_patch.pp_schedules.forward_backward_pipelining_without_interleaving
        # print(f"🟡 orig_fwd_backward_func: {orig_fwd_backward_func}")
        # print(f"🟡 orig_fwd_backward_func location: {orig_fwd_backward_func.__module__}.{orig_fwd_backward_func.__name__}")

        # from d2.runtime.megatron_patch.forward_backward_func import forward_backward_pipelining_without_interleaving
        # losses_reduced = forward_backward_pipelining_without_interleaving(
        losses_reduced = orig_fwd_backward_func(
            forward_step_func=forward_step,
            data_iterator=batch_generator,
            model=self.train_module,
            num_microbatches=n_micro_batch,
            seq_length=total_seqlen,  # no use, since variable_seq_lengths=True
            micro_batch_size=1,  # no use when input_shapes was set
            forward_only=forward_only,
        )
        grad_sample = unwrap_model(self.train_module[0]).decoder.layers[-1].self_attention.linear_proj.weight.main_grad.clone()

        # when testing numerical correctness, instead of running optimizer step, reset grads.
        for tm in self.train_module:
            for param in unwrap_model(tm).parameters():
                param.main_grad.zero_()
        return losses_reduced, grad_sample

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
                tf_config.recompute_method = gradient_checkpointing_cfg.get("activations_checkpoint_method", "mlp")
                tf_config.recompute_granularity = gradient_checkpointing_cfg.get(
                    "activations_checkpoint_granularity", "selective"
                )
                tf_config.recompute_num_layers = gradient_checkpointing_cfg.get("activations_checkpoint_num_layers", -1)
                tf_config.recompute_modules = gradient_checkpointing_cfg.get("activations_checkpoint_recompute_modules", ['mlp'])

        add_optimization_config_to_tf_config(tf_config)

        if self.rank == 0:
            print(f"TF config: {tf_config}")

        self.hf_config = hf_config
        self.tf_config = tf_config


def init_megatron_e2e_test(
    world_size: int, cp_size: int, tp_size: int, pp_size: int, dp_size: int, worker_cls=MegatronE2eWorker
):
    parallel_config = ParallelConfig(
        tensor_model_parallel_size=tp_size,
        pipeline_model_parallel_size=pp_size,
        context_parallel_size=cp_size,
    )
    rank = int(os.environ.get("RANK"))
    worker = worker_cls(rank, world_size)
    worker.init_comm(parallel_config)

    dp_rank = mpu.get_data_parallel_rank()
    pp_rank = mpu.get_pipeline_model_parallel_rank()
    cp_rank = mpu.get_context_parallel_rank()
    tp_rank = mpu.get_tensor_model_parallel_rank()
    print(f"[Rank {rank}] WLBLLM Communication groups initialized: {dp_rank=}, {pp_rank=}, {cp_rank=}, {tp_rank=}, {worker.as_rank=}, {worker.as_world_size=}")
    return worker


from global_batch_provider import setup_global_batch, get_next_batch
from dataclasses import dataclass

def flatten(a):
    return [y for x in a for y in x]

def pad_doc_lens(doc_lens: list[int], cp_size: int, dp_size: int):
    if len(doc_lens) < dp_size:
        # Pad the doc_lens to dp_size
        doc_lens += [512] * (dp_size - len(doc_lens))
        pass
    # TODO(HACK): This is a hack to ensure the doc_lens is divisible by cp_size*2.
    if sum(doc_lens) % (cp_size * 2 * 8) != 0:
        sum_of_doc_lens = sum(doc_lens)
        doc_lens[-1] += (cp_size * 2 * 8) - sum_of_doc_lens % (cp_size * 2 * 8)
        # assert doc_lens[-1] > 0
        pass
    assert sum(doc_lens) % (cp_size * 2 * 8) == 0, f"sum(doc_lens)={sum(doc_lens)} must be divisible by {cp_size * 2 * 8}"
    assert sum(doc_lens) % (cp_size * 2) == 0, f"sum(doc_lens)={sum(doc_lens)} must be divisible by {cp_size * 2}"
    return doc_lens



def test(args):
    rank = os.environ.get("RANK")
    rank = int(rank)

    output_dir = args.output_dir
    benchmark_log_path = os.path.join(output_dir, "benchmark.raw.jsonl")
    benchmark_final_path = os.path.join(output_dir, "benchmark.json")

    should_profile_memory_history = False
    if should_profile_memory_history:
        torch.cuda.memory._record_memory_history()
        mem_snapshots_dir = os.path.join(output_dir, "mem_snapshots")
        os.makedirs(mem_snapshots_dir, exist_ok=True)
        print(f"🟡 Will save mem snapshots to: {mem_snapshots_dir}")
        pass
    
    log_memory_usage("test start")

    config_path = os.path.join(output_dir, "config.json")
    with open(config_path, "w") as f:
        # Namespace to dict
        args_dict = vars(args)
        json.dump(args_dict, f, indent=2)
    
    memory_usage_dir = os.path.join(output_dir, "mem-log")
    os.makedirs(memory_usage_dir, exist_ok=True)
    enable_memory_usage_logging(memory_usage_dir)

    os.environ["WLBLLM_MODE"] = "1"
    seed = args.seed

    batch_size = args.num_batches
    num_microbatch = args.num_microbatch
    
    # Setup model configs
    model_path = args.model_path
    num_layers = args.num_layers
    dtype = torch.bfloat16
    element_size = dtype.itemsize
    # TODO: (Refactor) This is a hack to set the number of layers. It should be properly set in the HuggingFace config, not here
    if num_layers is not None:
        # See `megatron_test_utils.py` for more details.
        os.environ["NUM_LAYERS"] = str(num_layers)
    
    # Setup testing scales
    num_tokens = args.num_tokens   
    max_sample_id = args.max_sample_id
    total_seq_len = args.num_tokens
    cp_size = args.cp_size
    tp_size = args.tp_size
    pp_size = args.pp_size
    world_size = args.num_nodes * args.num_gpus_per_node
    assert world_size % (tp_size * pp_size * cp_size) == 0, f"world_size: {world_size} % (tp_size * pp_size * cp_size) == 0, {world_size} % {tp_size * pp_size * cp_size} != 0"

    assert world_size == int(os.environ.get("WORLD_SIZE")), f"world_size: {world_size} != WORLD_SIZE: {os.environ.get('WORLD_SIZE')}"
    dp_size = world_size // (tp_size * pp_size * cp_size)

    config = dict(
        mode="wlbllm", 
        nodes=args.num_nodes,
        num_gpus_per_node=args.num_gpus_per_node,
        tp_size=tp_size, dp_size=dp_size, cp_size=cp_size, 
        num_tokens=num_tokens, model_path=model_path, num_layers=num_layers, 
        max_sample_id=max_sample_id, up_sample_factor=args.up_sample_factor, filter_threshold=args.filter_threshold, filter_ratio=args.filter_ratio, 
        elongate_factor=args.elongate_factor,
        sample_name=args.sample_name,
        change_long_doc_ratio=args.change_long_doc_ratio,
    )


    # Prepare files

    # Setup the get batch logic
    # - each batch will contain num_tokens tokens.
    # - if cp is specified, then this num_tokens will be on one CP group.
    setup_global_batch(
        num_tokens,
        up_sample_factor=args.up_sample_factor,
        elongate_factor=args.elongate_factor,
        filter_threshold=args.filter_threshold,
        filter_ratio=args.filter_ratio,
        should_add_debug_cases=args.should_add_debug_cases,
        change_long_doc_ratio=args.change_long_doc_ratio,
        sample_name=args.sample_name,
    )

    # for _ in range(20):
    #     print(f"🟡 get_next_batch: {get_next_batch(int(num_microbatch * batch_size * 2))}")
    # exit(0)

    # Setup the model and configuration
    try:
        # First try with local_files_only to use cached version
        hf_config = AutoConfig.from_pretrained(model_path, local_files_only=True)
    except Exception as e:
        print(f"Local cache not found for {model_path}, downloading... Error: {e}")
        # Fallback to downloading with cache_dir specified
        hf_config = AutoConfig.from_pretrained(model_path, cache_dir="./models/")
    hidden_size_q = hf_config.hidden_size

    hidden_size_kv = hidden_size_q
    if hasattr(hf_config, "num_key_value_heads"):
        hidden_size_kv = (hidden_size_kv * hf_config.num_key_value_heads //
                          hf_config.num_attention_heads)

    wlbllm.megatron_patch.dot_product_attention.monkey_patch()
    wlbllm.megatron_patch.backends.monkey_patch()

    log_memory_usage("before init_megatron_e2e_test", force=True)
    worker: MegatronE2eWorker = init_megatron_e2e_test(
        world_size, cp_size, tp_size, pp_size, dp_size, 
        MegatronE2eWorker,
    )
    log_memory_usage("after init_megatron_e2e_test", force=True)
    worker.set_config(dtype=dtype)

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
    # set again to potentially adapt to the ray launch case.
    set_random_seed(seed, set_megatron=False)


    log_memory_usage("init worker done", force=True)

    # Check rank correctness
    dp_rank = mpu.get_data_parallel_rank()
    pp_rank = mpu.get_pipeline_model_parallel_rank()
    cp_rank = mpu.get_context_parallel_rank()
    tp_rank = mpu.get_tensor_model_parallel_rank()
    cp_group = mpu.get_context_parallel_group()

    # Setup the WLBLLM registry
    wlbllm.registry.clear()
    wlbllm.registry.set("cp_group", cp_group)
    wlbllm.registry.set("cp_stream", torch.cuda.current_stream())
    wlbllm.registry.set("num_microbatch", num_microbatch)
    def swap_metadata_fn(counter: int):
        wlb_metadata = wlbllm.registry.get(counter)
        for key, value in wlb_metadata.items():
            # print(f"🟡 swap_metadata_fn[{counter}]: swapping {key}")
            wlbllm.registry.set(key, value)
        return wlb_metadata

        
    wlbllm.registry.set("swap_metadata_fn", swap_metadata_fn)

    # Setup the microbatches
    final_durations_ms = []
    all_seq_lens = []
    for sample_idx in range(max_sample_id):

        # Get the unbalanced microbatches from the data loader
        ENABLE_BALANCED_FLOS_NO_DEFER = True

        batch_size_x2 = int(batch_size * 2)

        if ENABLE_BALANCED_FLOS_NO_DEFER:
            
            unbalanced_micro_batches = get_next_batch(batch_size_x2 * num_microbatch)
            print(f"🟡 unbalanced_micro_batches: {unbalanced_micro_batches}")
            
            all_seq_lens.append(unbalanced_micro_batches)
            my_batch_ranks = list(range(dp_rank, dp_size * num_microbatch, dp_size))

            assert isinstance(batch_size_x2, int) and batch_size_x2 > 0
            print(f"🟡 unbalanced_micro_batches: {unbalanced_micro_batches}")
            my_batch_ranks = list(range(dp_rank, dp_size * num_microbatch, dp_size))
            print(f"🟡 my_batch_ranks: {my_batch_ranks}")
            balanced_seq_lens, new_batch = d2.planner.wlb_planner.balance_data_for_wlbllm(
                dp_size * num_microbatch, my_batch_ranks, total_seq_len, batch_size, 
                unbalanced_micro_batches, 
                ENABLE_BALANCED_FLOS_NO_DEFER=ENABLE_BALANCED_FLOS_NO_DEFER,
                # TODO: (Refactor) This is a hack to pass the model config to the WLBLLM planner.
                model_config=hf_config, 
            )
        else:
            unbalanced_micro_batches = []
            for i in range(num_microbatch):
                a = get_next_batch(batch_size_x2 * num_microbatch)
                b = a[
                    dp_rank * (batch_size_x2):
                    (dp_rank + 1) * (batch_size_x2)
                ]
                unbalanced_micro_batches.append(b)
                pass
            balanced_seq_lens = [unbalanced_micro_batches[dp_rank]]
            new_batch = unbalanced_micro_batches

            pass
        # print(f"🟡[sample_idx={sample_idx}] balanced_seq_lens: {balanced_seq_lens}, {len(balanced_seq_lens) = }")
        # print(f"🟡[sample_idx={sample_idx}] new_batch: {new_batch}")

        print(f"🟡 {dp_size =} * {pp_size =}, {dp_rank =}, {my_batch_ranks =}")
        print(f"🟡 balanced_seq_lens: {balanced_seq_lens}, {len(balanced_seq_lens) = }")
        assert len(balanced_seq_lens) == num_microbatch, f"len(balanced_seq_lens) == num_microbatch, {len(balanced_seq_lens)} != {num_microbatch}"
        print(f"🟡 new_batch: {new_batch}")
        # all_seq_lens.append(new_batch)
        
        microbatches = []
        # all_seq_lens.append(balanced_seq_lens)
        for mb_idx, seq_lens in enumerate(balanced_seq_lens):
            # doc_lens = flatten(seq_lens)
            # TODO: (Refactor) doc lens must satisfies the TP requirement
            doc_lens = (seq_lens)
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
            
            print(f"🟡 balance_data_for_wlbllm[{mb_idx}]: doc_lens: {doc_lens}, seq_lens: {seq_lens}")
            context_length = sum(doc_lens) # maximum possible context length is just the num_tokens
            print(f"🟡 context_length: {context_length}")
            
            # wlbllm.registry.set("global_tensor_length", (num_tokens * cp_size * 2))
            wlbllm.registry.set("global_tensor_length", context_length)

            num_tokens_this_rank = context_length // cp_size
            assert num_tokens_this_rank * cp_size == context_length, f"num_tokens_this_rank * cp_size == context_length, {num_tokens_this_rank * cp_size} != {context_length}"
            doc_shards = wlbllm.utils.compute_per_doc_cp_shard_doc_len(
                doc_lens, context_length, cp_size
            )

            if rank % 8 == 1:
                print(f"🟡 doc_shards: {doc_shards}")
            
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

            # Only take the number of tokens in this rank seq_lens 
            # is already the dp_rank's doc_lens. 
            # So here we only take the cp-shard of it.
            input_ids = torch.randint(10, 1000, (num_tokens_this_rank,))
            print(f"🟡 input_ids: {input_ids.shape}")
            position_ids = torch.arange(num_tokens_this_rank)
            print(f"🟡 position_ids: {position_ids.shape}")
            wlb_metadata = dict(
                doc_lens=doc_lens,
                doc_shards=doc_shards,
                kv_idx_list=kv_idx_list,
                cu_seqlens_q_list=cu_seqlens_q_list,
                cu_seqlens_kv_list=cu_seqlens_k_list,
                max_seqlen_q_list=max_seqlen_q_list,
                max_seqlen_kv_list=max_seqlen_k_list,
                # global_tensor_length: 
                # global_tensor_length=(num_tokens * cp_size * 2),
                # TODO: What this value should be?
                global_tensor_length=(context_length),
                # context_length
            )
            packed_seq_params = WLBPackedSeqParams(
                qkv_format="thd",
                cu_seqlens_q=cu_seqlens_q_list[-1],
                cu_seqlens_kv=cu_seqlens_k_list[-1],
                max_seqlen_q=max_seqlen_q_list[-1].item(),
                max_seqlen_kv=max_seqlen_k_list[-1].item(),
            )
            packed_seq_params.wlb_metadata = wlb_metadata
            microbatch = {
                "input_ids": input_ids,
                "position_ids": position_ids,
                "packed_seq_params": packed_seq_params,
            }
            microbatches.append(microbatch)
            wlbllm.registry.set(mb_idx, wlb_metadata)
            if rank % 8 == 1:
                print(f"🟡 wlbllm.registry.set mb_idx: {mb_idx}, wlb_metadata: {wlb_metadata}")

        
        
        set_random_seed(seed, set_megatron=True)
        
        rank = torch.distributed.get_rank()
        
        max_warmup_cnt = 1
        try:
            if sample_idx == 0:
                max_warmup_cnt = int(os.environ.get("EXPERIMENT_0TH_SAMPLE_WARMUP_TIMES", 1))
            else:
                max_warmup_cnt = int(os.environ.get("EXPERIMENT_WARMUP_TIMES", 0))
                pass
        except:
            pass

        
        max_repeat_cnt = 1
        try:
            max_repeat_cnt = int(os.environ.get("EXPERIMENT_REPEAT_TIMES", 1))
        except:
            pass


        log_memory_usage("complete sample setup")
        durations_ms = []
        for repeat_idx in range(max_repeat_cnt + max_warmup_cnt):
            wlbllm.registry.set("forward_cnt", 0)
            wlbllm.registry.set("backward_cnt", 0)
    
            print(f"[Rank {rank}] [repeat {repeat_idx}] Start running wlbllm")

            is_warmup = repeat_idx < max_warmup_cnt
            should_log_memory_during_warmup = (
                os.environ.get("EXPERIMENT_SHOULD_LOG_MEMORY_DURING_WARMUP", "0") == "1"
            )

            memory_logging_ctx = nullcontext()
            if is_warmup and should_log_memory_during_warmup:
                memory_logging_ctx = log_memory_usage_context()
                pass
            
            log_memory_usage("before forward_backward_batch")
            config_name = f"n{args.num_nodes}t{args.num_tokens}b{args.num_batches}mb{args.num_microbatch}-cp{args.cp_size}pp{args.pp_size}tp{args.tp_size}"
            with torch.cuda.nvtx.range(f"wlbllm({config_name})[sample={sample_idx}][repeat={repeat_idx}]"):
                # with memory_logging_ctx:
                #     torch.cuda.synchronize(); torch.distributed.barrier(); start_time = time.time()
                #     if True:

                #         with torch.profiler.profile(
                #             activities=[
                #                 torch.profiler.ProfilerActivity.CPU,
                #                 torch.profiler.ProfilerActivity.CUDA,
                #             ],
                #             profile_memory=True,
                #             record_shapes=True,
                #             with_stack=True,
                #         ) as prof:
                #             loss, grad = worker.forward_backward_batch(
                #                 microbatches=microbatches,
                #                 forward_only=False,
                #                 mode="orig_reimpl", # actually wlbllm
                #                 with_dummy=False,
                #             )
                        
                #     if rank == 0:
                #         print("Dumping memory snapshot")
                #         mem_snapshot_output_path = os.path.join(mem_snapshots_dir, f"memory_profile.rank{rank}.pickle")
                #         memory_timeline_output_path = os.path.join(mem_snapshots_dir, f"memory_profile.rank{rank}.html")
                #         memory_timeline_output_raw = os.path.join(mem_snapshots_dir, f"memory_profile.rank{rank}.json.gz")
                #         torch.cuda.memory._dump_snapshot(mem_snapshot_output_path)
                #         prof.export_memory_timeline(memory_timeline_output_path, device=torch.cuda.current_device())
                #         prof.export_memory_timeline(memory_timeline_output_raw, device=torch.cuda.current_device())
                #         print("Memory snapshot dumped")
                #     exit(0)

                    torch.cuda.synchronize(); torch.distributed.barrier(); start_time = time.time()
                    loss, grad = worker.forward_backward_batch(
                        microbatches=microbatches,
                        forward_only=False,
                        mode="orig_reimpl", # actually wlbllm
                        with_dummy=False,
                    )
                    torch.cuda.synchronize(); torch.distributed.barrier(); end_time = time.time()

                    duration = end_time - start_time
                    duration_ms = duration * 1000
                    print(f"⚪ [Rank {rank}] [repeat {repeat_idx}] Finish running wlbllm: {duration_ms:.2f} ms")
                    if repeat_idx >= max_warmup_cnt:
                        durations_ms.append(duration_ms)
                        pass
            time.sleep(1)
            log_memory_usage(f"forward_backward_batch:done(sample_id={sample_idx},repeat={repeat_idx})")

        average_duration_ms = sum(durations_ms) / len(durations_ms) if durations_ms else 0
        
        final_durations_ms.append(average_duration_ms)

        if rank == 0:
            with open(benchmark_log_path, "a") as f:
                f.write(json.dumps({
                    "sample_id": sample_idx,
                    "duration_ms": average_duration_ms,
                    "duration_list": durations_ms,
                    "samples": unbalanced_micro_batches,
                }) + "\n")

            print(f"🟡 Write benchmark log to {benchmark_log_path}")

        pass

    

    print("=" * 20 + "wlbllm with pp done")
    if rank == 0:
        from datetime import datetime
        import pytz
        pst = pytz.timezone('US/Pacific')
        timestamp = datetime.now(pst).strftime("%Y-%m-%d %H:%M:%S PST")
        with open(benchmark_final_path, "w") as f:
            benchmark_data = {
                "test_file": __file__,
                "args": str(args),
                "timestamp": timestamp,
                "config": config,
                "samples": [],
            }
            
            for idx in range(len(final_durations_ms)):
                samples = new_batch
                duration = final_durations_ms[idx]
                benchmark_data["samples"].append({
                    "sample_id": idx,
                    "duration_ms": duration,
                    "samples": samples,
                })
            
            with open(benchmark_final_path, "w") as f:
                json.dump(benchmark_data, f, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    
    parser.add_argument("--num-tokens", type=int, default=1024)
    parser.add_argument("--num-batches", type=float, default=1)
    parser.add_argument("--num-microbatch", type=int, default=1)
    parser.add_argument("--num-nodes", type=int, default=1)
    parser.add_argument("--num-gpus-per-node", type=int, default=4)
    parser.add_argument("--cp-size", type=int, default=2)
    parser.add_argument("--tp-size", type=int, default=1)
    parser.add_argument("--pp-size", type=int, default=4)
    
    parser.add_argument("--model-path", type=str, default="deepseek-ai/DeepSeek-R1-Distill-Llama-8B")
    parser.add_argument("--num-layers", type=int, default=8)
    parser.add_argument("--output-dir", type=str, default="./logs/")

    parser.add_argument("--max-sample-id", type=int, default=3)
    parser.add_argument("--should-add-debug-cases", action="store_true")
    parser.add_argument("--sample-name", type=str, default="wlbllm", choices=["wlbllm", "prolong"])
    parser.add_argument("--change-long-doc-ratio", type=float, default=0.0)

    parser.add_argument("--up-sample-factor", type=int, default=4)
    parser.add_argument("--elongate-factor", type=int, default=1)
    parser.add_argument("--filter-threshold", type=int, default=65536)
    parser.add_argument("--filter-ratio", type=float, default=0.50)

    args = parser.parse_args()
    print("args: ", args)
    
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    error_log_dir = os.path.join(output_dir, "error_logs")
    os.makedirs(error_log_dir, exist_ok=True)
    try:
        test(args)
    except Exception as e:
        rank = os.environ.get("RANK")
        print(f"🟡 Error: {e}")
        with open(os.path.join(error_log_dir, f"error.{rank}.log"), "w") as file:
            import traceback
            tb = traceback.extract_tb(e.__traceback__)
            for filename, lineno, func, text in tb:
                path = os.path.abspath(filename)
                print(f"{path}:{lineno}: in {func}", file=file)
                if text:
                    print(f"    {text}", file=file)
            # error in red
            print(f"{type(e)}: {e}", file=file)
            print(f"🟡 Write error log to {os.path.join(error_log_dir, f'error.{rank}.log')}")
        raise e