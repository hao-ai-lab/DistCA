"""
Debug example:
NVTE_ALLOW_NONDETERMINISTIC_ALGO=0 torchrun --nnodes 1 --nproc_per_node 2 test_megatron_e2e_pipeline_with_cp.py --num-gpus-per-node 2 --pp-size 2 --num-microbatch 2

Planner example:
NVTE_ALLOW_NONDETERMINISTIC_ALGO=0 torchrun --nnodes 1 --nproc_per_node 4 test_megatron_e2e_pipeline_with_cp.py --num-gpus-per-node 4 --pp-size 2 --num-microbatch 2 --use-planner

Planner + CP layout example:
NVTE_ALLOW_NONDETERMINISTIC_ALGO=0 torchrun --nnodes 1 --nproc_per_node 4 test_megatron_e2e_pipeline_with_cp.py --num-gpus-per-node 4 --pp-size 2 --num-microbatch 2 --use-planner --num-batches 1 --num-tokens 2048

Nsys + CP layout example:
NVTE_ALLOW_NONDETERMINISTIC_ALGO=1 NUM_LAYERS=4 \
nsys profile -o /home/pangbo/nsys_reports/pp_16k.nsys-rep \
--trace=cuda,nvtx,osrt,cudnn,cublas --force-overwrite true \
torchrun --nnodes 1 --nproc_per_node 4 \
test_megatron_e2e_pipeline_with_cp.py --num-gpus-per-node 4 --pp-size 2 --num-microbatch 2 --use-planner --num-batches 1 --num-tokens 16384
"""

import argparse
from functools import partial
import os
import time
import json
from global_batch_provider import setup_global_batch, get_next_batch
import megatron.core.parallel_state as mpu
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.pipeline_parallel.schedules import get_forward_backward_func
import torch
from transformers import AutoConfig

from d2.runtime.compute_metadata import get_attn_metadata
from d2.runtime.megatron.packed_seq_params import arg_to_cuda, PingPangSingleStepPackedSeqParams, PingPangPackedSeqParams
from d2.runtime.megatron.forward_backward_func import forward_backward_pipelining_without_interleaving as forward_backward_func
from d2.planner.planner import cp_list_to_mlp_list

from test_util import ParallelConfig, init_worker_torch_distributed, create_qkv_dispatch_pipeline_tick
from test_megatron_e2e import MegatronE2eWorker as BaseMegatronE2eWorker, set_random_seed
from megatron_test_utils import (
    gptmodel_forward, make_batch_generator, unwrap_model,
)
import d2.mem
from contextlib import nullcontext


# --------------------------------
# Better traceback formatting
# --------------------------------
from d2.utils.traceback import enable_clickable_excepthook, enable_trace_calls
enable_clickable_excepthook()


import time
start_time__ = time.time()

import psutil, os
rank = int(os.environ.get("RANK", os.environ.get("SLURM_PROCID","0")))
local = int(os.environ.get("LOCAL_RANK", os.environ.get("SLURM_LOCALID","0")))
p = psutil.Process(os.getpid())
p.cpu_affinity([local * 16, (local + 1) * 16])  # pin to core based on local rank
print(f"[{rank}] allowed CPUs:", p.cpu_affinity())

# ----------------
# Taskset confirm
# ----------------
import check_cpu_binding
aff, mems = check_cpu_binding.check_cpu_binding()
print(f"CPUS={aff} MEMS={mems}")


class MegatronE2eWorker(BaseMegatronE2eWorker):
    def __init__(self, rank: int, world_size: int):
        super().__init__(rank, world_size)
        local_rank = int(os.getenv("LOCAL_RANK"))
        torch.cuda.set_device(local_rank)
        torch.set_default_device(torch.device("cuda", local_rank))

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
                    assert isinstance(psp, PackedSeqParams)

        # forward_backward_func = get_forward_backward_func()
        pp_size = self.tf_config.pipeline_model_parallel_size
        pp_rank = mpu.get_pipeline_model_parallel_rank()
        n_micro_batch = len(microbatches) - pp_size + 1
        # thd layout
        total_seqlen = microbatches[0]['input_ids'].shape[0]

        def loss_func(output):
            # NOTE: this is a dummy loss function.
            loss = output.mean()
            return loss, {'loss': loss}

        def forward_step(batch_iter, model):
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
                packed_seq_params, labels=input_ids.unsqueeze(0),
            )
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
        batch_generator = make_batch_generator(
            microbatches if with_dummy else microbatches[pp_rank:],
            vpp_size=len(self.train_module)
        )
        # if mpu.get_pipeline_model_parallel_world_size() > 1:

        torch.cuda.synchronize()
        from d2.runtime.attn_kernels.ops import nvshmem_barrier_all
        nvshmem_barrier_all()

        # from d2.utils.traceback import TraceFunctions
        # tracer = TraceFunctions("d2/d2/runtime/")
        # with tracer:
        if with_dummy:
            losses_reduced = forward_backward_func(
                forward_step_func=forward_step,
                data_iterator=batch_generator,
                model=self.train_module,
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
            losses_reduced = get_forward_backward_func()(
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


def init_megatron_e2e_test(
    hidden_size_q: int, hidden_size_kv: int, num_heads: int, num_tokens: int,
    world_size: int, max_cp_degree: int, tp_size: int, pp_size: int,
    dtype, worker_cls=MegatronE2eWorker
):
    token_bytes_q = hidden_size_q * dtype.itemsize // tp_size
    token_bytes_kv = hidden_size_kv * dtype.itemsize // tp_size
    max_tokens_query = num_tokens * (world_size // tp_size)
    max_tokens_key_value = num_tokens * (world_size // tp_size)
    buffer_size = (
        token_bytes_q * max_tokens_query * 3 +
        # lse_norm. TODO: the factor of 2 might be removed
        num_heads * torch.float32.itemsize * 2 * max_tokens_query +
        token_bytes_kv * max_tokens_key_value * max_cp_degree * 2
    )
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
        tensor_model_parallel_size=tp_size,
        pipeline_model_parallel_size=pp_size,
    )

    log_memory_usage("before init_worker_torch_distributed", force=True)
    worker = init_worker_torch_distributed(
        world_size, buffer_size,
        worker_cls, parallel_config
    )
    log_memory_usage("after init_worker_torch_distributed (also init_nvshmem)", force=True)
    print("Communication groups initialized")
    return worker


def create_pp_microbatches(
    num_microbatch: int, pp_degree: int, as_rank: int,
    as_world_size: int, total_seq_len: int, num_seqs: int,
    max_cp_degree: int, hidden_size_q_tp: int,
    hidden_size_k_tp: int, element_size: int,
    num_head_in_dtype: int, tp_size: int, dp_size: int,
    num_token_per_rank: int,
    num_batches: int = None,
    use_planner: bool=False,
    return_seq_lens: bool=False
):
    # print("Create pp microbatches")
    tick_per_rank_doc_lens = None
    bwd_metadata = []
    microbatches = []
    if use_planner:
        print("Enable planner. Get real batch.")
    else:
        print("No planner. Use random batch.")


    start_time = time.time()
    all_original_seq_lens = []
    loop_start_time = time.time()
    for i in range(num_microbatch + pp_degree - 1):
        # For the last few ticks (drain-out ticks)
        # add a dummy forward microbatch at PP rank 0.
        add_dummy_forward = i >= num_microbatch
        start_time = time.time()
        print(f"🟡 tick_per_rank_doc_lens: {tick_per_rank_doc_lens}")
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
        if rank == 1:
            print(f"🟡 fa_fwd_params: {fa_fwd_params}")
        all_original_seq_lens.append(original_tick_per_rank_doc_lens)
        end_time = time.time()
        print(f"🟡 create_qkv_dispatch_pipeline_tick duration: {end_time - start_time} seconds")
        
        # For MLP-CP, we need to transfer List[List[int]] from CP layout back to DP, so each rank knows its number of tokens.
        #   Example1 DP case:
        # tick_per_rank_doc_lens cp list: List[List[int]] = [[8], [8], [8], [8], [256, 256],[128, 384],[512], [10, 502] ]
        # tick_per_rank_doc_lens mlp list : [[8], [8], [8], [8], [256, 256],[128, 384],[512], [10, 502] ]
        #   Example2 CP case:
        # tick_per_rank_doc_lens cp list: List[List[int]] = [[8], [8], [8], [8], [256, 768],[512, 10, 502] ]
        # tick_per_rank_doc_lens mlp list: [[8], [8], [8], [8], [256, 128, 128], [256, 256], [512], [10, 502]]
        start_time = time.time()
        tick_per_rank_doc_lens_after_cp_transfer = cp_list_to_mlp_list(tick_per_rank_doc_lens, as_world_size, num_token_per_rank)
        
        this_rank_num_tokens = sum(tick_per_rank_doc_lens_after_cp_transfer[as_rank])
        bwd_packed_seq_params = PackedSeqParams(
            qkv_format="thd", **fa_bwd_params[as_rank]
        )
        tensor_doc_lens = torch.tensor(tick_per_rank_doc_lens_after_cp_transfer[as_rank], dtype=torch.int32)
        mlp_packed_seq_params = get_attn_metadata(tensor_doc_lens, get_packed_seq_params=True)
        end_time = time.time()
        print(f"🟡 get_attn_metadata duration: {end_time - start_time} seconds")

        # Create packed_params. Note that we do not add backward params here.
        start_time = time.time()
        ping_pang_params = PingPangSingleStepPackedSeqParams(
            qkv_format="thd",
            **fa_fwd_params[as_rank],
            qkv_fwd_metadata=qkv_fwd_fa2a_metadata.get_slice(as_rank),
            attn_out_fwd_metadata=attn_out_fwd_fa2a_metadata.get_slice(as_rank),
            mlp_packed_seq_params=mlp_packed_seq_params,
        )
        # print(f"🟡 [bid = {i}] ping_pang_params", ping_pang_params)

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
            (qkv_bwd_fa2a_metadata.get_slice(as_rank), attn_out_qkv_bwd_fa2a_metadata.get_slice(as_rank), bwd_packed_seq_params)
        )
        end_time = time.time()
        print(f"🟡 create_pp_microbatches - append microbatch duration: {end_time - start_time} seconds")



    loop_end_time = time.time()
    print(f"🟡 create_pp_microbatches - first for loop duration: {loop_end_time - loop_start_time} seconds")


    pp_rank = as_rank // dp_size
    dp_rank = as_rank % dp_size

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
    print(f"🟡 create_pp_microbatches - second for loop duration: {end_time - start_time} seconds")
    ret = microbatches
    if return_seq_lens:
        ret = (microbatches, all_original_seq_lens)
    return ret


from contextlib import contextmanager
@contextmanager
def time_me(msg):
    rank = torch.distributed.get_rank()
    print(f"⚪ [Rank {rank}] start {msg}")
    torch.cuda.synchronize(); torch.distributed.barrier(); 
    start_time = time.time()
    yield
    torch.cuda.synchronize(); torch.distributed.barrier(); 
    end_time = time.time()
    duration_ms = ((end_time - start_time) * 1000)
    print(f"⚪ [Rank {rank}] finish {msg}, duration: {duration_ms} ms")


import d2.mem
def log_memory_usage(message: str, force:bool = False):
    d2.mem.log_memory_usage(message, force=force)


def test(args):
    seed = args.seed
    # test scale
    num_nodes = args.num_nodes
    num_tokens = args.num_tokens
    dpcp_size = args.cp_size
    num_seqs = args.num_seqs
    num_batches = args.num_batches
    num_microbatch = args.num_microbatch
    num_layers = args.num_layers
    if num_layers is not None:
        # See `megatron_test_utils.py` for more details.
        os.environ["NUM_LAYERS"] = str(num_layers)

    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    benchmark_log_path = os.path.join(output_dir, "benchmark.raw.jsonl")
    benchmark_log_path__d2 = os.path.join(output_dir, "benchmark.raw.d2.jsonl")
    benchmark_log_path__baseline = os.path.join(output_dir, "benchmark.raw.baseline.jsonl")
    benchmark_log_path__baseline_with_dummy = os.path.join(output_dir, "benchmark.raw.baseline_with_dummy.jsonl")
    benchmark_final_path = os.path.join(output_dir, "benchmark.json")
    network_inspect_path = os.path.join(output_dir, "network_inspect.jsonl")
    network_inspect_summary_path = os.path.join(output_dir, "network_inspect.summary.jsonl")
    microbatch_log_path = os.path.join(output_dir, "microbatch.log")
    os.environ["EXPERIMENT_OUTPUT_DIR"] = output_dir

    config_path = os.path.join(output_dir, "config.json")
    with open(config_path, "w") as f:
        # Namespace to dict
        args_dict = vars(args)
        json.dump(args_dict, f, indent=2)

    memory_usage_dir = os.path.join(output_dir, "memory_usage")
    d2.mem.set_memory_usage_log_file(memory_usage_dir)
    d2.mem.enable_memory_usage_logging(memory_usage_dir)

    log_memory_usage("enter test", force=True)

    # parallelization
    tp_size = args.tp_size
    pp_size = args.pp_size
    world_size = args.num_nodes * args.num_gpus_per_node
    assert world_size % (tp_size * pp_size) == 0
    _dp_size = world_size // (tp_size * pp_size)
    assert dpcp_size == _dp_size, f"dpcp_size: {dpcp_size} != _dp_size: {_dp_size}"

    assert num_microbatch >= pp_size, f"num_microbatch need bigger than pp_size. Current num_microbatch: {num_microbatch}, pp size: {pp_size}"

    # Set num_batches. 
    # If None, we use MLP-DP. will get DP number of new batches per tick.
    # If set, num_batches < dp_size && dp_size % num_batches == 0, Will get num_batches number of List per tick.
    num_token_per_rank = num_tokens * num_batches // dpcp_size
    total_seq_len = num_tokens 

    dtype = torch.bfloat16
    element_size = dtype.itemsize

    max_sample_id = args.max_sample_id
    model_path = args.model_path

    should_log_memory_during_warmup = (
        os.environ.get("EXPERIMENT_SHOULD_LOG_MEMORY_DURING_WARMUP", "1") == "1"
    )

    print(f"tp_size: {tp_size}, pp_size: {pp_size}, dpcp_size: {dpcp_size}, world_size: {world_size}, num_tokens_per_rank: {num_token_per_rank}, total_seq_len: {total_seq_len}, num_batches: {num_batches}")

    should_balance_ping_pong = os.environ.get("EXPERIMENT_BALANCE_PING_PONG", "0") == "1"
    print(f"should_balance_ping_pong: {should_balance_ping_pong}")

    balance_ping_pong_batch_size = None
    if should_balance_ping_pong:
        balance_ping_pong_batch_size = dict(
            mb=num_microbatch,
            batch_size=num_batches,
        )
    setup_global_batch(
        total_seq_len=num_tokens,
        up_sample_factor=args.up_sample_factor,
        elongate_factor=args.elongate_factor,
        filter_threshold=args.filter_threshold,
        filter_ratio=args.filter_ratio,
        should_add_debug_cases=args.should_add_debug_cases,
        change_long_doc_ratio=args.change_long_doc_ratio,
        sample_name=args.sample_name,
        balance_ping_pong_batch_size=balance_ping_pong_batch_size,
    )
    # for _ in range(20):
    #     print(f"🟡 get_next_batch: {get_next_batch(int(num_microbatch * num_batches * 2))}")
    # exit(0)
    

    # Use local cache to avoid HuggingFace rate limiting
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

    log_memory_usage("before init_megatron_e2e_test", force=True)

    worker: MegatronE2eWorker = init_megatron_e2e_test(
        hidden_size_q, hidden_size_kv, hf_config.num_attention_heads, num_tokens,
        world_size, dpcp_size, tp_size, pp_size,
        dtype, MegatronE2eWorker
    )
    log_memory_usage("after init_megatron_e2e_test", force=True)

    enable_gradient_checkpointing = False
    gradient_checkpointing_kwargs = {}
    if os.environ.get("EXPERIMENT_ADD_SELECTIVE_CKPT", "0") == "1":
        enable_gradient_checkpointing = True
        gradient_checkpointing_kwargs = dict(
            # activations_checkpoint_method="mlp",
            activations_checkpoint_granularity="selective",
            activations_checkpoint_num_layers=None, # num-layers
            activations_checkpoint_recompute_modules = ["mlp"],
        )
    print(f"🟡 [Rank {worker.rank}] Adding selective checkpoint ?: {gradient_checkpointing_kwargs}")
    worker.set_config(
        dtype=dtype,
        enable_gradient_checkpointing=enable_gradient_checkpointing,
        gradient_checkpointing_kwargs=gradient_checkpointing_kwargs
    )
    log_memory_usage("before worker.init", force=True)
    worker.init(model_path, seed=seed)
    log_memory_usage("after worker.init", force=True)
    rank = torch.distributed.get_rank()
    
    # set again to potentially adapt to the ray launch case.
    set_random_seed(seed, set_megatron=False)

    as_world_size = worker.as_world_size
    as_rank = worker.as_rank
    rank = torch.distributed.get_rank()

    # Check rank correctness
    dp_rank = mpu.get_data_parallel_rank()
    pp_rank = mpu.get_pipeline_model_parallel_rank()
    assert as_rank == dp_rank + pp_rank * dpcp_size

    hidden_size_q_tp = hidden_size_q // tp_size
    hidden_size_k_tp = hidden_size_kv // tp_size
    num_head_in_dtype = (hf_config.num_attention_heads *
                         torch.float32.itemsize // element_size // tp_size)


    # for _ in range(20):
    #     print(f"🟡 get_next_batch: {get_next_batch(num_batches * 2)}")    
    final_durations_ms = [] # only for d2
    for sample_idx in range(max_sample_id):
        os.environ["__PRG__INTERNAL__EXPERIMENT_SAMPLE_ID"] = str(sample_idx)
        # this total_seq_len is token per rank.
        # Some explanations of the parameters inside `create_pp_microbatches`:
        # - num_microbatch: 
        #   Iterate `num_microbatch + pp_degree - 1` to create each tick's metadata
        # - num_batches: 
        #   For each tick, getting `num_batches` number of list from the data loader (GLOBAL_BATCH iterator). 
        #   This is the parameter controlling the number of batches per tick.
        # 
        start_time = time.time()

        print(f"""create_pp_microbatches(num_microbatch={num_microbatch}, pp_degree={pp_size}, as_rank={as_rank}, as_world_size={as_world_size}, total_seq_len={total_seq_len}, num_seqs={num_seqs}, max_cp_degree={dpcp_size}, hidden_size_q_tp={hidden_size_q_tp}, hidden_size_k_tp={hidden_size_k_tp}, element_size={element_size}, num_head_in_dtype={num_head_in_dtype}, tp_size={tp_size}, dp_size={dpcp_size}, num_token_per_rank={num_token_per_rank}, num_batches={num_batches}, use_planner={args.use_planner}, return_seq_lens=True)""")
        microbatches_0, tick_per_rank_doc_lens_0 = create_pp_microbatches(
            num_microbatch, pp_size, as_rank,
            as_world_size, total_seq_len, num_seqs, dpcp_size,
            hidden_size_q_tp, hidden_size_k_tp, element_size, num_head_in_dtype,
            tp_size, dpcp_size, 
            num_token_per_rank, num_batches, args.use_planner,  
            return_seq_lens=True,
        )
        end_time = time.time()
        duration = (end_time - start_time)
        print(f"⚪ [Rank {rank}] [sample {sample_idx}] create_pp_microbatches(0): {duration} seconds")

        start_time = time.time()
        microbatches_1, tick_per_rank_doc_lens_1 = create_pp_microbatches(
            num_microbatch, pp_size, as_rank,
            as_world_size, total_seq_len, num_seqs, dpcp_size,
            hidden_size_q_tp, hidden_size_k_tp, element_size, num_head_in_dtype,
            tp_size, dpcp_size, 
            num_token_per_rank, num_batches, args.use_planner,
            return_seq_lens=True,
        )
        end_time = time.time()
        duration = (end_time - start_time)
        print(f"⚪ [Rank {rank}] [sample {sample_idx}] create_pp_microbatches(1): {duration} seconds")

        seq_lens = [tick_per_rank_doc_lens_0, tick_per_rank_doc_lens_1]
        # print(f"🟡 [sample_idx = {sample_idx}] seq_lens is: {seq_lens}")

        loop_start_time = time.time()
        set_random_seed(seed, set_megatron=True)
        microbatches = []
        orig_impl_microbatches = []
        for mb_0, mb_1 in zip(microbatches_0, microbatches_1):
            # if rank % 8 == 2:
            # FIXME: Print this to another file, and only rank 0 prints it.
            if rank == 0:
                with open(microbatch_log_path, "a") as f:
                    f.write(f"🟡 [sample_idx = {sample_idx}] mb_0: {mb_0}\n")
                    f.write(f"🟡 [sample_idx = {sample_idx}] mb_1: {mb_1}\n")

            mb_0_psp = mb_0["packed_seq_params"]
            mb_1_psp = mb_1["packed_seq_params"]
            mb_0_mlp_psp = mb_0_psp.mlp_packed_seq_params
            mb_1_mlp_psp = mb_1_psp.mlp_packed_seq_params
            mb_0_psp.dispatcher_id = 0
            mb_1_psp.dispatcher_id = 1
            ping_pong_params = PingPangPackedSeqParams(
                seq_params = [mb_0_psp, mb_1_psp],
                mlp_layout_seq_params = [mb_0_mlp_psp, mb_1_mlp_psp],
                max_seqlen_q = max(mb_0_mlp_psp.max_seqlen_q, mb_1_mlp_psp.max_seqlen_q),
                max_seqlen_kv = max(mb_0_mlp_psp.max_seqlen_kv, mb_1_mlp_psp.max_seqlen_kv),
            )
            num_tokens = sum(mb["position_ids"].numel() for mb in [mb_0, mb_1])
            input_ids = torch.randint(10, 1000, (num_tokens,))
            mb = {
                "input_ids": input_ids,
                "position_ids": torch.concat([mb_0["position_ids"], mb_1["position_ids"]]),
                "packed_seq_params": ping_pong_params,
            }
            if rank == 0:
                with open(microbatch_log_path, "a") as f:
                    f.write(f"🟡 [sample_idx = {sample_idx}] input_ids: {input_ids.shape}\n")
            microbatches.append(mb)

            cu_seqlens_q = torch.concat([
                mb_0_mlp_psp.cu_seqlens_q, mb_1_mlp_psp.cu_seqlens_q[1:] + mb_0_mlp_psp.cu_seqlens_q[-1]
            ])
            cu_seqlens_kv = torch.concat([
                mb_0_mlp_psp.cu_seqlens_kv, mb_1_mlp_psp.cu_seqlens_kv[1:] + mb_0_mlp_psp.cu_seqlens_kv[-1]
            ])
            packed_seq_params = PackedSeqParams(
                qkv_format="thd",
                cu_seqlens_q = cu_seqlens_q,
                cu_seqlens_kv = cu_seqlens_kv,
                max_seqlen_q = ping_pong_params.max_seqlen_q,
                max_seqlen_kv = ping_pong_params.max_seqlen_kv,
            )
            orig_mb = {
                "input_ids": mb["input_ids"],
                "position_ids": mb["position_ids"],
                "packed_seq_params": packed_seq_params,
            }
            orig_impl_microbatches.append(orig_mb)
        loop_end_time = time.time()
        print(f"⚪ create_pp_microbatches - constructing PingPangPackedSeqParams: {loop_end_time - loop_start_time} seconds")

        log_memory_usage("complete microbatches construction", force=True)

        should_run_baseline_with_dummy = False
        should_run_baseline = False
        should_run_d2 = True

        n_warmup = 1
        try:
            if sample_idx == 0:
                n_warmup = int(os.environ.get("EXPERIMENT_0TH_SAMPLE_WARMUP_TIMES", 1))
            else:
                n_warmup = int(os.environ.get("EXPERIMENT_WARMUP_TIMES", 0))
                pass
        except:
            pass
        n_repeats = 1
        try:
            n_repeats = int(os.environ.get("EXPERIMENT_REPEAT_TIMES", 1))
        except:
            pass

        

        if should_run_baseline_with_dummy:

            durations = []
            for _ in range(n_repeats + n_warmup):

                mem_ctx = nullcontext()
                if _ < n_warmup and should_log_memory_during_warmup:
                    mem_ctx = d2.mem.log_memory_usage_context()
                    pass

                print(f"⚪ [Rank {rank}] [sample {sample_idx}] Start baseline dummy {_}")
                with torch.cuda.nvtx.range(f"baseline_dummy[sample={sample_idx}][repeat={_}]"):
                    with mem_ctx:
                        torch.cuda.synchronize(); torch.distributed.barrier(); start_time = time.time()
                        loss_orig_reimpl, grad_orig_reimpl = worker.forward_backward_batch(
                            microbatches=orig_impl_microbatches,

                            forward_only=False,
                            mode="orig_reimpl",
                            with_dummy=True,
                        )

                        torch.cuda.synchronize(); torch.distributed.barrier(); end_time = time.time()
                        duration_ms = (end_time - start_time) * 1000
                        durations.append(duration_ms)
                        print(f"⚪ [Rank {rank}] [sample {sample_idx}] baseline dummy {_}: {duration_ms} ms")
                time.sleep(1)
            
            if rank == 0:
                with open(benchmark_log_path__baseline_with_dummy, "a") as f:
                    f.write(json.dumps({
                        "sample_id": sample_idx,
                        "duration_ms": duration_ms,
                        "duration_list": durations,
                        "seq_lens": seq_lens,
                    }) + "\n")

        if should_run_baseline:
            durations = []
            for _ in range(n_repeats + n_warmup):
                mem_ctx = nullcontext()
                if _ < n_warmup and should_log_memory_during_warmup:
                    mem_ctx = d2.mem.log_memory_usage_context()
                    pass

                print(f"⚪ [Rank {rank}] [sample {sample_idx}] Start baseline {_}")
                with torch.cuda.nvtx.range(f"baseline[sample={sample_idx}][repeat={_}]"):
                    with mem_ctx:
                        torch.cuda.synchronize(); torch.distributed.barrier(); start_time = time.time()
                        loss_orig, grad_orig = worker.forward_backward_batch(
                            microbatches=orig_impl_microbatches,
                            forward_only=False,
                            mode="orig_reimpl",
                            with_dummy=False,
                        )
                        torch.cuda.synchronize(); torch.distributed.barrier(); end_time = time.time()
                        duration_ms = (end_time - start_time) * 1000
                        durations.append(duration_ms)
                        print(f"⚪ [Rank {rank}] [sample {sample_idx}] baseline {_}: {duration_ms} ms")
                time.sleep(1)
            
            if rank == 0:
                with open(benchmark_log_path__baseline, "a") as f:
                    f.write(json.dumps({
                        "sample_id": sample_idx,
                        "duration_ms": duration_ms,
                        "duration_list": durations,
                        "seq_lens": seq_lens,
                    }) + "\n")

        
        
        if should_run_d2:
            print(f"Prepare to run d2 with total runs: {n_repeats = } + {n_warmup = } = {n_repeats + n_warmup = }")
            durations = []
            for _ in range(n_repeats + n_warmup):
                mem_ctx = nullcontext()
                if _ < n_warmup and should_log_memory_during_warmup:
                    mem_ctx = d2.mem.log_memory_usage_context()
                    pass

                config_name = f"n{num_nodes}t{num_tokens}b{num_batches}mb{num_microbatch}-cp{dpcp_size}pp{pp_size}tp{tp_size}"
                print(f"⚪ [Rank {rank}] [sample {sample_idx}] Start pingpong dummy {_}")
                with torch.cuda.nvtx.range(f"d2({config_name})[sample={sample_idx}][repeat={_}]"):
                    with mem_ctx:
                        torch.cuda.synchronize(); torch.distributed.barrier(); start_time = time.time()
                        
                        if True:
                            loss_reduced, grad_sample = worker.forward_backward_batch(
                                microbatches=microbatches,
                                forward_only=False,
                                mode="ping_pong",
                                with_dummy=True,
                            )
                        torch.cuda.synchronize(); torch.distributed.barrier(); end_time = time.time()
                        duration_ms = (end_time - start_time) * 1000
                        durations.append(duration_ms)
                        print(f"⚪ [Rank {rank}] [sample {sample_idx}] pingpong with dummy {_}: {duration_ms} ms")
                time.sleep(1)
                
            final_durations_ms.append(duration_ms)
            if rank == 0:
                with open(benchmark_log_path, "a") as f:
                    f.write(json.dumps({
                        "sample_id": sample_idx,
                        "duration_ms": duration_ms,
                        "duration_list": durations,
                        "seq_lens": seq_lens,
                    }) + "\n")
        
            
            # print(f"{loss_reduced=}, {loss_orig_reimpl=}, {loss_orig=}")
        # torch.testing.assert_close(grad_orig_reimpl, grad_orig)
        # if worker.as_rank == 1:
        #     torch.testing.assert_close(grad_orig_reimpl, grad_sample, rtol=1.1e-3, atol=1.1e-3)
    
    torch.cuda.synchronize()
    print(f"⚪ [Rank {rank}] [sample {sample_idx}] finish pingpong")
    print(f"🟡 [Rank {rank}] [sample {sample_idx}] Write benchmark log to {benchmark_log_path}")

    print("=" * 20 + "forward_backward_batch attention server, done")

    benchmark_final_path = os.path.join(output_dir, "benchmark.json")
    config = dict(
        mode="d2", 
        nodes=args.num_nodes,
        num_gpus_per_node=args.num_gpus_per_node,
        tp_size=tp_size, dp_size=1, cp_size=dpcp_size, 
        num_tokens=num_tokens, model_path=model_path, num_layers=num_layers, 
        max_sample_id=max_sample_id, up_sample_factor=args.up_sample_factor, filter_threshold=args.filter_threshold, filter_ratio=args.filter_ratio, 
        elongate_factor=args.elongate_factor,
        sample_name=args.sample_name,
        change_long_doc_ratio=args.change_long_doc_ratio,
    )
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
                # samples = new_batch
                samples = []
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
    parser.add_argument("--num-tokens", type=int, default=1024)
    parser.add_argument("--num-batches", type=int, default=1)  # this is for cp. set num_batches and num_tokens to control cp doc length.
    parser.add_argument("--cp-size", type=int, default=2)
    parser.add_argument("--num-seqs", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-nodes", type=int, default=1)
    parser.add_argument("--num-gpus-per-node", type=int, default=4)
    parser.add_argument("--tp-size", type=int, default=1)
    parser.add_argument("--pp-size", type=int, default=4)
    parser.add_argument("--num-microbatch", type=int, default=2)
    parser.add_argument("--use-planner", action="store_true")

    # dataset sampling settings
    parser.add_argument("--up-sample-factor", type=int, default=4)
    parser.add_argument("--elongate-factor", type=int, default=1)
    parser.add_argument("--filter-threshold", type=int, default=65536)
    parser.add_argument("--filter-ratio", type=float, default=0.50)
    parser.add_argument("--sample-name", type=str, default="wlbllm")
    parser.add_argument("--change-long-doc-ratio", type=float, default=0.0)

    parser.add_argument("--model-path", type=str, default="./models/codellama/CodeLlama-34b-hf")
    parser.add_argument("--num-layers", type=int, default=8)
    parser.add_argument("--max-sample-id", type=int, default=3)
    parser.add_argument("--should-add-debug-cases", action="store_true")

    parser.add_argument("--output-dir", type=str, default="./logs/")

    args = parser.parse_args()
    print("args: ", args)
    test(args)
