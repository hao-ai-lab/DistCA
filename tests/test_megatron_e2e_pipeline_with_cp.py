"""
Debug example:
NVTE_ALLOW_NONDETERMINISTIC_ALGO=0 torchrun --nnodes 1 --nproc_per_node 2 test_megatron_e2e_pipeline.py --num-gpus-per-node 2 --pp-size 2 --num-microbatch 2

Planner example:
NVTE_ALLOW_NONDETERMINISTIC_ALGO=0 torchrun --nnodes 1 --nproc_per_node 4 test_megatron_e2e_pipeline.py --num-gpus-per-node 4 --pp-size 2 --num-microbatch 2 --use-planner

Planner + CP layout example:
NVTE_ALLOW_NONDETERMINISTIC_ALGO=0 torchrun --nnodes 1 --nproc_per_node 4 test_megatron_e2e_pipeline.py --num-gpus-per-node 4 --pp-size 2 --num-microbatch 2 --use-planner --num-batches 1 --num-tokens 2048
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
from d2.runtime.megatron_patch.packed_seq_params import arg_to_cuda, PingPangSingleStepPackedSeqParams, PingPangPackedSeqParams
from d2.runtime.megatron_patch.forward_backward_func import forward_backward_pipelining_without_interleaving as forward_backward_func
from d2.planner.planner import cp_list_to_mlp_list

from test_util import ParallelConfig, init_worker_torch_distributed, create_qkv_dispatch_pipeline_tick
from test_megatron_e2e import MegatronE2eWorker as BaseMegatronE2eWorker, set_random_seed
from megatron_test_utils import (
    gptmodel_forward, make_batch_generator, unwrap_model,
)


# --------------------------------
# Better traceback formatting
# --------------------------------
# TODO: Put this into some debug file
import sys, traceback, os

RED = "\033[31m"
BLUE = "\033[34m"
RESET = "\033[0m"

def clickable_excepthook(exc_type, exc_value, tb):
    for filename, lineno, func, text in traceback.extract_tb(tb):
        path = os.path.abspath(filename)
        print(f"{path}:{lineno}: in {func}")
        if text:
            print(f"    {text}")
    # error in red
    print(f"{RED}{exc_type.__name__}: {exc_value}{RESET}")

if os.environ.get("EXPERIMENT_PYTHON_BETTER_TRACEBACK", "1") == "1":
    sys.excepthook = clickable_excepthook

# --------------------------------
# Know where I get stuck
# --------------------------------
# TODO: Put this into some debug file

import sys

should_trace_calls = os.environ.get("EXPERIMENT_PYTHON_DEBUG_TRACE_CALLS", "0") == "1"
def trace_calls(frame, event, arg):
    if event == "call":
        code = frame.f_code
        print(f"--> Enter {code.co_name} ({code.co_filename}:{frame.f_lineno})")
    elif event == "return":
        code = frame.f_code
        print(f"<-- Exit {code.co_name} ({code.co_filename}:{frame.f_lineno})")
    return trace_calls

import sys

class TraceFunctions:
    def __init__(self, filter_path=None):
        self.filter_path = filter_path
        self._oldtrace = None
        self.indent = 0

    def _trace(self, frame, event, arg):
        if event in ("call", "return"):
            code = frame.f_code
            filename = code.co_filename
            if self.filter_path and self.filter_path not in filename:
                return
            if event == "call":
                print(f"{BLUE}{' ' * self.indent}--> Enter {code.co_name} ({filename}:{frame.f_lineno}){RESET}")
                self.indent += 1
            elif event == "return":
                print(f"{BLUE}{' ' * self.indent}<-- Exit  {code.co_name} ({filename}:{frame.f_lineno}){RESET}")
                self.indent -= 1
        return self._trace

    def __enter__(self):
        self._oldtrace = sys.gettrace()
        if not should_trace_calls:
            return self
        sys.settrace(self._trace)
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        sys.settrace(self._oldtrace)

# if should_trace_calls:
#     print("🟡 Enabling python debug trace calls.")
#     sys.settrace(trace_calls)




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

    worker = init_worker_torch_distributed(
        world_size, buffer_size,
        worker_cls, parallel_config
    )
    print("Communication groups initialized")
    return worker


def create_pp_microbatches(num_microbatch: int, pp_degree: int, as_rank: int,
                           as_world_size: int, total_seq_len: int, num_seqs: int,
                           max_cp_degree: int, hidden_size_q_tp: int,
                           hidden_size_k_tp: int, element_size: int,
                           num_head_in_dtype: int, tp_size: int, dp_size: int,
                           num_token_per_rank: int,
                           num_batches: int = None,
                           use_planner: bool=False,
                           return_seq_lens: bool=False
):
    tick_per_rank_doc_lens = None
    bwd_metadata = []
    microbatches = []
    if use_planner:
        print("Enable planner. Get real batch.")
    else:
        print("No planner. Use random batch.")


    all_original_seq_lens = []
    for i in range(num_microbatch + pp_degree - 1):
        # For the last few ticks (drain-out ticks)
        # add a dummy forward microbatch at PP rank 0.
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
        
        # For MLP-CP, we need to transfer List[List[int]] from CP layout back to DP, so each rank knows its number of tokens.
        #   Example1 DP case:
        # tick_per_rank_doc_lens cp list: List[List[int]] = [[8], [8], [8], [8], [256, 256],[128, 384],[512], [10, 502] ]
        # tick_per_rank_doc_lens mlp list : [[8], [8], [8], [8], [256, 256],[128, 384],[512], [10, 502] ]
        #   Example2 CP case:
        # tick_per_rank_doc_lens cp list: List[List[int]] = [[8], [8], [8], [8], [256, 768],[512, 10, 502] ]
        # tick_per_rank_doc_lens mlp list: [[8], [8], [8], [8], [256, 128, 128], [256, 256], [512], [10, 502]]

        tick_per_rank_doc_lens = cp_list_to_mlp_list(tick_per_rank_doc_lens, as_world_size, num_token_per_rank)
        this_rank_num_tokens = sum(tick_per_rank_doc_lens[as_rank])
        bwd_packed_seq_params = PackedSeqParams(
            qkv_format="thd", **fa_bwd_params[as_rank]
        )
        tensor_doc_lens = torch.tensor(tick_per_rank_doc_lens[as_rank], dtype=torch.int32)
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
            (qkv_bwd_fa2a_metadata.get_slice(as_rank), attn_out_qkv_bwd_fa2a_metadata.get_slice(as_rank), bwd_packed_seq_params)
        )

    pp_rank = as_rank // dp_size
    dp_rank = as_rank % dp_size

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

def test(args):
    seed = args.seed
    # test scale
    num_tokens = args.num_tokens
    cp_size = args.cp_size
    num_seqs = args.num_seqs
    num_batches = args.num_batches

    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    benchmark_log_path = os.path.join(output_dir, "benchmark.raw.jsonl")
    benchmark_log_path__d2 = os.path.join(output_dir, "benchmark.raw.d2.jsonl")
    benchmark_log_path__baseline = os.path.join(output_dir, "benchmark.raw.baseline.jsonl")
    benchmark_log_path__baseline_with_dummy = os.path.join(output_dir, "benchmark.raw.baseline_with_dummy.jsonl")
    benchmark_final_path = os.path.join(output_dir, "benchmark.json")
    
    config_path = os.path.join(output_dir, "config.json")
    with open(config_path, "w") as f:
        # Namespace to dict
        args_dict = vars(args)
        json.dump(args_dict, f, indent=2)

    # parallelization
    tp_size = args.tp_size
    pp_size = args.pp_size
    world_size = args.num_nodes * args.num_gpus_per_node
    assert world_size % (tp_size * pp_size) == 0
    dp_size = world_size // (tp_size * pp_size)

    assert args.num_microbatch >= pp_size, "num_microbatch need bigger than pp_size. Current num_microbatch: {args.num_microbatch}, pp size: {pp_size}"

    # Set num_batches. 
    # If None, we use MLP-DP. will get DP number of new batches per tick.
    # If set, num_batches < dp_size && dp_size % num_batches == 0, Will get num_batches number of List per tick.
    if False:
        # Legacy logic to take batches -- this is not good!!
        if not args.num_batches:
            num_batches = dp_size
            num_token_per_rank = args.num_tokens
            total_seq_len = args.num_tokens # when no CP, total_seq_len == args.num_tokens
            print(f"MLP-DP, num_batches: {num_batches}, num_token_per_rank: {num_token_per_rank}, total_seq_len: {total_seq_len}")
        else:
            assert args.num_batches <= dp_size, "num-batches must be <= dp_size"
            assert dp_size % args.num_batches == 0, "dp_size must be divisible by num-batches"
            num_batches = args.num_batches
            num_token_per_rank = args.num_tokens * num_batches // dp_size
            total_seq_len = args.num_tokens * num_batches  # total_seq_len is max possible seq_len.
            if args.num_batches <= dp_size:
                print(f"MLP-CP, num_batches: {num_batches}, num_token_per_rank: {num_token_per_rank}, total_seq_len: {total_seq_len}")
    else:
        num_token_per_rank = num_tokens * num_batches // dp_size
        total_seq_len = num_tokens 
        pass
    dtype = torch.bfloat16
    element_size = dtype.itemsize

    max_sample_id = args.max_sample_id
    model_path = args.model_path
    num_layers = args.num_layers
    if num_layers is not None:
        os.environ["NUM_LAYERS"] = str(num_layers)

    hf_config = AutoConfig.from_pretrained(model_path)
    hidden_size_q = hf_config.hidden_size

    hidden_size_kv = hidden_size_q
    if hasattr(hf_config, "num_key_value_heads"):
        hidden_size_kv = (hidden_size_kv * hf_config.num_key_value_heads //
                          hf_config.num_attention_heads)

    worker: MegatronE2eWorker = init_megatron_e2e_test(
        hidden_size_q, hidden_size_kv, hf_config.num_attention_heads, num_tokens,
        world_size, cp_size, tp_size, pp_size,
        dtype, MegatronE2eWorker
    )
    worker.set_config(dtype=dtype)
    worker.init(model_path, seed=seed)
    # set again to potentially adapt to the ray launch case.
    set_random_seed(seed, set_megatron=False)

    as_world_size = worker.as_world_size
    as_rank = worker.as_rank

    # Check rank correctness
    dp_rank = mpu.get_data_parallel_rank()
    pp_rank = mpu.get_pipeline_model_parallel_rank()
    assert as_rank == dp_rank + pp_rank * dp_size

    hidden_size_q_tp = hidden_size_q // tp_size
    hidden_size_k_tp = hidden_size_kv // tp_size
    num_head_in_dtype = (hf_config.num_attention_heads *
                         torch.float32.itemsize // element_size // tp_size)

    setup_global_batch(
        total_seq_len=num_tokens,
        up_sample_factor=args.up_sample_factor,
        elongate_factor=args.elongate_factor,
        filter_threshold=args.filter_threshold,
        filter_ratio=args.filter_ratio,
    )

    # for _ in range(20):
    #     print(f"🟡 get_next_batch: {get_next_batch(num_batches * 2)}")    

    for sample_idx in range(max_sample_id):
        # this total_seq_len is token per rank.
        # Some explanations of the parameters inside `create_pp_microbatches`:
        # - num_microbatch: 
        #   Iterate `num_microbatch + pp_degree - 1` to create each tick's metadata
        # - num_batches: 
        #   For each tick, getting `num_batches` number of list from the data loader (GLOBAL_BATCH iterator). 
        #   This is the parameter controlling the number of batches per tick.
        # 
        
        microbatches_0, tick_per_rank_doc_lens_0 = create_pp_microbatches(
            args.num_microbatch, pp_size, as_rank,
            as_world_size, total_seq_len, num_seqs, cp_size,
            hidden_size_q_tp, hidden_size_k_tp, element_size, num_head_in_dtype,
            tp_size, dp_size, 
            num_token_per_rank, num_batches, args.use_planner,  
            return_seq_lens=True,
        )
        microbatches_1, tick_per_rank_doc_lens_1 = create_pp_microbatches(
            args.num_microbatch, pp_size, as_rank,
            as_world_size, total_seq_len, num_seqs, cp_size,
            hidden_size_q_tp, hidden_size_k_tp, element_size, num_head_in_dtype,
            tp_size, dp_size, 
            num_token_per_rank, num_batches, args.use_planner,
            return_seq_lens=True,
        )
        seq_lens = [tick_per_rank_doc_lens_0, tick_per_rank_doc_lens_1]
        print(f"In test_megatron_e2e_pipeline_with_cp.py, seq_lens is: {seq_lens}")
        set_random_seed(seed, set_megatron=True)
        microbatches = []
        orig_impl_microbatches = []
        for mb_0, mb_1 in zip(microbatches_0, microbatches_1):
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
            print(f"🟡 input_ids: {input_ids.shape}")
            microbatches.append(mb)
            packed_seq_params = PackedSeqParams(
                qkv_format="thd",
                cu_seqlens_q = torch.concat([
                    mb_0_mlp_psp.cu_seqlens_q, mb_1_mlp_psp.cu_seqlens_q[1:] + mb_0_mlp_psp.cu_seqlens_q[-1]
                ]),
                cu_seqlens_kv = torch.concat([
                    mb_0_mlp_psp.cu_seqlens_kv, mb_1_mlp_psp.cu_seqlens_kv[1:] + mb_0_mlp_psp.cu_seqlens_kv[-1]
                ]),
                max_seqlen_q = ping_pong_params.max_seqlen_q,
                max_seqlen_kv = ping_pong_params.max_seqlen_kv,
            )
            orig_mb = {
                "input_ids": mb["input_ids"],
                "position_ids": mb["position_ids"],
                "packed_seq_params": packed_seq_params,
            }
            orig_impl_microbatches.append(orig_mb)



        should_run_baseline_with_dummy = False
        should_run_baseline = True
        should_run_d2 = True

        n_repeats = 2

        rank = torch.distributed.get_rank()

        if should_run_baseline_with_dummy:

            durations = []
            for _ in range(n_repeats):
                print(f"⚪ [Rank {rank}] [sample {sample_idx}] Start baseline dummy {_}")
                with torch.cuda.nvtx.range(f"baseline_dummy[sample={sample_idx}][repeat={_}]"):
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
            for _ in range(n_repeats):
                print(f"⚪ [Rank {rank}] [sample {sample_idx}] Start baseline {_}")
                with torch.cuda.nvtx.range(f"baseline[sample={sample_idx}][repeat={_}]"):
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
            durations = []
            for _ in range(n_repeats):
                print(f"⚪ [Rank {rank}] [sample {sample_idx}] Start pingpong dummy {_}")
                with torch.cuda.nvtx.range(f"d2[sample={sample_idx}][repeat={_}]"):
                    torch.cuda.synchronize(); torch.distributed.barrier(); start_time = time.time()
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

    parser.add_argument("--model-path", type=str, default="deepseek-ai/DeepSeek-R1-Distill-Llama-8B")
    parser.add_argument("--num-layers", type=int, default=8)
    parser.add_argument("--max-sample-id", type=int, default=3)


    parser.add_argument("--output-dir", type=str, default="./logs/")

    args = parser.parse_args()
    print("args: ", args)
    test(args)
