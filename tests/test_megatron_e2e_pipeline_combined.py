"""
Debug example:
NVTE_ALLOW_NONDETERMINISTIC_ALGO=1 torchrun --nnodes 1 --nproc_per_node 2 test_megatron_e2e_pipeline_combined.py --num-gpus-per-node 2 --pp-size 2 --num-microbatch 2

Planner example:
NVTE_ALLOW_NONDETERMINISTIC_ALGO=1 torchrun --nnodes 1 --nproc_per_node 4 test_megatron_e2e_pipeline_combined.py --num-gpus-per-node 4 --pp-size 2 --num-microbatch 2 --use-planner

Planner + CP layout example:
NVTE_ALLOW_NONDETERMINISTIC_ALGO=1 torchrun --nnodes 1 --nproc_per_node 4 test_megatron_e2e_pipeline_combined.py --num-gpus-per-node 4 --pp-size 2 --num-microbatch 2 --use-planner --num-batches 1 --num-tokens 2048
"""
import argparse
from functools import partial
import os
import time

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

import d2.planner.wlb_planner

try:
    import wlbllm
    import wlbllm.utils
    import wlbllm.registry
except ImportError:
    print("""⚠️ WLBLLM is not installed. This only affects if you're testing WLBLLM tests. To install:

    cd d2/baseline/wlbllm_original
    pip install -e .
    """)
    # exit(1)


NVTE_ALLOW_NONDETERMINISTIC_ALGO = os.environ.get("NVTE_ALLOW_NONDETERMINISTIC_ALGO", "1")
if NVTE_ALLOW_NONDETERMINISTIC_ALGO == "0":
    print("⚠️⚠️⚠️ Forcefully setting NVTE_ALLOW_NONDETERMINISTIC_ALGO to 1 to ensure attention is flops-efficient. This flag has introduced various hard-to-debug performance issues. If you really need to debug, set it back to 0 in your code.")
    os.environ["NVTE_ALLOW_NONDETERMINISTIC_ALGO"] = "1"


def debug_print(*args, **kwargs):
    rank = int(os.environ.get("RANK", 0))
    print(f"🟡 [Rank {rank}]", *args, **kwargs)
    return

from d2.utils.traceback import (
    enable_clickable_excepthook, 
    enable_trace_calls,
    should_enable_clickable_excepthook,
    should_trace_calls,
)

if should_enable_clickable_excepthook:
    enable_clickable_excepthook()
if should_trace_calls:
    enable_trace_calls()


# --------------------------------
# MegatronE2eWorker
# --------------------------------
class MegatronE2eWorker(BaseMegatronE2eWorker):
    def __init__(self, rank: int, world_size: int):
        super().__init__(rank, world_size)
        local_rank = int(os.getenv("LOCAL_RANK"))
        torch.cuda.set_device(local_rank)
        torch.set_default_device(torch.device("cuda", local_rank))
        self.init_comm_mode = None

    def set_init_comm_mode(self, mode: str):
        self.init_comm_mode = mode

    def init_comm(self, buffer_size: int, parallel_config: ParallelConfig, local_rank: int):
        assert self.init_comm_mode is not None, "init_comm_mode is not set. Should be either 'd2' or 'wlbllm'."
        if self.init_comm_mode == "d2":
            super().init_comm(buffer_size, parallel_config, local_rank)
        elif self.init_comm_mode == "wlbllm":
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
            # self.as_world_size = parallel_config.pipeline_model_parallel_size * parallel_config.data_parallel_size 
            # self.as_rank = 
            # TODO: What are the sizes of the AS group should be?
            dp_size = mpu.get_data_parallel_world_size()
            pp_size = mpu.get_pipeline_model_parallel_world_size()
            dp_rank = mpu.get_data_parallel_rank()
            pp_rank = mpu.get_pipeline_model_parallel_rank()
            self.as_world_size = dp_size * pp_size
            self.as_rank = dp_rank * pp_size + pp_rank
            debug_print(f"WLBLLM comm init finished - {self.as_world_size = }, {self.as_rank =}. This may fail if the ordering of the ranks changes underlying.")
        return
        

    def forward_backward_batch(
        self, microbatches: list[dict], forward_only: bool=False, mode: str="ping_pong", with_dummy: bool=True):
        # TODO: refactor this to ensure all names / modes are defined in one place.
        if mode == "d2":
            mode = "ping_pong"
            pass

        # TODO: What are the debug modes for?
        debug = (mode != "ping_pong" and mode != "wlbllm")
        # debug = False
        debug_fwd_impl = mode if debug else None

        microbatches = [{
            k: arg_to_cuda(v) for k, v in microbatch.items()
        } for microbatch in microbatches]
        if "orig" in mode or "wlbllm" in mode:
            for mb in microbatches:
                psp = mb["packed_seq_params"]
                if isinstance(psp, PingPangSingleStepPackedSeqParams):
                    mb["packed_seq_params"] = mb["packed_seq_params"].mlp_packed_seq_params
                psp = mb["packed_seq_params"]
                assert isinstance(psp, PackedSeqParams)

        # forward_backward_func = get_forward_backward_func()
        pp_size = self.tf_config.pipeline_model_parallel_size
        pp_rank = mpu.get_pipeline_model_parallel_rank()
        n_micro_batch = len(microbatches) - pp_size + 1
        # thd layout
        total_seqlen = microbatches[0]['input_ids'].shape[0]

        def loss_func(logits):
            loss = logits.sum()  # no gradient, but can trigger backward
            # Print the memory usage here
            # log_memory_usage("loss_func")
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

        assert mode in ["ping_pong", "orig_reimpl", "single_sided", "wlbllm"]

        for module in self.train_module:
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
        if mode == "ping_pong":
            from d2.runtime.attn_kernels.ops import nvshmem_barrier_all
            nvshmem_barrier_all()
        else:
            torch.cuda.synchronize()
        
        if with_dummy:
            # raise RuntimeError("At performance test, we should not use dummy backward.")
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
                    # TODO: Why skipping when orig is in mode?
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
    world_size: int, cp_degree: int, tp_size: int, pp_size: int,
    dtype, worker_cls=MegatronE2eWorker, mode='d2' # 'd2' or 'wlbllm' or 'baseline
):
    token_bytes_q = hidden_size_q * dtype.itemsize // tp_size
    token_bytes_kv = hidden_size_kv * dtype.itemsize // tp_size
    max_tokens_query = num_tokens * (world_size // tp_size)
    max_tokens_key_value = num_tokens * (world_size // tp_size)
    
    # TODO: Buffer size with env var.
    buffer_size = (
        token_bytes_q * max_tokens_query * 3 +
        # lse_norm. TODO: the factor of 2 might be removed
        num_heads * torch.float32.itemsize * 2 * max_tokens_query +
        token_bytes_kv * max_tokens_key_value * cp_degree * 2
    )
    debug_print(f'{buffer_size = }')
    parallel_config = ParallelConfig(
        tensor_model_parallel_size=tp_size,
        pipeline_model_parallel_size=pp_size,
    )

    # worker = init_worker_torch_distributed(
    #     world_size, buffer_size, worker_cls, parallel_config
    # ) # originally called from test_util.py, but since we have different comm group initialization process for wlbllm, we need to use the original one.
    WORLD_SIZE = os.environ.get("WORLD_SIZE")
    RANK = os.environ.get("RANK")
    LOCAL_RANK = os.environ.get("LOCAL_RANK")
    rank = int(RANK)
    local_rank = int(LOCAL_RANK)
    assert world_size == int(WORLD_SIZE), f"world_size: {world_size} != WORLD_SIZE: {WORLD_SIZE}. RANK: {RANK}, LOCAL_RANK: {LOCAL_RANK}"
    
    worker = worker_cls(rank, world_size)
    worker.set_init_comm_mode(mode)
    debug_print(f"init_comm_mode: {mode}")
    if parallel_config is not None:
        worker.init_comm(buffer_size, parallel_config, local_rank)
    else:
        worker.init_comm(buffer_size, local_rank)
    
    print("Communication groups initialized")
    return worker


def create_pp_microbatches(num_microbatch: int, pp_degree: int, as_rank: int,
                           as_world_size: int, total_seq_len: int, num_seqs: int,
                           max_cp_degree: int, hidden_size_q_tp: int,
                           hidden_size_k_tp: int, element_size: int,
                           num_head_in_dtype: int, tp_size: int, dp_size: int,
                           num_token_per_rank: int,
                           num_batches: int = None,
                           use_planner: bool = False, # TODO: should deprecate this flag.
                           return_original_doclen: bool = False,
                           
                           ):
    tick_per_rank_doc_lens: Optional[list[list[int]]] = None
    bwd_metadata = []
    microbatches = []
    if use_planner:
        debug_print("Enable planner. Get real batch.")
    else:
        debug_print("No planner. Use random batch.")
    for i in range(num_microbatch + pp_degree - 1):
        # For the last few ticks (drain-out ticks)
        # add a dummy forward microbatch at PP rank 0.
        add_dummy_forward = i >= num_microbatch

        (
            fa_fwd_params, fa_bwd_params,
            qkv_fwd_fa2a_metadata, qkv_bwd_fa2a_metadata,
            attn_out_fwd_fa2a_metadata, attn_out_qkv_bwd_fa2a_metadata,
            tick_per_rank_doc_lens,
            orig_cur_tick_per_rank_doc_lens,
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
            return_original_doclen=True,
        )
        
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
    
    # Prepare the return values.
    # - microbatches: list[dict]
    # - (if specified `return_original_doclen``) orig_cur_tick_per_rank_doc_lens: list[list[int]]
    ret = (microbatches, )
    if return_original_doclen:
        ret += (orig_cur_tick_per_rank_doc_lens,)
    return ret


def test(args):
    
    should_run_baseline_dummy = True
    should_run_baseline_no_dummy = True
    should_run_wlbllm = False
    should_run_pingpong_dummy = True


    mode = args.mode
    model_path = args.model_path
    num_layers = args.num_layers
    output_dir = args.output_dir
    num_batches = args.num_batches
    # TODO: (Refactor) This is a hack to set the number of layers. It should be properly set in the HuggingFace config, not here
    if num_layers is not None:
        # See `megatron_test_utils.py` for more details.
        os.environ["NUM_LAYERS"] = str(num_layers)

    
    seed = args.seed
    # test scale
    num_tokens = args.num_tokens
    max_cp_degree = args.cp_degree
    num_seqs = args.num_seqs
    num_microbatch = args.num_microbatch
    
    # parallelization
    tp_size = args.tp_size
    pp_size = args.pp_size
    cp_size = cp_degree = args.cp_degree
    world_size = args.num_nodes * args.num_gpus_per_node
    assert world_size % (tp_size * pp_size * cp_size) == 0, (
        f"world_size: {world_size} = num_nodes: {args.num_nodes} * num_gpus_per_node: {args.num_gpus_per_node} must be divisible by tp_size: {tp_size} * pp_size: {pp_size} * cp_size: {cp_size}. "
        f"Did you set the correct number of nodes, GPUs per node, and parallelism sizes? "
        f"Check your environment variables: NNODES, GPUS_PER_NODE, TP_SIZE, PP_SIZE, CP_SIZE"
    )
    dp_size = world_size // (tp_size * pp_size * cp_size)

    assert num_microbatch >= pp_size, f"num_microbatch need bigger than pp_size. Current num_microbatch: {num_microbatch}, pp size: {pp_size}"

    # Set num_batches. 
    # If None, we use MLP-DP. will get DP number of new batches per tick.
    # If set, num_batches < dp_size && dp_size % num_batches == 0, Will get num_batches number of List per tick.
    # TODO: (Refactor) Rename all `total_seq_len` to just `per_rank_seq_len`. This creates a huge miscommunication.
    if num_batches == None:
        num_batches = dp_size
        num_token_per_rank = args.num_tokens
        total_seq_len = args.num_tokens # when no CP, total_seq_len == args.num_tokens
        print(f"MLP-DP, num_batches: {num_batches}, num_token_per_rank: {num_token_per_rank}, total_seq_len: {total_seq_len}")
    else:
        # assert args.num_batches <= dp_size, f"num-batches ({args.num_batches}) must be <= dp_size ({dp_size})"
        assert num_batches >= dp_size, f"num-batches ({num_batches}) must be >= dp_size ({dp_size})"
        assert num_batches % dp_size == 0, f"num-batches ({num_batches}) must be divisible by dp_size ({dp_size})"
        num_token_per_rank = args.num_tokens * num_batches // dp_size
        total_seq_len = args.num_tokens * num_batches  # total_seq_len is max possible seq_len.
        if num_batches <= dp_size:
            print(f"MLP-CP, num_batches: {num_batches}, num_token_per_rank: {num_token_per_rank}, total_seq_len: {total_seq_len}")
    dtype = torch.bfloat16
    element_size = dtype.itemsize

    hf_config = AutoConfig.from_pretrained(model_path)
    hidden_size_q = hf_config.hidden_size

    hidden_size_kv = hidden_size_q
    if hasattr(hf_config, "num_key_value_heads"):
        hidden_size_kv = (hidden_size_kv * hf_config.num_key_value_heads //
                          hf_config.num_attention_heads)

    debug_print(f"hidden_size_q: {hidden_size_q}, hidden_size_kv: {hidden_size_kv}, hf_config.num_attention_heads: {hf_config.num_attention_heads}. mode: {mode}")
    worker: MegatronE2eWorker = init_megatron_e2e_test(
        hidden_size_q, hidden_size_kv, hf_config.num_attention_heads, num_tokens,
        world_size, cp_size, tp_size, pp_size,
        dtype, MegatronE2eWorker, mode=mode
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

    from global_batch_provider import setup_global_batch
    setup_global_batch(total_seq_len=total_seq_len)
        
    # this total_seq_len is token per rank.
    # Some explanations of the parameters inside `create_pp_microbatches`:
    # - num_microbatch: 
    #   Iterate `num_microbatch + pp_degree - 1` to create each tick's metadata
    # - num_batches: 
    #   For each tick, getting `num_batches` number of list from the data loader (GLOBAL_BATCH iterator). 
    #   This is the parameter controlling the number of batches per tick.

    
    # ---------------------------
    # Prepare the microbatches
    # ---------------------------
    # microbatch - dict(
    #   position_ids: tensor, 
    #   packed_seq_params: PingPangSingleStepPackedSeqParams, 
    # )
    # orig_cur_tick_per_rank_doc_lens_x -> prepare for wlbllm to use.

    microbatches_0, orig_cur_tick_per_rank_doc_lens_0 = create_pp_microbatches(
        args.num_microbatch, pp_size, as_rank,
        as_world_size, total_seq_len, num_seqs, max_cp_degree,
        hidden_size_q_tp, hidden_size_k_tp, element_size, num_head_in_dtype,
        tp_size, dp_size, 
        num_token_per_rank, num_batches, args.use_planner,  return_original_doclen=True,
    )
    microbatches_1, orig_cur_tick_per_rank_doc_lens_1 = create_pp_microbatches(
        args.num_microbatch, pp_size, as_rank,
        as_world_size, total_seq_len, num_seqs, max_cp_degree,
        hidden_size_q_tp, hidden_size_k_tp, element_size, num_head_in_dtype,
        tp_size, dp_size, 
        num_token_per_rank, num_batches, args.use_planner, return_original_doclen=True,
    )
    set_random_seed(seed, set_megatron=True)
    orig_impl_microbatches = []
    wlbllm_microbatches = []
    wlbllm_metadatas = [] # used to register for wlbllm.registry.
    d2_microbatches = []

    

    # TODO: (In progress) Still have some bugs undone.
    if should_run_wlbllm:
        # Constrcut WLBLLM microbatches
        # - use the `orig_cur_tick_per_rank_doc_lens_0` and `orig_cur_tick_per_rank_doc_lens_1` ticks.
        # - create balanced microbatches by reordering.
        # - construct the microbatch: dict(input_ids, position_ids, packed_seq_params: PackedSeqParams)
        # - register these variables as global variables into the wlbllm utils. 
        #     This is a historic hack from us to enable WLBLLM: 
        #     variables like cp_stream, etc, are slightly hard to pass down the chain. 
        all_doc_lens = orig_cur_tick_per_rank_doc_lens_0 + orig_cur_tick_per_rank_doc_lens_1
        debug_print(f"wlbllm all_doc_lens: {all_doc_lens}")

        
        wlb_plan_size = dp_size * pp_size
        wlb_plan_rank = dp_rank + pp_rank * dp_size
        _, new_batch = d2.planner.wlb_planner.balance_data_for_wlbllm(
            wlb_plan_size, wlb_plan_rank, total_seq_len, 
            # Here, the number of batches we pass is (num_batches * num_microbatch).
            # Inside the function, we use (num_batches * num_microbatch * 2) to distribute the workload.
            # This is becuase `orig_cur_tick_per_rank_doc_lens_0` has num_microbatch * batches, elements.
            # and we are taking 2 of them.
            num_batches * num_microbatch, 
            all_doc_lens,
            ENABLE_BALANCED_FLOS_NO_DEFER=True,
            model_config=hf_config, # TODO: (Refactor) This is a hack to pass the model config to the WLBLLM planner.
        )
        # TODO: Log the workload of differnet batches...
        debug_print(f"wlbllm: taking {num_batches * num_microbatch} batches, getting new_batch: {new_batch}")
        assert len(new_batch) == num_batches * num_microbatch, f"len(new_batch) must be equal to num_batches * num_microbatch. Current len(new_batch): {len(new_batch)}, num_batches: {num_batches}, num_microbatch: {num_microbatch}"

        # Take this DP-rank's all PP batches, and organizes them into a list of microbatch(es).
        wlb_my_dp_rank_microbatches: list[list[int]] = new_batch[dp_rank * pp_size: (dp_rank + 1) * pp_size]
        debug_print(f"wlbllm: taking {dp_rank * pp_size} to {dp_rank + pp_size} batches, getting wlb_my_batches: {wlb_my_dp_rank_microbatches}")
        debug_print(f"wlbllm: dp_rank: {dp_rank} - token_per_batch: {[sum(mb) for mb in wlb_my_dp_rank_microbatches]}")

        cp_rank = mpu.get_context_parallel_rank()
        assert cp_size == mpu.get_context_parallel_world_size()

        wlbllm.registry.clear()
        cp_stream = torch.cuda.current_stream()
        wlbllm.registry.set("cp_stream", cp_stream)

        for idx, wlb_mb in enumerate(wlb_my_dp_rank_microbatches):
            assert isinstance(wlb_mb, list) and isinstance(wlb_mb[0], int), f"wlb_mb must be a list of ints. Current wlb_mb: {wlb_mb}"
            
            doc_lens = wlb_mb
            num_tokens: int = sum(doc_lens)
            position_ids = torch.arange(num_tokens, dtype=torch.int64)
            
            local_context_length = num_tokens // cp_size
            context_length = local_context_length * cp_size

            doc_shards = wlbllm.utils.compute_per_doc_cp_shard_doc_len(
                doc_lens, context_length, cp_size,
            )
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

            mb = {
                "input_ids": torch.randint(10, 1000, (num_tokens,)),
                "position_ids": position_ids,
                "packed_seq_params": packed_seq_params,
            }
            wlbllm_microbatches.append(mb)

            # Now actually registering the wlbllm variables into the registry.
            # The registry takes care of the context of the variables. 

            # TODO: (HACK) I'm really sorry I should not have done this.
            def register_wlbllm_variables():
                wlbllm.registry.set("doc_lens", doc_lens)
                wlbllm.registry.set("doc_shards", doc_shards)
                wlbllm.registry.set("kv_idx_list", kv_idx_list)
                wlbllm.registry.set("cp_stream", cp_stream)
                wlbllm.registry.set("cu_seqlens_q_list", cu_seqlens_q_list)
                wlbllm.registry.set("cu_seqlens_kv_list", cu_seqlens_k_list)
                wlbllm.registry.set("max_seqlen_q_list", max_seqlen_q_list)
                wlbllm.registry.set("max_seqlen_kv_list", max_seqlen_k_list)
                # wlbllm.registry.set("global_tensor_length", (total_seq_len * cp_size * 2))
                wlbllm.registry.set("global_tensor_length", (total_seq_len * cp_size))
            wlbllm_metadatas.append(register_wlbllm_variables)
            pass



    # Construct D2 and baseline microbatches
    
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
        d2_microbatches.append(mb)
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

    def log_micro_batches(mb_list):
        rank = torch.distributed.get_rank()
        microbatch_log_dir = os.path.join(output_dir, f"microbatch_logs")
        os.makedirs(microbatch_log_dir, exist_ok=True)
        microbatch_log_file = os.path.join(microbatch_log_dir, f"{rank}.log")
        with open(microbatch_log_file, "a") as f:
            for mb in mb_list:
                f.write(f"{mb}\n")
        return
    
    # debug_print(f"orig_impl_microbatches[0]: {orig_impl_microbatches[0]}")
    # debug_print(f"microbatches[0]: {microbatches[0]}")
    time.sleep(2)

    
    stack_tracer = TraceFunctions(
        filter_path="/mnt/weka/home/yonghao.zhuang/jd/d2/"
    )
    stack_tracer.__enter__()


    if should_run_baseline_dummy:
        debug_print("Start Forward backward baseline with dummy")
        start_time = time.time()
        torch.cuda.synchronize()
        torch.distributed.barrier()
        loss_orig_reimpl, grad_orig_reimpl = worker.forward_backward_batch(
            microbatches=orig_impl_microbatches,
            forward_only=False,
            mode="orig_reimpl",
            with_dummy=True,
        )
        torch.cuda.synchronize()
        torch.distributed.barrier()
        end_time = time.time()
        duration = end_time - start_time
        duration_ms = duration * 1000
        debug_print(f"⚪ Finish Baseline with dummy in {duration_ms:.2f} ms")
    else:
        debug_print(f"⚪ Skipping Baseline with dummy")
        pass


    if should_run_baseline_no_dummy:
        debug_print("Start Forward backward baseline without dummy")
        start_time = time.time()
        torch.cuda.synchronize()
        torch.distributed.barrier()
        loss_orig, grad_orig = worker.forward_backward_batch(
            microbatches=orig_impl_microbatches,
            forward_only=False,
            mode="orig_reimpl",
            with_dummy=False,
        )
        torch.cuda.synchronize()
        torch.distributed.barrier()
        end_time = time.time()
        duration = end_time - start_time
        duration_ms = duration * 1000
        debug_print(f"⚪ Finish Baseline without dummy in {duration_ms:.2f} ms")
    else:
        debug_print(f"⚪ Skipping Baseline without dummy")
        pass


    if should_run_wlbllm:
        with wlbllm.megatron_patch.dot_product_attention.monkey_patch_context():
            with wlbllm.megatron_patch.backends.monkey_patch_context():
                debug_print("Start Forward backward wlbllm")
                start_time = time.time()
                torch.cuda.synchronize()
                torch.distributed.barrier()
                # Patch the attention and all stuff...
                loss_wlbllm, grad_wlbllm = worker.forward_backward_batch(
                    microbatches=wlbllm_microbatches,
                    forward_only=False,
                    mode="wlbllm",
                    with_dummy=False,
                )
                torch.cuda.synchronize()
                torch.distributed.barrier()
                end_time = time.time()
                duration = end_time - start_time
                duration_ms = duration * 1000
                debug_print(f"⚪ Finish WLBLLM in {duration_ms:.2f} ms")
    else:
        debug_print(f"⚪ Skipping WLBLLM")
        pass

    
    if should_run_pingpong_dummy:
        debug_print("Start Forward backward pingpong with dummy")
        for idx in range(3):
            time.sleep(2)
            torch.cuda.synchronize()
            torch.distributed.barrier()
            start_time = time.time()
            loss_reduced, grad_sample = worker.forward_backward_batch(
                microbatches=d2_microbatches,
                forward_only=False,
                mode="ping_pong",
                with_dummy=True,
            )
            torch.cuda.synchronize()
            torch.distributed.barrier()
            end_time = time.time()
            duration = end_time - start_time
            duration_ms = duration * 1000
            debug_print(f"⚪ [idx={idx}] Finish D2 with dummy in {duration_ms:.2f} ms")
    else:
        debug_print(f"⚪ Skipping D2 with dummy")
        pass

    torch.cuda.synchronize()
    stack_tracer.__exit__(None, None, None)

    is_close = False
    try:
        # torch.testing.assert_close(grad_orig_reimpl, grad_orig)
        print(f"{loss_reduced=}, {loss_orig_reimpl=}, {loss_orig=}")
        torch.testing.assert_close(grad_orig_reimpl, grad_sample, rtol=1.1e-3, atol=1.1e-3)
        is_close = True
    except AssertionError as e:
        debug_print(f"⚪ Gradients are not close: {e}")
        is_close = False
    except Exception as e:
        debug_print(f"👻 Unexpected error. Maybe we skipped some tests?: {e}")
        is_close = False

    prompt = "🟢" if is_close else "⚠️"
    print(f"{prompt} Passing e2e pipeline pingpong test. loss tensor is_close = {is_close}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-tokens", type=int, default=1024)
    parser.add_argument("--num-batches", type=int)  # this is for cp. set num_batches and num_tokens to control cp doc length.
    parser.add_argument("--num-seqs", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-nodes", type=int, default=1)
    parser.add_argument("--num-gpus-per-node", type=int, default=4)
    parser.add_argument("--cp-degree", type=int, default=2)
    parser.add_argument("--tp-size", type=int, default=1)
    parser.add_argument("--pp-size", type=int, default=4)
    parser.add_argument("--num-microbatch", type=int, default=2)
    parser.add_argument("--use-planner", action="store_true")


    parser.add_argument("--mode", type=str, default="d2", choices=["d2", "baseline", "wlbllm"])
    parser.add_argument("--model-path", type=str, default="deepseek-ai/DeepSeek-R1-Distill-Llama-8B")
    parser.add_argument("--num-layers", type=int, default=8)

    parser.add_argument("--output-dir", type=str, default="./logs/")
    args = parser.parse_args()
    
    try:
        test(args)
    except Exception as e:
        # then log into a special file: failure reason
        rank = os.environ.get("RANK", 0)
        failure_log_dir = os.path.join(args.output_dir, f"failure_logs")
        os.makedirs(failure_log_dir, exist_ok=True)
        failure_reason_file = os.path.join(failure_log_dir, f"failed.{rank}.log")
        with open(failure_reason_file, "w") as f:
            # source failure
            traceback.print_exc(file=f)


        raise e
            
