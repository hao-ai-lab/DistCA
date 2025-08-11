"""
NVTE_ALLOW_NONDETERMINISTIC_ALGO=0 \
torchrun --nnodes 1 --nproc_per_node 4 test_e2e_anchor.py \
    --num-nodes=1 --num-gpus-per-node=4 --tp-size=2 \
    --num-tokens 16384 --num-layers 4
"""
from datetime import datetime
import os
import gc
import pytz
import json
import os
import os
import time
import rich
import argparse
import sys
from megatron.core import mpu
from megatron.core.optimizer import get_megatron_optimizer
from megatron.core.pipeline_parallel.schedules import get_forward_backward_func
from omegaconf import OmegaConf
import torch
from transformers import AutoConfig, AutoTokenizer, AutoProcessor

from d2.runtime.megatron_patch.packed_seq_params import arg_to_cuda, PingPangPackedSeqParams
from d2.runtime.inplace_metadata import mlp_layout_packed_params

from test_util import MegatronBaseWorker, ParallelConfig, init_worker_torch_distributed
from test_pingpang_layer import create_one_batch, get_single_step_packed_seq_params
from megatron_test_utils import (
    get_megatron_optimizer_param_scheduler, get_model, get_torch_device, gptmodel_forward,
    hf_to_mcore_config, init_mcore_model, init_megatron_optim_config,
    make_batch_generator, print_model_size, update_model_config, unwrap_model,
)
from contextlib import contextmanager

@contextmanager
def nvtx_range(name):
    torch.cuda.nvtx.range_push(name)
    yield
    torch.cuda.nvtx.range_pop()

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
        override_transformer_config = OmegaConf.create()
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

        def loss_func(output):
            # NOTE: this is a dummy loss function.
            loss = ((output - 1)**2).mean()
            return loss, {'loss': loss}

        def forward_step(batch_iter, model):
            batch = next(batch_iter)
            input_ids = batch['input_ids']
            position_ids = batch['position_ids']
            attention_mask = None
            packed_seq_params = batch['packed_seq_params']
            # returns "hidden_states" if not model.post_process (not the last layer)
            # returns "logits" when label is None.
            torch.cuda.nvtx.range_push("forward_step")
            output = gptmodel_forward(
                model, input_ids, attention_mask, position_ids, self.tf_config.sequence_parallel, packed_seq_params
            )
            torch.cuda.nvtx.range_pop()
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
    token_bytes_q = hidden_size_q * dtype.itemsize
    token_bytes_kv = hidden_size_kv * dtype.itemsize
    max_tokens_query = num_tokens * world_size
    max_tokens_key_value = num_tokens * world_size
    buffer_size = (
        token_bytes_q * max_tokens_query +
        token_bytes_kv * max_tokens_key_value * max_cp_degree * 2
    )
    parallel_config = ParallelConfig(
        tensor_model_parallel_size=tp_size
    )
    worker = init_worker_torch_distributed(
        world_size, buffer_size, worker_cls, parallel_config
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
iterated_plans = []

def setup_global_batch(
    total_seq_len, 
    up_sample_factor=2,
):
    global GLOBAL_BATCH
    if GLOBAL_BATCH is not None:
        return

    batches = []
    for i in range(1000):
        batches.append([total_seq_len])
    GLOBAL_BATCH = iter(batches)
    return


def get_next_batch(dp_size):
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


from test_util import (
    create_raw_qkv_dispatch,
    create_qkv_dispatch_with_custom_mapping,
)

from d2.runtime.inplace_metadata import (
    compute_metadata,
    compute_metadata_kv,
    compute_attn_layout_seqlens,
)

# D2 specific imports
from d2.runtime.fast_alltoall_metadata import compute_fa2a_metadata_from_logical_metadata


def create_qkv_dispatch(
    world_size: int, total_seq_len: int, num_seqs: int, max_cp_degree: int,
    return_intermediate: bool=False, return_mlp_no_shard_seq_lens: bool=False
):
    (cp_seq_lens, num_cp_shards, cp_query_dst,
     kv_to_q_mapping, kv_to_q_rank, kv_context_size,
     q_to_num_kv_seq, q_to_num_kv_tokens,
     seq_lens) = create_raw_qkv_dispatch(
        world_size, total_seq_len, num_seqs, max_cp_degree,
        return_mlp_no_shard_seq_lens
    )
    fwd_q_metadata, rev_q_metadata, q_intermediates = compute_metadata(
        cp_seq_lens, cp_query_dst, return_intermediate=True
    )
    _, q_seq_to_dst, _ = q_intermediates
    pad_len = torch.max(num_cp_shards)
    fwd_k_metadata, rev_k_metadata, kv_intermediates = compute_metadata_kv(
        kv_to_q_mapping, kv_to_q_rank, kv_context_size, q_to_num_kv_seq,
        q_to_num_kv_tokens, cp_seq_lens, num_cp_shards, cp_query_dst,
        q_seq_to_dst.squeeze(2), pad_len,
        return_intermediate=True
    )
    attention_metadata = compute_attn_layout_seqlens(
        cp_seq_lens, q_to_num_kv_tokens, cp_query_dst, shard_to_tuple=True
    )
    ret = (
        fwd_q_metadata, rev_q_metadata, fwd_k_metadata, rev_k_metadata, attention_metadata
    )
    if return_intermediate:
        intermediates = q_intermediates + kv_intermediates
        ret += (intermediates,)
    ret += seq_lens
    return ret


# ========== D2 Specific Functions ==========

def test_create_qkv_dispatch_balanced_flops(
    world_size_, total_seq_len_, seq_lens, max_cp_degree_, 
    verbose=False, return_intermediate=False, return_mlp_no_shard_seq_lens=False,
    return_items=False, replan_iter=1,
):
    K = 1024

    from d2.planner.equal_flops import (
        batch_to_items, 
        plan_relocation,
        item_to_intermediate_tensors,
        postprocess_items,
    )

    items_list = seq_lens
    
    rank = torch.distributed.get_rank()
    if rank == 0:
        rich.print(f"Generate Sample ID={ITERATION_ID}: {items_list}")

    total_seq_len = max(sum(batch) for batch in items_list)
    assert total_seq_len == total_seq_len_, f"This test forces total_seq_len = {total_seq_len_}, got {total_seq_len=}"

    items = batch_to_items(items_list)
    for _ in range(replan_iter):
        items = plan_relocation(items, verbose=False, plot=False)
    if replan_iter > 1:
        items = postprocess_items(items)
    

    world_info, (items, info_mapping, info_list), (seq_lens, cp_num, cp_dst, seq_shard_lens) = item_to_intermediate_tensors(items)    

    world_size = world_info["world_size"]

    assert world_size == world_size_

    ret = create_qkv_dispatch_with_custom_mapping(
        world_size, 
        seq_lens,
        cp_num,
        cp_dst,
        seq_shard_lens,
        verbose=verbose, return_intermediate=return_intermediate,
    )
    if return_mlp_no_shard_seq_lens:
        ret += (seq_lens,)
    if return_items:
        ret += (items,)
    return ret


def create_one_batch_balanced_flops(
    world_size: int, total_seq_len: int, num_seqs: int, max_cp_degree: int,
    hidden_size_q: int, hidden_size_k: int, element_size: int,
    return_items: bool=False, replan_iter: int=1,
):
    ret = test_create_qkv_dispatch_balanced_flops(
        world_size, total_seq_len, num_seqs, max_cp_degree,
        return_intermediate=True, return_mlp_no_shard_seq_lens=True,
        return_items=return_items, replan_iter=replan_iter,
    )
    if return_items:
        (
            fwd_q_metadata, rev_q_metadata, fwd_k_metadata, rev_k_metadata,
            attention_metadata_attn_layout, intermediates, seq_lens, items
        ) = ret
    else:
        (
            fwd_q_metadata, rev_q_metadata, fwd_k_metadata, rev_k_metadata,
            attention_metadata_attn_layout, intermediates, seq_lens
        ) = ret
    # NOTE: this already adds prepended zeros and is sharded to tuples (remove padding seqs)
    (cu_seqlens_q, cu_seqlens_kv, max_seqlen_q, max_seqlen_kv,
     num_local_seqs_recv) = attention_metadata_attn_layout

    (qkv_fwd_fa2a_metadata, qkv_rev_fa2a_metadata,
     attn_out_fwd_fa2a_metadata, attn_out_rev_fa2a_metadata,
    ) = compute_fa2a_metadata_from_logical_metadata(
        fwd_q_metadata, rev_q_metadata, fwd_k_metadata, rev_k_metadata,
        intermediates, total_seq_len, hidden_size_q, hidden_size_k,
        element_size,
    )
    logical_metadata = (
        fwd_q_metadata, rev_q_metadata, fwd_k_metadata, rev_k_metadata,
    )
    fa2a_metadata = (
        qkv_fwd_fa2a_metadata, qkv_rev_fa2a_metadata,
        attn_out_fwd_fa2a_metadata, attn_out_rev_fa2a_metadata,
    )
    
    attn_metadata = (
        cu_seqlens_q, cu_seqlens_kv, max_seqlen_q, max_seqlen_kv,
    )
    raw_seq_lens = seq_lens

    """
    Override the metadata

        🟢 [Rank 0] qkv_fwd_fa2a_metadata:
        FastAlltoAllMetadata(
            fa2a_metadata=(
                tensor([[       0, 0],
                [       0,        0]]),
                tensor([[0,        0],
                [       0, 0]]),
                tensor([[       0,        0],
                [0,        0]]),
                tensor([[0,        0],
                [       0, 0]])
            ),
            send_memcpy_metadata=(
                (tensor([       0, 16777216]), tensor([0])),
                (tensor([[33554432, 37748736],
                [41943040,        0]]), tensor([[33554432],
                [       0]])),
                (tensor([[46137344, 50331648],
                [54525952,        0]]), tensor([[41943040],
                [       0]]))
            ),
            recv_memcpy_metadata=(
                (tensor([       0, 16777216]), tensor([0])),
                (tensor([33554432, 41943040, 37748736]), tensor([33554432])),
                (tensor([46137344, 54525952, 50331648]), tensor([41943040]))
            ),
            my_rank_send_offset=[0, 0],
            my_rank_recv_offset=[0, 0],
            my_rank_send_sz=[58720256, 50331648],
            seq_lens=[
                SeqLens(send_seqlens=(tensor([16384, 16384]), tensor([32768])), recv_seqlens=(tensor([16384, 16384]), tensor([32768]))),
                SeqLens(send_seqlens=(tensor([16384, 16384]), tensor([32768])), recv_seqlens=(tensor([16384, 16384, 16384]), tensor([32768])))
            ],
            tensor_shape=[
                LogicalShape(send_shape=((32768, 512), (32768, 512)), recv_shape=((32768, 512), (32768, 512))),
                LogicalShape(send_shape=((2, 32768, 128), (2, 32768, 128)), recv_shape=((49152, 128), (32768, 128)))
            ],
            kv_replica_mask=(tensor([[1, 1],
                [1, 0]], dtype=torch.int8), tensor([[1, 0]], dtype=torch.int8)),
            single_stream=False
        )
        🟢 [Rank 0] qkv_rev_fa2a_metadata:
        FastAlltoAllMetadata(
            fa2a_metadata=(
                tensor([[       0, 0],
                [       0,        0]]),
                tensor([[0,        0],
                [       0, 0]]),
                tensor([[       0,        0],
                [0,        0]]),
                tensor([[0,        0],
                [       0, 0]])
            ),
            send_memcpy_metadata=(
                (tensor([       0, 16777216]), tensor([0])),
                (tensor([33554432, 41943040, 37748736]), tensor([33554432])),
                (tensor([46137344, 54525952, 50331648]), tensor([41943040]))
            ),
            recv_memcpy_metadata=(
                (tensor([       0, 16777216]), tensor([0])),
                (tensor([[33554432, 37748736],
                [41943040,        0]]), tensor([[33554432],
                [       0]])),
                (tensor([[46137344, 50331648],
                [54525952,        0]]), tensor([[41943040],
                [       0]]))
            ),
            my_rank_send_offset=[0, 0],
            my_rank_recv_offset=[0, 0],
            my_rank_send_sz=[58720256, 50331648],
            seq_lens=(
                SeqLens(send_seqlens=(tensor([16384, 16384]), tensor([32768])), recv_seqlens=(tensor([16384, 16384]), tensor([32768]))),
                SeqLens(send_seqlens=(tensor([16384, 16384, 16384]), tensor([32768])), recv_seqlens=(tensor([16384, 16384]), tensor([32768])))
            ),
            tensor_shape=(
                LogicalShape(send_shape=((32768, 512), (32768, 512)), recv_shape=((32768, 512), (32768, 512))),
                LogicalShape(send_shape=((49152, 128), (32768, 128)), recv_shape=((2, 32768, 128), (2, 32768, 128)))
            ),
            kv_replica_mask=(tensor([[1, 1],
                [1, 0]], dtype=torch.int8), tensor([[1, 0]], dtype=torch.int8)),
            single_stream=False
        )
        🟢 [Rank 0] attn_out_fwd_fa2a_metadata:
        FastAlltoAllMetadata(
            fa2a_metadata=(
                tensor([[       0, 0],
                [       0,        0]]),
                tensor([[0,        0],
                [       0, 0]]),
                tensor([[       0,        0],
                [0,        0]]),
                tensor([[0,        0],
                [       0, 0]])
            ),
            send_memcpy_metadata=((tensor([       0, 16777216]), tensor([0])),),
            recv_memcpy_metadata=((tensor([       0, 16777216]), tensor([0, 0])),),
            my_rank_send_offset=[0, 0],
            my_rank_recv_offset=[0, 0],
            my_rank_send_sz=[33554432, 33554432],
            seq_lens=(SeqLens(send_seqlens=(tensor([16384, 16384]), tensor([32768])), recv_seqlens=(tensor([16384, 16384]), tensor([32768]))),),
            tensor_shape=(LogicalShape(send_shape=((32768, 512), (32768, 512)), recv_shape=((32768, 512), (32768, 512))),),
            kv_replica_mask=None,
            single_stream=False
        )
        🟢 [Rank 0] attn_out_rev_fa2a_metadata:
        FastAlltoAllMetadata(
            fa2a_metadata=(
                tensor([[       0, 0],
                [       0,        0]]),
                tensor([[0,        0],
                [       0, 0]]),
                tensor([[       0,        0],
                [0,        0]]),
                tensor([[0,        0],
                [       0, 0]])
            ),
            send_memcpy_metadata=((tensor([       0, 16777216]), tensor([0, 0])),),
            recv_memcpy_metadata=((tensor([       0, 16777216]), tensor([0])),),
            my_rank_send_offset=[0, 0],
            my_rank_recv_offset=[0, 0],
            my_rank_send_sz=[33554432, 33554432],
            seq_lens=[SeqLens(send_seqlens=(tensor([16384, 16384]), tensor([32768])), recv_seqlens=(tensor([16384, 16384]), tensor([32768])))],
            tensor_shape=[LogicalShape(send_shape=((32768, 512), (32768, 512)), recv_shape=((32768, 512), (32768, 512)))],
            kv_replica_mask=None,
            single_stream=False
        )
    """

    # # Rewrite the metadata
    # from torch import tensor
    # qkv_fwd_fa2a_metadata.fa2a_metadata = (
    #     tensor([[       0, 0],
    #     [       0,        0]]),
    #     tensor([[0,        0],
    #     [       0, 0]]),
    #     tensor([[       0,        0],
    #     [0,        0]]),
    #     tensor([[0,        0],
    #     [       0, 0]])
    # )

    # qkv_rev_fa2a_metadata.fa2a_metadata = (
    #     tensor([[       0, 0],
    #     [       0,        0]]),
    #     tensor([[0,        0],
    #     [       0, 0]]),
    #     tensor([[       0,        0],
    #     [0,        0]]),
    #     tensor([[0,        0],
    #     [       0, 0]])
    # )

    # attn_out_fwd_fa2a_metadata.fa2a_metadata = (
    #     tensor([[       0, 0],
    #     [       0,        0]]),
    #     tensor([[0,        0],
    #     [       0, 0]]),
    #     tensor([[       0,        0],
    #     [0,        0]]),
    #     tensor([[0,        0],
    #     [       0, 0]])
    # )

    # attn_out_rev_fa2a_metadata.fa2a_metadata = (
    #     tensor([[       0, 0],
    #     [       0,        0]]),
    #     tensor([[0,        0],
    #     [       0, 0]]),
    #     tensor([[       0,        0],
    #     [0,        0]]),
    #     tensor([[0,        0],
    #     [       0, 0]])
    # )


    
    rank = torch.distributed.get_rank()
    if rank == 0:
        rich.print(f"🟢 [Rank {rank}] items:", items)
        rich.print(f"🟢 [Rank {rank}] cu_seqlens_q: {cu_seqlens_q}")
        rich.print(f"🟢 [Rank {rank}] cu_seqlens_kv: {cu_seqlens_kv}")
        rich.print(f"🟢 [Rank {rank}] max_seqlen_q: {max_seqlen_q}")
        rich.print(f"🟢 [Rank {rank}] max_seqlen_kv: {max_seqlen_kv}")
        rich.print(f"🟢 [Rank {rank}] seq_lens: {seq_lens}")

        rich.print(f"🟢 [Rank {rank}] qkv_fwd_fa2a_metadata:", qkv_fwd_fa2a_metadata)
        rich.print(f"🟢 [Rank {rank}] qkv_rev_fa2a_metadata:", qkv_rev_fa2a_metadata)
        rich.print(f"🟢 [Rank {rank}] attn_out_fwd_fa2a_metadata:", attn_out_fwd_fa2a_metadata)
        rich.print(f"🟢 [Rank {rank}] attn_out_rev_fa2a_metadata:", attn_out_rev_fa2a_metadata)

    
    # exit(0)

    ret_vals = (logical_metadata, fa2a_metadata, attn_metadata, raw_seq_lens)
    if return_items:
        ret_vals += (items,)
    return ret_vals



def test(args):
    seed = args.seed
    num_tokens = args.num_tokens
    max_cp_degree = args.cp_degree
    num_seqs = args.num_seqs
    tp_size = args.tp_size
    world_size = args.num_nodes * args.num_gpus_per_node
    total_seq_len = args.num_tokens
    num_layers = args.num_layers
    model_path = args.model_path
    max_sample_id = args.max_sample_id
    up_sample_factor = args.up_sample_factor
    replan_iter = args.replan_iter
    if num_layers is not None:
        os.environ["NUM_LAYERS"] = str(num_layers)

    mode = args.mode
    output_file = args.output_file
    if output_file is None:
        # then store in current directory
        now = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = os.path.join(os.getcwd(), f"{now}.{mode}.json")
    output_dir = os.path.dirname(output_file)
    os.makedirs(output_dir, exist_ok=True)

    dtype = torch.bfloat16
    element_size = dtype.itemsize

    setup_global_batch(total_seq_len, up_sample_factor)

    hf_config = AutoConfig.from_pretrained(model_path)
    hidden_size_q = hf_config.hidden_size

    hidden_size_kv = hidden_size_q
    if hasattr(hf_config, "num_key_value_heads"):
        hidden_size_kv = (hidden_size_kv * hf_config.num_key_value_heads //
                          hf_config.num_attention_heads)

    worker: MegatronE2eWorker = init_megatron_e2e_test(
        hidden_size_q, hidden_size_kv, num_tokens,
        world_size, max_cp_degree, tp_size,
        dtype, MegatronE2eWorker
    )
    worker.set_config(dtype=dtype)
    worker.init(model_path, seed=seed)
    # set again to potentially adapt to the ray launch case.
    set_random_seed(seed, set_megatron=False)

    rank = worker.rank
    as_rank = worker.as_rank
    as_world_size = worker.as_world_size

    hidden_size_q_tp = hidden_size_q // tp_size
    hidden_size_k_tp = hidden_size_kv // tp_size

    # TODO(Refactor): Properly refactor this into a function and we call it multiple times
    max_sample_id = max_sample_id
    sample_times = []
    for sample_id in range(max_sample_id):
        
        
        _seq_lens = get_next_batch(as_world_size * 2)

        if mode == "baseline":
            # Set forward function mode based on test mode
            # normal_forward_fn = (mode == "baseline")
            normal_forward_fn = True
            # TODO: Adding proper support for context parallel in megatron.
            # Baseline mode: Use simple batch generation
            seq_lens = _seq_lens[2 * as_rank] + _seq_lens[2 * as_rank + 1]

            total_seq_len_x2 = total_seq_len * 2
            input_ids = torch.randint(100, 10000, (as_world_size, total_seq_len_x2))
            input_ids_local = input_ids[as_rank]
            
            # Use normal packed seq params for baseline
            seq_lens_local = torch.tensor(seq_lens, dtype=torch.int32)
            packed_seq_params = mlp_layout_packed_params(seq_lens_local)
            
            position_ids = torch.arange(total_seq_len, dtype=torch.int64).repeat(as_world_size, 2)
            position_ids_local = position_ids[as_rank]

            microbatch = {
                "input_ids": input_ids_local,
                "position_ids": position_ids_local,
                "packed_seq_params": packed_seq_params,
            }
            
        elif mode == "d2":
            normal_forward_fn = False
            # D2 mode: Use balanced flops planning and ping-pang parameters
            seq_lens_0 = _seq_lens[:as_world_size]
            seq_lens_1 = _seq_lens[as_world_size:]
            seq_lens = seq_lens_0 + seq_lens_1 # Not used for logging in D2 mode
            _, fa2a_metadata_0, attn_metadata_0, raw_seq_lens_0, plan_items_0 = create_one_batch_balanced_flops(
                as_world_size, total_seq_len, seq_lens_0, max_cp_degree,
                hidden_size_q_tp, hidden_size_k_tp, element_size,
                replan_iter=replan_iter,
                return_items=True, 
            )
            _, fa2a_metadata_1, attn_metadata_1, raw_seq_lens_1, plan_items_1 = create_one_batch_balanced_flops(
                as_world_size, total_seq_len, seq_lens_1, max_cp_degree,
                hidden_size_q_tp, hidden_size_k_tp, element_size,
                replan_iter=replan_iter,
                return_items=True, 
            )
            iterated_plans.append([plan_items_0, plan_items_1])

            # if rank == 0:
            #     rich.print(f"🟢 [Rank {rank}] fa2a_metadata_0: {fa2a_metadata_0}")
            #     rich.print(f"🟢 [Rank {rank}] fa2a_metadata_1: {fa2a_metadata_1}")
            #     rich.print(f"🟢 [Rank {rank}] attn_metadata_0: {attn_metadata_0}")
            #     rich.print(f"🟢 [Rank {rank}] attn_metadata_1: {attn_metadata_1}")
            #     rich.print(f"🟢 [Rank {rank}] raw_seq_lens_0: {raw_seq_lens_0}")
            #     rich.print(f"🟢 [Rank {rank}] raw_seq_lens_1: {raw_seq_lens_1}")
            #     rich.print(f"🟢 [Rank {rank}] plan_items_0: {plan_items_0}")
            #     rich.print(f"🟢 [Rank {rank}] plan_items_1: {plan_items_1}")

            set_random_seed(seed, set_megatron=False)
            input_ids = torch.randint(0, 100, (as_world_size, total_seq_len * 2))
            position_ids = torch.arange(total_seq_len).repeat(as_world_size, 2)
            input_ids_local = input_ids[as_rank]
            position_ids_local = position_ids[as_rank]
            
            ping_pang_params_0 = get_single_step_packed_seq_params(
                fa2a_metadata_0, attn_metadata_0, as_rank
            )
            ping_pang_params_1 = get_single_step_packed_seq_params(
                fa2a_metadata_1, attn_metadata_1, as_rank
            )

            # NOTE: we don't consider that seq_lens var has padding because our data generation
            # guarantees so. However, in practice, this is not true.
            mlp_seq_params_0 = mlp_layout_packed_params(raw_seq_lens_0[as_rank])
            mlp_seq_params_1 = mlp_layout_packed_params(raw_seq_lens_1[as_rank])
            
            from d2.runtime.megatron_patch.packed_seq_params import PingPangPackedSeqParams
            packed_seq_params = PingPangPackedSeqParams(
                seq_params=[ping_pang_params_0, ping_pang_params_1],
                mlp_layout_seq_params=[mlp_seq_params_0, mlp_seq_params_1],
                max_seqlen_q=torch.tensor([total_seq_len * 2], dtype=torch.int32)[0],
                max_seqlen_kv=torch.tensor([total_seq_len * 2], dtype=torch.int32)[0],
                qkv_format="thd",
            )
            
            microbatch = {
                "input_ids": input_ids_local,
                "position_ids": position_ids_local,
                "packed_seq_params": packed_seq_params,
            }
        else:
            raise ValueError(f"Unknown mode: {mode}")

        # Move all to cuda
        with nvtx_range("microbatch_to_cuda"):
            microbatch = {
                k: arg_to_cuda(v) for k, v in microbatch.items()
            }

        microbatches = [microbatch]

        if sample_id == 0:
            # Warmup for 10 iterations
            for _ in range(10):
                ref = worker.forward_backward_batch(
                    microbatches=microbatches,
                    normal_forward_fn=normal_forward_fn,
                    forward_only=False,
                )
            time.sleep(3)
            torch.cuda.synchronize()
            torch.distributed.barrier()
            if rank == 0:
                print("=" * 20 + "warmup done")
        
        # Real Experiment
        N = 4
        torch.cuda.synchronize()
        torch.distributed.barrier()

        start_time = time.time()
        torch.cuda.nvtx.range_push(f"sample_{sample_id}(repeat={N})")
        for _ in range(N):
            with nvtx_range(f"forward_backward_batch(iter={_})"):
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
            rich.print(f"[Sample ID=({2*sample_id+1}-{2*sample_id+2})] seq_lens = {seq_lens}")
            rich.print(f"[Sample ID=({2*sample_id+1}-{2*sample_id+2})] Mode={mode} forward_backward_batch: avg_time_per_iteration = {avg_duration_ms:.2f} ms")

        time.sleep(2) # to ensure the profile sees a better profiling result
        torch.cuda.synchronize()
        torch.distributed.barrier()

    torch.cuda.synchronize()
    torch.distributed.barrier()
    print("=" * 20 + "forward_backward_batch attention server, done")

    if rank == 0:    
        pst = pytz.timezone('US/Pacific')
        timestamp = datetime.now(pst).strftime("%Y-%m-%d %H:%M:%S PST")
        now_ts = datetime.now(pst).strftime("%Y%m%d_%H%M%S")
        
        rich.print(f"🟢 Test {__file__} passed")
        dp_size = as_world_size
        config = dict(
            mode=mode, 
            tp_size=tp_size, 
            dp_size=dp_size, 
            num_tokens=num_tokens, 
            model_path=model_path, 
            num_layers=num_layers, 
            max_sample_id=max_sample_id, 
            up_sample_factor=up_sample_factor, 
            replan_iter=replan_iter,
        )
        rich.print(f"🟢 Test Config: ", config)
        rich.print(f"🟢 Test DateTime: ", timestamp)
        
        # Prepare benchmark data
        benchmark_data = {
            "test_file": __file__,
            "timestamp": timestamp,
            "config": config,
            "samples": [],
            "plans": [],
        }
        
        for idx in range(len(sample_times)):
            samples = iterated_samples[idx]
            duration = sample_times[idx]
            plans = []
            if mode == "d2":
                plans = iterated_plans[idx]
            # total_flops_factor = 
            rich.print(f"🟢 Sample {idx}: {samples}, duration: {duration:.2f} ms")
            benchmark_data["samples"].append({
                "sample_id": idx,
                "samples": samples,
                "duration_ms": duration,
                "plans": plans,
            })
        
        # Write benchmark results to file
        with open(output_file, 'w') as f:
            json.dump(benchmark_data, f, indent=2)
        
        rich.print(f"🟢 Benchmark results saved to: {output_file}")

        # for idx, (sample, duration) in enumerate(zip(iterated_samples, sample_times)):
        #     rich.print(f"🟢 Sample {idx}: {sample}, duration: {duration} ms")
    
    # Cleanup and exit
    
    
    # NO BARRIER - this is what was causing the hang!
    # Instead, each process cleans up independently and exits
    rich.print(f"🟢 [Rank {rank}] Finished test and prepare to exit.")

    del worker
    torch.distributed.barrier()
    rich.print(f"🟢 [Rank {rank}] Barrier passed.")
    torch.distributed.destroy_process_group()
    rich.print(f"🟢 [Rank {rank}] Process group destroyed.")



    def exit_program():
        print(f"[Rank {rank}] Starting aggressive cleanup process...")
        # Prevent nsys to exit...
        # Clear CUDA cache first
        try:
            if torch.cuda.is_available():
                print(f"[Rank {rank}] Clearing CUDA cache...")
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                torch.cuda.ipc_collect()
                print(f"[Rank {rank}] CUDA cleanup completed")
        except Exception as e:
            print(f"[Rank {rank}] Error in CUDA cleanup: {e}")
        
        # Force garbage collection
        try:
            
            gc.collect()
            print(f"[Rank {rank}] Garbage collection completed")
        except Exception as e:
            print(f"[Rank {rank}] Error in garbage collection: {e}")
        
        # Just force exit immediately - don't try to clean up distributed stuff
        # as it often hangs in complex setups like Megatron + NVSHMEM
        print(f"[Rank {rank}] Force exiting now...")
        
        # Immediate force exit with os._exit (bypasses Python cleanup)
        os._exit(0)

    if args.force_exit:
        exit_program()

    rich.print(f"❄️ [Rank {rank}] Right before exit.")
    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, choices=["baseline", "d2"], default="baseline", 
                        help="Test mode: 'baseline' for simple batch generation, 'd2' for balanced flops planning")
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
    parser.add_argument("--output-file", type=str, default=None)
    parser.add_argument("--replan-iter", type=int, default=1, help="(d2 only) Number of replan iterations for planner")
    parser.add_argument("--force-exit", action="store_true", default=False, help="Force exit the program")
    args = parser.parse_args()
    test(args)