"""
Megatron E2E Base Test. 

Note: You should use `test_e2e_baseline.py` for testing. 
For testing, we want to forward x2 batches at a time, 
instead of 1 batch at a time for this one.

# Debug

```bash
NVSHMEM_IB_ENABLE_IBGDA=true NVTE_ALLOW_NONDETERMINISTIC_ALGO=0 \
torchrun --nnodes=1 --nproc_per_node=4 --node_rank=0 --master_addr=<master_addr> \
    --master_port=29500 test_e2e_base.py --num-nodes=1 --num-gpus-per-node=4 --tp-size=1 --num-tokens 4096
```

# Benchmark

# ⚪ Not Tested: Node = 2, TP = 8, CPDP = 2, SeqLen = 16k 
```bash

NVSHMEM_IB_ENABLE_IBGDA=true NVTE_ALLOW_NONDETERMINISTIC_ALGO=0 \
torchrun --nnodes=2 --nproc_per_node=8 --node_rank=0 --master_addr=<master_addr> --master_port=29500 \
    test_e2e_base.py --num-nodes=2 --num-gpus-per-node=8 --tp-size=8 --num-tokens 16384

NVSHMEM_IB_ENABLE_IBGDA=true NVTE_ALLOW_NONDETERMINISTIC_ALGO=0 \
torchrun --nnodes=2 --nproc_per_node=8 --node_rank=1 --master_addr=<master_addr> --master_port=29500 \
    test_e2e_base.py --num-nodes=2 --num-gpus-per-node=8 --tp-size=8 --num-tokens 16384

```

# ⚪ Not Tested: Node = 2, TP = 8, CPDP = 2, SeqLen = 64k, num_layers = 4
```bash
NUM_LAYERS=4 \
NVSHMEM_IB_ENABLE_IBGDA=true NVTE_ALLOW_NONDETERMINISTIC_ALGO=0 \
    torchrun --nnodes=2 --nproc_per_node=8 --node_rank=0 --master_addr=<master_addr> --master_port=29500 \
        test_e2e_base.py --num-nodes=2 --num-gpus-per-node=8 --tp-size=8 --num-tokens 65536

NUM_LAYERS=4 \
NVSHMEM_IB_ENABLE_IBGDA=true NVTE_ALLOW_NONDETERMINISTIC_ALGO=0 \
    torchrun --nnodes=2 --nproc_per_node=8 --node_rank=1 --master_addr=<master_addr> --master_port=29500 \
        test_e2e_base.py --num-nodes=2 --num-gpus-per-node=8 --tp-size=8 --num-tokens 65536
```

# ⚪ Not Tested: Node = 2, TP = 8, CPDP = 2, SeqLen = 96k, num_layers = 4
```bash
NUM_LAYERS=4 \
NVSHMEM_IB_ENABLE_IBGDA=true NVTE_ALLOW_NONDETERMINISTIC_ALGO=0 \
    torchrun --nnodes=2 --nproc_per_node=8 --node_rank=0 --master_addr=<master_addr> --master_port=29500 \
        test_e2e_base.py --num-nodes=2 --num-gpus-per-node=8 --tp-size=8 --num-tokens 98304

NUM_LAYERS=4 \
NVSHMEM_IB_ENABLE_IBGDA=true NVTE_ALLOW_NONDETERMINISTIC_ALGO=0 \
    torchrun --nnodes=2 --nproc_per_node=8 --node_rank=1 --master_addr=<master_addr> --master_port=29500 \
        test_e2e_base.py --num-nodes=2 --num-gpus-per-node=8 --tp-size=8 --num-tokens 98304
```


# ⚪ Not Tested: Node = 2, TP = 8, CPDP = 2, SeqLen = 128k, num_layers = 4
```bash
NUM_LAYERS=4 \
NVSHMEM_IB_ENABLE_IBGDA=true NVTE_ALLOW_NONDETERMINISTIC_ALGO=0 \
    torchrun --nnodes=2 --nproc_per_node=8 --node_rank=0 --master_addr=fs-mbz-gpu-463 --master_port=29500 \
        test_e2e_base.py --num-nodes=2 --num-gpus-per-node=8 --tp-size=8 --num-tokens 131072

NUM_LAYERS=4 \
NVSHMEM_IB_ENABLE_IBGDA=true NVTE_ALLOW_NONDETERMINISTIC_ALGO=0 \
    torchrun --nnodes=2 --nproc_per_node=8 --node_rank=1 --master_addr=fs-mbz-gpu-463 --master_port=29500 \
        test_e2e_base.py --num-nodes=2 --num-gpus-per-node=8 --tp-size=8 --num-tokens 131072
```

# Profile: 

# ⚪ Not Tested: Node = 2, TP = 8, CPDP = 2, SeqLen = 16k 
#   (testing 8k does not make comp/comm overlap)
```bash
mkdir -p nsys-profile
# export NVSHMEM_DEBUG=DEBUG

NVSHMEM_IB_ENABLE_IBGDA=true NVTE_ALLOW_NONDETERMINISTIC_ALGO=0 \
nsys profile --force-overwrite=true -o nsys-profile/test_d2_e2e.n0.t16k.nsys-rep -t cuda,nvtx \
torchrun --nnodes=2 --nproc_per_node=8 --node_rank=0 --master_addr=<master_addr> --master_port=29500 \
    test_e2e_base.py --num-nodes=2 --num-gpus-per-node=8 --tp-size=8 --num-tokens 16384

NVSHMEM_IB_ENABLE_IBGDA=true NVTE_ALLOW_NONDETERMINISTIC_ALGO=0 \
nsys profile --force-overwrite=true -o nsys-profile/test_d2_e2e.n1.t16k.nsys-rep -t cuda,nvtx \
torchrun --nnodes=2 --nproc_per_node=8 --node_rank=1 --master_addr=<master_addr> --master_port=29500 \
    test_e2e_base.py --num-nodes=2 --num-gpus-per-node=8 --tp-size=8 --num-tokens 16384
```
"""

import time
import rich
import argparse

from megatron.core import mpu
from megatron.core.optimizer import get_megatron_optimizer
from megatron.core.pipeline_parallel.schedules import get_forward_backward_func
from omegaconf import OmegaConf
import torch
from transformers import AutoConfig, AutoTokenizer, AutoProcessor

from d2.runtime.megatron_patch.packed_seq_params import arg_to_cuda, PingPangPackedSeqParams
from d2.runtime.inplace_metadata import mlp_layout_packed_params

from test_util import MegatronBaseWorker, ParallelConfig, init_worker_torch_distributed
from test_pingpong_layer import create_one_batch, get_single_step_packed_seq_params
from megatron_test_utils import (
    get_megatron_optimizer_param_scheduler, get_model, get_torch_device, gptmodel_forward,
    hf_to_mcore_config, init_mcore_model, init_megatron_optim_config,
    make_batch_generator, print_model_size, update_model_config, unwrap_model,
)


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

def setup_global_batch(total_seq_len):
    global GLOBAL_BATCH
    if GLOBAL_BATCH is not None:
        return
    
    GLOBAL_BATCH = batch_documents(
        sample_wlbllm_docs_upsample(
            size=10000,
            upsample_long_factor=2,
            filter_threshold=10000,
            filter_ratio=0.09,
        ), max_ctx_length=total_seq_len
    )
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
)

from d2.runtime.inplace_metadata import (
    compute_metadata,
    compute_metadata_kv,
    compute_attn_layout_seqlens,
)


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


def test(args):
    seed = args.seed
    num_tokens = args.num_tokens
    max_cp_degree = args.cp_degree
    num_seqs = args.num_seqs
    tp_size = args.tp_size
    world_size = args.num_nodes * args.num_gpus_per_node
    total_seq_len = args.num_tokens

    normal_forward_fn = True

    dtype = torch.bfloat16
    element_size = dtype.itemsize

    model_path = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
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
    max_sample_id = 10
    sample_times = []
    for sample_id in range(max_sample_id):
        (
            fwd_q_metadata, rev_q_metadata, fwd_k_metadata, rev_k_metadata,
            attention_metadata_attn_layout, intermediates, seq_lens
        ) = create_qkv_dispatch(
            as_world_size, total_seq_len, num_seqs, max_cp_degree,
            return_intermediate=True, return_mlp_no_shard_seq_lens=True
        )

        # TODO(Refactor): Remove the `as_` part of the dependency.
        # thd layout's hidden size input is "t,1,h"
        # tensors = torch.randn(
        #     (as_world_size, total_seq_len, 1, hidden_size_q), dtype=dtype
        # )
        # tensor_shard = tensors[as_rank]
        input_ids = torch.randint(100, 10000, (as_world_size, total_seq_len))
        input_ids_local = input_ids[as_rank]
        # 1. normal forward. Need to provide the PackedSeqParams
        seq_lens_local = seq_lens[as_rank][:num_seqs]
        packed_seq_params = mlp_layout_packed_params(seq_lens_local)
        
        position_ids = torch.arange(total_seq_len, dtype=torch.int64).repeat(as_world_size, 2)
        position_ids_local = position_ids[as_rank]

        microbatch = {
            # "input_ids": tensor_shard,
            "input_ids": input_ids_local,
            "position_ids": position_ids_local,
            "packed_seq_params": packed_seq_params,
        }
        microbatches = [microbatch]

        if sample_id == 0:
            # Warmup
            for _ in range(5):
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
        N = 5
        torch.cuda.synchronize()

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
            rich.print(f"[Sample ID=({2*sample_id+1}-{2*sample_id+2})] forward_backward_batch: avg_time_per_iteration = {avg_duration_ms} ms")

        time.sleep(2) # to ensure the profile sees a better profiling result
        torch.cuda.synchronize()
        torch.distributed.barrier()

    torch.cuda.synchronize()
    torch.distributed.barrier()
    print("=" * 20 + "forward_backward_batch attention server, done")

    if rank == 0:
        from datetime import datetime
        import pytz
        pst = pytz.timezone('US/Pacific')
        timestamp = datetime.now(pst).strftime("%Y-%m-%d %H:%M:%S PST")
        rich.print(f"🟢 Test {__file__} passed")
        dp_size = world_size // tp_size
        config = dict(tp_size=tp_size, dp_size=dp_size, num_tokens=num_tokens)
        rich.print(f"🟢 HF Config: {hf_config}")
        rich.print(f"🟢 Test Config: ", config)
        rich.print(f"🟢 Test DateTime: ", timestamp)
        for idx in range(len(sample_times)):
            samples = iterated_samples[2*idx: 2*idx+2]
            duration = sample_times[idx]
            # total_flops_factor = 
            rich.print(f"🟢 Sample {idx}: {samples}, duration: {duration} ms")

        # for idx, (sample, duration) in enumerate(zip(iterated_samples, sample_times)):
        #     rich.print(f"🟢 Sample {idx}: {sample}, duration: {duration} ms")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-tokens", type=int, default=1024)
    parser.add_argument("--cp-degree", type=int, default=2)
    parser.add_argument("--num-seqs", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-nodes", type=int, default=1)
    parser.add_argument("--num-gpus-per-node", type=int, default=2)
    parser.add_argument("--tp-size", type=int, default=1)
    args = parser.parse_args()
    test(args)
