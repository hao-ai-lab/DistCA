"""
Instantiating Megatron with ray so that we can easily create a single worker to do the scheduling.
"""
import argparse
import os

from d2.runtime import inplace_metadata
from megatron.core import mpu
from megatron.core.optimizer import get_megatron_optimizer
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.pipeline_parallel.schedules import get_forward_backward_func
from omegaconf import OmegaConf
from tensordict import TensorDict
import torch
from transformers import AutoConfig, AutoTokenizer, AutoProcessor

from d2.runtime.megatron_patch.packed_seq_params import arg_to_cuda, PingPangSingleStepPackedSeqParams, PingPangPackedSeqParams
from d2.runtime.inplace_metadata import (
    compute_attn_layout_seqlens, mlp_layout_packed_params,
)
from d2.runtime.megatron_patch.forward_backward_func import forward_backward_pipelining_without_interleaving as forward_backward_func

from test_util import MegatronBaseWorker, ParallelConfig, init_worker_torch_distributed, create_qkv_dispatch, create_fast_a2a_metadata_from_qkv_dispatch, create_qkv_dispath_with_backward
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
        local_rank = int(os.getenv("LOCAL_RANK"))
        torch.cuda.set_device(local_rank)
        torch.set_default_device(torch.device("cuda", local_rank))
        self.dtype = torch.bfloat16
        self.enable_gradient_checkpointing = False
        self.gradient_checkpointing_kwargs = {}

    def init_comm(self, stride_q: int, stride_kv: int, max_tokens_query: int, max_tokens_key_value: int,
                  parallel_config: ParallelConfig):
        super().init_comm(
            stride_q, stride_kv, max_tokens_query,
            max_tokens_key_value, parallel_config
        )

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
            k: arg_to_cuda(v) for k, v in microbatch.items()
        } for microbatch in microbatches]
        for module in self.train_module:
            unwrap_model(module).set_debug(normal_forward_fn)
        assert len(self.train_module) == 1, "only support one module"

        # forward_backward_func = get_forward_backward_func()
        n_micro_batch = len(microbatches)
        # thd layout
        total_seqlen = microbatches[0]['input_ids'].shape[0]

        def loss_func(output):
            # NOTE: this is a dummy loss function.
            loss = ((output - 1)**2).mean()
            return loss, {'loss': loss}

        dummy_microbatch = microbatches[0]  # FIXME: this is important for all-to-all

        def wrap_iter(batch_iter):
            if False:  #mpu.is_pipeline_first_stage() or mpu.is_pipeline_last_stage():
                # hardcode here: pipeline parallel add this number of dummy forwards
                for _ in range(mpu.get_data_parallel_rank()):
                    print('yield pre dummy')
                    yield dummy_microbatch
                yield from batch_iter
                for _ in range(mpu.get_pipeline_model_parallel_world_size() - mpu.get_data_parallel_rank() - 1):
                    print('yield post dummy')
                    yield dummy_microbatch
            else:
                while True:
                    yield dummy_microbatch

        def forward_step(batch_iter, model):
            batch = next(batch_iter)
            input_ids = batch['input_ids']
            print(f'{input_ids.shape=}')
            position_ids = batch['position_ids']
            attention_mask = None
            packed_seq_params = batch['packed_seq_params']
            # returns "hidden_states" if not model.post_process (not the last layer)
            # returns "logits" when label is None.
            output = gptmodel_forward(
                model, input_ids, attention_mask, position_ids, self.tf_config.sequence_parallel, packed_seq_params, labels=input_ids.unsqueeze(0),
            )
            return output, loss_func

        def dummy_backward_step(model):
            unwrap_model(model).dummy_backward(dummy_microbatch['packed_seq_params'])

        batch_generator = make_batch_generator(microbatches, vpp_size=len(self.train_module))
        batch_generator = wrap_iter(batch_generator)
        if mpu.get_pipeline_model_parallel_world_size() > 1:
            losses_reduced = forward_backward_func(
                forward_step_func=forward_step,
                data_iterator=batch_generator,
                model=self.train_module,
                num_microbatches=n_micro_batch,
                seq_length=total_seqlen,  # no use, since variable_seq_lengths=True
                micro_batch_size=1,  # no use when input_shapes was set
                forward_only=forward_only,
                dummy_bwd_func=dummy_backward_step,
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
        print(f'{losses_reduced=}')
        return losses_reduced

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
        print(f"train_module: {len(train_module)}")
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
    hidden_size_q: int, hidden_size_kv: int, num_heads: int, num_tokens: int,
    world_size: int, max_cp_degree: int, tp_size: int, pp_size: int,
    dtype, worker_cls=MegatronE2eWorker
):
    token_bytes_q = hidden_size_q * dtype.itemsize
    token_bytes_kv = hidden_size_kv * dtype.itemsize
    max_tokens_query = num_tokens * world_size
    max_tokens_key_value = num_tokens * world_size
    buffer_size = (
        token_bytes_q * max_tokens_query * 3 +
        num_heads * torch.float32.itemsize * 2 * max_tokens_query +
        token_bytes_kv * max_tokens_key_value * max_cp_degree * 2
    )
    print(f'{buffer_size=}', flush=True)
    parallel_config = ParallelConfig(
        tensor_model_parallel_size=tp_size,
        pipeline_model_parallel_size=pp_size,
    )
    worker = init_worker_torch_distributed(
        world_size, buffer_size, worker_cls, parallel_config
    )
    print("Communication groups initialized")
    return worker


def test(args):
    seed = args.seed
    num_tokens = args.num_tokens
    max_cp_degree = args.cp_degree
    num_seqs = args.num_seqs
    tp_size = args.tp_size
    pp_size = args.pp_size
    world_size = args.num_nodes * args.num_gpus_per_node
    total_seq_len = args.num_tokens

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
        hidden_size_q, hidden_size_kv, hf_config.num_attention_heads, num_tokens,
        world_size, max_cp_degree, tp_size, pp_size,
        dtype, MegatronE2eWorker
    )
    worker.set_config(dtype=dtype)
    worker.init(model_path, seed=seed)
    # set again to potentially adapt to the ray launch case.
    set_random_seed(seed, set_megatron=False)

    rank = as_rank = worker.as_rank
    as_world_size = worker.as_world_size

    hidden_size_q_tp = hidden_size_q // tp_size
    hidden_size_k_tp = hidden_size_kv // tp_size

    (
        # fwd_q_metadata, rev_q_metadata, fwd_k_metadata, rev_k_metadata,
        # attention_metadata_attn_layout, intermediates, seq_lens
        fwd_metadata_q, bwd_metadata_q, fwd_metadata_kv, bwd_metadata_kv,
        fa_fwd_params, fa_bwd_params,
        qkv_fwd_fa2a_metadata, qkv_bwd_fa2a_metadata,
        attn_out_fwd_fa2a_metadata, attn_out_qkv_bwd_fa2a_metadata,
        seq_lens,
    ) = create_qkv_dispath_with_backward(
        world_size, total_seq_len, num_seqs, max_cp_degree,
        hidden_size_q_tp, hidden_size_k_tp, element_size, hf_config.num_attention_heads * torch.float32.itemsize // element_size,
        return_mlp_no_shard_seq_lens=True,
        fixed_seq_lens=True,
    )

    set_random_seed(seed, set_megatron=False)
    # input_ids = torch.randint(0, 100, (as_world_size, total_seq_len))
    # position_ids = torch.arange(total_seq_len).repeat(as_world_size, 1)
    # input_ids_local = input_ids[as_rank]
    # position_ids_local = position_ids[as_rank]
    # print(input_ids_local.shape, position_ids_local.shape)
    input_ids_local = torch.randint(0, 100, (total_seq_len,))
    position_ids_local = torch.arange(total_seq_len)
    (cu_seqlens_q, cu_seqlens_kv, max_seqlen_q, max_seqlen_kv, *_) = fa_bwd_params
    bwd_packed_seq_params = PackedSeqParams(
        cu_seqlens_q=cu_seqlens_q[rank],
        cu_seqlens_kv=cu_seqlens_kv[rank],
        max_seqlen_q=max_seqlen_q[rank],
        max_seqlen_kv=max_seqlen_kv[rank],
    )
    mlp_packed_seq_params = mlp_layout_packed_params(seq_lens[rank][:num_seqs])
    (cu_seqlens_q, cu_seqlens_kv, max_seqlen_q, max_seqlen_kv, *_) = fa_fwd_params

    ping_pang_params = PingPangSingleStepPackedSeqParams(
        qkv_format="thd",
        cu_seqlens_q=cu_seqlens_q[rank],
        cu_seqlens_kv=cu_seqlens_kv[rank],
        max_seqlen_q=max_seqlen_q[rank],
        max_seqlen_kv=max_seqlen_kv[rank],
        qkv_fwd_metadata=qkv_fwd_fa2a_metadata.get_slice(rank),
        qkv_bwd_metadata=qkv_bwd_fa2a_metadata.get_slice(rank),
        attn_out_fwd_metadata=attn_out_fwd_fa2a_metadata.get_slice(rank),
        attn_out_bwd_metadata=attn_out_qkv_bwd_fa2a_metadata.get_slice(rank),
        bwd_packed_seq_params=bwd_packed_seq_params,
        mlp_packed_seq_params=mlp_packed_seq_params,
    )
    # ping_pang_params_0 = get_single_step_packed_seq_params(
    #     fa2a_metadata_0, attn_metadata_0, as_rank
    # )
    # ping_pang_params_1 = get_single_step_packed_seq_params(
    #     fa2a_metadata_1, attn_metadata_1, as_rank
    # )

    # NOTE: we don't consider that seq_lens var has padding because our data generation
    # guarantees so. However, in practice, this is not true.
    # mlp_seq_params_0 = mlp_layout_packed_params(raw_seq_lens_0[as_rank])
    # mlp_seq_params_1 = mlp_layout_packed_params(raw_seq_lens_1[as_rank])
    # ping_pang_params = PingPangPackedSeqParams(
    #     seq_params=[ping_pang_params_0, ping_pang_params_1],
    #     mlp_layout_seq_params=[mlp_seq_params_0, mlp_seq_params_1],
    #     max_seqlen_q=torch.tensor([total_seq_len * 2], dtype=torch.int32)[0],
    #     max_seqlen_kv=torch.tensor([total_seq_len * 2], dtype=torch.int32)[0],
    #     qkv_format="thd",
    # )
    microbatch = {
        "input_ids": input_ids_local,
        "position_ids": position_ids_local,
        "packed_seq_params": ping_pang_params,
    }
    # print(rank, microbatch["packed_seq_params"])
    microbatches = [microbatch, microbatch, microbatch, microbatch]
    worker.forward_backward_batch(
        microbatches=microbatches,
        normal_forward_fn=False,
        forward_only=False,
    )
    # microbatches = [microbatch, microbatch, microbatch, microbatch, microbatch, microbatch, microbatch, microbatch]
    # worker.forward_backward_batch(
    #     microbatches=microbatches,
    #     normal_forward_fn=False,
    #     forward_only=False,
    # )
    print("=" * 20 + "forward_backward_batch attention server, done")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-tokens", type=int, default=1024)
    parser.add_argument("--cp-degree", type=int, default=2)
    parser.add_argument("--num-seqs", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-nodes", type=int, default=1)
    parser.add_argument("--num-gpus-per-node", type=int, default=2)
    parser.add_argument("--tp-size", type=int, default=1)
    parser.add_argument("--pp-size", type=int, default=2)
    args = parser.parse_args()
    test(args)
