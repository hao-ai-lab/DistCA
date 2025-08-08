"""
Instantiating Megatron with ray so that we can easily create a single worker to do the scheduling.
"""
import argparse

from megatron.core import mpu
from megatron.core.optimizer import get_megatron_optimizer
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.pipeline_parallel.schedules import get_forward_backward_func
from omegaconf import OmegaConf
import ray
from tensordict import TensorDict
import torch
from transformers import AutoConfig, AutoTokenizer, AutoProcessor

from d2.runtime.megatron_patch.packed_seq_params import arg_to_cuda, PingPangSingleStepPackedSeqParams, PingPangPackedSeqParams
from d2.runtime.inplace_metadata import (
    compute_attn_layout_seqlens, mlp_layout_packed_params,
)

from test_util import MegatronBaseWorker, ParallelConfig
from deprecated.test_dispatch_qkv import create_testcase_qkv
from deprecated.test_megatron_layer import create_pg, get_seqlen_shard
from megatron_test_utils import (
    get_megatron_optimizer_param_scheduler, get_model, get_torch_device, gptmodel_forward,
    hf_to_mcore_config, init_mcore_model, init_megatron_optim_config,
    make_batch_generator, print_model_size, update_model_config, unwrap_model,
)


def set_random_seed(seed):
    """Set worker side random seed."""
    import random

    import numpy as np
    import torch

    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if get_torch_device().device_count() > 0:
        from megatron.core import tensor_parallel

        tensor_parallel.model_parallel_cuda_manual_seed(seed)


class MegatronE2eWorker(MegatronBaseWorker):
    def __init__(self, rank: int, world_size: int):
        super().__init__(rank, world_size)
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
        set_random_seed(42)

    def set_config(self, dtype=torch.bfloat16, enable_gradient_checkpointing=False, gradient_checkpointing_kwargs={}):
        self.dtype = dtype
        self.enable_gradient_checkpointing = enable_gradient_checkpointing
        self.gradient_checkpointing_kwargs = gradient_checkpointing_kwargs

    def init(self, model_path):
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


def init_test(args, hidden_size_q, hidden_size_kv, worker_cls=MegatronE2eWorker):
    ray.init()
    workers = create_pg(args.num_nodes, args.num_gpus_per_node, worker_cls)
    print("Workers created")
    stride_q = hidden_size_q * torch.float16.itemsize // args.tp_size
    stride_kv = hidden_size_kv * torch.float16.itemsize * 2 // args.tp_size
    world_size = len(workers)
    # NOTE: a reason very likely causing the hanging is that
    # max_tokens_query and max_tokens_key_value are not large enough (nvshmem buffer not enough)
    max_tokens_query = args.num_tokens * world_size
    max_tokens_key_value = args.num_tokens * world_size
    parallel_config = ParallelConfig(
        tensor_model_parallel_size=args.tp_size
    )
    ray.get([worker.init_comm.remote(
        stride_q, stride_kv, max_tokens_query, max_tokens_key_value, parallel_config
    ) for worker in workers])
    print("Communication groups initialized")
    return workers


def test(args):
    seed = args.seed
    num_tokens = args.num_tokens
    max_cp_degree = args.cp_degree
    num_seqs = args.num_seqs

    model_path = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
    hf_config = AutoConfig.from_pretrained(model_path)
    hidden_size_q = hf_config.hidden_size
    hidden_size_kv = hidden_size_q * 2
    if hasattr(hf_config, "num_key_value_heads"):
        hidden_size_kv = (hidden_size_kv * hf_config.num_key_value_heads //
                          hf_config.num_attention_heads)

    workers = init_test(args, hidden_size_q, hidden_size_kv, worker_cls=MegatronE2eWorker)
    world_size = len(workers)
    refs = []
    for worker in workers:
        # NOTE: gradient checkpointing is currently disabled
        worker.set_config.remote()
        ref = worker.init.remote(model_path)
        refs.append(ref)
    ray.get(refs)
    print("=" * 20 + "model init done")

    seq_len = args.num_tokens
    refs = []
    # easiest case: all sequences run attention locally.
    (
        fwd_q_metadata_0, rev_q_metadata_0,
        fwd_kv_metadata_0, rev_kv_metadata_0,
        sp_kv_dst_0, sp_seq_lens_0, sp_query_dst_0, cp_dst_kv_len_0, seq_lens_0,
    ) = create_testcase_qkv(seed, world_size, num_tokens, max_cp_degree, num_seqs)
    (
        fwd_q_metadata_1, rev_q_metadata_1,
        fwd_kv_metadata_1, rev_kv_metadata_1,
        sp_kv_dst_1, sp_seq_lens_1, sp_query_dst_1, cp_dst_kv_len_1, seq_lens_1,
    ) = create_testcase_qkv(seed, world_size, num_tokens, max_cp_degree, num_seqs)
    (cu_seqlens_q_pp_0, cu_seqlens_kv_pp_0, max_seqlen_q_pp_0, max_seqlen_kv_pp_0, num_local_seqs_recv_pp_0) = compute_attn_layout_seqlens(
        sp_seq_lens_0, cp_dst_kv_len_0, sp_query_dst_0
    )
    (cu_seqlens_q_pp_1, cu_seqlens_kv_pp_1, max_seqlen_q_pp_1, max_seqlen_kv_pp_1, num_local_seqs_recv_pp_1) = compute_attn_layout_seqlens(
        sp_seq_lens_1, cp_dst_kv_len_1, sp_query_dst_1
    )
    input_ids = torch.randint(0, 100, (world_size, seq_len * 2))
    position_ids = torch.arange(seq_len).repeat(world_size, 2)
    for rank, worker in enumerate(workers):
        input_ids_local = input_ids[rank]
        position_ids_local = position_ids[rank]
        fwd_q_metadata_0_local = fwd_q_metadata_0.get_slice(rank)
        rev_q_metadata_0_local = rev_q_metadata_0.get_slice(rank)
        fwd_kv_metadata_0_local = fwd_kv_metadata_0.get_slice(rank)
        rev_kv_metadata_0_local = rev_kv_metadata_0.get_slice(rank)
        fwd_q_metadata_1_local = fwd_q_metadata_1.get_slice(rank)
        rev_q_metadata_1_local = rev_q_metadata_1.get_slice(rank)
        fwd_kv_metadata_1_local = fwd_kv_metadata_1.get_slice(rank)
        rev_kv_metadata_1_local = rev_kv_metadata_1.get_slice(rank)
        (cu_seqlens_q_0, cu_seqlens_kv_0, max_seqlen_q_0, max_seqlen_kv_0, num_seq_0) = get_seqlen_shard(
            cu_seqlens_q_pp_0, cu_seqlens_kv_pp_0, max_seqlen_q_pp_0, max_seqlen_kv_pp_0, num_local_seqs_recv_pp_0, rank
        )
        (cu_seqlens_q_1, cu_seqlens_kv_1, max_seqlen_q_1, max_seqlen_kv_1, num_seq_1) = get_seqlen_shard(
            cu_seqlens_q_pp_1, cu_seqlens_kv_pp_1, max_seqlen_q_pp_1, max_seqlen_kv_pp_1, num_local_seqs_recv_pp_1, rank
        )
        packed_seq_params_stage_0 = PingPangSingleStepPackedSeqParams(
            qkv_format="thd",
            cu_seqlens_q=cu_seqlens_q_0,
            cu_seqlens_kv=cu_seqlens_kv_0,
            max_seqlen_q=max_seqlen_q_0,
            max_seqlen_kv=max_seqlen_kv_0,
            mlp_to_attn_metadata=fwd_q_metadata_0_local,
            attn_to_mlp_metadata=rev_q_metadata_0_local,
            mlp_to_attn_kv_metadata=fwd_kv_metadata_0_local,
            mlp_to_attn_kv_grad_metadata=rev_kv_metadata_0_local,
        )
        packed_seq_params_stage_1 = PingPangSingleStepPackedSeqParams(
            qkv_format="thd",
            cu_seqlens_q=cu_seqlens_q_1,
            cu_seqlens_kv=cu_seqlens_kv_1,
            max_seqlen_q=max_seqlen_q_1,
            max_seqlen_kv=max_seqlen_kv_1,
            mlp_to_attn_metadata=fwd_q_metadata_1_local,
            attn_to_mlp_metadata=rev_q_metadata_1_local,
            mlp_to_attn_kv_metadata=fwd_kv_metadata_1_local,
            mlp_to_attn_kv_grad_metadata=rev_kv_metadata_1_local,
        )
        # NOTE: we don't consider that seq_lens var has padding because our data generation
        # guarantees so. However, in practice, this is not true.
        mlp_seq_params_0 = mlp_layout_packed_params(seq_lens_0[rank])
        mlp_seq_params_1 = mlp_layout_packed_params(seq_lens_1[rank])
        packed_seq_params = PingPangPackedSeqParams(
            seq_params=[packed_seq_params_stage_0, packed_seq_params_stage_1],
            mlp_layout_seq_params=[mlp_seq_params_0, mlp_seq_params_1],
            debug=False, do_gather=False,
            max_seqlen_q=torch.tensor([seq_len * 2], dtype=torch.int32)[0],
            max_seqlen_kv=torch.tensor([seq_len * 2], dtype=torch.int32)[0],
        )
        microbatch = {
            "input_ids": input_ids_local,
            "position_ids": position_ids_local,
            "packed_seq_params": packed_seq_params,
        }
        # print(rank, microbatch["packed_seq_params"])
        microbatches = [microbatch]
        ref = worker.forward_backward_batch.remote(
            microbatches=microbatches,
            normal_forward_fn=False,
            forward_only=False,
        )
        refs.append(ref)
    ray.get(refs)
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
    args = parser.parse_args()
    test(args)
