"""
Debug example:
NVTE_ALLOW_NONDETERMINISTIC_ALGO=0 torchrun --nnodes 1 --nproc_per_node 2 test_megatron_e2e_pipeline.py --num-gpus-per-node 2 --pp-size 2 --num-microbatch 2
"""
import argparse
from functools import partial
import os

from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.pipeline_parallel.schedules import get_forward_backward_func
import torch
from transformers import AutoConfig

from d2.runtime.megatron_patch.packed_seq_params import arg_to_cuda, PingPangSingleStepPackedSeqParams, PingPangPackedSeqParams
from d2.runtime.inplace_metadata import mlp_layout_packed_params
from d2.runtime.megatron_patch.forward_backward_func import forward_backward_pipelining_without_interleaving as forward_backward_func

from test_util import ParallelConfig, init_worker_torch_distributed, create_qkv_dispatch_pipeline_tick
from test_megatron_e2e import MegatronE2eWorker as BaseMegatronE2eWorker, set_random_seed
from megatron_test_utils import (
    gptmodel_forward, make_batch_generator, unwrap_model,
)


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
        n_micro_batch = len(microbatches) - self.as_world_size + 1
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
                model, input_ids, attention_mask, position_ids, self.tf_config.sequence_parallel,
                packed_seq_params, labels=input_ids.unsqueeze(0),
            )
            return output, loss_func

        def dummy_backward_step(model, dummy_bwd_iter, skip: bool):
            next_iter_args = next(dummy_bwd_iter)
            if skip:
                return
            unwrap_model(model).dummy_backward(next_iter_args)

        assert len(self.train_module) == 1, "only support one module"

        dummy_bwd_packed_seq_params = [
            microbatch['packed_seq_params'] for microbatch in
            (microbatches[-self.as_world_size + self.as_rank + 1:][:self.as_world_size - self.as_rank - 1] + microbatches[:self.as_rank])
        ]
        dummy_bwd_packed_seq_params = dummy_bwd_packed_seq_params[self.as_rank:] + dummy_bwd_packed_seq_params[:self.as_rank]

        assert mode in ["ping_pong", "orig_reimpl", "single_sided"]

        for module in self.train_module:
            debug = (mode != "ping_pong")
            debug_fwd_impl = mode if debug else None
            unwrap_model(module).set_debug(debug=debug, debug_fwd_impl=debug_fwd_impl)
            unwrap_model(module).train()
        assert len(self.train_module) == 1, "only support one module"

        dummy_bwd_packed_seq_params_iter = iter(dummy_bwd_packed_seq_params)
        batch_generator = make_batch_generator(
            microbatches if with_dummy else microbatches[self.as_rank:],
            vpp_size=len(self.train_module)
        )
        # if mpu.get_pipeline_model_parallel_world_size() > 1:
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
        grad_sample = unwrap_model(self.train_module[0]).decoder.layers[-2].self_attention.linear_proj.weight.main_grad.clone()

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
    token_bytes_q = hidden_size_q * dtype.itemsize
    token_bytes_kv = hidden_size_kv * dtype.itemsize
    max_tokens_query = num_tokens * world_size
    max_tokens_key_value = num_tokens * world_size
    buffer_size = (
        token_bytes_q * max_tokens_query * 3 +
        # lse_norm. TODO: the factor of 2 might be removed
        num_heads * torch.float32.itemsize * 2 * max_tokens_query +
        token_bytes_kv * max_tokens_key_value * max_cp_degree * 2
    )
    print(f'{buffer_size=}', flush=True)
    parallel_config = ParallelConfig(
        tensor_model_parallel_size=tp_size,
        pipeline_model_parallel_size=pp_size,
    )
    # TODO: to support TP, merge with main, this still uses world_size instead of as_world_size
    worker = init_worker_torch_distributed(
        world_size, buffer_size, worker_cls, parallel_config
    )
    print("Communication groups initialized")
    return worker


def create_pp_microbatches(num_microbatch: int, pp_degree: int, rank: int,
                           as_world_size: int, total_seq_len: int, num_seqs: int,
                           max_cp_degree: int, hidden_size_q_tp: int,
                           hidden_size_k_tp: int, element_size: int,
                           num_head_in_dtype: int):
    tick_seq_lens = None
    bwd_metadata = []
    microbatches = []
    for i in range(num_microbatch + pp_degree - 1):
        # For the last few ticks (drain-out ticks)
        # add a dummy forward microbatch at PP rank 0.
        add_dummy_forward = i >= num_microbatch

        (
            fa_fwd_params, fa_bwd_params,
            qkv_fwd_fa2a_metadata, qkv_bwd_fa2a_metadata,
            attn_out_fwd_fa2a_metadata, attn_out_qkv_bwd_fa2a_metadata,
            tick_seq_lens,
        ) = create_qkv_dispatch_pipeline_tick(
            as_world_size, total_seq_len, num_seqs, max_cp_degree,
            hidden_size_q_tp, hidden_size_k_tp, element_size, num_head_in_dtype,
            ref_seq_lens=tick_seq_lens,
            add_dummy=add_dummy_forward,
        )
        this_rank_num_tokens = tick_seq_lens[rank].sum().item()

        (cu_seqlens_q, cu_seqlens_kv, max_seqlen_q, max_seqlen_kv, *_) = fa_bwd_params
        bwd_packed_seq_params = PackedSeqParams(
            cu_seqlens_q=cu_seqlens_q[rank],
            cu_seqlens_kv=cu_seqlens_kv[rank],
            max_seqlen_q=max_seqlen_q[rank],
            max_seqlen_kv=max_seqlen_kv[rank],
        )
        mlp_packed_seq_params = mlp_layout_packed_params(tick_seq_lens[rank][:num_seqs])
        (cu_seqlens_q, cu_seqlens_kv, max_seqlen_q, max_seqlen_kv, *_) = fa_fwd_params

        # Create packed_params. Note that we do not add backward params here.
        ping_pang_params = PingPangSingleStepPackedSeqParams(
            qkv_format="thd",
            cu_seqlens_q=cu_seqlens_q[rank],
            cu_seqlens_kv=cu_seqlens_kv[rank],
            max_seqlen_q=max_seqlen_q[rank],
            max_seqlen_kv=max_seqlen_kv[rank],
            qkv_fwd_metadata=qkv_fwd_fa2a_metadata.get_slice(rank),
            attn_out_fwd_metadata=attn_out_fwd_fa2a_metadata.get_slice(rank),
            mlp_packed_seq_params=mlp_packed_seq_params,
        )

        input_ids_local = torch.randint(0, 100, (this_rank_num_tokens,))
        position_ids_local = torch.arange(this_rank_num_tokens)
        microbatch = {
            "input_ids": input_ids_local,
            "position_ids": position_ids_local,
            "packed_seq_params": ping_pang_params,
        }
        microbatches.append(microbatch)

        # store the corresponding bwd metadata (for later ticks)
        bwd_metadata.append(
            (qkv_bwd_fa2a_metadata.get_slice(rank), attn_out_qkv_bwd_fa2a_metadata.get_slice(rank), bwd_packed_seq_params)
        )

    # put bwd metadata to the corresponding side
    for i, microbatch in enumerate(microbatches):
        # When mb_i is computed on rank at forward tick t, assume the backward right after this forward is at tick t'.
        # mb_i requires another (pp_degree - 1 - rank) forward steps to start mb_i backward,
        # and another (pp_degree - 1 - rank) backward steps to compute mb_i backward on this rank.
        # Since in 1F1B, every forward follows a backward, so backward tick
        # t' + 2 * (pp_degree - 1 - rank) is the one to compute mb_i on this rank.
        # Note that at the end of PP warmup, forward tick is pp_degree - 1, while backward tick
        # is 0. Therefore, t - t' = pp_degree - 1, and thus
        # t' + 2 * (pp_degree - 1 - rank) == t + pp_degree - 1 - rank * 2
        bwd_metadata_idx = (i + pp_degree - 1 - rank * 2) % len(bwd_metadata)
        qkv_bwd_metadata, attn_out_bwd_metadata, bwd_packed_seq_params = bwd_metadata[bwd_metadata_idx]
        packed_seq_params = microbatch["packed_seq_params"]
        packed_seq_params.qkv_bwd_metadata = qkv_bwd_metadata
        packed_seq_params.attn_out_bwd_metadata = attn_out_bwd_metadata
        packed_seq_params.bwd_packed_seq_params = bwd_packed_seq_params
    return microbatches


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
    set_random_seed(seed, set_megatron=True)

    rank = as_rank = worker.as_rank
    as_world_size = worker.as_world_size

    hidden_size_q_tp = hidden_size_q // tp_size
    hidden_size_k_tp = hidden_size_kv // tp_size
    num_head_in_dtype = (hf_config.num_attention_heads *
                         torch.float32.itemsize // element_size)

    microbatches_0 = create_pp_microbatches(
        args.num_microbatch, pp_size, worker.as_rank,
        as_world_size, total_seq_len, num_seqs, max_cp_degree,
        hidden_size_q_tp, hidden_size_k_tp, element_size, num_head_in_dtype
    )
    microbatches_1 = create_pp_microbatches(
        args.num_microbatch, pp_size, worker.as_rank,
        as_world_size, total_seq_len, num_seqs, max_cp_degree,
        hidden_size_q_tp, hidden_size_k_tp, element_size, num_head_in_dtype
    )
    microbatches = []
    orig_impl_microbatches = []
    for mb_0, mb_1 in zip(microbatches_0, microbatches_1):
        mb_0_psp = mb_0["packed_seq_params"]
        mb_1_psp = mb_1["packed_seq_params"]
        mb_0_mlp_psp = mb_0_psp.mlp_packed_seq_params
        mb_1_mlp_psp = mb_1_psp.mlp_packed_seq_params
        ping_pong_params = PingPangPackedSeqParams(
            seq_params = [mb_0_psp, mb_1_psp],
            mlp_layout_seq_params = [mb_0_mlp_psp, mb_1_mlp_psp],
            max_seqlen_q = max(mb_0_mlp_psp.max_seqlen_q, mb_1_mlp_psp.max_seqlen_q),
            max_seqlen_kv = max(mb_0_mlp_psp.max_seqlen_kv, mb_1_mlp_psp.max_seqlen_kv),
        )
        mb = {
            "input_ids": torch.concat([mb_0["input_ids"], mb_1["input_ids"]]),
            "position_ids": torch.concat([mb_0["position_ids"], mb_1["position_ids"]]),
            "packed_seq_params": ping_pong_params,
        }
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

    loss_orig_reimpl, grad_orig_reimpl = worker.forward_backward_batch(
        microbatches=orig_impl_microbatches,
        forward_only=False,
        mode="orig_reimpl",
        with_dummy=True,
    )
    loss_orig, grad_orig = worker.forward_backward_batch(
        microbatches=orig_impl_microbatches,
        forward_only=False,
        mode="orig_reimpl",
        with_dummy=False,
    )
    print("finish baseline")
    loss_reduced, grad_sample = worker.forward_backward_batch(
        microbatches=microbatches,
        forward_only=False,
        mode="ping_pong",
        with_dummy=True,
    )
    print(f"{loss_reduced=}, {loss_orig_reimpl=}, {loss_orig=}")
    torch.testing.assert_close(grad_orig_reimpl, grad_orig)
    torch.testing.assert_close(grad_orig_reimpl, grad_sample, rtol=1e-3, atol=1e-3)

    print("=" * 20 + "forward_backward_batch attention server, done")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-tokens", type=int, default=1024)
    parser.add_argument("--cp-degree", type=int, default=2)
    parser.add_argument("--num-seqs", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-nodes", type=int, default=1)
    parser.add_argument("--num-gpus-per-node", type=int, default=4)
    parser.add_argument("--tp-size", type=int, default=1)
    parser.add_argument("--pp-size", type=int, default=4)
    parser.add_argument("--num-microbatch", type=int, default=2)
    args = parser.parse_args()
    test(args)
