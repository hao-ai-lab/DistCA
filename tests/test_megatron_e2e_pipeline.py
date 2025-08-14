"""
Instantiating Megatron with ray so that we can easily create a single worker to do the scheduling.

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

from d2.runtime.megatron_patch.packed_seq_params import arg_to_cuda, PingPangSingleStepPackedSeqParams
from d2.runtime.inplace_metadata import mlp_layout_packed_params
from d2.runtime.megatron_patch.forward_backward_func import forward_backward_pipelining_without_interleaving as forward_backward_func

from test_util import ParallelConfig, init_worker_torch_distributed, create_qkv_dispath_with_backward
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
                model, input_ids, attention_mask, position_ids, self.tf_config.sequence_parallel, packed_seq_params, labels=input_ids.unsqueeze(0),
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
        grad_sample = unwrap_model(self.train_module[0]).decoder.layers[-1].self_attention.linear_proj.weight.main_grad.clone()

        for param in unwrap_model(self.train_module[0]).parameters():
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

    seq_lens = None
    bwd_metadata = []

    def get_microbatch(dummy_first):
        nonlocal seq_lens
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
            last_seq_lens=seq_lens,
            dummy_first=dummy_first,
            dummy_except_first=seq_lens is None,
            reverse=True,
        )
        actual_total_seq_len = seq_lens[rank].sum().item()

        rev_rank = as_world_size - rank - 1
        input_ids_local = torch.randint(0, 100, (actual_total_seq_len,))
        position_ids_local = torch.arange(actual_total_seq_len)
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
            # qkv_bwd_metadata=qkv_bwd_fa2a_metadata.get_slice(rev_rank),
            attn_out_fwd_metadata=attn_out_fwd_fa2a_metadata.get_slice(rank),
            # attn_out_bwd_metadata=attn_out_qkv_bwd_fa2a_metadata.get_slice(rev_rank),
            # bwd_packed_seq_params=bwd_packed_seq_params,
            mlp_packed_seq_params=mlp_packed_seq_params,
        )
        bwd_metadata.append(
            (qkv_bwd_fa2a_metadata.get_slice(rank), attn_out_qkv_bwd_fa2a_metadata.get_slice(rank), bwd_packed_seq_params)
        )

        microbatch = {
            "input_ids": input_ids_local,
            "position_ids": position_ids_local,
            "packed_seq_params": ping_pang_params,
        }
        return microbatch
    # print(rank, microbatch["packed_seq_params"])
    microbatches = [get_microbatch(dummy_first=False) for _ in range(args.num_microbatch)] + [
        get_microbatch(dummy_first=True) for _ in range(as_world_size - 1)
    ]
    for i, microbatch in enumerate(microbatches):
        qkv_bwd_metadata, attn_out_bwd_metadata, bwd_packed_seq_params = bwd_metadata[(i + as_world_size - 1 - as_rank * 2) % len(bwd_metadata)]
        packed_seq_params = microbatch["packed_seq_params"]
        packed_seq_params.qkv_bwd_metadata = qkv_bwd_metadata
        packed_seq_params.attn_out_bwd_metadata = attn_out_bwd_metadata
        packed_seq_params.bwd_packed_seq_params = bwd_packed_seq_params
    loss_one_sided, grad_one_sided = worker.forward_backward_batch(
        microbatches=microbatches,
        forward_only=False,
        mode="single_sided",
        with_dummy=True,
    )
    loss_orig_reimpl, grad_orig_reimpl = worker.forward_backward_batch(
        microbatches=microbatches,
        forward_only=False,
        mode="orig_reimpl",
        with_dummy=True,
    )
    loss_orig, grad_orig = worker.forward_backward_batch(
        microbatches=microbatches,
        forward_only=False,
        mode="orig_reimpl",
        with_dummy=False,
    )
    print(f"{loss_one_sided=}, {loss_orig_reimpl=}, {loss_orig=}")
    torch.testing.assert_close(grad_orig_reimpl, grad_orig)
    torch.testing.assert_close(grad_orig_reimpl, grad_one_sided, rtol=1e-3, atol=1e-3)

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
