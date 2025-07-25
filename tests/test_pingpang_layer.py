import argparse

import torch
import ray

from d2.runtime.inplace_metadata import compute_attn_layout_seqlens
from d2.runtime.megatron_patch.packed_seq_params import PackedSeqParams, PingPangPackedSeqParams, PingPangSingleStepPackedSeqParams

from test_dispatch_qkv import create_testcase_qkv
from test_megatron_layer import init_test, MegatronLayerWorker, get_seqlen_shard


class PingPangLayerWorker(MegatronLayerWorker):
    def __init__(self, rank: int, world_size: int):
        super().__init__(rank, world_size)
        self.stream = torch.cuda.Stream()

    def forward_ping_pang(self, tensor_input: torch.Tensor, packed_seq_params: PingPangPackedSeqParams):
        packed_seq_params = packed_seq_params.to_device()
        tensor_input = tensor_input.cuda()
        self.layer.train()
        # TODO: if not debug, add communication stream here.
        return self.layer.ping_pang_forward(tensor_input, packed_seq_params=packed_seq_params)


@torch.no_grad()
def test_dp(workers, seed, num_tokens, max_cp_degree, num_seqs, hidden_size, debug=True):
    world_size = len(workers)
    # Create two splits for ping-pong
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

    # Create tensor input
    tensor_input = torch.randn(world_size, num_tokens * 2, hidden_size, dtype=torch.float16)
    ref_ans_handles = []
    args = []
    ans_handles = []
    for rank in range(world_size):
        # of shape (num_tokens, 1, hidden_size)
        tensor_input_local = tensor_input[rank].unsqueeze(1)
        fwd_q_metadata_0_local = fwd_q_metadata_0.get_slice(rank)
        rev_q_metadata_0_local = rev_q_metadata_0.get_slice(rank)
        fwd_kv_metadata_0_local = fwd_kv_metadata_0.get_slice(rank)
        rev_kv_metadata_0_local = rev_kv_metadata_0.get_slice(rank)
        fwd_q_metadata_1_local = fwd_q_metadata_1.get_slice(rank)
        rev_q_metadata_1_local = rev_q_metadata_1.get_slice(rank)
        fwd_kv_metadata_1_local = fwd_kv_metadata_1.get_slice(rank)
        rev_kv_metadata_1_local = rev_kv_metadata_1.get_slice(rank)

        # Create packed seq params metadata
        # Normal forward. No layout switch. Batches are in the normal data parallel on each rank.
        seq_lens_local_0 = seq_lens_0[rank]
        seq_lens_local_1 = seq_lens_1[rank]
        seq_lens_local = torch.cat([seq_lens_local_0, seq_lens_local_1], dim=0)
        cu_seqlens_q = torch.cat([
            torch.zeros((1,), dtype=seq_lens_local.dtype, device=seq_lens_local.device),
            seq_lens_local.cumsum(dim=0)
        ])
        cu_seqlens_kv = cu_seqlens_q.clone()
        max_seqlen_q = seq_lens_local.max()
        max_seqlen_kv = max_seqlen_q.clone()
        packed_seq_params_normal = PackedSeqParams(
            qkv_format="thd",
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_kv=cu_seqlens_kv,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_kv=max_seqlen_kv,
        )
        # Ping-pong forward. cu_seqlens is in a special layout.
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
        packed_seq_params_ping_pang = PingPangPackedSeqParams(
            debug=debug,
            seq_params=[packed_seq_params_stage_0, packed_seq_params_stage_1],
        )

        # Compute the reference answer and test result
        ref_ans_handle = workers[rank].forward_normal.remote(
            tensor_input_local, packed_seq_params_normal
        )

        args.append(
            (tensor_input_local, packed_seq_params_ping_pang)
        )
        ref_ans_handles.append(ref_ans_handle)
    ref_output = ray.get(ref_ans_handles)
    ref_debug = [ref[1] for ref in ref_output]
    ref_ans = [ref[0] for ref in ref_output]

    for rank in range(world_size):
        ans_handle = workers[rank].forward_ping_pang.remote(*args[rank])
        ans_handles.append(ans_handle)
    ans = ray.get(ans_handles)
    torch.testing.assert_close(ref_ans, ans)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-tokens", type=int, default=1024)
    parser.add_argument("--cp-degree", type=int, default=2)
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--num-seqs", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-nodes", type=int, default=1)
    parser.add_argument("--num-gpus-per-node", type=int, default=2)
    parser.add_argument("--tp-size", type=int, default=1)
    args = parser.parse_args()
    workers = init_test(args, worker_cls=PingPangLayerWorker)
    test_dp(workers, args.seed, args.num_tokens, args.cp_degree, args.num_seqs, args.hidden_size)
