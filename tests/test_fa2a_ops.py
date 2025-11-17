import torch

from d2.runtime.attn_kernels.ops import (
    a2a_memcpy_non_cp, a2a_memcpy_cp
)
from d2.runtime.compute_metadata import from_planner_output

from test_util import create_random_shard_info
from test_fa2a_metadata import (
    simulate_fa2a_copy_non_cp, simulate_fa2a_copy_cp,
    simulate_fa2a,
)


def test(args):
    world_size = args.world_size
    num_doc = args.num_docs
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    hidden_size_q = args.hidden_size_q
    hidden_size_k = args.hidden_size_k
    total_seq_len = args.num_tokens
    max_cp_degree: int = args.max_seq_shard
    torch.manual_seed(args.seed)
    dtype = torch.float16

    element_size = dtype.itemsize
    lse_size = 0    # we do not test pp here so it can be 0
    scheduler_output, _ = create_random_shard_info(args.seed, world_size, num_doc, max_cp_degree,
                                                   min_shard_len=8, tot_num_token=total_seq_len)
    qkv_fwd_fa2a_metadata, qkv_bwd_fa2a_metadata, _, _, _ = from_planner_output(
        world_size, scheduler_output, hidden_size_q, hidden_size_k, lse_size, element_size,
        is_pipeline_tick=False,
    )

    tensor_q = torch.rand((world_size, total_seq_len, hidden_size_q), device=device, dtype=dtype) + 1    # + 1 to avoid zeros
    tensor_k = torch.rand((world_size, total_seq_len, hidden_size_k), device=device, dtype=dtype) + 1
    tensor_v = torch.rand((world_size, total_seq_len, hidden_size_k), device=device, dtype=dtype) + 1
    element_size = tensor_q.element_size()

    tot_send_bytes = qkv_fwd_fa2a_metadata.fa2a_metadata[1].sum(dim=1) // element_size
    max_send_bytes = int(torch.max(tot_send_bytes).item())
    tot_recv_bytes = qkv_fwd_fa2a_metadata.fa2a_metadata[3].sum(dim=1) // element_size
    max_recv_bytes = int(torch.max(tot_recv_bytes).item())

    src_buffer = torch.zeros(
        (world_size, max_send_bytes), dtype=dtype, device=device
    )
    dst_buffer = torch.zeros(
        (world_size, max_recv_bytes), dtype=dtype, device=device
    )
    # use rank 0 for the test
    # FWD memory transfer
    # test send cpy, cp and non cp
    for rank in range(world_size):
        metadata_slice = qkv_fwd_fa2a_metadata.get_slice(rank).normalize()
        q_src_offsets, k_src_offsets, v_src_offsets = metadata_slice.send_memcpy_metadata
        q_src_seqlen = metadata_slice.seq_lens[0].send_seqlens
        kv_src_seqlen = metadata_slice.seq_lens[1].send_seqlens
        q_shard = tensor_q[rank]
        k_shard = tensor_k[rank]
        v_shard = tensor_v[rank]
        src_shard = src_buffer[rank]
        src_shard_cor = src_shard.clone()
        send_mask = metadata_slice.kv_replica_mask
        max_cp = metadata_slice.kv_replica_mask.shape[1]
        # Q shard
        src_shard_cor = simulate_fa2a_copy_non_cp(
            q_shard.flatten(), src_shard_cor, q_src_offsets, q_src_seqlen,
            hidden_size_q, element_size, is_send=True
        )
        src_shard_out = src_shard.clone()
        a2a_memcpy_non_cp(
            q_shard.cuda(), q_src_offsets.long().cuda(), q_src_seqlen.long().cuda(),
            to_nvshmem=True, buffer=src_shard_out.cuda()
        )
        torch.cuda.synchronize()
        torch.testing.assert_close(src_shard_cor, src_shard_out)
        print(f"copying rank {rank} q shard to src buffer done.")
        # K shard
        src_shard_cor = simulate_fa2a_copy_cp(
            k_shard.flatten(), src_shard_cor, k_src_offsets,
            kv_src_seqlen, hidden_size_k, element_size, max_cp,
            send_mask, is_send=True,
        )
        do_shard = send_mask.to(torch.int8).cuda()
        a2a_memcpy_cp(
            k_shard.cuda(), do_shard, k_src_offsets.long().cuda(),
            kv_src_seqlen.long().cuda(), to_nvshmem=True,
            buffer=src_shard_out
        )
        torch.cuda.synchronize()
        torch.testing.assert_close(src_shard_cor, src_shard_out)
        print(f"copying rank {rank} k shard to src buffer done.")
        src_shard_cor = simulate_fa2a_copy_cp(
            v_shard.flatten(), src_shard_cor, v_src_offsets,
            kv_src_seqlen, hidden_size_k, element_size, max_cp,
            send_mask, is_send=True,
        )
        src_buffer[rank] = src_shard_cor
        a2a_memcpy_cp(
            v_shard.cuda(), do_shard, v_src_offsets.long().cuda(),
            kv_src_seqlen.long().cuda(), to_nvshmem=True,
            buffer=src_shard_out
        )
        torch.cuda.synchronize()
        torch.testing.assert_close(src_shard_cor, src_shard_out)
        print(f"copying rank {rank} v shard to src buffer done.")


    dst_buffer = simulate_fa2a(
        src_buffer, dst_buffer, qkv_fwd_fa2a_metadata.fa2a_metadata, element_size
    )
    dst_q = [
        torch.zeros(qkv_fwd_fa2a_metadata.tensor_shape[0].recv_shape[r], device=device, dtype=dtype)
        for r in range(world_size)
    ]
    dst_k = [
        torch.zeros(qkv_fwd_fa2a_metadata.tensor_shape[1].recv_shape[r], device=device, dtype=dtype)
        for r in range(world_size)
    ]
    dst_v = [
        torch.zeros(qkv_fwd_fa2a_metadata.tensor_shape[1].recv_shape[r], device=device, dtype=dtype)
        for r in range(world_size)
    ]
    # Test recv copy non cp
    for rank in range(world_size):
        metadata_slice = qkv_fwd_fa2a_metadata.get_slice(rank).normalize()
        q_recv_offsets, k_recv_offsets, v_recv_offsets = metadata_slice.recv_memcpy_metadata
        recv_seqlen_q = metadata_slice.seq_lens[0].recv_seqlens
        recv_seqlen_kv = metadata_slice.seq_lens[1].recv_seqlens

        dst_q_shard = dst_q[rank].reshape(-1, hidden_size_q)
        dst_k_shard = dst_k[rank].reshape(-1, hidden_size_k)
        dst_v_shard = dst_v[rank].reshape(-1, hidden_size_k)
        dst_shard = dst_buffer[rank]
        # Q
        dst_q_shard_cor = dst_q_shard.clone().flatten()
        dst_q_shard_cor = simulate_fa2a_copy_non_cp(
            dst_q_shard_cor, dst_shard, q_recv_offsets,
            recv_seqlen_q, hidden_size_q, element_size, is_send=False,
        )
        dst_q_shard_out = dst_q_shard.clone()
        a2a_memcpy_non_cp(
            dst_q_shard_out, q_recv_offsets.long().cuda(), 
            recv_seqlen_q.long().cuda(), to_nvshmem=False, buffer=dst_shard.cuda()
        )
        torch.cuda.synchronize()
        torch.testing.assert_close(dst_q_shard_cor, dst_q_shard_out.flatten())
        print(f"copying rank {rank} q shard from dst buffer done.")
        dst_q[rank] = dst_q_shard_out
        # K
        dst_k_shard_cor = dst_k_shard.clone().flatten()
        dst_k_shard_cor = simulate_fa2a_copy_non_cp(
            dst_k_shard_cor, dst_shard, k_recv_offsets,
            recv_seqlen_kv, hidden_size_k, element_size, is_send=False,
        )
        dst_k_shard_out = dst_k_shard.clone()
        a2a_memcpy_non_cp(
            dst_k_shard_out, k_recv_offsets.long().cuda(),
            recv_seqlen_kv.long().cuda(), to_nvshmem=False, buffer=dst_shard.cuda()
        )
        torch.cuda.synchronize()
        torch.testing.assert_close(dst_k_shard_cor, dst_k_shard_out.flatten())
        print(f"copying rank {rank} k shard from dst buffer done.")
        dst_k[rank] = dst_k_shard_out
        # V
        dst_v_shard_cor = dst_v_shard.clone().flatten()
        dst_v_shard_cor = simulate_fa2a_copy_non_cp(
            dst_v_shard_cor, dst_shard, v_recv_offsets,
            recv_seqlen_kv, hidden_size_k, element_size, is_send=False,
        )
        dst_v_shard_out = dst_v_shard.clone()
        a2a_memcpy_non_cp(
            dst_v_shard_out, v_recv_offsets.long().cuda(),
            recv_seqlen_kv.long().cuda(), to_nvshmem=False, buffer=dst_shard.cuda()
        )
        torch.testing.assert_close(dst_v_shard_cor, dst_v_shard_out.flatten())
        dst_v[rank] = dst_v_shard_out

    # Test send copy of backward
    grad_buffer = torch.zeros_like(dst_buffer)
    for rank in range(world_size):
        metadata_slice = qkv_bwd_fa2a_metadata.get_slice(rank).normalize()
        dst_shard = grad_buffer[rank]

        attn_q_shard = dst_q[rank]
        attn_k_shard = dst_k[rank]
        attn_v_shard = dst_v[rank]
        q_send_offsets, k_send_offsets, v_send_offsets = metadata_slice.send_memcpy_metadata

        a2a_memcpy_non_cp(
            attn_q_shard, q_send_offsets, metadata_slice.seq_lens[0].send_seqlens,
            to_nvshmem=True, buffer=dst_shard
        )

        copy_seq_mask = metadata_slice.kv_grad_send_dedup.main_copy_mask
        a2a_memcpy_non_cp(
            attn_k_shard, k_send_offsets, metadata_slice.seq_lens[1].send_seqlens,
            to_nvshmem=True, buffer=dst_shard, copy_seq_mask=copy_seq_mask
        )
        a2a_memcpy_non_cp(
            attn_v_shard, v_send_offsets, metadata_slice.seq_lens[1].send_seqlens,
            to_nvshmem=True, buffer=dst_shard, copy_seq_mask=copy_seq_mask
        )
        torch.testing.assert_close(dst_shard, dst_buffer[rank])
        print(f"copying rank {rank} grads to send buffer done.")

    # Test recv copy cp (kv grad)
    grad_k = torch.zeros(
        (world_size, max_cp_degree, total_seq_len, hidden_size_k),
        device=device, dtype=dtype
    )
    grad_v = grad_k.clone()
    for rank in range(world_size):
        metadata_slice = qkv_bwd_fa2a_metadata.get_slice(rank).normalize()
        grad_q_recv_offsets, grad_k_recv_offsets, grad_v_recv_offsets = metadata_slice.recv_memcpy_metadata
        grad_seqlen_kv = metadata_slice.seq_lens[1].recv_seqlens

        grad_k_shard = grad_k[rank]
        grad_v_shard = grad_v[rank]
        # assume grad equals the original value
        grad_shard = src_buffer[rank]
        max_cp = metadata_slice.kv_replica_mask.shape[1]
        # K
        grad_send_mask = metadata_slice.kv_replica_mask
        grad_do_shard = grad_send_mask.to(torch.int8).cuda()

        grad_k_shard_cor = grad_k_shard.flatten(start_dim=1).clone()
        grad_k_shard_cor = simulate_fa2a_copy_cp(
            grad_k_shard_cor, grad_shard, grad_k_recv_offsets, grad_seqlen_kv,
            hidden_size_k, element_size, max_cp, grad_send_mask, is_send=False,
        )
        grad_k_shard_cor = grad_k_shard_cor.reshape_as(grad_k_shard)
        torch.testing.assert_close(
            grad_k_shard_cor.sum(dim=0) / (grad_k_shard_cor > 0).sum(dim=0),
            tensor_k[rank]
        )
        grad_k_shard_out = grad_k_shard.clone()
        a2a_memcpy_cp(
            grad_k_shard_out.cuda(), grad_do_shard,
            grad_k_recv_offsets.long().cuda(),
            grad_seqlen_kv.long().cuda(),
            to_nvshmem=False,
            buffer=grad_shard.cuda()
        )
        torch.cuda.synchronize()
        torch.testing.assert_close(grad_k_shard_cor, grad_k_shard_out)
        print(f"copying rank {rank} k shard from grad buffer done.")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--world-size', type=int, default=2)
    parser.add_argument('--num-docs', type=int, default=4)
    parser.add_argument('--num-tokens', type=int, default=128)
    parser.add_argument('--hidden-size-q', type=int, default=64)
    parser.add_argument('--hidden-size-k', type=int, default=16)
    parser.add_argument('--max-seq-shard', type=int, default=2)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()
    test(args)
