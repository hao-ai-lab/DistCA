# test script for attn worker
from typing import List

from flash_attn import flash_attn_varlen_func
import numpy as np
import ray
import torch

from attn.src.attn_worker import AttentionWorker, ScheduleMetadata
from attn.src.attn_kernels.ops import nvshmem_init

class Requester:
    def __init__(self, requester_rank, num_heads, hidden_size):
        self.requester_rank = requester_rank
        self.num_heads = num_heads
        self.hidden_size = hidden_size
        # temporary solution: a cupy communicator for each worker
        self.communicators = {}

    def init_nvshmem(self, uuid, rank, world_size):
        nvshmem_init(uuid, rank, world_size)

    def launch_comm(self, metadatas: List[ScheduleMetadata]):
        # TODO: this is a temporary solution. Should be nvshmem.
        # Should use a better order to send the tensors.
        for worker_rank, metadata in enumerate(metadatas):
            pass

def test_communication(communication_methd):
    hidden_size = 128
    num_heads = 32
    num_heads_k = 8
    dtype = torch.float16

    # create attention server
    worker_cls = ray.remote(num_gpus=1)(AttentionWorker)
    worker = worker_cls.remote(
        dropout_p=0.1,
        softmax_scale=1.0,
        hidden_size=hidden_size,
        num_heads=num_heads,
        num_heads_k=num_heads_k,
        dtype=dtype,
    )
    workers = [worker]

    # create requester
    total_requestors = 2
    # req_gpus = 0 if communication_methd == "dummy" else 1
    req_gpus = 1
    requestor_cls = ray.remote(num_gpus=req_gpus)(Requester)
    requestors = [
        requestor_cls.remote(requester_rank=i, num_heads=num_heads, hidden_size=hidden_size)
        for i in range(total_requestors)
    ]

    # Create communication groups between requestors and workers
    uuid = ray.get(workers[0].get_nvshmem_uuid.remote())
    refs = []
    world_size = len(workers) + len(requestors)
    for i, actor in enumerate(workers + requestors):
        refs.append(
            actor.init_nvshmem.remote(
                uuid, i, world_size
            )
        )
    ray.get(refs)
    print("communication group created")

    # # construct metadata
    # seq_info = np.array([
    #     [0, 10, 10, 0, 0],
    #     [1, 10, 10, 0, 0],
    #     [0, 30, 30, 0, 0],
    # ])

    # metadata = ScheduleMetadata(
    #     communication_method=communication_methd,
    #     seq_info=seq_info,
    #     tp_degree=2,
    # )

    # # launch the run
    # ref = worker.remote_attn.remote(metadata)
    # req_ref = [
    #     requestor.launch_comm.remote(metadata)
    #     for requestor in requestors
    # ]

    # exec_status = ray.get(ref)
    # print("done")

def test_tp():
    hidden_size = 128
    num_heads = 32
    num_heads_k = 8
    dropout_p = 0.0
    softmax_scale = 1.0
    dtype = torch.float16

    # create attention server
    worker_cls = ray.remote(num_gpus=1)(AttentionWorker)
    worker = worker_cls.remote(
        dropout_p=dropout_p,
        softmax_scale=softmax_scale,
        hidden_size=hidden_size,
        num_heads=num_heads,
        num_heads_k=num_heads_k,
        dtype=dtype,
    )

    # construct metadata
    # each row has:
    # worker_id, seq_len_q, seq_len_k, q_read_addr, kv_read_addr
    seq_info = np.array([
        [0, 10, 10, 0, 0],
        [1, 10, 10, 0, 0],
        [0, 10, 10, 0, 0],
    ])

    def create_qkv_tp(total_q_len, total_k_len, num_heads, num_heads_k, hidden_size, dtype, tp_rank, tp_degree):
        torch.manual_seed(0)
        q = torch.randn(
            (total_q_len, num_heads, hidden_size), device="cuda", dtype=dtype)
        k = torch.randn(
            (total_k_len, num_heads_k, hidden_size), device="cuda", dtype=dtype)
        v = torch.randn(
            (total_k_len, num_heads_k, hidden_size), device="cuda", dtype=dtype)
        q, k, v = [
            tensor.reshape(
                tensor.shape[0], tp_degree, tensor.shape[1] // tp_degree, tensor.shape[2]
            )[:, tp_rank, :, :].contiguous()
            for tensor in [q, k, v]
        ]
        return q, k, v

    create_qkv_fn = lambda tp_rank, tp_degree: create_qkv_tp(
        total_q_len=int(seq_info[:, 1].sum()),
        total_k_len=int(seq_info[:, 2].sum()),
        num_heads=num_heads,
        num_heads_k=num_heads_k,
        hidden_size=hidden_size,
        dtype=dtype,
        tp_rank=tp_rank,
        tp_degree=tp_degree,
    )

    # Get the answer
    q, k, v = create_qkv_fn(0, 1)
    max_seqlen_q = seq_info[:, 1].max()
    max_seqlen_k = seq_info[:, 2].max()
    # TODO: add 0 for the cumsum, or add a dummy seq info at the beginning?
    max_seqlen_q = torch.tensor(max_seqlen_q, device='cuda', dtype=torch.int32)
    max_seqlen_k = torch.tensor(max_seqlen_k, device='cuda', dtype=torch.int32)

    cu_seqlens_q = np.cumsum(seq_info[:, 1])
    cu_seqlens_k = np.cumsum(seq_info[:, 2])
    cu_seqlens_q = torch.from_numpy(cu_seqlens_q).cuda().to(torch.int32)
    cu_seqlens_k = torch.from_numpy(cu_seqlens_k).cuda().to(torch.int32)
    # prepend zero
    cu_seqlens_q, cu_seqlens_k = [
        torch.cat([torch.tensor([0], device='cuda', dtype=torch.int32), tensor])
        for tensor in [cu_seqlens_q, cu_seqlens_k]
    ]
    answer = flash_attn_varlen_func(
        q, k, v, cu_seqlens_q, cu_seqlens_k,
        max_seqlen_q, max_seqlen_k,
        dropout_p=dropout_p,
        softmax_scale=softmax_scale,
        causal=True,
    )

    # for TP=2, launch two runs
    metadata = ScheduleMetadata(
        communication_method="dummy",
        seq_info=seq_info,
        tp_degree=2,
        dummy_gen_fn=lambda : create_qkv_fn(0, 2),
    )
    val_tp0 = ray.get(worker.remote_attn.remote(metadata, debug=True))

    metadata = ScheduleMetadata(
        communication_method="dummy",
        seq_info=seq_info,
        tp_degree=2,
        dummy_gen_fn=lambda : create_qkv_fn(1, 2),
    )
    val_tp1 = ray.get(worker.remote_attn.remote(metadata, debug=True))

    torch.testing.assert_close(
        torch.concatenate([val_tp0, val_tp1], dim=1),
        answer,
        atol=1e-4,
        rtol=1e-4,
    )
    print("tp numerical test passed")


def test_sp():
    hidden_size = 128
    num_heads = 32
    num_heads_k = 8
    dropout_p = 0.0
    softmax_scale = 1.0
    dtype = torch.float16

    # create attention server
    worker_cls = ray.remote(num_gpus=1)(AttentionWorker)
    worker = worker_cls.remote(
        dropout_p=dropout_p,
        softmax_scale=softmax_scale,
        hidden_size=hidden_size,
        num_heads=num_heads,
        num_heads_k=num_heads_k,
        dtype=dtype,
    )

    # construct metadata
    # each row has:
    # worker_id, seq_len_q, seq_len_k, q_read_addr, kv_read_addr
    seq_info_full = np.array([
        [0, 10, 10, 0, 0],
    ])

    def create_qkv_sp(q_lens, total_k_len, num_heads, num_heads_k, hidden_size, dtype, sp_rank, sp_degree):
        # FIXME: this is a SP without load balance.
        torch.manual_seed(0)
        total_q_len = q_lens.sum().item()
        q = torch.randn(
            (total_q_len, num_heads, hidden_size), device="cuda", dtype=dtype)
        k = torch.randn(
            (total_k_len, num_heads_k, hidden_size), device="cuda", dtype=dtype)
        v = torch.randn(
            (total_k_len, num_heads_k, hidden_size), device="cuda", dtype=dtype)
        q_sp_indices = []
        k_sp_indices = []
        cul_len = 0
        for q_len in q_lens:
            q_len_sp = q_len // sp_degree
            q_sp_indices.extend(range(cul_len + q_len_sp * sp_rank, cul_len + q_len_sp * (sp_rank + 1)))
            k_sp_indices.extend(range(cul_len, cul_len + q_len_sp * (sp_rank + 1)))
            cul_len += q_len
        q_sp_indices = torch.tensor(q_sp_indices, device='cuda', dtype=torch.int64)
        q_sp_indices = q_sp_indices.repeat(num_heads, hidden_size, 1).permute(2, 0, 1)
        k_sp_indices = torch.tensor(k_sp_indices, device='cuda', dtype=torch.int64)
        k_sp_indices = k_sp_indices.repeat(num_heads_k, hidden_size, 1).permute(2, 0, 1)
        q = torch.gather(q, dim=0, index=q_sp_indices)
        k = torch.gather(k, dim=0, index=k_sp_indices)
        v = torch.gather(v, dim=0, index=k_sp_indices)
        print(f"{sp_rank, sp_degree, q.shape, k.shape, v.shape}")
        return q, k, v

    create_qkv_fn = lambda sp_rank, sp_degree: create_qkv_sp(
        q_lens=seq_info_full[:, 1],
        total_k_len=int(seq_info_full[:, 2].sum()),
        num_heads=num_heads,
        num_heads_k=num_heads_k,
        hidden_size=hidden_size,
        dtype=dtype,
        sp_rank=sp_rank,
        sp_degree=sp_degree,
    )

    # Get the answer
    q, k, v = create_qkv_fn(sp_rank=0, sp_degree=1)
    max_seqlen_q = seq_info_full[:, 1].max()
    max_seqlen_k = seq_info_full[:, 2].max()
    # TODO: add 0 for the cumsum, or add a dummy seq info at the beginning?
    max_seqlen_q = torch.tensor(max_seqlen_q, device='cuda', dtype=torch.int32)
    max_seqlen_k = torch.tensor(max_seqlen_k, device='cuda', dtype=torch.int32)

    cu_seqlens_q = np.cumsum(seq_info_full[:, 1])
    cu_seqlens_k = np.cumsum(seq_info_full[:, 2])
    cu_seqlens_q = torch.from_numpy(cu_seqlens_q).cuda().to(torch.int32)
    cu_seqlens_k = torch.from_numpy(cu_seqlens_k).cuda().to(torch.int32)
    # prepend zero
    cu_seqlens_q, cu_seqlens_k = [
        torch.cat([torch.tensor([0], device='cuda', dtype=torch.int32), tensor])
        for tensor in [cu_seqlens_q, cu_seqlens_k]
    ]
    answer = flash_attn_varlen_func(
        q, k, v, cu_seqlens_q, cu_seqlens_k,
        max_seqlen_q, max_seqlen_k,
        dropout_p=dropout_p,
        softmax_scale=softmax_scale,
        causal=True,
    )

    # for SP=2, launch two runs
    seq_info_sp = seq_info_full.copy()
    seq_info_sp[:, 1] = seq_info_sp[:, 1] // 2  # q is sharded, k and v are replicated
    # TODO: this is a non-load balance version
    seq_info_sp_rank_0 = seq_info_sp.copy()
    seq_info_sp_rank_0[:, 2] = seq_info_sp_rank_0[:, 2] // 2
    metadata = ScheduleMetadata(
        communication_method="dummy",
        seq_info=seq_info_sp_rank_0,
        tp_degree=1,
        dummy_gen_fn=lambda : create_qkv_fn(0, 2),
    )
    val_sp0 = ray.get(worker.remote_attn.remote(metadata, debug=True))

    seq_info_sp_rank_1 = seq_info_sp.copy()
    metadata = ScheduleMetadata(
        communication_method="dummy",
        seq_info=seq_info_sp_rank_1,
        tp_degree=1,
        dummy_gen_fn=lambda : create_qkv_fn(1, 2),
    )
    val_sp1 = ray.get(worker.remote_attn.remote(metadata, debug=True))

    torch.testing.assert_close(
        torch.concatenate([val_sp0, val_sp1], dim=0),
        answer,
        atol=1e-4,
        rtol=1e-4,
    )
    print("sp numerical test passed")


def main():
    ray.init()
    # test_communication("dummy")
    # test_tp()
    test_sp()


if __name__ == "__main__":
    main()
