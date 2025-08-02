from typing import List, Tuple

import ray
import torch

from d2.runtime.attn_kernels.ops import dispatch_no_cp_tensor, nvshmem_init, nvshmem_barrier_all, nvshmem_get_unique_id, DispatcherWrapper
from d2.runtime.inplace_metadata import compute_metadata, Metadata

from test_util import gen_seq_lens
from test_comm_metadata import orchestrate_simulate


@ray.remote(num_gpus=1)
class Worker:
    def __init__(self, rank: int, world_size: int):
        self.rank = rank
        self.world_size = world_size
        self.nvshmem_initialized = False
        self.dispatcher = None

    def init_comm(self, uid: torch.Tensor, stride: int, max_tokens_query: int, max_tokens_key_value: int):
        nvshmem_init(uid, self.rank, self.world_size)
        DispatcherWrapper.init(
            q_stride=stride,
            kv_stride=stride,
            max_tokens_query=max_tokens_query,
            max_tokens_key_value=max_tokens_key_value,
            rank=self.rank,
            world_size=self.world_size,
            uid=uid,
        )
        self.nvshmem_initialized = True

    def test(self, seed: int, tensor: torch.Tensor,
             fwd_metadata: Metadata, rev_metadata: Metadata):
        fwd_metadata = fwd_metadata.cuda().normalize_dtype()
        rev_metadata = rev_metadata.cuda().normalize_dtype()
        tensor = tensor.cuda()

        hidden_size = tensor.shape[-1]
        torch.manual_seed(seed)

        num_tokens = tensor.shape[0]

        # forward communication buffer
        num_received_tokens = fwd_metadata.num_recv_tokens[-1]
        dst_tensor = torch.zeros(
            (num_received_tokens, hidden_size), dtype=tensor.dtype, device=tensor.device
        )
        torch.cuda.synchronize()
        dst_tensor = self.orchestrate(
            tensor, fwd_metadata,
            dst_tensor=dst_tensor,
        )
        torch.cuda.synchronize()
        print(f"rank {self.rank} fwd done", flush=True)

        # reverse communication buffer
        back_tensor_shape = (num_tokens, hidden_size,)
        back_tensor = torch.zeros(
            back_tensor_shape, dtype=tensor.dtype, device=tensor.device
        )
        back_tensor = self.orchestrate(
            dst_tensor, rev_metadata,
            dst_tensor=back_tensor,
        )
        torch.cuda.synchronize()
        print(f"rank {self.rank} bwd communication done", flush=True)
        torch.testing.assert_close(tensor, back_tensor)
        torch.cuda.synchronize()
        return dst_tensor, back_tensor

    @torch.no_grad()
    def orchestrate(self,
                    tensor: torch.Tensor,
                    metadata: Metadata,
                    dst_tensor: torch.Tensor):
        # print(f"rank {self.rank} orchestrate tensor {tensor[:, :2]=}, {dst_id=}, {dst_offset=}, {dst_tensor.shape=}, {num_recv_tokens=}")
        dispatch_no_cp_tensor(DispatcherWrapper.get_instance(), tensor, dst_tensor, metadata)
        nvshmem_barrier_all()
        return dst_tensor


def create_testcase(
    seed: int, world_size: int, num_tokens: int, num_seqs: int, hidden_size: int
) -> Tuple[Metadata, Metadata, torch.Tensor, torch.Tensor, torch.Tensor]:
    torch.manual_seed(seed)
    # Init seq lens and dispatch tensor.
    seq_lens = gen_seq_lens(world_size, num_seqs, num_tokens).long()

    min_rank = 0
    global_dispatch = torch.randint(min_rank, world_size, (world_size, num_seqs), dtype=torch.int64)

    tensor = torch.randn(world_size, num_tokens, hidden_size, dtype=torch.float16)
    fwd_metadata, rev_metadata = compute_metadata(seq_lens, global_dispatch)

    max_recv_tokens = fwd_metadata.num_recv_tokens.max()
    output_tensor = torch.zeros((world_size, max_recv_tokens, hidden_size), dtype=tensor.dtype, device=tensor.device)
    output_tensor = orchestrate_simulate(tensor, output_tensor, fwd_metadata)

    back_tensor = torch.zeros((world_size, num_tokens, hidden_size), dtype=output_tensor.dtype, device=output_tensor.device)
    back_tensor = orchestrate_simulate(output_tensor, back_tensor, rev_metadata)

    back_tensor_dedup = back_tensor
    # print(f"fwd_metadata={fwd_metadata}, rev_metadata={rev_metadata}")
    return fwd_metadata, rev_metadata, tensor, output_tensor, back_tensor_dedup

@torch.no_grad()
def test(seed, world_size, num_tokens, num_seqs, workers: List[ray.ObjectRef], hidden_size: int=128):
    fwd_metadata, rev_metadata, tensor, output_tensor, back_tensor_dedup = create_testcase(seed, world_size, num_tokens, num_seqs, hidden_size)
    assert len(workers) == world_size
    refs = []

    for rank, worker in enumerate(workers):
        tensor_slice = tensor[rank]
        fwd_metadata_slice = fwd_metadata.get_slice(rank)
        rev_metadata_slice = rev_metadata.get_slice(rank)
        ref = worker.test.remote(
            seed, tensor_slice,
            fwd_metadata_slice, rev_metadata_slice,
        )
        refs.append(ref)

    results = ray.get(refs)
    dst_tensor_outs = [r[0] for r in results]
    # The first dimension of each value in the list should be padded to the same as dst_tensor's dim 1
    max_dim1 = output_tensor.shape[1]
    dst_tensor_outs = [
        torch.cat([r, torch.zeros(max_dim1 - r.shape[0], *r.shape[1:], dtype=r.dtype, device=r.device)], dim=0).unsqueeze(0)
        for r in dst_tensor_outs
    ]
    dst_tensor_out = torch.cat(dst_tensor_outs, dim=0)
    assert dst_tensor_out.shape == output_tensor.shape
    torch.testing.assert_close(dst_tensor_out, output_tensor.to(dst_tensor_out.device))
    back_tensor_out = torch.cat([r[1].unsqueeze(0) for r in results], dim=0)
    torch.testing.assert_close(back_tensor_out, back_tensor_dedup.to(back_tensor_out.device))


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--world-size", type=int, default=2)
    parser.add_argument("--num-tokens", type=int, default=1024)
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--num-seqs", type=int, default=4)
    args = parser.parse_args()
    ray.init()
    workers = [
        Worker.remote(i, args.world_size)
        for i in range(args.world_size)
    ]

    uid = nvshmem_get_unique_id()
    stride = args.hidden_size * torch.float16.itemsize
    max_tokens_query = args.num_tokens * args.world_size
    max_tokens_key_value = args.num_tokens * args.world_size
    ray.get([
        worker.init_comm.remote(uid, stride, max_tokens_query, max_tokens_key_value)
        for worker in workers
    ])

    print("init done")
    test(0, args.world_size, args.num_tokens, args.num_seqs, workers, args.hidden_size)
    print("test done")

