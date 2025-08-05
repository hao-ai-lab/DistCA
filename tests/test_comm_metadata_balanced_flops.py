"""Test whether forward and backward metadata generation works correctly."""

import torch
import rich

from d2.runtime.inplace_metadata import (
    Metadata, compute_metadata
)

from test_util import (
    gen_seq_lens, 
    create_qkv_dispatch, 
    orchestrate_simulate, 
    create_qkv_dispatch_with_custom_mapping,
)




def test_create_qkv_dispatch_balanced_flops_1(
    world_size_, total_seq_len_, num_seqs_, max_cp_degree_, 
    verbose=False,
):

    K = 1024

    world_size = 4
    assert world_size == world_size_, f"This test forces world_size = 4"
    
    total_seq_len = 16 * K
    assert total_seq_len == total_seq_len_, f"This test forces total_seq_len = 16K"

    num_seqs = 8
    assert num_seqs == num_seqs_, f"This test forces num_seqs = 8"

    max_cp_degree = 8
    assert max_cp_degree == max_cp_degree_, f"This test forces max_cp_degree = 8"
    

    seq_lens_list = [
        [16*K],
        [8*K] * 2,
        [4*K] * 4,
        [2*K] * 8,
    ]
    rich.print("seq_lens_list =", seq_lens_list)

    cp_num_list = [
        [8],
        [1, 1],
        [1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1, 1],
    ]
    rich.print("cp_num_list =", cp_num_list)

    cp_dst_list = [
        [[0,1,2,3,3,2,1,0]],
        [[1], [1]],
        [[2], [2], [2], [2]],
        [[3], [3], [3], [3], [3], [3], [3], [3]],
    ]
    rich.print("cp_dst_list =", cp_dst_list)

    seq_shard_lens_list = [
        [ [2*K, 2*K, 2*K, 2*K, 2*K, 2*K, 2*K, 2*K] ] ,
        [ [8*K], [8*K] ] ,
        [ [4*K], [4*K], [4*K], [4*K] ] ,
        [ [2*K], [2*K], [2*K], [2*K], [2*K], [2*K], [2*K], [2*K] ] ,
    ]
    rich.print("seq_shard_lens_list =", seq_shard_lens_list)

    num_seqs = max(len(seq_lens) for seq_lens in seq_lens_list)
    max_cp_degree = max(cp_num for cp_nums in cp_num_list for cp_num in cp_nums)

    seq_lens = torch.zeros((world_size, num_seqs), dtype=torch.int64)
    for i in range(len(seq_lens_list)):
        for j in range(len(seq_lens_list[i])):
            seq_lens[i, j] = seq_lens_list[i][j]
    rich.print("seq_lens =", seq_lens)
    
    cp_num = torch.zeros((world_size, num_seqs), dtype=torch.int64)
    for i in range(len(cp_num_list)):
        for j in range(len(cp_num_list[i])):
            cp_num[i, j] = cp_num_list[i][j]
    rich.print("cp_num =", cp_num)

    cp_dst = torch.ones((world_size, num_seqs, max_cp_degree), dtype=torch.int64) * -1
    for i in range(len(cp_dst_list)):
        for j in range(len(cp_dst_list[i])):
            for k in range(len(cp_dst_list[i][j])):
                cp_dst[i, j, k] = cp_dst_list[i][j][k]
    rich.print("cp_dst =", cp_dst)

    seq_shard_lens = torch.zeros((world_size, num_seqs, max_cp_degree), dtype=torch.int64)
    for i in range(len(seq_shard_lens_list)):
        for j in range(len(seq_shard_lens_list[i])):
            for k in range(len(seq_shard_lens_list[i][j])):
                seq_shard_lens[i, j, k] = seq_shard_lens_list[i][j][k]
    rich.print("seq_shard_lens =", seq_shard_lens)
    
    # breakpoint()
    
    ret = create_qkv_dispatch_with_custom_mapping(
        world_size, 
        seq_lens,
        cp_num,
        cp_dst,
        seq_shard_lens,
        # total_seq_len, num_seqs, max_cp_degree,
        verbose=verbose,
    )
    return ret



def test_create_qkv_dispatch_balanced_flops_2(
    world_size_, total_seq_len_, num_seqs_, max_cp_degree_, 
    verbose=False,
):

    K = 1024
    
    world_size = 4
    assert world_size == world_size_, f"This test forces world_size = 4"
    
    total_seq_len = 16 * K
    assert total_seq_len == total_seq_len_, f"This test forces total_seq_len = 16K"

    num_seqs = 8
    assert num_seqs == num_seqs_, f"This test forces num_seqs = 8"

    max_cp_degree = 8
    assert max_cp_degree == max_cp_degree_, f"This test forces max_cp_degree = 8"
    

    from d2.planner.equal_flops import (
        batch_to_items, 
        plan_relocation,
        item_to_intermediate_tensors,
    )
    # items_after_planning = [
    #     {'q': 8192, 'kv': 8192, 'gpuid': 1, 'seqid': 0, 'src_gpuid': 1, 'is_original': True},
    #     {'q': 8192, 'kv': 8192, 'gpuid': 1, 'seqid': 1, 'src_gpuid': 1, 'is_original': True},
    #     {'q': 4096, 'kv': 4096, 'gpuid': 2, 'seqid': 0, 'src_gpuid': 2, 'is_original': True},
    #     {'q': 4096, 'kv': 4096, 'gpuid': 2, 'seqid': 1, 'src_gpuid': 2, 'is_original': True},
    #     {'q': 4096, 'kv': 4096, 'gpuid': 2, 'seqid': 2, 'src_gpuid': 2, 'is_original': True},
    #     {'q': 4096, 'kv': 4096, 'gpuid': 2, 'seqid': 3, 'src_gpuid': 2, 'is_original': True},
    #     {'q': 2048, 'kv': 2048, 'gpuid': 3, 'seqid': 0, 'src_gpuid': 3, 'is_original': True},
    #     {'q': 2048, 'kv': 2048, 'gpuid': 3, 'seqid': 1, 'src_gpuid': 3, 'is_original': True},
    #     {'q': 2048, 'kv': 2048, 'gpuid': 3, 'seqid': 2, 'src_gpuid': 3, 'is_original': True},
    #     {'q': 2048, 'kv': 2048, 'gpuid': 3, 'seqid': 3, 'src_gpuid': 3, 'is_original': True},
    #     {'q': 2048, 'kv': 2048, 'gpuid': 3, 'seqid': 4, 'src_gpuid': 3, 'is_original': True},
    #     {'q': 2048, 'kv': 2048, 'gpuid': 3, 'seqid': 5, 'src_gpuid': 3, 'is_original': True},
    #     {'q': 2048, 'kv': 2048, 'gpuid': 3, 'seqid': 6, 'src_gpuid': 3, 'is_original': True},
    #     {'q': 2048, 'kv': 2048, 'gpuid': 3, 'seqid': 7, 'src_gpuid': 3, 'is_original': True},
    #     {'q': 3754, 'kv': 3754, 'gpuid': 0, 'seqid': 0, 'src_gpuid': 0, 'is_original': False},
    #     {'q': 3754, 'kv': 16384, 'gpuid': 0, 'seqid': 0, 'src_gpuid': 0, 'is_original': False},
    #     {'q': 1706, 'kv': 5460, 'gpuid': 2, 'seqid': 0, 'src_gpuid': 0, 'is_original': False},
    #     {'q': 1706, 'kv': 12630, 'gpuid': 2, 'seqid': 0, 'src_gpuid': 0, 'is_original': False},
    #     {'q': 2730, 'kv': 8190, 'gpuid': 3, 'seqid': 0, 'src_gpuid': 0, 'is_original': False},
    #     {'q': 2730, 'kv': 10924, 'gpuid': 3, 'seqid': 0, 'src_gpuid': 0, 'is_original': False}
    # ]

    
    # {'world_size': 4, 'num_seqs': 8, 'max_cp_degree': 6}

    world_size = 4
    num_seqs = 8
    max_cp_degree = 6

    seq_lens = torch.tensor([
        [16384,     0,     0,     0,     0,     0,     0,     0],
        [ 8192,  8192,     0,     0,     0,     0,     0,     0],
        [ 4096,  4096,  4096,  4096,     0,     0,     0,     0],
        [ 2048,  2048,  2048,  2048,  2048,  2048,  2048,  2048]
    ])

    cp_num = torch.tensor([
        [6, 0, 0, 0, 0, 0, 0, 0],
        [1, 1, 0, 0, 0, 0, 0, 0],
        [1, 1, 1, 1, 0, 0, 0, 0],
        [1, 1, 1, 1, 1, 1, 1, 1]
    ])

    cp_dst = torch.tensor([
        [[ 0,  2,  3,  3,  2,  0],
         [-1, -1, -1, -1, -1, -1],
         [-1, -1, -1, -1, -1, -1],
         [-1, -1, -1, -1, -1, -1],
         [-1, -1, -1, -1, -1, -1],
         [-1, -1, -1, -1, -1, -1],
         [-1, -1, -1, -1, -1, -1],
         [-1, -1, -1, -1, -1, -1]],

        [[ 1, -1, -1, -1, -1, -1],
         [ 1, -1, -1, -1, -1, -1],
         [-1, -1, -1, -1, -1, -1],
         [-1, -1, -1, -1, -1, -1],
         [-1, -1, -1, -1, -1, -1],
         [-1, -1, -1, -1, -1, -1],
         [-1, -1, -1, -1, -1, -1],
         [-1, -1, -1, -1, -1, -1]],

        [[ 2, -1, -1, -1, -1, -1],
         [ 2, -1, -1, -1, -1, -1],
         [ 2, -1, -1, -1, -1, -1],
         [ 2, -1, -1, -1, -1, -1],
         [-1, -1, -1, -1, -1, -1],
         [-1, -1, -1, -1, -1, -1],
         [-1, -1, -1, -1, -1, -1],
         [-1, -1, -1, -1, -1, -1]],

        [[ 3, -1, -1, -1, -1, -1],
         [ 3, -1, -1, -1, -1, -1],
         [ 3, -1, -1, -1, -1, -1],
         [ 3, -1, -1, -1, -1, -1],
         [ 3, -1, -1, -1, -1, -1],
         [ 3, -1, -1, -1, -1, -1],
         [ 3, -1, -1, -1, -1, -1],
         [ 3, -1, -1, -1, -1, -1]]
    ])

    seq_shard_lens = torch.tensor([
        [[3754, 1706, 2732, 2732, 1706, 3754],
         [   0,    0,    0,    0,    0,    0],
         [   0,    0,    0,    0,    0,    0],
         [   0,    0,    0,    0,    0,    0],
         [   0,    0,    0,    0,    0,    0],
         [   0,    0,    0,    0,    0,    0],
         [   0,    0,    0,    0,    0,    0],
         [   0,    0,    0,    0,    0,    0]],

        [[8192,    0,    0,    0,    0,    0],
         [8192,    0,    0,    0,    0,    0],
         [   0,    0,    0,    0,    0,    0],
         [   0,    0,    0,    0,    0,    0],
         [   0,    0,    0,    0,    0,    0],
         [   0,    0,    0,    0,    0,    0],
         [   0,    0,    0,    0,    0,    0],
         [   0,    0,    0,    0,    0,    0]],

        [[4096,    0,    0,    0,    0,    0],
         [4096,    0,    0,    0,    0,    0],
         [4096,    0,    0,    0,    0,    0],
         [4096,    0,    0,    0,    0,    0],
         [   0,    0,    0,    0,    0,    0],
         [   0,    0,    0,    0,    0,    0],
         [   0,    0,    0,    0,    0,    0],
         [   0,    0,    0,    0,    0,    0]],

        [[2048,    0,    0,    0,    0,    0],
         [2048,    0,    0,    0,    0,    0],
         [2048,    0,    0,    0,    0,    0],
         [2048,    0,    0,    0,    0,    0],
         [2048,    0,    0,    0,    0,    0],
         [2048,    0,    0,    0,    0,    0],
         [2048,    0,    0,    0,    0,    0],
         [2048,    0,    0,    0,    0,    0]]
    ])

    
    ret = create_qkv_dispatch_with_custom_mapping(
        world_size, 
        seq_lens,
        cp_num,
        cp_dst,
        seq_shard_lens,
        verbose=verbose,
    )
    return ret


def test_create_qkv_dispatch_balanced_flops_3(
    world_size_, total_seq_len_, num_seqs_, max_cp_degree_, 
    verbose=False,
):

    K = 1024
    
    world_size = 4
    assert world_size == world_size_, f"This test forces world_size = 4"
    
    total_seq_len = 16 * K
    assert total_seq_len == total_seq_len_, f"This test forces total_seq_len = 16K"

    num_seqs = 8
    assert num_seqs == num_seqs_, f"This test forces num_seqs = 8"

    max_cp_degree = 8
    assert max_cp_degree == max_cp_degree_, f"This test forces max_cp_degree = 8"
    

    from d2.planner.equal_flops import (
        batch_to_items, 
        plan_relocation,
        item_to_intermediate_tensors,
    )

    items = batch_to_items([
        [16 * K] * 1,
        [8 * K] * 2,
        [4 * K] * 4,
        [2 * K] * 8, 
    ])
    items = plan_relocation(items, verbose=False, plot=False)

    world_info, (items, info_mapping, info_list), (seq_lens, cp_num, cp_dst, seq_shard_lens) = item_to_intermediate_tensors(items)    

    world_size = world_info["world_size"]
    num_seqs = world_info["num_seqs"]
    max_cp_degree = world_info["max_cp_degree"]

    
    ret = create_qkv_dispatch_with_custom_mapping(
        world_size, 
        seq_lens,
        cp_num,
        cp_dst,
        seq_shard_lens,
        verbose=verbose,
    )
    return ret



def test_query_dispatch(args):
    rich.print("⚪ Testing query dispatch")

    torch.manual_seed(args.seed)
    world_size = args.world_size
    num_seqs = args.num_seqs
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    hidden_size = args.hidden_size
    total_seq_len = args.num_tokens
    seq_len = gen_seq_lens(world_size, num_seqs, total_seq_len).long().to(device)
    global_dispatch = torch.randint(0, world_size, (world_size, num_seqs),
                                    device=device)


    tensor = torch.rand((world_size, total_seq_len, hidden_size), device=device) + 1    # + 1 to avoid zeros
    fwd_metadata, rev_metadata = compute_metadata(seq_len, global_dispatch)

    # forward
    max_recv_tokens = fwd_metadata.num_recv_tokens.max()
    output_tensor = torch.zeros((world_size, max_recv_tokens, hidden_size),
                                device=device, dtype=tensor.dtype)
    output_tensor = orchestrate_simulate(tensor, output_tensor, fwd_metadata)

    # reverse
    rev_metadata.num_recv_tokens.max()
    rev_tensor = torch.zeros((world_size, total_seq_len, hidden_size),
                             device=device, dtype=output_tensor.dtype)
    rev_tensor = orchestrate_simulate(output_tensor, rev_tensor, rev_metadata)
    rev_tensor = rev_tensor.reshape(world_size, total_seq_len, hidden_size)
    torch.testing.assert_close(tensor, rev_tensor)
    rich.print("🟢 Test query dispatch passed")


from typing import Union

test_fn_list = [
    test_create_qkv_dispatch_balanced_flops_1,
    test_create_qkv_dispatch_balanced_flops_2,
]

def test_qkv_dispatch(args, test_fn):
    world_size = args.world_size
    num_seqs = args.num_seqs
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    hidden_size = args.hidden_size
    total_seq_len = args.num_tokens
    max_cp_degree: int = args.max_seq_shard
    torch.manual_seed(args.seed)

    rich.print("⚪ Testing qkv dispatch with test_fn =", test_fn.__name__)

    # TODO: Make sure the metadata is correctly defined via the interface.
    (fwd_q_metadata, rev_q_metadata, fwd_k_metadata, rev_k_metadata, _) = test_fn(
        world_size_=world_size,
        total_seq_len_=total_seq_len,
        num_seqs_=num_seqs,
        max_cp_degree_=max_cp_degree,
        verbose=True,
    )

    # Test that Query is correctly sent.
    tensor = torch.rand((world_size, total_seq_len, hidden_size), device=device) + 1    # + 1 to avoid zeros
    # forward
    max_recv_tokens = fwd_q_metadata.num_recv_tokens.max()
    output_tensor = torch.zeros((world_size, max_recv_tokens, hidden_size),
                                device=device, dtype=tensor.dtype)
    output_tensor = orchestrate_simulate(tensor, output_tensor, fwd_q_metadata)

    # reverse
    rev_q_metadata.num_recv_tokens.max()
    rev_tensor = torch.zeros((world_size, total_seq_len, hidden_size),
                             device=device, dtype=output_tensor.dtype)
    rev_tensor = orchestrate_simulate(output_tensor, rev_tensor, rev_q_metadata)
    rev_tensor = rev_tensor.reshape(world_size, total_seq_len, hidden_size)
    torch.testing.assert_close(tensor, rev_tensor)

    # Test that key-value is correctly sent.
    tensor = torch.rand((world_size, total_seq_len, hidden_size), device=device) + 1
    max_recv_tokens_kv = fwd_k_metadata.num_recv_tokens.max()
    output_tensor = torch.zeros((world_size, max_recv_tokens_kv, hidden_size),
                                device=device, dtype=tensor.dtype)
    output_tensor = orchestrate_simulate(tensor, output_tensor, fwd_k_metadata)
    # reverse
    rev_tensor = torch.zeros((world_size, total_seq_len * max_cp_degree, hidden_size), device=device)
    rev_tensor = orchestrate_simulate(output_tensor, rev_tensor, rev_k_metadata)
    rev_tensor = rev_tensor.reshape(world_size, max_cp_degree, total_seq_len, hidden_size)

    rev_tensor_mask = rev_tensor != 0
    rev_tensor_dedup = rev_tensor.sum(dim=1) / rev_tensor_mask.int().sum(dim=1)

    torch.testing.assert_close(rev_tensor_mask * tensor.unsqueeze(1), rev_tensor)
    torch.testing.assert_close(tensor, rev_tensor_dedup)
    rich.print("🟢 Test qkv dispatch passed")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="Test metadata generation")
    parser.add_argument('--world_size', type=int, default=4)
    parser.add_argument('--num_seqs', type=int, default=8, help='Number of sequences per rank')
    parser.add_argument('--max_seq_shard', type=int, default=8, help='Number of shards per sequence')
    parser.add_argument('--num_tokens', type=int, default=16 * 1024, help='Total number of tokens per rank')
    parser.add_argument('--hidden_size', type=int, default=128, help='Hidden size of the tensor')
    parser.add_argument('--seed', type=int, default=0, help='Random seed for reproducibility')
    args = parser.parse_args()
    rich.print(f"Testing {__file__} with args =", args)

    test_query_dispatch(args)


    test_create_qkv_dispatch_balanced_flops_1(
        world_size_=args.world_size,
        total_seq_len_=args.num_tokens,
        num_seqs_=args.num_seqs,
        max_cp_degree_=args.max_seq_shard,
        verbose=True,
    )

    test_qkv_dispatch(args, test_fn=test_create_qkv_dispatch_balanced_flops_1)

    test_create_qkv_dispatch_balanced_flops_2(
        world_size_=args.world_size,
        total_seq_len_=args.num_tokens,
        num_seqs_=args.num_seqs,
        max_cp_degree_=args.max_seq_shard,
        verbose=True,
    )

    test_qkv_dispatch(args, test_fn=test_create_qkv_dispatch_balanced_flops_2)

    test_create_qkv_dispatch_balanced_flops_3(
        world_size_=args.world_size,
        total_seq_len_=args.num_tokens,
        num_seqs_=args.num_seqs,
        max_cp_degree_=args.max_seq_shard,
        verbose=True,
    )

    test_qkv_dispatch(args, test_fn=test_create_qkv_dispatch_balanced_flops_3)

