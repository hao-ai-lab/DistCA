import torch

from d2.runtime.attn_kernels.ops import _ops

def test_one_case(
        num_token: int, hidden_size: int, num_seq_raw: int, max_num_copies: int,
        dtype: torch.dtype, seed: int, logging: bool=False):
    torch.manual_seed(seed)
    tensor = torch.randn((num_token, hidden_size), dtype=dtype, device='cuda')

    copy_freq = 0.4
    num_main_copies = 0
    num_seq = num_token
    while num_seq * 4 >= num_token:
        while not num_main_copies:
            has_copies_mask = torch.rand(num_seq_raw) < copy_freq
            num_main_copies = has_copies_mask.sum().item()
        # Assign a random number of copies (from 1 to max_num_copies) to those sequences
        raw_seq_num_copies = torch.zeros(num_seq_raw, dtype=torch.long)
        if num_main_copies > 0:
            copies_for_mains = torch.randint(1, max_num_copies + 1, (num_main_copies,))
            raw_seq_num_copies[has_copies_mask] = copies_for_mains
        num_seq = int(num_seq_raw + raw_seq_num_copies.sum().item())

    ## Create seqlens
    # 0.1 ~ 1.1 to guarantee not too short sequences
    seq_len_factor = torch.rand((num_seq_raw,)) + 0.1
    seq_len_and_copies = list(enumerate(seq_len_factor.tolist()))
    for seq_idx, num_copies in enumerate(raw_seq_num_copies):
        seq_len_and_copies.extend([seq_len_and_copies[seq_idx]] * num_copies.item())
    seq_len_sum = sum([x[1] for x in seq_len_and_copies])
    seq_len_and_copies = [(x[0], int(x[1] / seq_len_sum * num_token)) for x in seq_len_and_copies]
    # Adjust the first not repeated sequence to make the sum exactly num_token
    diff = num_token - sum([x[1] for x in seq_len_and_copies])
    for i in range(num_seq_raw):
        if raw_seq_num_copies[i] == 0:
            seq_len_and_copies[i] = (seq_len_and_copies[i][0], seq_len_and_copies[i][1] + diff)
            break

    ## randomly reorder sequences.
    seq_len_perm = torch.randperm(num_seq)
    seq_len_and_copies = [seq_len_and_copies[i] for i in seq_len_perm]

    ## identify the main copy
    main_copy_id = {}
    seq_cur_copy_id = {}
    num_copies = torch.zeros((num_seq,), dtype=torch.int32, device='cuda')
    seq_lens = torch.zeros((num_seq,), dtype=torch.int64, device='cuda')
    copy_start_id = torch.zeros((num_seq, max_num_copies), dtype=torch.int64, device='cuda')
    copy_seq_id = torch.zeros((num_seq, max_num_copies), dtype=torch.int64, device='cuda')
    cur_num_token = 0

    for permuted_seq_id, (raw_seq_id, seq_len) in enumerate(seq_len_and_copies):
        seq_lens[permuted_seq_id] = seq_len
        if raw_seq_num_copies[raw_seq_id] == 0:
            # not has a copy
            pass
        elif raw_seq_id not in main_copy_id:
            # has a copy
            main_copy_id[raw_seq_id] = permuted_seq_id
            seq_cur_copy_id[raw_seq_id] = 0

            num_copies[permuted_seq_id] = raw_seq_num_copies[raw_seq_id]
        else:
            main_id = main_copy_id[raw_seq_id]
            cur_copy_id = seq_cur_copy_id[raw_seq_id]
            seq_cur_copy_id[raw_seq_id] += 1

            copy_start_id[main_id, cur_copy_id] = cur_num_token
            copy_seq_id[main_id, cur_copy_id] = permuted_seq_id
            assert seq_lens[main_id] == seq_len
        cur_num_token += seq_len
    assert cur_num_token == num_token

    seq_start = torch.cumsum(torch.cat([torch.zeros(1, dtype=torch.long, device=seq_lens.device), seq_lens[:-1]]), dim=0)
    if logging:
        print("num_seq:", num_seq, "num_token:", num_token, "hidden_size:", hidden_size)
        print("num_copies:", num_copies)
        print("seq_lens:", seq_lens)
        print("copy_start_id:", copy_start_id)
        print("copy_seq_id:", copy_seq_id)
        print("seq_start:", seq_start)

    ## Create the reference answer
    target_tensor = tensor.clone()
    cur_num_token = 0
    torch.cuda.synchronize()
    for seq_id, (raw_seq_id, seq_len) in enumerate(seq_len_and_copies):
        assert cur_num_token == seq_start[seq_id]
        if raw_seq_num_copies[raw_seq_id] == 0:
            pass
        elif main_copy_id[raw_seq_id] != seq_id:
            main_id = main_copy_id[raw_seq_id]
            main_start_token = seq_start[main_id]
            target_tensor[main_start_token:main_start_token + seq_len] += target_tensor[
                cur_num_token:cur_num_token + seq_len]
        cur_num_token += seq_len
    torch.cuda.synchronize()

    ## Run test on a clone
    cloned_tensor = tensor.clone()
    torch.cuda.synchronize()
    _ops.fast_a2a_grad_acc(cloned_tensor, num_copies, copy_start_id, seq_lens)
    torch.cuda.synchronize()
    torch.testing.assert_close(cloned_tensor, target_tensor, atol=1e-3, rtol=1e-3)


if __name__ == "__main__":
    for num_token in [4096, 8192]:
        for num_seq_raw in [4, 8, 16, 32]:
            with torch.no_grad():
                test_one_case(
                    num_token=num_token,
                    hidden_size=128,
                    num_seq_raw=num_seq_raw,
                    max_num_copies=4,
                    dtype=torch.float16,
                    seed=42
                )
    print("All test cases passed!")
