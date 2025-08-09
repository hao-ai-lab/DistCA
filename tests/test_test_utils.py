import rich
from test_util import create_raw_qkv_dispatch

def test_create_raw_qkv_dispatch():
    world_size = 4
    total_seq_len = 1024
    num_seqs = 4
    max_cp_degree = 1
    return_mlp_no_shard_seq_lens = False

    (cp_seq_lens, num_cp_shards, cp_query_dst,
        kv_to_q_mapping, kv_to_q_rank, kv_context_size,
        q_to_num_kv_seq, q_to_num_kv_tokens,
        seq_lens) = create_raw_qkv_dispatch(
        world_size, total_seq_len, num_seqs, max_cp_degree,
        return_mlp_no_shard_seq_lens
    )
    rich.print(f"cp_seq_lens = {cp_seq_lens}")
    rich.print(f"num_cp_shards = {num_cp_shards}")
    rich.print(f"cp_query_dst = {cp_query_dst}")
    rich.print(f"kv_to_q_mapping = {kv_to_q_mapping}")
    rich.print(f"kv_to_q_rank = {kv_to_q_rank}")
    rich.print(f"kv_context_size = {kv_context_size}")
    rich.print(f"q_to_num_kv_seq = {q_to_num_kv_seq}")
    rich.print(f"q_to_num_kv_tokens = {q_to_num_kv_tokens}")
    rich.print(f"seq_lens = {seq_lens}")
    return

(
            fwd_q_metadata, rev_q_metadata, fwd_k_metadata, rev_k_metadata,
            attention_metadata_attn_layout, intermediates, seq_lens
        ) = create_qkv_dispatch(
            as_world_size, total_seq_len, num_seqs, max_cp_degree,
            return_intermediate=True, return_mlp_no_shard_seq_lens=True
        )


if __name__ == "__main__":
    test_create_raw_qkv_dispatch()