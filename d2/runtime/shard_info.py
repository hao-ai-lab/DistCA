import collections
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import torch


@dataclass
class ShardInfo:
    """
    Metadata of a single shard of a sequence.

    Attributes:
        rid: The Rank ID where the MLP (Multi-Layer Perceptron) computation for this shard is performed.
        dispatch_rid: The Rank ID where the Attention computation for this shard is performed.
        logical_sid: The logical order ID of this shard in the original complete sequence (starting from 0).
        shard_len: The length of this shard, usually referring to the number of tokens.
    """
    rid: int
    dispatch_rid: int
    logical_sid: int
    shard_len: int



def handle_planner_metadata(
    world_size: int,
    sequence_plans: List[List[ShardInfo]]
) -> Tuple[torch.Tensor, ...]:
    """
    Args:
        world_size: Total number of ranks.
        sequence_plans: List containing all sequence shard information.

    Returns:
        A tuple containing all computed metadata tensors.
    """
    if not sequence_plans or not any(sequence_plans) or world_size <= 0:
        raise ValueError("Invalid input! Sequence_plans cannot be None. world_size need bigger than 0.")

    all_shards_with_seq_id = []
    for seq_idx, sequence in enumerate(sequence_plans):
        for shard in sequence:
            all_shards_with_seq_id.append((seq_idx, shard))

    # Sort in a globally deterministic order: first by sequence_id, then by logical_sid
    all_shards_with_seq_id.sort(key=lambda item: (item[0], item[1].logical_sid))

    rank_shard_counters = torch.zeros(world_size, dtype=torch.int64)

    # hash table, map shard to rank_level_local_sid
    # key: (seq_idx, logical_sid), value: rank_level_local_sid.
    shard_to_rank_local_sid = {}
    for seq_idx, shard in all_shards_with_seq_id:
        rank_id = shard.rid
        rank_level_local_sid = int(rank_shard_counters[rank_id])
        shard_to_rank_local_sid[(seq_idx, shard.logical_sid)] = rank_level_local_sid
        rank_shard_counters[rank_id] += 1

    mlp_num_shards = rank_shard_counters
    max_shards_per_rank = int(mlp_num_shards.max())

    max_shards_per_seq = max(len(seq) for seq in sequence_plans)

    mlp_q_dispatch = torch.ones((world_size, max_shards_per_rank), dtype=torch.int64) * -1
    cp_seq_lens = torch.zeros((world_size, max_shards_per_rank), dtype=torch.int64)
    kv_context_size = torch.zeros((world_size, max_shards_per_rank), dtype=torch.int64)
    q_to_num_kv_seq = torch.ones((world_size, max_shards_per_rank), dtype=torch.int64) * -1
    kv_to_q_mapping = torch.ones((world_size, max_shards_per_rank, max_shards_per_seq, 2), dtype=torch.int64) * -1
    kv_to_q_rank = torch.ones((world_size, max_shards_per_rank, max_shards_per_seq), dtype=torch.int64) * -1

    for seq_idx, original_sequence in enumerate(sequence_plans):
        sequence = sorted(original_sequence, key=lambda s: s.logical_sid)
        sequence_cumsum = [0] + [s.shard_len for s in sequence]
        sequence_cumsum = torch.tensor(sequence_cumsum, dtype=torch.int64).cumsum(dim=0)

        for logical_sid, shard_info in enumerate(sequence):
            # mlp rank of current shard
            p_rank = shard_info.rid
            # rank index of current shard
            p_local_sid = shard_to_rank_local_sid[(seq_idx, logical_sid)]

            mlp_q_dispatch[p_rank, p_local_sid] = shard_info.dispatch_rid
            cp_seq_lens[p_rank, p_local_sid] = shard_info.shard_len
            q_to_num_kv_seq[p_rank, p_local_sid] = logical_sid + 1
            kv_context_size[p_rank, p_local_sid] = sequence_cumsum[logical_sid]

            # loop all related Q Shards of current KV Shard.
            for q_idx, q_logical_sid in enumerate(range(logical_sid, len(sequence))):
                dest_q_shard_info = sequence[q_logical_sid]
                dest_q_computed_sid = shard_to_rank_local_sid[(seq_idx, q_logical_sid)]

                # kv_to_q_mapping: record the (rid, sid) of the target Q
                kv_to_q_mapping[p_rank, p_local_sid, q_idx, 0] = dest_q_shard_info.rid
                kv_to_q_mapping[p_rank, p_local_sid, q_idx, 1] = dest_q_computed_sid

                # kv_to_q_rank: record the index of current KV of the target Q
                kv_to_q_rank[p_rank, p_local_sid, q_idx] = logical_sid

    q_to_num_kv_tokens = kv_context_size + cp_seq_lens

    return (
        mlp_num_shards,
        mlp_q_dispatch,
        cp_seq_lens,
        kv_to_q_mapping,
        kv_to_q_rank,
        kv_context_size,
        q_to_num_kv_seq,
        q_to_num_kv_tokens,
    )


# Transfer Items output from planner to List[List[ShardInfo]]
def items_into_shardinfos(data: List[Dict[str, Any]]) -> List[List[ShardInfo]]:
    sequences_map = collections.defaultdict(list)


    for shard_dict in data:
        sequence_key = (shard_dict['seqid']) # sequence_key = (shard_dict['src_gpuid'], shard_dict['seqid'])
        sequences_map[sequence_key].append(shard_dict)

    all_sequences: List[List[ShardInfo]] = []

    for sequence_key, shard_dicts in sequences_map.items():
        shard_dicts.sort(key=lambda d: d['shard_id'])

        current_sequence_shards: List[ShardInfo] = []
        for shard_dict in shard_dicts:
            shard_info = ShardInfo(
                rid=shard_dict['src_gpuid'],
                dispatch_rid=shard_dict['gpuid'],
                logical_sid=shard_dict['shard_id'],
                shard_len=shard_dict['q'],
                # document_id=shard_dict['seqid']
            )
            current_sequence_shards.append(shard_info)
        
        all_sequences.append(current_sequence_shards)
        
    return all_sequences