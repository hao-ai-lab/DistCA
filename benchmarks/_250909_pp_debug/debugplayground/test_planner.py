# %%
from typing import List
from types import SimpleNamespace


class Item:
    def __init__(self,
                 model_config,
                 seq_len, seqid, gpuid, src_gpuid,
                 *item_dicts,
                 is_original=True):
        self.model_config = model_config
        assert len(item_dicts) == 1 or len(item_dicts) == 2, "item_dicts must contain exactly 1 or 2 items"
        for item in item_dicts:
            assert 'q' in item and 'kv' in item, "Each item must have 'q' and 'kv' keys"

        self.seq_len = seq_len
        self.seqid = seqid
        self.gpuid = gpuid
        self.src_gpuid = src_gpuid
        self.is_original = is_original
        self.shard_id = None

        self.items = [item for item in item_dicts]
        if len(self.items) == 1:
            self.complete = True
            self.complete_item = self.items[0]
        else:
            self.complete = False
            self.head = self.items[0]
            self.tail = self.items[1]
        self.total_flops = self.get_flops()

    def get_flops(self):
        flops = 0
        for item in self.items:
            q = int(item['q'])
            kv = int(item['kv'])
            flops += sum(kv - i for i in range(q))
        return flops

    def get_q_size(self, length):
        hidden_size = self.model_config.hidden_size
        q_size = length * hidden_size
        return q_size

    def get_kv_size(self, length):
        hidden_size = self.model_config.hidden_size
        num_attention_heads = self.model_config.num_attention_heads
        num_key_value_heads = self.model_config.num_key_value_heads

        head_dim = hidden_size // num_attention_heads
        kv_size = 2 * length * num_key_value_heads * head_dim
        return kv_size

    def get_priority(self, local_surplus, remote_deficit):
        max_possible_flops = min(local_surplus, remote_deficit, self.total_flops)
        q_len = max_possible_flops / (self.seq_len + 1)
        kv_len = self.seq_len + 1  # this can be improve if use memcpy instead of transfering all kvs.
        communication_volume = self.get_q_size(q_len) + self.get_kv_size(kv_len)
        # latency = communication_volume / transfer_speed, if we know current rank_id, remote rank_id, network speed. We can compute communication latency.
        priority = communication_volume / max_possible_flops
        # maybe we can have a matrix in the planner to compute the latency of communication.
        return priority

    def split_item(self, flops, remote_gpuid, verbose=False):
        """
        Split the Item according to the specified amount of FLOPs.

        - If flops >= self.total_flops, move entire item to remote_gpuid.
        - If flops < self.total_flops, split the Item and return a new Item containing the head and tail chunks.
          The original Item (self) will be updated accordingly.

        Args:
            flops (float): The amount of computation (FLOPs) to move.
            remote_gpuid (int): The target GPU ID.

        Returns:
            Item or None: The newly created Item object; returns None if move entire item.
            actual_flops_moved (float): The actual amount of FLOPs moved.
        """
        # If the current Item is empty or has no FLOPs to move, splitting is not possible
        if not self.items or self.total_flops <= 0:
            return None

        def rlog(message):
            if verbose:
                print(message)

        # If the requested flops is greater than or equal to the total FLOPs, move the whole Item
        if flops >= self.total_flops:
            # Update the gpuid and is_original status to reflect the move
            self.gpuid = remote_gpuid
            self.is_original = False
            # There is no new items. Return None.
            rlog(f"    - [bold]Move entire item[/bold]: Actual Moving ({self.total_flops:.2f} FLOPs) to satisfy need.")
            return None, self.total_flops

        else:
            flops_per_q = self.seq_len + 1
            q_to_move = int(flops / flops_per_q)

            if q_to_move <= 0:
                return None, 0
            moved_flops_actual = q_to_move * flops_per_q
            rlog(
                f"    - [bold]Splitting item[/bold]: Actual Moving q={q_to_move * 2} ({moved_flops_actual:.2f} FLOPs) to satisfy need.")

            # TODO(Pb): Current flops_per_q only support even seq_len. Need to support odd seq_len in the future.
            if self.complete:
                # Split current Item
                head_dict = {}
                tail_dict = {}
                head_dict['q'] = (self.complete_item['q'] + 1) // 2 - q_to_move
                head_dict['kv'] = head_dict['q']

                tail_dict['q'] = self.complete_item['q'] // 2 - q_to_move
                tail_dict['kv'] = self.complete_item['kv']  # Unchanged

                self.head = head_dict
                self.tail = tail_dict
                self.complete = False
                self.complete_item = None
            else:
                # Split head
                self.head['q'] = self.head['q'] - q_to_move
                self.head['kv'] = self.head['kv'] - q_to_move

                # Split tail
                self.tail['q'] = self.tail['q'] - q_to_move
                self.tail['kv'] = self.tail['kv']  # Unchanged
            self.items = [self.head, self.tail]
            rlog(
                f"Origin flops {self.total_flops}, moved flops: {moved_flops_actual}, current flops: {self.get_flops()}")

            rlog(
                f"    - [debug] total_flops(before)={self.total_flops}, get_flops()={self.get_flops()}, moved_flops_actual={moved_flops_actual}, sum={self.get_flops() + moved_flops_actual}")
            assert self.total_flops == self.get_flops() + moved_flops_actual, f"Total flops should be equal. This error is mostly because of odd doc length. Currently, we only support even doc length. Please pad to even."
            self.total_flops = self.get_flops()
            rlog(
                f"    - [bold]Splitting item[/bold]: Actual Moving q={q_to_move} ({moved_flops_actual:.2f} FLOPs) to satisfy need.")

            new_head_dict = {}
            new_tail_dict = {}
            new_head_dict['q'] = q_to_move
            new_head_dict['kv'] = self.head['kv'] + new_head_dict['q']

            new_tail_dict['q'] = q_to_move
            new_tail_dict['kv'] = self.tail['kv'] - self.tail['q']

            rlog(f"    - Created head chunk: q={new_head_dict['q']}, kv={new_head_dict['kv']}, on GPU {remote_gpuid}")
            rlog(f"    - Created tail chunk: q={new_tail_dict['q']}, kv={new_tail_dict['kv']}, on GPU {remote_gpuid}")

            newly_split_item = Item(
                self.model_config,
                self.seq_len, self.seqid, remote_gpuid, self.src_gpuid,
                new_head_dict, new_tail_dict,
                is_original=False
            )
            assert newly_split_item.total_flops == moved_flops_actual, "Total moved flops should be equal"
            return newly_split_item, moved_flops_actual

    def to_dicts(self):
        output_dicts = []
        for item_chunk in self.items:
            if item_chunk['q'] <= 0:
                continue
            item_dict = {
                'q': item_chunk['q'],
                'kv': item_chunk['kv'],
                'gpuid': self.gpuid,
                'seqid': self.seqid,
                'src_gpuid': self.src_gpuid,
                'is_original': self.is_original,
                'shard_id': self.shard_id
            }
            output_dicts.append(item_dict)
        return output_dicts

    def __repr__(self):
        if self.complete:
            return (f"<Item seqid={self.seqid} src_gpuid = {self.src_gpuid} gpuid={self.gpuid} complete="
                    f"q={self.complete_item['q']} kv={self.complete_item['kv']}>")
        else:
            return (f"<Item seqid={self.seqid} src_gpuid = {self.src_gpuid} gpuid={self.gpuid} split="
                    f"head(q={self.head['q']}, kv={self.head['kv']}) "
                    f"tail(q={self.tail['q']}, kv={self.tail['kv']})>")


def batch_to_items_with_dummy(batches: List[List[int]], num_tokens_per_rank: int, as_world_size: int,
                              model_config: dict):
    print(
        f"Entering batch_to_items_with_dummy: {batches = }, {num_tokens_per_rank = }, {as_world_size = }, {model_config = }")

    items = []
    seqid = 0
    rank_budgets = [num_tokens_per_rank] * as_world_size
    

    for batch in batches:
        current_rank_idx = 0
        if len(batch) == 1 and batch[0] < num_tokens_per_rank:
            while current_rank_idx < as_world_size and rank_budgets[current_rank_idx] == 0:
                current_rank_idx += 1
            # assert current_rank_idx < len(rank_budgets), f"current_rank_idx should be less than as_world_size. {current_rank_idx = }, {as_world_size = }, {len(rank_budgets) = }, {batch = }"
            assert rank_budgets[current_rank_idx] == num_tokens_per_rank, f"dummy doc should put on a empty rank. {rank_budgets = }, {current_rank_idx = }"
            # dummy doc, this rank only put this dummy item.
            doc_len = batch[0]
            item_dict = {'q': doc_len, 'kv': doc_len}
            new_item = Item(model_config, doc_len, seqid, current_rank_idx, current_rank_idx,
                            item_dict, is_original=True)
            items.append(new_item)
            current_rank_idx += 1
            seqid += 1
        else:
            for doc_length in batch:
                doc_len = doc_length
                remaining_len = doc_len
                head_prefix_len = 0
                tail_suffix_start_pos = doc_len
                is_original_flag = True

                while remaining_len > 0:
                    while current_rank_idx < as_world_size and rank_budgets[current_rank_idx] == 0:
                        current_rank_idx += 1
                    if current_rank_idx >= as_world_size:
                        raise ValueError(f"Not enough space for seqid {seqid}, remaining_len {remaining_len}.")

                    chunk_size = min(remaining_len, rank_budgets[current_rank_idx])

                    if remaining_len == doc_len and chunk_size == doc_len:
                        item_dict = {'q': doc_len, 'kv': doc_len}
                        new_item = Item(model_config, doc_len, seqid, current_rank_idx, current_rank_idx,
                                        item_dict, is_original=True)
                        items.append(new_item)
                    else:
                        q_len_head = chunk_size // 2
                        q_len_tail = chunk_size - q_len_head

                        new_head_kv = head_prefix_len + q_len_head
                        head_item = {'q': q_len_head, 'kv': new_head_kv}
                        tail_item = {'q': q_len_tail, 'kv': tail_suffix_start_pos}

                        new_item = Item(model_config, doc_len, seqid, current_rank_idx, current_rank_idx,
                                        head_item, tail_item, is_original=is_original_flag)
                        items.append(new_item)

                        head_prefix_len = new_head_kv
                        tail_suffix_start_pos -= q_len_tail
                        is_original_flag = False

                    rank_budgets[current_rank_idx] -= chunk_size
                    remaining_len -= chunk_size

                seqid += 1

    return items


def test_batch_to_items_with_dummy():
    # Test with real data from logs
    batches = [
        [1376, 4080, 2288, 3376, 5536, 2976, 2400, 2400, 3248, 3408, 64, 4240, 5424, 3680, 28496, 4912, 3728, 2816, 64,
         2848, 5776, 5424, 4928, 4320, 4784, 576, 4528, 3712, 2144, 2688, 1104, 560, 4976, 2912, 2896, 7296, 2912, 4000,
         1136, 1600, 3456, 1408, 5408, 688, 5824, 320, 3696, 416, 992, 4736, 848, 816, 1472, 960, 16, 2752, 1024, 768,
         3888, 1456, 3824, 6016, 624, 1072, 2016, 4560, 2336, 3312, 1456, 6192, 2080, 2368, 4720, 4768, 3712, 3632,
         3184, 3344, 4992, 1360],
        [1520, 848, 2096, 4896, 1552, 5040, 3040, 3184, 2720, 2208, 4000, 5120, 224, 1904, 3520, 4448, 2288, 5728, 1024,
         2768, 928, 1712, 5872, 4752, 5616, 272, 128, 3216, 6240, 3168, 5760, 3440, 4096, 5856, 1104, 8288, 5024, 5328,
         1808, 5280, 5248, 3776, 5984, 2400, 704, 5008, 1744, 4400, 736, 912, 4416, 4208, 2736, 1696, 2272, 2464, 640,
         4672, 4096, 5952, 2240, 1120, 3056, 3616, 4096, 4864, 1904, 3680, 1536, 1680, 4448, 6032, 944, 5952, 192, 4320,
         4400, 3248, 1152, 2080, 1504],
        [512, 3296, 3360, 5424, 1280, 4496, 976, 1088, 2864, 5776, 3312, 640, 3888, 1216, 2928, 6080, 2592, 2480, 5024,
         4896, 1296, 2432, 1936, 1936, 3568, 11872, 3440, 2544, 1152, 5328, 64, 4320, 3392, 53024, 4496, 5424, 11936,
         1792, 5472, 1856, 2784, 5056, 368, 672, 1872, 61984],
        [14512, 256, 3328, 5680, 192, 1632, 3984, 2592, 5392, 64, 4336, 992, 5472, 400, 3392, 3232, 64, 1312, 2384,
         2544, 3088, 2128, 1664, 1632, 1792, 6176, 1760, 3904, 4992, 3040, 2096, 4960, 4160, 3664, 1456, 9552, 4528,
         1216, 1824, 2560, 5152, 5248, 2992, 1712, 3200, 4608, 4512, 400, 160, 3760, 5248, 2800, 256, 3712, 4064, 2544,
         3216, 1936, 1328, 3536, 1104, 2928, 3648, 416, 528, 1968, 5904, 992, 5632, 5760, 3360, 2320, 2528, 1184,
         35536],
        [8], [8], [8], [8], [8], [8], [8], [8], [8], [8], [8], [8], [8], [8], [8], [8], [8], [8], [8], [8], [8], [8],
        [8], [8], [8], [8], [8], [8]]
    num_tokens_per_rank = 65536
    as_world_size = 32
    model_config = SimpleNamespace(hidden_size=4096, num_attention_heads=1, num_key_value_heads=4, num_hidden_layers=1)
    list_items = batch_to_items_with_dummy(batches=batches,
                                           num_tokens_per_rank=num_tokens_per_rank,
                                           as_world_size=as_world_size,
                                           model_config=model_config)
    print(list_item)


test_batch_to_items_with_dummy()
# %%
