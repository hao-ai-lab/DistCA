import torch
import random
from itertools import accumulate

class doc_shard:
    def __init__(self, shard_len, shard_id, doc_id, doc_len, prefix_len):
        """
        Initialize the doc_shard object with shard length, shard ID, prefix length, and document length.
        """
        self.shard_len = shard_len
        self.shard_id = shard_id
        self.doc_id = doc_id
        self.doc_len = doc_len
        self.prefix_len = prefix_len
    
    def __repr__(self):
        """
        String representation of the doc_shard object.
        """
        return f"doc_shard(shard_len={self.shard_len}, shard_id={self.shard_id}, doc_id={self.doc_id}, doc_len={self.doc_len}, prefix_len={self.prefix_len})"

# =============== Per-Doc CP Sharding ==================
def compute_per_doc_cp_shard_doc_len(doc_lens, context_length, cp_size, eval_workload=False):
    """
    Compute the per-document sharding for CP (Column Parallel) sharding.
    Each document is divided into chunks of 2 * cp_size.
    """
    n_doc = len(doc_lens)
    doc_shards = [[] for _ in range(2 * cp_size)] # (2 * cp_size, <=n_doc)
    workload_shards = [0 for _ in range(2 * cp_size)]
    remainder_idx = 0
    for doc_id, doc in enumerate(doc_lens):
        chunk_size = doc // (2 * cp_size)
        tmp_length = [chunk_size] * (2 * cp_size)
        n_ramainder = doc - chunk_size * (2 * cp_size)
        while n_ramainder > 0:
            tmp_length[remainder_idx] += 1
            remainder_idx += 1
            remainder_idx = remainder_idx % (2 * cp_size)
            n_ramainder -= 1

        assert sum(tmp_length) == doc, f"Total length {sum(tmp_length)} must equals document length {doc}."
    
        # construct the doc_shard
        prefix_len = 0
        for i in range(2 * cp_size):
            if tmp_length[i] == 0:
                doc_shards[i].append(None)
            else:
                doc_shard_i = doc_shard(tmp_length[i], i, doc_id, doc, prefix_len)
                doc_shards[i].append(doc_shard_i)
                prefix_len += tmp_length[i]

                if eval_workload == True:
                    workload = (2 * prefix_len + 1 + tmp_length[i]) * tmp_length[i]
                    workload_shards[i] += workload
            
    if eval_workload == True:
        # print per-rank workload:
        per_rank_workload = []
        for i in range(cp_size):
            total = workload_shards[i] + workload_shards[2 * cp_size - 1 - i]
            per_rank_workload.append(total)
        
        # normalize to min
        min_workload = min(per_rank_workload)
        total_workload = sum(per_rank_workload) / 1024 / 1024 / 1024
        for i in range(len(per_rank_workload)):
            per_rank_workload[i] = round(per_rank_workload[i] / min_workload, 4)
        print(f"Per_doc Normalized Workload: {per_rank_workload}, total_workload: {total_workload}")

    return doc_shards

def compute_per_doc_metadate_combined(context_length, q, k, v, doc_lens, doc_shards, cp_size, rank, d_out=None):
    """
    Compute the metadata (e.g., cumulative sequence lengths) for per-document CP.
    """
    # ============== Compute metadata =================
    chunk_size = context_length // (2 * cp_size)
    global_cu_lens =  [0] + list(accumulate(doc_lens))

    local_q_chunks = []
    local_k_chunks = []
    local_v_chunks = []
    local_d_out_chunks = []
    cu_seqlens_q_list = []
    max_seqlen_q_list = []
    cu_seqlens_k_list = []
    max_seqlen_k_list = []
    kv_idx_list = []
    for chunk_id in range(2):
        if chunk_id == 0:
            chunk_index = rank
        else:
            chunk_index = 2 * cp_size - 1 - rank

        this_doc_shards = doc_shards[chunk_index]
        this_chunk_docs = []

        local_q_list = []
        local_k_list = []
        local_v_list = []
        local_d_out_list = []
        kv_len_list = []
        kv_idx = []

        for doc_shard_i in this_doc_shards:
            if doc_shard_i is None:
                continue
            else:
                this_chunk_docs.append(doc_shard_i.shard_len)
                q_chunk_start = global_cu_lens[doc_shard_i.doc_id] + doc_shard_i.prefix_len
                q_chunk_end = q_chunk_start + doc_shard_i.shard_len
                local_q_list.append(q[q_chunk_start:q_chunk_end, :, :]) # qkv input should have the same format
                local_k_list.append(k[q_chunk_start:q_chunk_end, :, :])
                local_v_list.append(v[q_chunk_start:q_chunk_end, :, :])
                if d_out is not None:
                    local_d_out_list.append(d_out[q_chunk_start:q_chunk_end, :, :])

                k_chunk_start = global_cu_lens[doc_shard_i.doc_id]
                k_chunk_end = k_chunk_start + doc_shard_i.prefix_len + doc_shard_i.shard_len
                kv_idx.append((k_chunk_start, k_chunk_end))
                kv_len_list.append(doc_shard_i.prefix_len + doc_shard_i.shard_len)
    
        assert sum(this_chunk_docs) == chunk_size, f"Total length {sum(this_chunk_docs)} must equals chunk_size {chunk_size}."

    
        local_q_chunks.append(torch.cat(local_q_list, dim=0))
        local_k_chunks.append(torch.cat(local_k_list, dim=0))
        local_v_chunks.append(torch.cat(local_v_list, dim=0))
        if d_out is not None:
            local_d_out_chunks.append(torch.cat(local_d_out_list, dim=0))
        cu_seqlens_q_list.append(torch.tensor([0] + list(accumulate(this_chunk_docs)), dtype=torch.int32).to(q.device))
        max_seqlen_q_list.append(torch.tensor([max(this_chunk_docs)], dtype=torch.int32).to(q.device))
        cu_seqlens_k_list.append(torch.tensor([0] + list(accumulate(kv_len_list)), dtype=torch.int32).to(q.device))
        max_seqlen_k_list.append(torch.tensor([max(kv_len_list)], dtype=torch.int32).to(q.device))
        kv_idx_list.append(kv_idx)

    local_q = torch.cat(local_q_chunks, dim=0).clone()
    local_k = torch.cat(local_k_chunks, dim=0).clone()
    local_v = torch.cat(local_v_chunks, dim=0).clone()
    local_q.requires_grad_(True)
    local_k.requires_grad_(True)
    local_v.requires_grad_(True)
    if d_out is not None:
        local_d_out = torch.cat(local_d_out_chunks, dim=0)
        local_d_out.requires_grad_(True)
    else:
        local_d_out = None

    return local_q, local_k, local_v, cu_seqlens_q_list, cu_seqlens_k_list, max_seqlen_q_list, max_seqlen_k_list, kv_idx_list, local_d_out

def get_per_doc_local_result(context_length, global_result, doc_lens, doc_shards, cp_size, rank, chunk_id):
    """
    Get the local result for per-doc CP based on the global result.
    """
    chunk_size = context_length // (2 * cp_size)
    if chunk_id == 0:
        chunk_index = rank
    else:
        chunk_index = 2 * cp_size - 1 - rank

    global_cu_lens =  [0] + list(accumulate(doc_lens))

    this_doc_shards = doc_shards[chunk_index]
    this_chunk_docs = []

    local_out_list = []

    for doc_shard_i in this_doc_shards:
        if doc_shard_i is None:
            continue
        else:
            this_chunk_docs.append(doc_shard_i.shard_len)
            chunk_start = global_cu_lens[doc_shard_i.doc_id] + doc_shard_i.prefix_len
            chunk_end = chunk_start + doc_shard_i.shard_len
            local_out_list.append(global_result[chunk_start:chunk_end, :, :])
    
    local_result = torch.cat(local_out_list, dim=0)

    return local_result

def per_doc_correctness_evaluate(global_out_ref, local_out, context_length, cp_size, rank, doc_lens, doc_shards, rtol=None, atol=None):
    out_chunks = []
    for chunk_id in range(2):
        chunk_result = get_per_doc_local_result(context_length, global_out_ref, doc_lens, doc_shards, cp_size, rank, chunk_id)
        out_chunks.append(chunk_result)
    ref_local_out = torch.cat(out_chunks, dim=0)
    torch.testing.assert_close(ref_local_out, local_out, rtol=rtol, atol=atol)

def kv_shuffle_for_per_doc_cp(context_length, k_tensor_list, v_tensor_list, doc_lens, doc_shards, cp_size):
    """
    This function has two usages:
    * (1) Use the kv tensors gathered from all ranks and shuffle them to original order (order in global kv tensor).
    * (2) It can also used to shuffle the result on each rank to compare with the original result.
    """
    chunk_size = context_length // (2 * cp_size)
    global_cu_lens =  [0] + list(accumulate(doc_lens))
    global_k = [[] for _ in range(len(doc_lens))]
    global_v = [[] for _ in range(len(doc_lens))]
    for chunk_id in range(2):
        rank_range = range(cp_size) if chunk_id == 0 else range(cp_size - 1, -1, -1)
        for rank in rank_range:
            if chunk_id == 0:
                chunk_index = rank
            else:
                chunk_index = 2 * cp_size - 1 - rank

            k_tensor = k_tensor_list[rank][chunk_id * chunk_size:(chunk_id + 1) * chunk_size, :, :]
            v_tensor = v_tensor_list[rank][chunk_id * chunk_size:(chunk_id + 1) * chunk_size, :, :] if v_tensor_list is not None else None


            this_doc_shards = doc_shards[chunk_index]
            offset = 0
            for doc_shard_i in this_doc_shards:
                if doc_shard_i is not None:
                    this_doc_k = k_tensor[offset:offset + doc_shard_i.shard_len, :, :]
                    this_doc_v = v_tensor[offset:offset + doc_shard_i.shard_len, :, :] if v_tensor is not None else None
                    offset += doc_shard_i.shard_len

                    global_k[doc_shard_i.doc_id].append(this_doc_k)
                    if v_tensor is not None:
                        global_v[doc_shard_i.doc_id].append(this_doc_v)

    # Concatenate the tensors for each chunk
    flat_k = [k_chunk for sub in global_k for k_chunk in sub]
    flat_v = [v_chunk for sub in global_v for v_chunk in sub] if v_tensor_list is not None else None

    # Concatenate the tensors for each chunk
    shuffled_k_tensor = torch.cat(flat_k, dim=0)
    if flat_v is not None:
        shuffled_v_tensor = torch.cat(flat_v, dim=0)
    else:
        shuffled_v_tensor = None

    assert shuffled_k_tensor.shape[0] == context_length, f"shuffled_k_tensor shape {shuffled_k_tensor.shape[0]} must equals context length {context_length}."
                    
    return shuffled_k_tensor, shuffled_v_tensor

def kv_unshuffle_for_per_doc_cp(context_length, k_tensor, v_tensor, doc_lens, doc_shards, cp_size):
    """
    Unshuffle the kv tensor for reducescatter in the per-doc backward.
    """
    chunk_size = context_length // (2 * cp_size)
    global_k = []
    global_v = []
    global_cu_lens =  [0] + list(accumulate(doc_lens))
    for rank in range(cp_size):
        for chunk_id in range(2):
            if chunk_id == 0:
                chunk_index = rank
            else:
                chunk_index = 2 * cp_size - 1 - rank

            this_doc_shards = doc_shards[chunk_index]
            offset = 0

            for doc_shard_i in this_doc_shards:
                if doc_shard_i is None:
                    continue
                else:
                    chunk_start = global_cu_lens[doc_shard_i.doc_id] + doc_shard_i.prefix_len
                    chunk_end = chunk_start + doc_shard_i.shard_len
                    global_k.append(k_tensor[chunk_start:chunk_end, :, :]) # qkv input should have the same format
                    global_v.append(v_tensor[chunk_start:chunk_end, :, :])
                    
    return torch.cat(global_k, dim=0), torch.cat(global_v, dim=0)

# ================= Per-Seq CP Sharding =================
def compute_per_seq_metadate_combined(context_length, q_tensor, k_tensor, v_tensor, doc_lens, cp_size, rank, d_out=None):
    """
    Compute the cumulative sequence lengths for per-sequence CP.
    """
    # ============== Split doc lens for sequence sharding =================
    chunk_size = context_length // (2 * cp_size)

    split_doc_lens = []
    prefix_lens = []
    cur_length = 0
    for i, doc_len in enumerate(doc_lens):
        if cur_length + doc_len <= chunk_size: 
            split_doc_lens.append(doc_len)
            prefix_lens.append(0)
            cur_length += doc_len
        else: # split the document
            split_doc_lens.append(chunk_size - cur_length)
            prefix_lens.append(0)
            cu_prefix = chunk_size - cur_length
            remained_length = doc_len - (chunk_size - cur_length)
            while remained_length > chunk_size:
                split_doc_lens.append(chunk_size)
                prefix_lens.append(cu_prefix)
                cu_prefix += chunk_size
                remained_length -= chunk_size
            if remained_length > 0:
                split_doc_lens.append(remained_length)
                prefix_lens.append(cu_prefix)
                cur_length = remained_length
            else:
                cur_length = 0
        
        if cur_length == chunk_size:
            cur_length = 0
    assert sum(split_doc_lens) == context_length, f"Total length {sum(split_doc_lens)} must equals context length {context_length}."
    
    cur_offset = 0
    doc_idx_list = [0] # to record the document index for each chunk
    for i, doc_len in enumerate(split_doc_lens):
        cur_length += doc_len
        if cur_length == chunk_size:
            doc_idx_list.append(i + 1)
            cur_length = 0
        elif cur_length > chunk_size:
            assert False, "cur_length > chunk_size, this should not happen."
        
    for i in range(len(doc_idx_list)-1):
        assert sum(split_doc_lens[doc_idx_list[i]:doc_idx_list[i+1]]) == chunk_size, f"error doc per chunk"
    
    # ============== Compute metadata =================
    local_q_chunks = []
    local_k_chunks = []
    local_v_chunks = []
    local_d_out_chunks = []
    cu_seqlens_q_list = []
    max_seqlen_q_list = []
    cu_seqlens_k_list = []
    max_seqlen_k_list = []
    k_offset_list = []
    for chunk_id in range(2):
        if chunk_id == 0:
            chunk_index = rank
        else:
            chunk_index = 2 * cp_size - 1 - rank
    
        this_chunk_docs = split_doc_lens[doc_idx_list[chunk_index]:doc_idx_list[chunk_index+1]]
        k_offset = chunk_index * chunk_size
        doc_id_split = doc_idx_list[chunk_index]

        local_q_chunks.append(q_tensor.chunk(2 * cp_size, dim=0)[chunk_index])
        local_k_chunks.append(k_tensor.chunk(2 * cp_size, dim=0)[chunk_index])
        local_v_chunks.append(v_tensor.chunk(2 * cp_size, dim=0)[chunk_index])
        if d_out is not None:
            local_d_out_chunks.append(d_out.chunk(2 * cp_size, dim=0)[chunk_index])

        cu_seqlens_q_list.append(torch.tensor([0] + list(accumulate(this_chunk_docs)), dtype=torch.int32).to(q_tensor.device))
        max_seqlen_q_list.append(torch.tensor([max(this_chunk_docs)], dtype=torch.int32).to(q_tensor.device))

        # check if the first doc is splitted
        if prefix_lens[doc_id_split] > 0:
            k_offset -= prefix_lens[doc_id_split]
            this_chunk_docs[0] += prefix_lens[doc_id_split]
            assert k_offset >= 0, f"error k_offset {k_offset} < 0"

        cu_seqlens_k_list.append(torch.tensor([0] + list(accumulate(this_chunk_docs)), dtype=torch.int32).to(q_tensor.device))
        max_seqlen_k_list.append(torch.tensor([max(this_chunk_docs)], dtype=torch.int32).to(q_tensor.device))
        k_offset_list.append(k_offset)

    local_q = torch.cat(local_q_chunks, dim=0)
    local_k = torch.cat(local_k_chunks, dim=0)
    local_v = torch.cat(local_v_chunks, dim=0)
    local_q.requires_grad_(True)
    local_k.requires_grad_(True)
    local_v.requires_grad_(True)
    if d_out is not None:
        local_d_out = torch.cat(local_d_out_chunks, dim=0)
        local_d_out.requires_grad_(True)
    else:
        local_d_out = None

    return local_q, local_k, local_v, cu_seqlens_q_list, cu_seqlens_k_list, max_seqlen_q_list, max_seqlen_k_list, k_offset_list, local_d_out

def per_seq_correctness_evaluate(global_out_ref, local_out, context_length, cp_size, rank, rtol=None, atol=None):
    chunk_size = context_length // (2 * cp_size)
    out_chunks = global_out_ref.chunk(2 * cp_size, dim=0)
    ref_local_out = torch.cat([out_chunks[rank], out_chunks[2 * cp_size - 1 - rank]], dim=0)
    torch.testing.assert_close(ref_local_out, local_out, rtol=rtol, atol=atol)

# ================= Others =================
def generate_doc_lens(avg_doc_len, std_doc_len, context_length, divide_cp=1):
    """
    Generate a list of document lengths based on average and standard deviation.
    """
    doc_lens = []
    cur_len = 0
    while cur_len <= context_length:
        doc_len = int(torch.normal(avg_doc_len, std_doc_len, size=(1,1)).item() * context_length)

        # Ensure doc_len is a multiple of cp_size
        if divide_cp > 1:
            doc_len = (doc_len // divide_cp) * divide_cp

        if doc_len <= 0:
            continue
        else:
            doc_lens.append(doc_len)
            cur_len += doc_len
    
    # Ensure the last document length does not exceed the context length
    if cur_len > context_length:
        doc_lens[-1] = context_length - sum(doc_lens[:-1])
    if doc_lens[-1] == 0:
        doc_lens = doc_lens[:-1]
    
    assert sum(doc_lens) == context_length, f"Total length {sum(doc_lens)} must equals context length {context_length}."
    for doc_len in doc_lens:
        assert doc_len % divide_cp == 0, f"Document length {doc_len} must be divisible by {divide_cp}."

    return doc_lens

def generate_doc_lens_1LNS(long_doc_ratio, long_doc_std_ratio, short_doc_len, short_doc_std, context_length, divide_cp=1):
    """
    Generate a list of document lengths based on average and standard deviation.
    The length pattern is 1LNS (1 Long, N Short).
    """
    doc_lens = []
    cur_len = 0
    doc_len = int(torch.normal(long_doc_ratio, long_doc_std_ratio, size=(1,1)).item() * context_length)
    if divide_cp > 1:
        doc_len = (doc_len // divide_cp) * divide_cp  # Ensure doc_len is a multiple of cp_size
    if doc_len > 0:
        doc_lens.append(doc_len)
        cur_len += doc_len

    while cur_len <= context_length:
        doc_len = int(torch.normal(short_doc_len, short_doc_std, size=(1,1)).item())

        # Ensure doc_len is a multiple of cp_size
        if divide_cp > 1:
            doc_len = (doc_len // divide_cp) * divide_cp

        if doc_len <= 0:
            continue
        else:
            doc_lens.append(doc_len)
            cur_len += doc_len
    
    # Ensure the last document length does not exceed the context length
    if cur_len > context_length:
        doc_lens[-1] = context_length - sum(doc_lens[:-1])
    if doc_lens[-1] == 0:
        doc_lens = doc_lens[:-1]
    
    assert sum(doc_lens) == context_length, f"Total length {sum(doc_lens)} must equals context length {context_length}."
    for doc_len in doc_lens:
        assert doc_len % divide_cp == 0, f"Document length {doc_len} must be divisible by {divide_cp}."

    # shuffle the doc_lens
    random.shuffle(doc_lens)

    return doc_lens

def compute_workload(cu_seqlens_q, cu_seqlens_k):
    """
    Compute the workload based on cumulative sequence lengths.
    """
    workload = 0
    for i in range(len(cu_seqlens_q) - 1):
        k_length = cu_seqlens_k[i + 1].item() - cu_seqlens_k[i].item()
        q_length = cu_seqlens_q[i + 1].item() - cu_seqlens_q[i].item()
        upper_k_length = k_length - q_length + 1

        workload += (upper_k_length + k_length) * q_length
    return workload
