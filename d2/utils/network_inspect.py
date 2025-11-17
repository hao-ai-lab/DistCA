import os
from typing import Tuple
import json

def print_2d_tensor(name: str, tensor, unit: str = None):
    """Print a 2D tensor with optional unit conversion.
    
    Args:
        name: Name of the tensor to print
        tensor: 2D tensor to print 
        unit: Optional unit to convert values to ('KB', 'MB', or 'GB')
    """
    print(f"🟡 {name} = ")
    
    # Convert to requested unit
    if unit == "KB":
        scale = 1024
        unit_str = " KB"
    elif unit == "MB": 
        scale = 1024 * 1024
        unit_str = " MB"
    elif unit == "GB":
        scale = 1024 * 1024 * 1024
        unit_str = " GB"
    else:
        scale = 1
        unit_str = ""
        
    tensor = tensor // scale if scale > 1 else tensor
        
    # Print each row
    for row in tensor.tolist():
        print(f"    {row}")
                    
def exclude_self_and_sum(t):
    for i in range(len(t)):
        t[i][i] = 0
    return t.sum(dim=1)


FastAllToAllMetadata_Tuple = Tuple[
    'qkv_fwd_fa2a_metadata',
    'qkv_rev_fa2a_metadata',
    'attn_out_fwd_fa2a_metadata',
    'attn_out_rev_fa2a_metadata',
]

# FIXME: Decouple the data logic from the output logic.
def inspect_network_metadata(metadata: 'FastAllToAllMetadata_Tuple', is_ping, sample_id, tolerance_factor, output_dir, rank, seq_len=None):
    qkv_fwd_metadata__send_transfer_sz = metadata[0].fa2a_metadata[1]
    qkv_fwd_metadata__recv_transfer_sz = metadata[0].fa2a_metadata[3]
    attn_out_fwd_metadata__send_transfer_sz = metadata[1].fa2a_metadata[1]
    attn_out_fwd_metadata__recv_transfer_sz = metadata[1].fa2a_metadata[3]
            
    # Print qkv_fwd_metadata
    # print_2d_tensor("qkv_fwd_metadata.send_transfer_sz", qkv_fwd_metadata__send_transfer_sz, unit="MB")
    # print_2d_tensor("qkv_fwd_metadata.recv_transfer_sz", qkv_fwd_metadata__recv_transfer_sz, unit="MB")
    
    # Print attn_out_fwd_metadata  
    # print_2d_tensor("attn_out_fwd_metadata.send_transfer_sz", attn_out_fwd_metadata__send_transfer_sz, unit="MB")
    # print_2d_tensor("attn_out_fwd_metadata.recv_transfer_sz", attn_out_fwd_metadata__recv_transfer_sz, unit="MB")

    # Calculate the demand for the DispatcherWrapper.buffer (including comm and local kv)
    max_buffer_budget_all_rank = (
        qkv_fwd_metadata__send_transfer_sz
        + qkv_fwd_metadata__recv_transfer_sz
        + attn_out_fwd_metadata__send_transfer_sz
        + attn_out_fwd_metadata__recv_transfer_sz
    ).max().item()

    # Calculate send size from me to others by subtracting diagonal (self-send) from total send
    qkv_fwd_metadata__send_transfer_sz_to_others = exclude_self_and_sum(qkv_fwd_metadata__send_transfer_sz)
    qkv_fwd_metadata__recv_transfer_sz_to_others = exclude_self_and_sum(qkv_fwd_metadata__recv_transfer_sz)
    
    # print_2d_tensor("qkv_fwd_metadata.send_transfer_sz_to_others", qkv_fwd_metadata__send_transfer_sz_to_others, unit="MB")
    # print_2d_tensor("qkv_fwd_metadata.recv_transfer_sz_to_others", qkv_fwd_metadata__recv_transfer_sz_to_others, unit="MB")

    attn_out_fwd_metadata__send_transfer_sz_to_others = exclude_self_and_sum(attn_out_fwd_metadata__send_transfer_sz)
    attn_out_fwd_metadata__recv_transfer_sz_to_others = exclude_self_and_sum(attn_out_fwd_metadata__recv_transfer_sz)

    # print_2d_tensor("attn_out_fwd_metadata.send_transfer_sz_to_others", attn_out_fwd_metadata__send_transfer_sz_to_others, unit="MB")
    # print_2d_tensor("attn_out_fwd_metadata.recv_transfer_sz_to_others", attn_out_fwd_metadata__recv_transfer_sz_to_others, unit="MB")
    
    # Expected send-recv time
    bandwidth_mb = 40 # MB/ms
    bandwidth = bandwidth_mb * 1024 * 1024  # Convert to bytes/ms
    send_time_ms = qkv_fwd_metadata__send_transfer_sz_to_others / bandwidth
    recv_time_ms = qkv_fwd_metadata__recv_transfer_sz_to_others / bandwidth
    # print_2d_tensor("send_time_ms", send_time_ms)
    # print_2d_tensor("recv_time_ms", recv_time_ms)

    max_send_time_ms = (send_time_ms.max().item())
    max_recv_time_ms = (recv_time_ms.max().item())

    max_comm_budget_all_rank = (
          qkv_fwd_metadata__send_transfer_sz_to_others 
        + qkv_fwd_metadata__recv_transfer_sz_to_others 
        + attn_out_fwd_metadata__send_transfer_sz_to_others 
        + attn_out_fwd_metadata__recv_transfer_sz_to_others
    ).max().item()

    if rank == 0:
        network_inspect_file = os.path.join(output_dir, "network_inspect.jsonl")
        with open(network_inspect_file, "a") as f:
            f.write(json.dumps({
                "sample_id": sample_id,
                "is_ping": is_ping,
                "tolerance_factor": tolerance_factor,
                "qkv_fwd_metadata__send_transfer_sz_mb": (qkv_fwd_metadata__send_transfer_sz // (1024 * 1024)).tolist(),
                "qkv_fwd_metadata__recv_transfer_sz_mb": (qkv_fwd_metadata__recv_transfer_sz // (1024 * 1024)).tolist(),
                "attn_out_fwd_metadata__send_transfer_sz_mb": (attn_out_fwd_metadata__send_transfer_sz // (1024 * 1024)).tolist(),
                "attn_out_fwd_metadata__recv_transfer_sz_mb": (attn_out_fwd_metadata__recv_transfer_sz // (1024 * 1024)).tolist(),

                "qkv_fwd_metadata__send_transfer_sz_mb_to_others": (qkv_fwd_metadata__send_transfer_sz_to_others // (1024 * 1024)).tolist(),
                "qkv_fwd_metadata__recv_transfer_sz_mb_from_others": (qkv_fwd_metadata__recv_transfer_sz_to_others // (1024 * 1024)).tolist(),

                "max_comm_budget_all_rank_mb": max_comm_budget_all_rank // (1024 * 1024),
                "max_buffer_budget_all_rank_mb": max_buffer_budget_all_rank // (1024 * 1024),
                "bandwidth_mb": bandwidth_mb,
                "send_time_ms": send_time_ms.tolist(),
                "recv_time_ms": recv_time_ms.tolist(),
                "max_send_time_ms": max_send_time_ms,
                "max_recv_time_ms": max_recv_time_ms,
                "seq_len": seq_len if seq_len is not None else None,
            }) + "\n")

        network_inspect_summary_file = os.path.join(output_dir, "network_inspect.summary.jsonl")
        with open(network_inspect_summary_file, "a") as f:
            f.write(json.dumps({
                "sample_id": sample_id,
                "is_ping": is_ping,
                "tolerance_factor": tolerance_factor,
                "qkv_fwd_send_mb": (qkv_fwd_metadata__send_transfer_sz_to_others // (1024 * 1024)).tolist(),
                "qkv_fwd_recv_mb": (qkv_fwd_metadata__recv_transfer_sz_to_others // (1024 * 1024)).tolist(),

                "max_comm_budget_all_rank_mb": max_comm_budget_all_rank // (1024 * 1024),
                "max_buffer_budget_all_rank_mb": max_buffer_budget_all_rank // (1024 * 1024),
                "send_time_ms": send_time_ms.tolist(),
                "recv_time_ms": recv_time_ms.tolist(),
                "max_send_time_ms": max_send_time_ms,
                "max_recv_time_ms": max_recv_time_ms,
                "seq_len": seq_len if seq_len is not None else None,
            }) + "\n")

    ret = dict(
        max_comm_budget_all_rank=max_comm_budget_all_rank,
        max_buffer_budget_all_rank=max_buffer_budget_all_rank,
    )
    return ret

# # Inspect both metadata sets
# inspect_network_metadata(fa2a_metadata_0, True, sample_id, tolerance_factor, output_dir, rank)
# inspect_network_metadata(fa2a_metadata_1, False, sample_id, tolerance_factor, output_dir, rank)