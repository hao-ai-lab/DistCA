import wlbllm
import wlbllm.registry
from wlbllm.per_doc_cp_attn import PerDocumentCPAttention
from wlbllm.per_seq_cp_attn import PerSequenceCPAttention
import rich
import time

def wlbllm_func__per_seq(*args, **kwargs):
    """
    Patch to 
        TransformerEngine/transformer_engine/pytorch/attention/dot_product_attention/backends.py
    
    This is a wrapper of PerSequenceCPAttention.apply()
    """
    (
        q, 
        k,
        v,
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlen_q, 
        max_seqlen_k,
        dropout_p,
    ) = args
    softmax_scale = kwargs["softmax_scale"]

    doc_lens = wlbllm.registry.get("doc_lens")
    # kv_idx_list = wlbllm.registry.get("kv_idx_list")
    k_offset_list = wlbllm.registry.get("k_offset_list")
    cp_group = wlbllm.registry.get("cp_group")
    cp_stream = wlbllm.registry.get("cp_stream")

    cu_seqlens_q_list = wlbllm.registry.get("cu_seqlens_q_list")
    cu_seqlens_kv_list = wlbllm.registry.get("cu_seqlens_kv_list")
    max_seqlen_q_list = wlbllm.registry.get("max_seqlen_q_list")
    max_seqlen_kv_list = wlbllm.registry.get("max_seqlen_kv_list")

    out = PerSequenceCPAttention.apply(
        q, k, v,
        cu_seqlens_q_list, cu_seqlens_kv_list,
        max_seqlen_q_list, max_seqlen_kv_list,
        k_offset_list, 
        dropout_p,
        softmax_scale, 
        "causal",
        cp_group,
        cp_stream,
        None, 
        None, 
        None,
    )
    # torch.cuda.synchronize()
    return out


def wlbllm_func__per_doc(*args, **kwargs):
    """
    Patch to 
        TransformerEngine/transformer_engine/pytorch/attention/dot_product_attention/backends.py
    
    This is a wrapper of PerDocumentCPAttention.apply()
    """
    # import traceback    
    # traceback.print_stack()
    # print("kwargs:", kwargs)
    # time.sleep(10)
    # exit(0)
    (
        q, 
        k,
        v,
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlen_q, 
        max_seqlen_k,
        dropout_p,
    ) = args
    softmax_scale = kwargs["softmax_scale"]

    doc_lens = wlbllm.registry.get("doc_lens")
    doc_shards = wlbllm.registry.get("doc_shards")
    kv_idx_list = wlbllm.registry.get("kv_idx_list")
    cp_group = wlbllm.registry.get("cp_group")
    cp_stream = wlbllm.registry.get("cp_stream")

    cu_seqlens_q_list = wlbllm.registry.get("cu_seqlens_q_list")
    cu_seqlens_kv_list = wlbllm.registry.get("cu_seqlens_kv_list")
    max_seqlen_q_list = wlbllm.registry.get("max_seqlen_q_list")
    max_seqlen_kv_list = wlbllm.registry.get("max_seqlen_kv_list")

    out = PerDocumentCPAttention.apply(
        q, 
        k, 
        v,
        cu_seqlens_q_list, cu_seqlens_kv_list,
        max_seqlen_q_list, max_seqlen_kv_list,
        doc_lens,
        doc_shards,
        kv_idx_list, 
        dropout_p,
        softmax_scale, 
        "causal",
        cp_group,
        cp_stream ,
        None, 
        None, 
        None,
    )
    # torch.cuda.synchronize()
    return out


def wlbllm_func(*args, **kwargs):
    mode = wlbllm.registry.get("mode")
    print(f"🟡 wlbllm_func: mode={mode}")
    if mode not in ["per-doc", "per-seq"]:
        mode = "per-doc"
        print(f"🟡 Warning: Invalid mode: {mode}, using per-doc mode as default.")
    if mode == "per-doc":
        return wlbllm_func__per_doc(*args, **kwargs)
    elif mode == "per-seq":
        return wlbllm_func__per_seq(*args, **kwargs)
    else:
        raise ValueError(f"Invalid mode: {mode}")
    return