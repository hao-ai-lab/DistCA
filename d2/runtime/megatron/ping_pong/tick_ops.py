from typing import Any, Dict, Optional, Union

from d2.runtime.megatron.ops import FusedCommAttn, TickSync, post_a2a_attn_out_with_lse
from d2.runtime.megatron.ops.fused_comm_attn import FlashAttnArgs
from d2.runtime.megatron.packed_seq_params import PingPangSingleStepPackedSeqParams
from d2.runtime.megatron.ping_pong.transformer_layer import TransformerLayer


def log_memory_usage(message: str):
    import d2.mem
    d2.mem.log_memory_usage(message)
    return


def forward_pre_core_attn(layer: TransformerLayer, args: Dict[str, Any]):
    log_memory_usage(f"(L{layer.layer_number}) forward_pre_core_attn:(start)")
    log_memory_usage(f"(L{layer.layer_number}) forward_pre_core_attn:(before pre core attn)")
    hidden_states = args.pop("hidden_states")
    query, key, value, residual, attn_mask_type = layer._forward_pre_core_attn(
        hidden_states,
        args["rotary_pos_emb"],
        args["rotary_pos_cos"],
        args["rotary_pos_sin"],
        args["mlp_packed_seq_params"],
        args["sequence_len_offset"],
    )
    log_memory_usage(f"(L{layer.layer_number}) forward_pre_core_attn:(after pre core attn)")

    log_memory_usage(f"(L{layer.layer_number}) forward_pre_core_attn:(before pre mlp to attn)")
    signal = layer._pre_mlp_to_attn(query, key, value, args["packed_seq_params"])
    log_memory_usage(f"(L{layer.layer_number}) forward_pre_core_attn:(after pre mlp to attn)")

    args["query"] = query
    args["key"] = key
    args["value"] = value
    args["residual"] = residual
    args["attn_mask_type"] = attn_mask_type
    args["signal"] = signal
    log_memory_usage(f"(L{layer.layer_number}) forward_pre_core_attn:(return)")
    return args


def layout_mlp_to_attn(layer: TransformerLayer, args: Dict[str, Any]):
    log_memory_usage(f"(L{layer.layer_number}) layout_mlp_to_attn:(start)")
    bwd_resend_qkv = args["packed_seq_params"].bwd_packed_seq_params is not None
    if not bwd_resend_qkv:
        # qkv are stored until attn_out to resend at backward.
        args.pop("query"), args.pop("key"), args.pop("value")
    signal = args.pop("signal")
    args["signal"] = layer._all_to_all(signal, args["packed_seq_params"], is_qkv=True)
    log_memory_usage(f"(L{layer.layer_number}) layout_mlp_to_attn:(end)")
    return args


def forward_core_attn(layer: TransformerLayer, args: Dict[str, Any]):
    # pop out to make sure the tensor is freed
    log_memory_usage(f"(L{layer.layer_number}) forward_core_attn:(start)")
    signal = args.pop("signal")
    packed_seq_params: PingPangSingleStepPackedSeqParams = args["packed_seq_params"]
    bwd_resend_qkv = packed_seq_params.bwd_packed_seq_params is not None
    
    if bwd_resend_qkv:
        log_memory_usage(f"(L{layer.layer_number}) forward_core_attn:(before FusedCommAttn)")
        signal = FusedCommAttn.apply(
            signal,
            packed_seq_params.qkv_fwd_metadata,
            packed_seq_params.qkv_bwd_metadata,
            packed_seq_params.attn_out_fwd_metadata,
            packed_seq_params.attn_out_bwd_metadata,
            packed_seq_params,
            packed_seq_params.bwd_packed_seq_params,
            packed_seq_params.dispatcher_id,
            FlashAttnArgs(
                num_heads_q=layer.config.num_attention_heads // layer.config.tensor_model_parallel_size,
                num_heads_kv=layer.config.num_query_groups // layer.config.tensor_model_parallel_size,
                head_dim=layer.config.hidden_size // layer.config.num_attention_heads,
                return_attn_probs=True,
                deterministic=True,
            ),
        )
        args["signal"] = signal
        log_memory_usage(f"(L{layer.layer_number}) forward_core_attn:(after FusedCommAttn)")
    else:
        log_memory_usage(f"(L{layer.layer_number}) forward_core_attn:(before post mlp to attn)")
        query, key, value = layer._post_mlp_to_attn(signal, args["packed_seq_params"])
        log_memory_usage(f"(L{layer.layer_number}) forward_core_attn:(after post mlp to attn)")

        log_memory_usage(f"(L{layer.layer_number}) forward_core_attn:(before forward core attn)")
        core_attn_out = layer._forward_core_attn(
            query, key, value,
            args["attention_mask"],
            args["attention_bias"],
            args["attn_mask_type"],
            args["packed_seq_params"],
        )
        log_memory_usage(f"(L{layer.layer_number}) forward_core_attn:(after forward core attn)")
        
        log_memory_usage(f"(L{layer.layer_number}) forward_core_attn:(before pre attn to mlp)")
        args["signal"] = layer._pre_attn_to_mlp(core_attn_out, args["packed_seq_params"])
        log_memory_usage(f"(L{layer.layer_number}) forward_core_attn:(after pre attn to mlp)")
    log_memory_usage(f"(L{layer.layer_number}) forward_core_attn:(end)")
    return args


def layout_attn_to_mlp(layer: TransformerLayer, args: Dict[str, Any]):
    signal = args.pop("signal")
    args["signal"] = layer._all_to_all(signal, args["packed_seq_params"], is_qkv=False)
    return args


def forward_post_core_attn(layer: TransformerLayer, args: Dict[str, Any]):
    log_memory_usage(f"(L{layer.layer_number}) forward_post_core_attn:(start)")
    signal = args.pop("signal")
    packed_seq_params: PingPangSingleStepPackedSeqParams = args["packed_seq_params"]
    bwd_resend_qkv = packed_seq_params.bwd_packed_seq_params is not None
    if bwd_resend_qkv:
        core_attn_out = post_a2a_attn_out_with_lse.apply(
            signal, args["query"], args["key"], args["value"],
            layer.config.num_attention_heads // layer.config.tensor_model_parallel_size,
            packed_seq_params.attn_out_fwd_metadata,
            packed_seq_params.attn_out_bwd_metadata,
            packed_seq_params.dispatcher_id,
        )
        args.pop("query"), args.pop("key"), args.pop("value")
    else:
        core_attn_out = layer._post_attn_to_mlp(signal, args["packed_seq_params"])
    residual = args.pop("residual")
    mlp_output, context = layer._forward_post_core_attn(
        core_attn_out,
        residual,
        args["context"],
        args["context_mask"],
    )
    args["hidden_states"] = mlp_output
    args["context"] = context
    log_memory_usage(f"(L{layer.layer_number}) forward_post_core_attn:(end)")
    return args


def tick_sync(compute_stream, comm_stream, arg_group_0, keys_0, arg_group_1, keys_1):
    log_memory_usage(f"(L?) tick_sync:(start)")
    if isinstance(keys_0, str):
        keys_0 = [keys_0]
    if isinstance(keys_1, str):
        keys_1 = [keys_1]
    tensors_0 = [arg_group_0[key] for key in keys_0]
    tensors_1 = [arg_group_1[key] for key in keys_1]
    tensors = tensors_0 + tensors_1
    out_tensors = TickSync.apply(compute_stream, comm_stream, *tensors)
    out_tensors_0 = out_tensors[:len(tensors_0)]
    out_tensors_1 = out_tensors[len(tensors_0):]
    for key, out_tensor in zip(keys_0, out_tensors_0):
        arg_group_0[key] = out_tensor
    for key, out_tensor in zip(keys_1, out_tensors_1):
        arg_group_1[key] = out_tensor
    log_memory_usage(f"(L?) tick_sync:(end)")
