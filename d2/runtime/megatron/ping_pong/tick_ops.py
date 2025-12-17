from typing import Any, Dict, Optional
import os
import torch

from d2.runtime.megatron.ops import FusedCommAttn, post_a2a_attn_out_with_lse
from d2.runtime.megatron.ops.stream_sync_fn import tick_sync_with_info
from d2.runtime.megatron.ops.fused_comm_attn import FlashAttnArgs
from d2.runtime.megatron.packed_seq_params import PingPangSingleStepPackedSeqParams
from d2.runtime.megatron.ping_pong.transformer_layer import TransformerLayer


def log_memory_usage(message: str):
    import d2.mem
    d2.mem.log_memory_usage(message)
    return


def _log_tensor_shapes(layer: TransformerLayer, where: str, **named_values: Any) -> None:
    """Debug helper: print shape/dtype/device for tensors (guarded by env var)."""
    if os.getenv("D2_LOG_TENSOR_SHAPES", "0") != "1":
        return

    def _emit(name: str, v: Any):
        if v is None:
            return
        if torch.is_tensor(v):
            print(
                f"[L{layer.layer_number}] {where}: {name} "
                f"shape={tuple(v.shape)} dtype={v.dtype} device={v.device} req_grad={getattr(v, 'requires_grad', False)}"
            )
            return
        if isinstance(v, (tuple, list)):
            for i, vv in enumerate(v):
                _emit(f"{name}[{i}]", vv)
            return
        if isinstance(v, dict):
            for kk, vv in v.items():
                _emit(f"{name}.{kk}", vv)
            return

    for k, v in named_values.items():
        _emit(k, v)


def forward_pre_core_attn(layer: TransformerLayer, args: Dict[str, Any]):
    log_memory_usage(f"(L{layer.layer_number}) forward_pre_core_attn:(start)")
    
    # Setup timing if enabled
    start_event = None
    end_event = None
    if os.getenv("UNIFIED_RECORD_TICK_TIMES", "0") == "1" and _current_tick_sample_id is not None:
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()
    
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

    args["query"] = query
    args["key"] = key
    args["value"] = value
    args["residual"] = residual
    args["attn_mask_type"] = attn_mask_type
    forward_pre_core_attn_comm(layer, args)
    
    
    # Record timing if enabled
    if start_event is not None and end_event is not None:
        end_event.record()
        _record_tick_timing("forward_pre_core_attn", layer.layer_number, start_event, end_event)
    
    log_memory_usage(f"(L{layer.layer_number}) forward_pre_core_attn:(return)")
    return args


def forward_pre_core_attn_comm(layer: TransformerLayer, args: Dict[str, Any]):
    log_memory_usage(f"(L{layer.layer_number}) forward_pre_core_attn:(before pre mlp to attn)")
    signal = layer._pre_mlp_to_attn(
        args["query"], args["key"], args["value"], args["packed_seq_params"],
    )
    log_memory_usage(f"(L{layer.layer_number}) forward_pre_core_attn:(after pre mlp to attn)")
    args["signal"] = signal
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


def forward_post_core_attn_comm(layer: TransformerLayer, args: Dict[str, Any]):
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
    args["core_attn_out"] = core_attn_out
    return args


def forward_post_core_attn(layer: TransformerLayer, args: Dict[str, Any]):
    log_memory_usage(f"(L{layer.layer_number}) forward_post_core_attn:(start)")

    # Setup timing if enabled
    start_event = None
    end_event = None
    if os.getenv("UNIFIED_RECORD_TICK_TIMES", "0") == "1" and _current_tick_sample_id is not None:
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()

    forward_post_core_attn_comm(layer, args)
    mlp_output, context = layer._forward_post_core_attn(
        args.pop("core_attn_out"),
        args.pop("residual"),
        args["context"],
        args["context_mask"],
    )
    args["hidden_states"] = mlp_output
    args["context"] = context
    
    
    # Record timing if enabled
    if start_event is not None and end_event is not None:
        end_event.record()
        _record_tick_timing("forward_post_core_attn", layer.layer_number, start_event, end_event)
    
    log_memory_usage(f"(L{layer.layer_number}) forward_post_core_attn:(end)")
    return args


def forward_post_then_pre_core_attn_cuda_graph(layer: TransformerLayer, args: Dict[str, Any]):
    log_memory_usage(f"(L{layer.layer_number}) forward_post_then_pre_core_attn:(start)")
    assert args["context"] is None and args["context_mask"] is None, "not supported in cudagraph"
    _log_tensor_shapes(layer, "post_then_pre_cuda_graph.entry", **args)
    forward_post_core_attn_comm(layer, args)
    _log_tensor_shapes(layer, "post_then_pre_cuda_graph.after_post_core_attn_comm", **args)
    log_memory_usage(f"(L{layer.layer_number}) forward_post_then_pre_core_attn:(before non core attn)")
    _log_tensor_shapes(
        layer,
        "post_then_pre_cuda_graph.pre__forward_pre_attn_cuda_graph",
        core_attn_out=args.get("core_attn_out", None),
        residual=args.get("residual", None),
        rotary_pos_emb=args.get("rotary_pos_emb", None),
        rotary_pos_cos=args.get("rotary_pos_cos", None),
        rotary_pos_sin=args.get("rotary_pos_sin", None),
        sequence_len_offset=args.get("sequence_len_offset", None),
    )
    query, key, value, residual, attn_mask_type = layer._forward_pre_attn_cuda_graph(
        (args.pop("core_attn_out"), args.pop("residual")),
        args["rotary_pos_emb"],
        args["rotary_pos_cos"],
        args["rotary_pos_sin"],
        args["mlp_packed_seq_params"],
        args["sequence_len_offset"],
    )
    log_memory_usage(f"(L{layer.layer_number}) forward_post_then_pre_core_attn:(after non core attn)")
    _log_tensor_shapes(
        layer,
        "post_then_pre_cuda_graph.post__forward_pre_attn_cuda_graph",
        query=query,
        key=key,
        value=value,
        residual=residual,
    )
    args["query"] = query
    args["key"] = key
    args["value"] = value
    args["residual"] = residual
    args["attn_mask_type"] = attn_mask_type
    forward_pre_core_attn_comm(layer, args)
    log_memory_usage(f"(L{layer.layer_number}) forward_pre_then_pre_core_attn:(return)")
    return args


def forward_pre_core_attn_cuda_graph(layer: TransformerLayer, args: Dict[str, Any]):
    log_memory_usage(f"(L{layer.layer_number}) forward_pre_core_attn:(start)")
    log_memory_usage(f"(L{layer.layer_number}) forward_pre_core_attn:(before pre core attn)")
    _log_tensor_shapes(layer, "pre_cuda_graph.entry", **args)
    hidden_states = args.pop("hidden_states")
    _log_tensor_shapes(
        layer,
        "pre_cuda_graph.pre__forward_pre_attn_cuda_graph",
        hidden_states=hidden_states,
        rotary_pos_emb=args.get("rotary_pos_emb", None),
        rotary_pos_cos=args.get("rotary_pos_cos", None),
        rotary_pos_sin=args.get("rotary_pos_sin", None),
        sequence_len_offset=args.get("sequence_len_offset", None),
    )
    query, key, value, residual, attn_mask_type = layer._forward_pre_attn_cuda_graph(
        (hidden_states,),
        args["rotary_pos_emb"],
        args["rotary_pos_cos"],
        args["rotary_pos_sin"],
        args["mlp_packed_seq_params"],
        args["sequence_len_offset"],
    )
    log_memory_usage(f"(L{layer.layer_number}) forward_pre_core_attn:(after pre core attn)")
    _log_tensor_shapes(
        layer,
        "pre_cuda_graph.post__forward_pre_attn_cuda_graph",
        query=query,
        key=key,
        value=value,
        residual=residual,
    )

    args["query"] = query
    args["key"] = key
    args["value"] = value
    args["residual"] = residual
    args["attn_mask_type"] = attn_mask_type
    forward_pre_core_attn_comm(layer, args)
    log_memory_usage(f"(L{layer.layer_number}) forward_pre_core_attn:(return)")
    return args


def forward_post_core_attn_cuda_graph(layer: TransformerLayer, args: Dict[str, Any]):
    _log_tensor_shapes(layer, "post_cuda_graph.entry", **args)
    forward_post_core_attn_comm(layer, args)
    _log_tensor_shapes(layer, "post_cuda_graph.after_post_core_attn_comm", **args)
    mlp_output = layer._forward_post_attn_cuda_graph(
        args.pop("core_attn_out"),
        args.pop("residual"),
        args["context"],
        args["context_mask"],
    )
    _log_tensor_shapes(layer, "post_cuda_graph.post__forward_post_attn_cuda_graph", hidden_states=mlp_output)
    args["hidden_states"] = mlp_output
    log_memory_usage(f"(L{layer.layer_number}) forward_post_core_attn:(end)")
    return args


def tick_sync(compute_stream, comm_stream, arg_group_0, keys_0, arg_group_1, keys_1, 
              layer_info="unknown", operation_info="unknown"):
    log_memory_usage(f"(L?) tick_sync:(start)")
    if isinstance(keys_0, str):
        keys_0 = [keys_0]
    if isinstance(keys_1, str):
        keys_1 = [keys_1]
    tensors_0 = [arg_group_0[key] for key in keys_0]
    tensors_1 = [arg_group_1[key] for key in keys_1]
    tensors = tensors_0 + tensors_1
    out_tensors = tick_sync_with_info(
        compute_stream, comm_stream,
        layer_info, operation_info,
        *tensors
    )
    out_tensors_0 = out_tensors[:len(tensors_0)]
    out_tensors_1 = out_tensors[len(tensors_0):]
    for key, out_tensor in zip(keys_0, out_tensors_0):
        arg_group_0[key] = out_tensor
    for key, out_tensor in zip(keys_1, out_tensors_1):
        arg_group_1[key] = out_tensor
    log_memory_usage(f"(L?) tick_sync:(end)")


def tick_nonca_compute(
    layer: TransformerLayer, prev_layer: Optional[TransformerLayer],
    arg_group: Dict[str, Any],  is_last_layer_post_attn: bool
):
    """
    Previous layer(if exists) post core attention + this layer's pre core attention.
    This is the maximum size for a cuda graph tracing, if not consider MoE
    """
    # Only run this layer's post-attention
    if is_last_layer_post_attn:
        arg_group = forward_post_core_attn(layer, arg_group)
        return arg_group
    # Previous layer's (if exists) post core attn and this layer's pre core attn
    if prev_layer is not None:
        arg_group = forward_post_core_attn(prev_layer, arg_group)
    arg_group = forward_pre_core_attn(layer, arg_group)
    return arg_group

def tick_nonca_compute_cuda_graph(
    layer: TransformerLayer, prev_layer: Optional[TransformerLayer],
    arg_group: Dict[str, Any], is_last_layer_post_attn: bool
):
    # print(f"-- use cudagraph --: {layer.layer_number}")
    if is_last_layer_post_attn:
        return forward_post_core_attn_cuda_graph(layer, arg_group)
    if prev_layer is None:
        # return forward_pre_core_attn(layer, arg_group)
        return forward_pre_core_attn_cuda_graph(layer, arg_group)
    return forward_post_then_pre_core_attn_cuda_graph(layer, arg_group)

# ========== Tick Operations Timing Collection ==========
# TODO: (Refactor) Move this to somewhere else for sole measurement purpose only.

# Global variables for timing collection
_tick_times = {}
_current_tick_sample_id = None
_pending_tick_events = []  # List of (sample_id, operation, layer_number, start_event, end_event) tuples


def setup_tick_timing():
    """Setup timing collection for tick operations."""
    if os.getenv("UNIFIED_RECORD_TICK_TIMES", "0") == "1":
        print("🟢 Tick operations timing collection enabled")
        return True
    return False


def set_current_tick_sample_id(sample_id):
    """Set the current sample ID for tick timing."""
    global _current_tick_sample_id
    _current_tick_sample_id = sample_id


def _record_tick_timing(operation: str, layer_number: int, start_event: torch.cuda.Event, end_event: torch.cuda.Event):
    """Record timing for a tick operation."""
    if _current_tick_sample_id is not None:
        _pending_tick_events.append((_current_tick_sample_id, operation, layer_number, start_event, end_event))






def sync_and_collect_tick_timing():
    """Synchronize all pending tick events and collect timing data."""
    global _pending_tick_events, _tick_times
    
    if not _pending_tick_events:
        return {}
    
    # Synchronize all events
    torch.cuda.synchronize()
    
    # Process all pending events
    for sample_id, operation, layer_number, start_event, end_event in _pending_tick_events:
        # Calculate duration in milliseconds
        duration_ms = start_event.elapsed_time(end_event)
        
        # Initialize sample data if needed
        if sample_id not in _tick_times:
            _tick_times[sample_id] = {
                "forward_pre_core_attn": [], 
                "forward_post_core_attn": []
            }
        
        # Store timing data
        if operation in _tick_times[sample_id]:
            _tick_times[sample_id][operation].append({
                "layer_number": layer_number,
                "duration_ms": duration_ms
            })
    
    # Clear pending events
    _pending_tick_events.clear()
    return _tick_times.copy()


def get_tick_times():
    """Get all tick timing data."""
    return _tick_times.copy()


def clear_tick_times():
    """Clear tick timing data."""
    global _tick_times, _pending_tick_events
    _tick_times.clear()
    _pending_tick_events.clear()
