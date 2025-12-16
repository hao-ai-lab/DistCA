from typing import Any, Optional, Union
from typing import Any, Iterable, Optional
import time
import torch
from torch import Tensor
import os
from megatron.core import tensor_parallel
from megatron.core.models.common.embeddings.rope_utils import (
    apply_rotary_pos_emb,
)
from d2.runtime.megatron.d2_rope import apply_rotary_pos_emb_d2, apply_rotary_pos_emb_d2_triton
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.transformer.transformer_layer import (
    TransformerLayer as MegatronTransformerLayer,
    TransformerLayerSubmodules,
)
from megatron.core import parallel_state

from d2.runtime.megatron.packed_seq_params import PingPangSingleStepPackedSeqParams, MLPLayoutPackedSeqParams
import rich


def log_memory_usage(message: str, comment: str = None):
    import d2.mem
    d2.mem.log_memory_usage(message, comment=comment)


def _log_tensor_shapes(layer: "TransformerLayer", where: str, **named_values: Any) -> None:
    """Debug helper: print shape/dtype/device for tensors (guarded by env var)."""
    # if os.getenv("D2_LOG_TENSOR_SHAPES", "0") != "1":
    #     return

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


class TransformerLayer(MegatronTransformerLayer):
    """Base transformer layer that splits the forward 3 steps: core attention, pre- and post- core attention."""
    def __init__(
        self,
        config: TransformerConfig,
        submodules: TransformerLayerSubmodules,
        layer_number: int = 1,
        hidden_dropout: Optional[float] = None,
    ):
        super().__init__(config, submodules, layer_number, hidden_dropout)

        # TODO(yonghao): this is a dev annotation for type hinting. remove it later.
        from megatron.core.transformer.attention import SelfAttention
        from megatron.core.extensions.transformer_engine import TEDotProductAttention
        self.self_attention: SelfAttention
        assert isinstance(self.self_attention.core_attention, TEDotProductAttention)
        self.pre_attn_cuda_graph = None
        self.post_attn_cuda_graph = None

    def init_pre_attn_cuda_graph(self, prev_layer: "Optional[TransformerLayer]", seq_len: int, device: torch.device, dtype: torch.dtype):
        # TODO: May need to also check `self.config.sequence_parallel_size`. If not, just assume SP == TP
        # use_sp = self.config.sequence_parallel
        tp = parallel_state.get_tensor_model_parallel_world_size()
        hidden_size_tp = self.config.hidden_size // tp
        print(f"---- init_pre_attn_cuda_graph: {self.layer_number=}, {tp=}, {self.config.hidden_size=}, {hidden_size_tp=}, {tp=}", flush=True)

        if prev_layer is None:
            # static_input = torch.zeros((seq_len, 1, self.config.hidden_size), device=device, dtype=dtype, requires_grad=True)
            static_input = torch.zeros((seq_len // tp, 1, hidden_size_tp * tp), device=device, dtype=dtype, requires_grad=True)
            print(f"---- init_pre_attn_cuda_graph: {self.layer_number=}, {static_input.shape=}", flush=True)
            self.pre_attn_cuda_graph = torch.cuda.make_graphed_callables(self.__pre_attn_cuda_graph, (static_input,))
            print(f"---- init_pre_attn_cuda_graph: {self.layer_number=}, {self.pre_attn_cuda_graph=}", flush=True)
            return
        
        # prev_layer.self_attention.query_projection_size // tp
        # static_core_attn_out = torch.zeros((seq_len * tp, 1, hidden_size_tp), device=device, dtype=dtype, requires_grad=True)
        static_core_attn_out = torch.zeros((seq_len, 1, hidden_size_tp), device=device, dtype=dtype, requires_grad=True)
        # static_residual = torch.zeros((seq_len // tp, 1, self.config.hidden_size), device=device, dtype=dtype, requires_grad=True)
        # static_residual = torch.zeros((seq_len, 1, self.config.hidden_size), device=device, dtype=dtype, requires_grad=True)
        static_residual = torch.zeros((seq_len // tp, 1, hidden_size_tp * tp), device=device, dtype=dtype, requires_grad=True)
        def post_then_pre_core_attn_cuda_graph(core_attn_out: Tensor, residual: Tensor):
            hidden_states = prev_layer._post_attn_cuda_graph(core_attn_out, residual)
            return self.__pre_attn_cuda_graph(hidden_states)
        print(f"---- init_pre_attn_cuda_graph: {self.layer_number=}, {static_core_attn_out.shape=}, {static_residual.shape=}", flush=True)
        self.pre_attn_cuda_graph = torch.cuda.make_graphed_callables(post_then_pre_core_attn_cuda_graph, (static_core_attn_out, static_residual))
        print(f"---- init_pre_attn_cuda_graph: {self.layer_number=}, {self.pre_attn_cuda_graph=}", flush=True)
        return
    
    def init_post_attn_cuda_graph(self, seq_len: int, device: torch.device, dtype: torch.dtype):
        tp = parallel_state.get_tensor_model_parallel_world_size()
        hidden_size_tp = self.config.hidden_size // tp
        # prev_layer.self_attention.query_projection_size // tp
        # static_core_attn_out = torch.zeros((seq_len * tp, 1, hidden_size_tp), device=device, dtype=dtype, requires_grad=True)
        static_core_attn_out = torch.zeros((seq_len, 1, hidden_size_tp), device=device, dtype=dtype, requires_grad=True)
        # static_residual = torch.zeros((seq_len, 1, self.config.hidden_size), device=device, dtype=dtype, requires_grad=True)
        static_residual = torch.zeros((seq_len // tp, 1, hidden_size_tp * tp), device=device, dtype=dtype, requires_grad=True)
        print(f"---- init_post_attn_cuda_graph: {self.layer_number=}, {static_core_attn_out.shape=}, {static_residual.shape=}", flush=True)
        self.post_attn_cuda_graph = torch.cuda.make_graphed_callables(self._post_attn_cuda_graph, (static_core_attn_out, static_residual))
        print(f"---- init_post_attn_cuda_graph: {self.layer_number=}, {self.post_attn_cuda_graph=}", flush=True)

    def _forward_pre_attn_cuda_graph(
        self,
        args: Iterable[torch.Tensor],
        rotary_pos_emb: Optional[Tensor] = None,
        rotary_pos_cos: Optional[Tensor] = None,
        rotary_pos_sin: Optional[Tensor] = None,
        packed_seq_params: Optional[PingPangSingleStepPackedSeqParams] = None,
        sequence_len_offset: Optional[Tensor] = None
    ):
        query, key, value, residual = self.pre_attn_cuda_graph(*args)

        log_memory_usage(f"(L{self.layer_number}) _forward_pre_core_attn:(init, before input layernorm)")

        assert rotary_pos_cos is None and rotary_pos_sin is None

        # For self attention we just duplicate the rotary_pos_emb if it isn't already
        if rotary_pos_emb is not None and not isinstance(rotary_pos_emb, tuple):
            rotary_pos_emb = (rotary_pos_emb,) * 2

        #### Some code in core_attention. This is because we don't want the pos embedding
        # being handled in the attention layout (the pos id will be hard to handle)
        inference_context = None

        log_memory_usage(f"(L{self.layer_number}) _forward_pre_core_attn:(after  qkv, before adjust_key_value_for_inference)")

        query, key, value, rotary_pos_emb, attn_mask_type = self.self_attention._adjust_key_value_for_inference(
            inference_context,
            query,
            key,
            value,
            rotary_pos_emb,
            rotary_pos_cos,
            rotary_pos_sin,
            sequence_len_offset,
        )
        if packed_seq_params is not None:
            query = query.squeeze(1)
            key = key.squeeze(1)
            value = value.squeeze(1)

        log_memory_usage(f"(L{self.layer_number}) _forward_pre_core_attn:(after adjust_key_value_for_inference, before rope)")

        # ================================================
        # relative positional embedding (rotary embedding)
        # ================================================
        if rotary_pos_emb is not None and not self.config.flash_decode:
            q_pos_emb, k_pos_emb = rotary_pos_emb

            if packed_seq_params is not None:
                if packed_seq_params.cu_seqlens_q_padded is not None:
                    cu_seqlens_q = packed_seq_params.cu_seqlens_q_padded
                else:
                    cu_seqlens_q = packed_seq_params.cu_seqlens_q
                if packed_seq_params.cu_seqlens_kv_padded is not None:
                    cu_seqlens_kv = packed_seq_params.cu_seqlens_kv_padded
                else:
                    cu_seqlens_kv = packed_seq_params.cu_seqlens_kv
            else:
                cu_seqlens_q = cu_seqlens_kv = None

            if q_pos_emb is not None:
                # TODO VIJAY: simplify
                query = apply_rotary_pos_emb(
                    query, q_pos_emb, config=self.config, cu_seqlens=cu_seqlens_q
                )
            if k_pos_emb is not None:
                key = apply_rotary_pos_emb(
                    key, k_pos_emb, config=self.config, cu_seqlens=cu_seqlens_kv
                )

            # TODO, can apply positional embedding to value_layer so it has
            # absolute positional embedding.
            # otherwise, only relative positional embedding takes effect
            # value_layer = apply_rotary_pos_emb(value_layer, k_pos_emb)
        log_memory_usage(f"(L{self.layer_number}) _forward_pre_core_attn:(after rope, before return)")
        return query, key, value, residual, attn_mask_type

    def _forward_post_attn_cuda_graph(
        self, core_attn_out: Tensor, residual: Tensor,
        context: Optional[Tensor] = None, context_mask: Optional[Tensor] = None,
    ):
        assert context is None and context_mask is None, "not supported in cudagraph"
        return self.post_attn_cuda_graph(core_attn_out, residual)

    
    def __pre_attn_cuda_graph(self, hidden_states: Tensor):
        if self.recompute_input_layernorm:
            self.input_layernorm_checkpoint = tensor_parallel.CheckpointWithoutOutput()
            input_layernorm_output = self.input_layernorm_checkpoint.checkpoint(
                self.input_layernorm, hidden_states
            )
        else:
            input_layernorm_output = self.input_layernorm(hidden_states)

        log_memory_usage(f"(L{self.layer_number}) _forward_pre_core_attn:(after input layernorm, before qkv)")

        # q, k, v
        query, key, value = self.self_attention.get_query_key_value_tensors(input_layernorm_output, None)
        return query, key, value, hidden_states

    def _post_attn_cuda_graph(self, core_attn_out: Tensor, residual: Tensor):
        attention_output_with_bias = self.self_attention.linear_proj(core_attn_out)

        log_memory_usage(f"(L{self.layer_number}) _forward_post_core_attn:(before layernorm)")
        if self.recompute_input_layernorm:
            # discard the output of the input layernorm and register the recompute
            # as a gradient hook of attention_output_with_bias[0]
            self.input_layernorm_checkpoint.discard_output_and_register_recompute(
                attention_output_with_bias[0]
            )

        log_memory_usage(f"(L{self.layer_number}) _forward_post_core_attn:(after layernorm)")

        # TODO: could we move `bias_dropout_add_exec_handler` itself
        # inside the module provided in the `bias_dropout_add_spec` module?
        with self.bias_dropout_add_exec_handler():
            hidden_states = self.self_attn_bda(self.training, self.config.bias_dropout_fusion)(
                attention_output_with_bias, residual, self.hidden_dropout
            )

        # Residual connection.
        residual = hidden_states

        # # Optional Layer norm after self-attention
        # pre_cross_attn_layernorm_output = self.pre_cross_attn_layernorm(hidden_states)

        # # Cross attention.
        # log_memory_usage(f"(L{self.layer_number}) _forward_post_core_attn:(cross attention)")
        # attention_output_with_bias = self.cross_attention(
        #     pre_cross_attn_layernorm_output,
        #     attention_mask=context_mask,
        #     key_value_states=context,
        #     inference_context=inference_context,
        # )
        # log_memory_usage(f"(L{self.layer_number}) _forward_post_core_attn:(after cross attention)")

        # if isinstance(attention_output_with_bias, dict) and "context" in attention_output_with_bias:
        #     context = attention_output_with_bias["context"]
        
        attention_output_with_bias = self.pre_cross_attn_layernorm(hidden_states)

        # TODO: could we move `bias_dropout_add_exec_handler` itself
        # inside the module provided in the `bias_dropout_add_spec` module?
        with self.bias_dropout_add_exec_handler():
            hidden_states = self.cross_attn_bda(self.training, self.config.bias_dropout_fusion)(
                attention_output_with_bias, residual, self.hidden_dropout
            )
        log_memory_usage(f"(L{self.layer_number}) _forward_post_core_attn:(after cross attn bda)")

        # Residual connection.
        residual = hidden_states

        # Optional Layer norm post the cross-attention.
        if self.recompute_pre_mlp_layernorm:
            self.pre_mlp_norm_checkpoint = tensor_parallel.CheckpointWithoutOutput()
            pre_mlp_layernorm_output = self.pre_mlp_norm_checkpoint.checkpoint(
                self.pre_mlp_layernorm, hidden_states
            )
        else:
            pre_mlp_layernorm_output = self.pre_mlp_layernorm(hidden_states)

        log_memory_usage(f"(L{self.layer_number}) _forward_post_core_attn:(after pre mlp layernorm)")

        mlp_output = self._forward_mlp(pre_mlp_layernorm_output, residual)
        log_memory_usage(f"(L{self.layer_number}) _forward_post_core_attn:(after mlp)")
        return mlp_output

    def _forward_pre_core_attn(
        self,
        hidden_states: Tensor,
        rotary_pos_emb: Optional[Tensor] = None,
        rotary_pos_cos: Optional[Tensor] = None,
        rotary_pos_sin: Optional[Tensor] = None,
        packed_seq_params: Optional[PingPangSingleStepPackedSeqParams] = None,
        sequence_len_offset: Optional[Tensor] = None,
    ):
        """
        Perform a forward pass through the attention layer and the layernorms before and after
        the attention operations.
        """
        log_memory_usage(f"(L{self.layer_number}) _forward_pre_core_attn:(init, before input layernorm)")
        _log_tensor_shapes(
            self,
            "_forward_pre_core_attn.entry",
            hidden_states=hidden_states,
            sequence_len_offset=sequence_len_offset,
            rotary_pos_emb=rotary_pos_emb,
        )

        residual = hidden_states
        # Optional Input Layer norm
        if self.recompute_input_layernorm:
            self.input_layernorm_checkpoint = tensor_parallel.CheckpointWithoutOutput()
            input_layernorm_output = self.input_layernorm_checkpoint.checkpoint(
                self.input_layernorm, hidden_states
            )
        else:
            input_layernorm_output = self.input_layernorm(hidden_states)
        _log_tensor_shapes(
            self,
            "_forward_pre_core_attn.after_input_ln",
            input_layernorm_output=input_layernorm_output,
            residual=residual,
        )

        log_memory_usage(f"(L{self.layer_number}) _forward_pre_core_attn:(after input layernorm, before qkv)")
        # Below code copied from megatron.core.transformer.attention.Attention.forward
        # rotary pos emb
        assert rotary_pos_cos is None and rotary_pos_sin is None

        # For self attention we just duplicate the rotary_pos_emb if it isn't already
        if rotary_pos_emb is not None and not isinstance(rotary_pos_emb, tuple):
            rotary_pos_emb = (rotary_pos_emb,) * 2
        # q, k, v
        log_memory_usage(f"(L{self.layer_number}) _forward_pre_core_attn:(before qkv)")
        query, key, value = self.self_attention.get_query_key_value_tensors(input_layernorm_output, None)
        _log_tensor_shapes(self, "_forward_pre_core_attn.after_qkv", query=query, key=key, value=value, rotary_pos_emb=rotary_pos_emb)

        #### Some code in core_attention. This is because we don't want the pos embedding
        # being handled in the attention layout (the pos id will be hard to handle)
        inference_context = None

        log_memory_usage(f"(L{self.layer_number}) _forward_pre_core_attn:(after  qkv, before adjust_key_value_for_inference)")

        query, key, value, rotary_pos_emb, attn_mask_type = self.self_attention._adjust_key_value_for_inference(
            inference_context,
            query,
            key,
            value,
            rotary_pos_emb,
            rotary_pos_cos,
            rotary_pos_sin,
            sequence_len_offset,
        )
        _log_tensor_shapes(
            self,
            "_forward_pre_core_attn.after_adjust_kv",
            query=query,
            key=key,
            value=value,
            rotary_pos_emb=rotary_pos_emb,
        )
        if packed_seq_params is not None:
            query = query.squeeze(1)
            key = key.squeeze(1)
            value = value.squeeze(1)
            _log_tensor_shapes(self, "_forward_pre_core_attn.after_packed_squeeze", query=query, key=key, value=value)

        log_memory_usage(f"(L{self.layer_number}) _forward_pre_core_attn:(after adjust_key_value_for_inference, before rope)")

        # ================================================
        # relative positional embedding (rotary embedding)
        # ================================================
        if rotary_pos_emb is not None and not self.config.flash_decode:
            q_pos_emb, k_pos_emb = rotary_pos_emb
            _log_tensor_shapes(self, "_forward_pre_core_attn.rope_inputs", q_pos_emb=q_pos_emb, k_pos_emb=k_pos_emb)

            if packed_seq_params is not None:
                if isinstance(packed_seq_params, MLPLayoutPackedSeqParams):
                    mlp_seq_param = packed_seq_params.mlp_layout_seq_params[0]
                    if mlp_seq_param.cu_seqlens_q_padded is not None:
                        cu_seqlens_q = mlp_seq_param.cu_seqlens_q_padded
                    else:
                        cu_seqlens_q = mlp_seq_param.cu_seqlens_q
                    if mlp_seq_param.cu_seqlens_kv_padded is not None:
                        cu_seqlens_kv = mlp_seq_param.cu_seqlens_kv_padded
                    else:
                        cu_seqlens_kv = mlp_seq_param.cu_seqlens_kv
                    shard_logical_range = packed_seq_params.shard_logical_range[0]
                    # Get max_seqlen from mlp_seq_param
                    max_seqlen_q = mlp_seq_param.max_seqlen_q
                    max_seqlen_kv = mlp_seq_param.max_seqlen_kv
                    
                    if os.getenv("D2_LOG_ROPE_METADATA", "0") == "1":
                        seqlens = cu_seqlens_q[1:] - cu_seqlens_q[:-1]
                        rich.print(f"[bold yellow]🟡 [L{self.layer_number}] RoPE Metadata:[/bold yellow]")
                        rich.print(f"  Physical Shard Lengths: {seqlens.tolist()}")
                        rich.print(f"  Logical Ranges: {shard_logical_range.tolist()}")
                else:
                    if packed_seq_params.cu_seqlens_q_padded is not None:
                        cu_seqlens_q = packed_seq_params.cu_seqlens_q_padded
                    else:
                        cu_seqlens_q = packed_seq_params.cu_seqlens_q
                    if packed_seq_params.cu_seqlens_kv_padded is not None:
                        cu_seqlens_kv = packed_seq_params.cu_seqlens_kv_padded
                    else:
                        cu_seqlens_kv = packed_seq_params.cu_seqlens_kv
                    shard_logical_range = None
            else:
                cu_seqlens_q = cu_seqlens_kv = None
                shard_logical_range = None

            if q_pos_emb is not None:
                if shard_logical_range is not None:
                    with torch.cuda.nvtx.range(f"RoPE.L{self.layer_number}.query_d2"):
                        final_indices = None
                        if isinstance(packed_seq_params, MLPLayoutPackedSeqParams):
                            final_indices = packed_seq_params.rope_final_indices
                        
                        if final_indices is not None:
                            query = apply_rotary_pos_emb_d2(
                                query, q_pos_emb, config=self.config, final_indices=final_indices, mscale=1.0
                            )
                        
                            # query = apply_rotary_pos_emb_d2_triton(
                            #     query, q_pos_emb, config=self.config, final_indices=final_indices, mscale=1.0
                            # )
                else:
                    query = apply_rotary_pos_emb(
                        query, q_pos_emb, config=self.config, cu_seqlens=cu_seqlens_q
                    )
                _log_tensor_shapes(self, "_forward_pre_core_attn.after_rope_query", query=query)

            if k_pos_emb is not None:
                if shard_logical_range is not None:
                    with torch.cuda.nvtx.range(f"RoPE.L{self.layer_number}.key_d2"):
                        final_indices = None
                        if isinstance(packed_seq_params, MLPLayoutPackedSeqParams):
                            final_indices = packed_seq_params.rope_final_indices
                        
                        if final_indices is not None:
                            key = apply_rotary_pos_emb_d2(
                                key, k_pos_emb, config=self.config, final_indices=final_indices, mscale=1.0
                            )

                            # key = apply_rotary_pos_emb_d2_triton(
                            #     key, k_pos_emb, config=self.config, final_indices=final_indices, mscale=1.0
                            # )
                else:
                    key = apply_rotary_pos_emb(
                        key, k_pos_emb, config=self.config, cu_seqlens=cu_seqlens_kv
                    )
                _log_tensor_shapes(self, "_forward_pre_core_attn.after_rope_key", key=key)

            # TODO, can apply positional embedding to value_layer so it has
            # absolute positional embedding.
            # otherwise, only relative positional embedding takes effect
            # value_layer = apply_rotary_pos_emb(value_layer, k_pos_emb)
        log_memory_usage(f"(L{self.layer_number}) _forward_pre_core_attn:(after rope, before return)")
        _log_tensor_shapes(self, "_forward_pre_core_attn.return", query=query, key=key, value=value, residual=residual)
        return query, key, value, residual, attn_mask_type

    def _forward_core_attn(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        attention_mask: Optional[Tensor] = None,
        attention_bias: Optional[Tensor] = None,
        attn_mask_type: Optional[AttnMaskType] = None,
        packed_seq_params: Optional[Union[PingPangSingleStepPackedSeqParams, MLPLayoutPackedSeqParams]] = None,
    ):
        """
        Copied from megatron.core.transformer.attention.Attention.forward
        """
        _log_tensor_shapes(
            self,
            "_forward_core_attn.entry",
            query=query,
            key=key,
            value=value,
            attention_mask=attention_mask,
            attention_bias=attention_bias,
        )

        # ==================================
        # core attention computation
        # ==================================
        should_d2_sync_time_core_attn = os.getenv("D2_SYNC_TIME_CORE_ATTN", "0") == "1"
        start_time = time.time()
        if should_d2_sync_time_core_attn:
            torch.cuda.synchronize()
            # torch.distributed.barrier()
        log_memory_usage(f"(L{self.layer_number}) _forward_core_attn:(start)")
        if self.self_attention.checkpoint_core_attention and self.training:
            log_memory_usage(f"(L{self.layer_number}) _forward_core_attn:(before checkpointed attention forward)")
            core_attn_out = self.self_attention._checkpointed_attention_forward(
                query,
                key,
                value,
                attention_mask,
                attn_mask_type=attn_mask_type,
                attention_bias=attention_bias,
                packed_seq_params=packed_seq_params,
            )
            log_memory_usage(f"(L{self.layer_number}) _forward_core_attn:(after checkpointed attention forward)")
        else:
            # Static batching attention kernel.
            # NOTE(yonghao): megatron.core.extensions.transformer_engine.TEDotProductAttention
            # core impl in te.pytorch.DotProductAttention
            # use `set_context_parallel_group` to disable context parallel
            #   cp_size = get_distributed_world_size(self.cp_group)
            log_memory_usage(f"(L{self.layer_number}) _forward_core_attn:(before core attention forward)")
            core_attn_out = self.self_attention.core_attention(
                query,
                key,
                value,
                attention_mask,
                attn_mask_type=attn_mask_type,
                attention_bias=attention_bias,
                packed_seq_params=packed_seq_params,
            )
            log_memory_usage(f"(L{self.layer_number}) _forward_core_attn:(after core attention forward)")
        _log_tensor_shapes(self, "_forward_core_attn.after_core_attention", core_attn_out=core_attn_out)
        if should_d2_sync_time_core_attn:
            torch.cuda.synchronize()
            # torch.distributed.barrier()
        end_time = time.time()
        duration = end_time - start_time
        duration_ms = duration * 1000
        
        if should_d2_sync_time_core_attn:
            print(f"🟡 TransformerLayer._forward_core_attn[{self.layer_number}] duration: {duration_ms:.3f} ms")

        if packed_seq_params is not None and packed_seq_params.qkv_format == 'thd':
            # reshape to same output shape as unpacked case
            # (t, np, hn) -> (t, b=1, h=np*hn)
            # t is the pack size = sum (sq_i)
            # note that batch is a dummy dimension in the packed case
            core_attn_out = core_attn_out.reshape(core_attn_out.size(0), 1, -1)
            _log_tensor_shapes(self, "_forward_core_attn.after_thd_reshape", core_attn_out=core_attn_out)
        log_memory_usage(f"(L{self.layer_number}) _forward_core_attn:(end)")
        return core_attn_out

    def _forward_post_core_attn(
        self,
        core_attn_out: Tensor,
        residual: Tensor,
        context: Optional[Tensor] = None,
        context_mask: Optional[Tensor] = None,
    ):
        inference_context = None
        _log_tensor_shapes(self, "_forward_post_core_attn.entry", core_attn_out=core_attn_out, residual=residual)
        attention_output_with_bias = self.self_attention.linear_proj(core_attn_out)
        _log_tensor_shapes(self, "_forward_post_core_attn.after_linear_proj", attention_output_with_bias=attention_output_with_bias)

        log_memory_usage(f"(L{self.layer_number}) _forward_post_core_attn:(before layernorm)")
        if self.recompute_input_layernorm:
            # discard the output of the input layernorm and register the recompute
            # as a gradient hook of attention_output_with_bias[0]
            self.input_layernorm_checkpoint.discard_output_and_register_recompute(
                attention_output_with_bias[0]
            )

        log_memory_usage(f"(L{self.layer_number}) _forward_post_core_attn:(after layernorm)")

        # TODO: could we move `bias_dropout_add_exec_handler` itself
        # inside the module provided in the `bias_dropout_add_spec` module?
        with self.bias_dropout_add_exec_handler():
            hidden_states = self.self_attn_bda(self.training, self.config.bias_dropout_fusion)(
                attention_output_with_bias, residual, self.hidden_dropout
            )
        _log_tensor_shapes(self, "_forward_post_core_attn.after_self_attn_bda", hidden_states=hidden_states)

        # Residual connection.
        residual = hidden_states
        _log_tensor_shapes(self, "_forward_post_core_attn.after_residual_connection", residual=residual)

        # Optional Layer norm after self-attention
        pre_cross_attn_layernorm_output = self.pre_cross_attn_layernorm(hidden_states)
        _log_tensor_shapes(self, "_forward_post_core_attn.after_pre_cross_ln", pre_cross_attn_layernorm_output=pre_cross_attn_layernorm_output)

        # Cross attention.
        log_memory_usage(f"(L{self.layer_number}) _forward_post_core_attn:(cross attention)")
        attention_output_with_bias = self.cross_attention(
            pre_cross_attn_layernorm_output,
            attention_mask=context_mask,
            key_value_states=context,
            inference_context=inference_context,
        )
        log_memory_usage(f"(L{self.layer_number}) _forward_post_core_attn:(after cross attention)")
        _log_tensor_shapes(self, "_forward_post_core_attn.after_cross_attention", attention_output_with_bias=attention_output_with_bias)

        if isinstance(attention_output_with_bias, dict) and "context" in attention_output_with_bias:
            context = attention_output_with_bias["context"]

        # TODO: could we move `bias_dropout_add_exec_handler` itself
        # inside the module provided in the `bias_dropout_add_spec` module?
        with self.bias_dropout_add_exec_handler():
            hidden_states = self.cross_attn_bda(self.training, self.config.bias_dropout_fusion)(
                attention_output_with_bias, residual, self.hidden_dropout
            )
        log_memory_usage(f"(L{self.layer_number}) _forward_post_core_attn:(after cross attn bda)")
        _log_tensor_shapes(self, "_forward_post_core_attn.after_cross_attn_bda", hidden_states=hidden_states)

        # Residual connection.
        residual = hidden_states

        # Optional Layer norm post the cross-attention.
        if self.recompute_pre_mlp_layernorm:
            self.pre_mlp_norm_checkpoint = tensor_parallel.CheckpointWithoutOutput()
            pre_mlp_layernorm_output = self.pre_mlp_norm_checkpoint.checkpoint(
                self.pre_mlp_layernorm, hidden_states
            )
        else:
            pre_mlp_layernorm_output = self.pre_mlp_layernorm(hidden_states)
        _log_tensor_shapes(self, "_forward_post_core_attn.after_pre_mlp_ln", pre_mlp_layernorm_output=pre_mlp_layernorm_output)

        log_memory_usage(f"(L{self.layer_number}) _forward_post_core_attn:(after pre mlp layernorm)")

        mlp_output = self._forward_mlp(pre_mlp_layernorm_output, residual)
        log_memory_usage(f"(L{self.layer_number}) _forward_post_core_attn:(after mlp)")
        _log_tensor_shapes(self, "_forward_post_core_attn.return", mlp_output=mlp_output, context=context)
        return mlp_output, context

    ######## Debug ########
    def forward_orig_impl(
        self,
        hidden_states: Tensor,
        attention_mask: Optional[Tensor] = None,
        context: Optional[Tensor] = None,
        context_mask: Optional[Tensor] = None,
        rotary_pos_emb: Optional[Tensor] = None,
        rotary_pos_cos: Optional[Tensor] = None,
        rotary_pos_sin: Optional[Tensor] = None,
        attention_bias: Optional[Tensor] = None,
        inference_context: Optional[Any] = None,
        packed_seq_params: Optional[PingPangSingleStepPackedSeqParams] = None,
        sequence_len_offset: Optional[Tensor] = None,
        *,
        inference_params: Optional[Any] = None,
        return_debug: bool = False,
    ):
        # import traceback
        # traceback.print_stack()
        # print(packed_seq_params)
        # exit(0)
        """Debug use. normal forward with output hooked."""
        
        layer_number = self.layer_number
        with torch.cuda.nvtx.range(f"forward[L={layer_number}]"):

            assert inference_params is None, "inference not supported yet"
            assert inference_context is None, "inference not supported yet"
            assert context is None, "cross-attention not supported yet"
            assert context_mask is None, "cross-attention not supported yet"

            setattr(packed_seq_params, "stream", torch.cuda.current_stream())
            # Enable RoPE.
            # rotary_pos_emb = None

            log_memory_usage(f"(L{self.layer_number}) _forward_orig_impl:(before pre core attn)")

            query, key, value, residual, attn_mask_type = self._forward_pre_core_attn(
                hidden_states,
                rotary_pos_emb,
                rotary_pos_cos,
                rotary_pos_sin,
                packed_seq_params,
                sequence_len_offset,
            )
            debug_tensors = [(query, key, value),]

            log_memory_usage(f"(L{self.layer_number}) _forward_orig_impl:(after pre core attn)")

            core_attn_out = self._forward_core_attn(
                query,
                key,
                value,
                attention_mask,
                attention_bias,
                attn_mask_type,
                packed_seq_params,
            )

            log_memory_usage(f"(L{self.layer_number}) _forward_orig_impl:(after core attention)")
            debug_tensors.append(core_attn_out)
            mlp_output, context = self._forward_post_core_attn(
                core_attn_out,
                residual,
                context,
                context_mask,
            )

            log_memory_usage(f"(L{self.layer_number}) _forward_orig_impl:(after post core attn)")

            return (mlp_output, context,) + (
                (debug_tensors,) if return_debug else ()
            )

