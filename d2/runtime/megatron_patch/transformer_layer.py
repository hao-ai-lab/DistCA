from typing import Any, List, Optional

import torch
import torch.distributed
from torch import Tensor

from megatron.core import tensor_parallel
from megatron.core.models.common.embeddings.rope_utils import (
    apply_rotary_pos_emb,
)
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.transformer.transformer_layer import (
    TransformerLayer as MegatronTransformerLayer,
    TransformerLayerSubmodules,
)

from d2.runtime.megatron_patch.packed_seq_params import PingPangPackedSeqParams, PingPangSingleStepPackedSeqParams
from d2.runtime.megatron_patch.dispatcher_wrapper import n_to_n_dispatch

#### Tool functions for splitting and gathering args ####
def _split_tensor(x: Optional[torch.Tensor], num_splits: int):
    if x is None:
        return (None,) * num_splits
    return x.split(x.shape[0] // num_splits, dim=0)

def _repack_args(args: List[List[torch.Tensor]], num_splits: int):
    assert all(len(a) == num_splits for a in args)
    return [
        [a[i] for a in args]
        for i in range(num_splits)
    ]

def _splits_all(tensors: List[torch.Tensor], num_splits: int):
    splits = [_split_tensor(t, num_splits) for t in tensors]
    return _repack_args(splits, num_splits)

def _gather_tensor(tensors: List[torch.Tensor], num_splits: int):
    assert len(tensors) == num_splits
    if any(t is None for t in tensors):
        assert all(t is None for t in tensors), "None tensors in gather_tensor"
        return None
    return torch.cat(tensors, dim=0)
####


class TransformerLayer(MegatronTransformerLayer):

    def __init__(
        self,
        config: TransformerConfig,
        submodules: TransformerLayerSubmodules,
        layer_number: int = 1,
        hidden_dropout: Optional[float] = None,
    ):
        super().__init__(config, submodules, layer_number, hidden_dropout)
        self.comm_event = torch.cuda.Event()

        # TODO(yonghao): this is a dev annotation for type hinting. remove it later.
        from megatron.core.transformer.attention import SelfAttention
        from megatron.core.extensions.transformer_engine import TEDotProductAttention
        self.self_attention: SelfAttention
        assert isinstance(self.self_attention.core_attention, TEDotProductAttention)

    def _layout_attn_to_mlp(
            self, attn_out: torch.Tensor,
            packed_seq_params: PingPangPackedSeqParams
        ):
        #### NOTE: this is a hack. We should fix this in the future.
        num_heads = attn_out.shape[1]
        head_dim = attn_out.shape[2]
        attn_out = attn_out.reshape(attn_out.shape[0], num_heads * head_dim)
        ####

        attn_out_mlp_layout, _ = n_to_n_dispatch.apply(
            attn_out, # query_in
            packed_seq_params.attn_to_mlp_metadata, # query_metadata
            packed_seq_params.mlp_to_attn_metadata, # rev_query_metadata
            None, # key_value_in
            None, # key_value_metadata
            None, # rev_key_value_metadata
            packed_seq_params.stream, # stream
            self.comm_event, # event
        )
        #### switch layout back
        attn_out_mlp_layout = attn_out_mlp_layout.reshape(-1, num_heads, head_dim)
        ####
        return attn_out_mlp_layout

    def _layout_mlp_to_attn(
        self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
        packed_seq_params: PingPangPackedSeqParams
    ):
        # TODO(yonghao): current we do a walk around: merge head dimension
        # into hidden dimension. In this way, we need the TP degree being
        # kept for attention and other parts.
        assert query.dim() == 3, f"{query.shape=}, should be tnh layout"
        num_q_heads = query.shape[1]
        num_kv_heads = key.shape[1]
        q_head_dim = query.shape[2]
        kv_head_dim = key.shape[2]
        query = query.reshape(query.shape[0], num_q_heads * q_head_dim)
        key = key.reshape(key.shape[0], num_kv_heads * kv_head_dim)
        value = value.reshape(value.shape[0], num_kv_heads * kv_head_dim)
        ####

        key_value = torch.cat([key, value], dim=-1).contiguous()
        query_attn_layout, key_value_attn_layout = n_to_n_dispatch.apply(
            query,  # query_in
            packed_seq_params.mlp_to_attn_metadata, # query_metadata
            packed_seq_params.attn_to_mlp_metadata, # rev_query_metadata
            key_value, # key_value_in
            packed_seq_params.mlp_to_attn_kv_metadata, # key_value_metadata
            packed_seq_params.mlp_to_attn_kv_grad_metadata, # rev_key_value_metadata
            packed_seq_params.stream, # stream
            self.comm_event, # event
        )
        key_attn_layout, value_attn_layout = key_value_attn_layout.split(key_value_attn_layout.shape[1] // 2, dim=1)

        #### switch layout back
        query_attn_layout = query_attn_layout.reshape(-1, num_q_heads, q_head_dim)
        key_attn_layout = key_attn_layout.reshape(-1, num_kv_heads, kv_head_dim)
        value_attn_layout = value_attn_layout.reshape(-1, num_kv_heads, kv_head_dim)

        return query_attn_layout, key_attn_layout, value_attn_layout

    def ping_pang_forward(
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
        packed_seq_params: Optional[PingPangPackedSeqParams] = None,
        sequence_len_offset: Optional[Tensor] = None,
        *,
        inference_params: Optional[Any] = None,
    ):
        """
        In-place Ping-Pang parallel
        """
        assert inference_params is None, "inference not supported yet"
        assert inference_context is None, "inference not supported yet"
        assert context is None, "cross-attention not supported yet"
        assert context_mask is None, "cross-attention not supported yet"

        debug = packed_seq_params.debug
        # NOTE: transformer_block.py sets layer_number to 1 for the first layer
        needs_split = debug or self.layer_number == 1
        needs_gather = debug    # NOTE: cannot infer if this is the last local layer or not.

        packed_seq_params_0 = packed_seq_params.seq_params[0]
        packed_seq_params_1 = packed_seq_params.seq_params[1]
        if debug:
            setattr(packed_seq_params_0, "stream", torch.cuda.current_stream())
            setattr(packed_seq_params_1, "stream", torch.cuda.current_stream())

        # 1. split input into two microbatches
        args = [hidden_states, attention_mask, context, context_mask, rotary_pos_emb,
                rotary_pos_cos, rotary_pos_sin, attention_bias, sequence_len_offset]
        if needs_split:
            args_0, args_1 = _splits_all(args, 2)
        else:
            args_0, args_1 = _repack_args(args, 2)
        (hidden_states_0, attention_mask_0, context_0, context_mask_0, rotary_pos_emb_0,
            rotary_pos_cos_0, rotary_pos_sin_0, attention_bias_0, sequence_len_offset_0) = args_0
        (hidden_states_1, attention_mask_1, context_1, context_mask_1, rotary_pos_emb_1,
            rotary_pos_cos_1, rotary_pos_sin_1, attention_bias_1, sequence_len_offset_1) = args_1

        # 2. pre-self-attention forward microbatch 0.
        # Ideally, this part should merge to the previous layer's post-self-attention to maximize
        # the communication-computation overlap.
        query_0, key_0, value_0, residual_0, attn_mask_type_0 = self._forward_pre_core_attn(
            hidden_states_0,
            rotary_pos_emb_0,
            rotary_pos_cos_0,
            rotary_pos_sin_0,
            packed_seq_params_0,
            sequence_len_offset_0,
        )

        # 3. pre-attention forward of microbatch 1, mlp2attn all2all of microbatch 0
        self.comm_event.wait(torch.cuda.current_stream())
        query_0, key_0, value_0 = self._layout_mlp_to_attn(query_0, key_0, value_0, packed_seq_params_0)
        query_1, key_1, value_1, residual_1, attn_mask_type_1 = self._forward_pre_core_attn(
            hidden_states_1,
            rotary_pos_emb_1,
            rotary_pos_cos_1,
            rotary_pos_sin_1,
            packed_seq_params_1,
            sequence_len_offset_1,
        )

        # 4. self-attention forward of microbatch 0, mlp2attn all2all of microbatch 1
        self.comm_event.wait(torch.cuda.current_stream())
        query_1, key_1, value_1 = self._layout_mlp_to_attn(query_1, key_1, value_1, packed_seq_params_1)
        core_attn_out_0 = self._forward_core_attn(
            query_0,
            key_0,
            value_0,
            attention_mask_0,
            attention_bias_0,
            attn_mask_type_0,
            packed_seq_params_0,
        )

        # 5. post-self-attention forward of microbatch 0, mlp2attn all2all of microbatch 1
        self.comm_event.wait(torch.cuda.current_stream())
        core_attn_out_0 = self._layout_attn_to_mlp(core_attn_out_0, packed_seq_params_0)
        core_attn_out_1 = self._forward_core_attn(
            query_1,
            key_1,
            value_1,
            attention_mask_1,
            attention_bias_1,
            attn_mask_type_1,
            packed_seq_params_1,
        )

        # 6. mlp forward of microbatch 0, mlp2attn all2all of microbatch 1
        self.comm_event.wait(torch.cuda.current_stream())
        core_attn_out_1 = self._layout_attn_to_mlp(core_attn_out_1, packed_seq_params_1)
        mlp_output_0, context_0 = self._forward_post_core_attn(
            core_attn_out_0,
            residual_0,
            context_0,
            context_mask_0,
        )

        self.comm_event.wait(torch.cuda.current_stream())
        mlp_output_1, context_1 = self._forward_post_core_attn(
            core_attn_out_1,
            residual_1,
            context_1,
            context_mask_1,
        )
        # concatenate the two microbatches to one.
        if needs_gather:
            output = _gather_tensor([mlp_output_0, mlp_output_1], num_splits=2)
            context = _gather_tensor([context_0, context_1], num_splits=2)
        else:
            output = [mlp_output_0, mlp_output_1]
            context = [context_0, context_1]
        return output, context

    def _forward_pre_core_attn(
        self,
        hidden_states: Tensor,
        rotary_pos_emb: Optional[Tensor] = None,
        rotary_pos_cos: Optional[Tensor] = None,
        rotary_pos_sin: Optional[Tensor] = None,
        packed_seq_params: Optional[PingPangPackedSeqParams] = None,
        sequence_len_offset: Optional[Tensor] = None,
    ):
        """
        Perform a forward pass through the attention layer and the layernorms before and after
        the attention operations.
        """
        residual = hidden_states
        # Optional Input Layer norm
        if self.recompute_input_layernorm:
            self.input_layernorm_checkpoint = tensor_parallel.CheckpointWithoutOutput()
            input_layernorm_output = self.input_layernorm_checkpoint.checkpoint(
                self.input_layernorm, hidden_states
            )
        else:
            input_layernorm_output = self.input_layernorm(hidden_states)
        # Below code copied from megatron.core.transformer.attention.Attention.forward
        # rotary pos emb
        assert rotary_pos_cos is None and rotary_pos_sin is None

        # For self attention we just duplicate the rotary_pos_emb if it isn't already
        if rotary_pos_emb is not None and not isinstance(rotary_pos_emb, tuple):
            rotary_pos_emb = (rotary_pos_emb,) * 2
        # q, k, v
        query, key, value = self.self_attention.get_query_key_value_tensors(input_layernorm_output, None)

        #### Some code in core_attention. This is because we don't want the pos embedding
        # being handled in the attention layout (the pos id will be hard to handle)
        inference_context = None

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
        return query, key, value, residual, attn_mask_type

    def _forward_core_attn(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        attention_mask: Optional[Tensor] = None,
        attention_bias: Optional[Tensor] = None,
        attn_mask_type: Optional[AttnMaskType] = None,
        packed_seq_params: Optional[PingPangPackedSeqParams] = None,
    ):
        """
        Copied from megatron.core.transformer.attention.Attention.forward
        """

        # ==================================
        # core attention computation
        # ==================================

        # TODO(yonghao): this core attention still has the CP communication.
        # Need to remove it since we can merge it with our all2all
        if self.self_attention.checkpoint_core_attention and self.training:
            core_attn_out = self.self_attention._checkpointed_attention_forward(
                query,
                key,
                value,
                attention_mask,
                attn_mask_type=attn_mask_type,
                attention_bias=attention_bias,
                packed_seq_params=packed_seq_params,
            )
        else:
            # Static batching attention kernel.
            # NOTE(yonghao): megatron.core.extensions.transformer_engine.TEDotProductAttention
            # core impl in te.pytorch.DotProductAttention
            # use `set_context_parallel_group` to disable context parallel
            #   cp_size = get_distributed_world_size(self.cp_group)
            core_attn_out = self.self_attention.core_attention(
                query,
                key,
                value,
                attention_mask,
                attn_mask_type=attn_mask_type,
                attention_bias=attention_bias,
                packed_seq_params=packed_seq_params,
            )

        if packed_seq_params is not None and packed_seq_params.qkv_format == 'thd':
            # reshape to same output shape as unpacked case
            # (t, np, hn) -> (t, b=1, h=np*hn)
            # t is the pack size = sum (sq_i)
            # note that batch is a dummy dimension in the packed case
            core_attn_out = core_attn_out.reshape(core_attn_out.size(0), 1, -1)
        return core_attn_out

    def _forward_post_core_attn(
        self,
        core_attn_out: Tensor,
        residual: Tensor,
        context: Optional[Tensor] = None,
        context_mask: Optional[Tensor] = None,
    ):
        inference_context = None
        attention_output_with_bias = self.self_attention.linear_proj(core_attn_out)
        if self.recompute_input_layernorm:
            # discard the output of the input layernorm and register the recompute
            # as a gradient hook of attention_output_with_bias[0]
            self.input_layernorm_checkpoint.discard_output_and_register_recompute(
                attention_output_with_bias[0]
            )

        # TODO: could we move `bias_dropout_add_exec_handler` itself
        # inside the module provided in the `bias_dropout_add_spec` module?
        with self.bias_dropout_add_exec_handler():
            hidden_states = self.self_attn_bda(self.training, self.config.bias_dropout_fusion)(
                attention_output_with_bias, residual, self.hidden_dropout
            )

        # Residual connection.
        residual = hidden_states

        # Optional Layer norm after self-attention
        pre_cross_attn_layernorm_output = self.pre_cross_attn_layernorm(hidden_states)

        # Cross attention.
        attention_output_with_bias = self.cross_attention(
            pre_cross_attn_layernorm_output,
            attention_mask=context_mask,
            key_value_states=context,
            inference_context=inference_context,
        )

        if isinstance(attention_output_with_bias, dict) and "context" in attention_output_with_bias:
            context = attention_output_with_bias["context"]

        # TODO: could we move `bias_dropout_add_exec_handler` itself
        # inside the module provided in the `bias_dropout_add_spec` module?
        with self.bias_dropout_add_exec_handler():
            hidden_states = self.cross_attn_bda(self.training, self.config.bias_dropout_fusion)(
                attention_output_with_bias, residual, self.hidden_dropout
            )

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

        mlp_output = self._forward_mlp(pre_mlp_layernorm_output, residual)
        return mlp_output, context

    #### Debug use.
    def forward_no_switch(
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
    ):
        """Debug use. normal forward with output hooked."""
        assert inference_params is None, "inference not supported yet"
        assert inference_context is None, "inference not supported yet"
        assert context is None, "cross-attention not supported yet"
        assert context_mask is None, "cross-attention not supported yet"

        setattr(packed_seq_params, "stream", torch.cuda.current_stream())

        query, key, value, residual, attn_mask_type = self._forward_pre_core_attn(
            hidden_states,
            rotary_pos_emb,
            rotary_pos_cos,
            rotary_pos_sin,
            packed_seq_params,
            sequence_len_offset,
        )
        debug_tensors = [(query, key, value),]

        core_attn_out = self._forward_core_attn(
            query,
            key,
            value,
            attention_mask,
            attention_bias,
            attn_mask_type,
            packed_seq_params,
        )
        debug_tensors.append(core_attn_out)
        mlp_output, context = self._forward_post_core_attn(
            core_attn_out,
            residual,
            context,
            context_mask,
        )

        return mlp_output, context, debug_tensors

    def forward_one_stage(
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
    ):
        """Debug use. Single stage of Ping-Pang parallel."""
        assert inference_params is None, "inference not supported yet"
        assert inference_context is None, "inference not supported yet"
        assert context is None, "cross-attention not supported yet"
        assert context_mask is None, "cross-attention not supported yet"

        setattr(packed_seq_params, "stream", torch.cuda.current_stream())

        query, key, value, residual, attn_mask_type = self._forward_pre_core_attn(
            hidden_states,
            rotary_pos_emb,
            rotary_pos_cos,
            rotary_pos_sin,
            packed_seq_params,
            sequence_len_offset,
        )
        debug_tensors = [(query, key, value),]

        query, key, value = self._layout_mlp_to_attn(query, key, value, packed_seq_params)
        debug_tensors.append((query, key, value))

        core_attn_out = self._forward_core_attn(
            query,
            key,
            value,
            attention_mask,
            attention_bias,
            attn_mask_type,
            packed_seq_params,
        )
        debug_tensors.append(core_attn_out)

        core_attn_out = self._layout_attn_to_mlp(core_attn_out, packed_seq_params)
        debug_tensors.append(core_attn_out)

        mlp_output, context = self._forward_post_core_attn(
            core_attn_out,
            residual,
            context,
            context_mask,
        )

        return mlp_output, context, debug_tensors
