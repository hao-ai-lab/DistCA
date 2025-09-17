import os
from typing import Any, Optional
import warnings

import torch
from torch import Tensor

from d2.runtime.megatron.base_transformer_layer import TransformerLayer as BaseTransformerLayer
from d2.runtime.megatron.packed_seq_params import PingPangPackedSeqParams, PingPangSingleStepPackedSeqParams
from d2.runtime.megatron.ops import TickSync, FusedCommAttn, post_a2a_attn_out_with_lse
from d2.runtime.megatron.ops.fused_comm_attn import FlashAttnArgs
from d2.runtime.megatron.ping_pong.utils import splits_all, repack_args, gather_tensor
from d2.runtime.dispatch_fn import (
    all_to_all, post_all2all_layout_transfer, pre_all2all_layout_transfer
)


class TransformerLayer(BaseTransformerLayer):
    ########## Attention Layout <-> MLP Layout Transformation ##########
    def _pre_attn_to_mlp(
        self, attn_out: torch.Tensor,
        packed_seq_params: PingPangSingleStepPackedSeqParams,
    ):
        """
        Communication between attention layout and mlp layout.
        This operation runs on the stream `packed_seq_params.stream`.
        """
        torch.cuda.nvtx.range_push("pre_attn_to_mlp")
        num_heads = self.config.num_attention_heads
        head_dim = self.config.hidden_size // self.config.num_attention_heads
        num_heads_local = num_heads // self.config.tensor_model_parallel_size
        attn_out = attn_out.view(attn_out.shape[0], num_heads_local * head_dim)
        ####
        signal = pre_all2all_layout_transfer.apply(
            attn_out, None, None,
            packed_seq_params.attn_out_fwd_metadata,
            packed_seq_params.attn_out_bwd_metadata,
            packed_seq_params.dispatcher_id,
            False,  # is_qkv
        )
        torch.cuda.nvtx.range_pop()
        return signal

    def _post_attn_to_mlp(
        self, signal: torch.Tensor,
        packed_seq_params: PingPangSingleStepPackedSeqParams,
    ):
        num_heads = self.config.num_attention_heads
        head_dim = self.config.hidden_size // num_heads
        tp_size = self.config.tensor_model_parallel_size
        num_heads_local = num_heads // tp_size
        torch.cuda.nvtx.range_push("post_attn_to_mlp")
        attn_out = post_all2all_layout_transfer.apply(
            signal,
            packed_seq_params.attn_out_fwd_metadata,
            packed_seq_params.attn_out_bwd_metadata,
            packed_seq_params.dispatcher_id,
            False,  # is_qkv
        )
        attn_out = attn_out.reshape(
            attn_out.shape[0], 1, num_heads_local * head_dim,
        ).contiguous()
        torch.cuda.nvtx.range_pop()
        return attn_out

    def _pre_mlp_to_attn(
        self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
        packed_seq_params: PingPangSingleStepPackedSeqParams,
    ):
        """
        Communication between attention layout and mlp layout.
        This operation runs on the stream `packed_seq_params.stream`.
        """

        torch.cuda.nvtx.range_push("mlp_to_attn")
        # TODO(yonghao): current we do a walk around: merge head dimension
        # into hidden dimension. In this way, we need the TP degree being
        # kept for attention and other parts.
        assert query.dim() in [3, 4], f"{query.shape=}, should be t1nh or tnh layout"
        num_q_heads = self.config.num_attention_heads
        num_kv_heads = self.config.num_query_groups or self.config.num_attention_heads
        head_dim = self.config.hidden_size // num_q_heads
        tp_size = self.config.tensor_model_parallel_size
        # TODO: write memcpy to nvshmem buffer kernel with stride support.
        query = query.reshape(query.shape[0], num_q_heads * head_dim // tp_size)
        key = key.reshape(key.shape[0], num_kv_heads * head_dim // tp_size)
        value = value.reshape(value.shape[0], num_kv_heads * head_dim // tp_size)
        ####
        signal = pre_all2all_layout_transfer.apply(
            query, key, value,
            packed_seq_params.qkv_fwd_metadata, # query_metadata
            packed_seq_params.qkv_bwd_metadata, # rev_query_metadata
            packed_seq_params.dispatcher_id,
            True,  # is_qkv
        )
        torch.cuda.nvtx.range_pop()

        return signal

    def _post_mlp_to_attn(
        self, signal: torch.Tensor,
        packed_seq_params: PingPangSingleStepPackedSeqParams,
    ):
        num_q_heads = self.config.num_attention_heads
        num_kv_heads = self.config.num_query_groups or self.config.num_attention_heads
        head_dim = self.config.hidden_size // num_q_heads

        tp_size = self.config.tensor_model_parallel_size
        num_q_heads_local = num_q_heads // tp_size
        num_kv_heads_local = num_kv_heads // tp_size
        torch.cuda.nvtx.range_push("post_mlp_to_attn")
        query, key, value = post_all2all_layout_transfer.apply(
            signal,
            packed_seq_params.qkv_fwd_metadata,
            packed_seq_params.qkv_bwd_metadata,
            packed_seq_params.dispatcher_id,
            True,  # is_qkv
        )
        query = query.view(query.shape[0], num_q_heads_local, head_dim).contiguous()
        key = key.view(key.shape[0], num_kv_heads_local, head_dim).contiguous()
        value = value.view(value.shape[0], num_kv_heads_local, head_dim).contiguous()
        torch.cuda.nvtx.range_pop()
        return query, key, value

    def _all_to_all(self, signal: torch.Tensor,
                    packed_seq_params: PingPangSingleStepPackedSeqParams,
                    is_qkv: bool,):
        metadatas = []
        if is_qkv:
            metadatas = [
                packed_seq_params.qkv_fwd_metadata,
                packed_seq_params.qkv_bwd_metadata,
            ]
        else:
            metadatas = [
                packed_seq_params.attn_out_fwd_metadata,
                packed_seq_params.attn_out_bwd_metadata,
            ]
        signal = all_to_all.apply(
            signal,
            *metadatas,
            packed_seq_params.dispatcher_id,
            packed_seq_params.stream,
        )
        return signal

    ########## Ping-Pong ##########
    def ping_pong_forward(
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
        needs_gather = debug or packed_seq_params.do_gather   # NOTE: cannot infer if this is the last local layer or not.

        packed_seq_params_0 = packed_seq_params.seq_params[0]
        packed_seq_params_1 = packed_seq_params.seq_params[1]
        mlp_packed_seq_params_0 = packed_seq_params.mlp_layout_seq_params[0]
        mlp_packed_seq_params_1 = packed_seq_params.mlp_layout_seq_params[1]
        compute_stream = torch.cuda.current_stream()
        if debug:
            setattr(packed_seq_params_0, "stream", compute_stream)
            setattr(packed_seq_params_1, "stream", compute_stream)
        else:
            setattr(packed_seq_params_0, "dispatcher_id", 0)
            setattr(packed_seq_params_1, "dispatcher_id", 1)


        # NOTE: DO NOT REMOVE THIS DEBUG TENSOR! Torch seems to have some
        # liveness issue, that deallocates the tensors too early before
        # communication kernels on the customized stream is done.
        # This list help increasing the tensor's liveness a bit.
        debug_tensors = None
        # 1. split input into two microbatches
        args = [hidden_states, attention_mask, context, context_mask, rotary_pos_emb,
                rotary_pos_cos, rotary_pos_sin, attention_bias, sequence_len_offset]
        if needs_split:
            args_0, args_1 = splits_all(args, 2)
        else:
            args_0, args_1 = repack_args(args, 2)
        (hidden_states_0, attention_mask_0, context_0, context_mask_0, rotary_pos_emb_0,
            rotary_pos_cos_0, rotary_pos_sin_0, attention_bias_0, sequence_len_offset_0) = args_0
        (hidden_states_1, attention_mask_1, context_1, context_mask_1, rotary_pos_emb_1,
            rotary_pos_cos_1, rotary_pos_sin_1, attention_bias_1, sequence_len_offset_1) = args_1

        comm_stream = packed_seq_params_0.stream
        assert comm_stream.stream_id == packed_seq_params_1.stream.stream_id

        # 2. pre-self-attention forward microbatch 0.
        ## compute,0
        torch.cuda.nvtx.range_push("pre_core_attn.0")
        query_0, key_0, value_0, residual_0, attn_mask_type_0 = self._forward_pre_core_attn(
            hidden_states_0,
            rotary_pos_emb_0,
            rotary_pos_cos_0,
            rotary_pos_sin_0,
            mlp_packed_seq_params_0,
            sequence_len_offset_0,
        )
        torch.cuda.nvtx.range_pop()
        # pre-communicate,0
        signal_0 = self._pre_mlp_to_attn(
            query_0, key_0, value_0, packed_seq_params_0
        )
        signal_0, hidden_states_1 = TickSync.apply(
            compute_stream, comm_stream, signal_0, hidden_states_1
        )

        # 3. pre-attention forward of microbatch 1, mlp2attn all2all of microbatch 0
        # NOTE: do not remove this debug tensor. see above.
        debug_tensors = (query_0, key_0, value_0)
        ## communicate,0
        signal_0 = self._all_to_all(
            signal_0, packed_seq_params_0, is_qkv=True,
        )
        ## compute,1
        torch.cuda.nvtx.range_push("pre_core_attn.1")
        query_1, key_1, value_1, residual_1, attn_mask_type_1 = self._forward_pre_core_attn(
            hidden_states_1,
            rotary_pos_emb_1,
            rotary_pos_cos_1,
            rotary_pos_sin_1,
            mlp_packed_seq_params_1,
            sequence_len_offset_1,
        )
        torch.cuda.nvtx.range_pop()
        # pre-communicate,1
        signal_1 = self._pre_mlp_to_attn(
            query_1, key_1, value_1, packed_seq_params_1
        )
        signal_0, signal_1 = TickSync.apply(
            compute_stream, comm_stream, signal_0, signal_1
        )
        # NOTE: do not remove this debug tensor. see above.
        debug_tensors = (query_1, key_1, value_1)

        # 4. self-attention forward of microbatch 0, mlp2attn all2all of microbatch 1
        ## communicate,1
        signal_1 = self._all_to_all(
            signal_1, packed_seq_params_1, is_qkv=True,
        )
        ## compute
        # post-communicate,0
        query_0, key_0, value_0 = self._post_mlp_to_attn(
            signal_0, packed_seq_params_0
        )
        torch.cuda.nvtx.range_push("core_attn.0")
        core_attn_out_0 = self._forward_core_attn(
            query_0,
            key_0,
            value_0,
            attention_mask_0,
            attention_bias_0,
            attn_mask_type_0,
            packed_seq_params_0,
        )
        torch.cuda.nvtx.range_pop()
        # pre-communicate,0
        signal_0 = self._pre_attn_to_mlp(core_attn_out_0, packed_seq_params_0)
        signal_1, signal_0 = TickSync.apply(
            compute_stream, comm_stream, signal_1, signal_0
        )
        # NOTE: do not remove this debug tensor. see above.
        debug_tensors = core_attn_out_0

        # 5. post-self-attention forward of microbatch 0, mlp2attn all2all of microbatch 1
        ## communicate,0
        signal_0 = self._all_to_all(
            signal_0, packed_seq_params_0, is_qkv=False,
        )
        ## compute,1
        # post-communicate,1
        query_1, key_1, value_1 = self._post_mlp_to_attn(signal_1, packed_seq_params_1)
        torch.cuda.nvtx.range_push("core_attn.1")
        core_attn_out_1 = self._forward_core_attn(
            query_1,
            key_1,
            value_1,
            attention_mask_1,
            attention_bias_1,
            attn_mask_type_1,
            packed_seq_params_1,
        )
        torch.cuda.nvtx.range_pop()
        # pre-communicate,1
        signal_1 = self._pre_attn_to_mlp(core_attn_out_1, packed_seq_params_1)
        signal_0, signal_1 = TickSync.apply(compute_stream, comm_stream, signal_0, signal_1)
        # NOTE: do not remove this debug tensor. see above.
        debug_tensors = core_attn_out_1

        # 6. mlp forward of microbatch 0, mlp2attn all2all of microbatch 1
        # communicate,1
        signal_1 = self._all_to_all(signal_1, packed_seq_params_1, is_qkv=False,)
        ## compute,0
        # post-communicate,0
        core_attn_out_0 = self._post_attn_to_mlp(signal_0, packed_seq_params_0)
        torch.cuda.nvtx.range_push("post_core_attn.0")
        mlp_output_0, context_0 = self._forward_post_core_attn(
            core_attn_out_0,
            residual_0,
            context_0,
            context_mask_0,
        )
        torch.cuda.nvtx.range_pop()
        # no pre-communicate for 0 now.
        signal_1, mlp_output_0 = TickSync.apply(compute_stream, comm_stream, signal_1, mlp_output_0)
        # NOTE: do not remove this debug tensor. see above.
        debug_tensors = None

        torch.cuda.nvtx.range_push("post_core_attn.1")
        # post-communicate,1
        core_attn_out_1 = self._post_attn_to_mlp(signal_1, packed_seq_params_1)
        mlp_output_1, context_1 = self._forward_post_core_attn(
            core_attn_out_1,
            residual_1,
            context_1,
            context_mask_1,
        )
        torch.cuda.nvtx.range_pop()
        # concatenate the two microbatches to one.
        if needs_gather:
            output = gather_tensor([mlp_output_0, mlp_output_1], num_splits=2)
            context = gather_tensor([context_0, context_1], num_splits=2)
        else:
            output = [mlp_output_0, mlp_output_1]
            context = [context_0, context_1]
        return output, context

    #### Debug use.
    def forward_ping_pong_single_sided(
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
        """Debug use. Single side of Ping-Pang parallel."""
        assert inference_params is None, "inference not supported yet"
        assert inference_context is None, "inference not supported yet"
        assert context is None, "cross-attention not supported yet"
        assert context_mask is None, "cross-attention not supported yet"

        setattr(packed_seq_params, "stream", torch.cuda.current_stream())
        backward_resend_qkv = packed_seq_params.bwd_packed_seq_params is not None

        if rotary_pos_emb is not None:
            rotary_pos_emb = None
            warnings.warn("forward_ping_pong_single_sided is only to debug a single ping-pong "
                          "stage's correctness, and does not have RoPE supported")

        query, key, value, residual, attn_mask_type = self._forward_pre_core_attn(
            hidden_states,
            rotary_pos_emb,
            rotary_pos_cos,
            rotary_pos_sin,
            packed_seq_params,
            sequence_len_offset,
        )
        debug_tensors = [(query, key, value),]

        signal = self._pre_mlp_to_attn(query, key, value, packed_seq_params)
        signal = self._all_to_all(signal, packed_seq_params, is_qkv=True)

        if backward_resend_qkv:
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
                    num_heads_q=self.config.num_attention_heads // self.config.tensor_model_parallel_size,
                    num_heads_kv=self.config.num_query_groups // self.config.tensor_model_parallel_size,
                    head_dim=self.config.hidden_size // self.config.num_attention_heads,
                    return_attn_probs=True,
                ),
            )
        else:
            query, key, value = self._post_mlp_to_attn(signal, packed_seq_params)
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

            signal = self._pre_attn_to_mlp(core_attn_out, packed_seq_params)

        signal = self._all_to_all(signal, packed_seq_params, is_qkv=False)

        if backward_resend_qkv:
            core_attn_out = post_a2a_attn_out_with_lse.apply(
                signal, query, key, value,
                self.config.num_attention_heads // self.config.tensor_model_parallel_size,
                packed_seq_params.attn_out_fwd_metadata,
                packed_seq_params.attn_out_bwd_metadata,
                packed_seq_params.dispatcher_id,
            )
        else:
            core_attn_out = self._post_attn_to_mlp(signal, packed_seq_params)
        debug_tensors.append(core_attn_out)

        mlp_output, context = self._forward_post_core_attn(
            core_attn_out,
            residual,
            context,
            context_mask,
        )

        return (mlp_output, context,) + (
            (debug_tensors,) if return_debug else ()
        )
