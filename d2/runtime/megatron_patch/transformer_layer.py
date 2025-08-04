from contextlib import nullcontext
from typing import Any, Dict, List, Optional, Union
import types

import torch
import torch.distributed
from torch import Tensor

from megatron.core import tensor_parallel
from megatron.core.inference.contexts import BaseInferenceContext
from megatron.core.models.common.embeddings.rope_utils import (
    apply_rotary_pos_emb,
)
from megatron.core.models.gpt.gpt_model import GPTModel
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.transformer.transformer_block import (
    TransformerBlock as MegatronTransformerBlock
)
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.transformer.transformer_layer import (
    TransformerLayer as MegatronTransformerLayer,
    TransformerLayerSubmodules,
)
from megatron.core.utils import WrappedTensor, make_viewless_tensor

from d2.runtime.megatron_patch.base_transformer_layer import TransformerLayer as BaseTransformerLayer
from d2.runtime.megatron_patch.packed_seq_params import PingPangPackedSeqParams, PingPangSingleStepPackedSeqParams
from d2.runtime.megatron_patch.dispatcher_wrapper import TickSync
from d2.runtime.megatron_patch.fast_dispatch_fn import (
    all_to_all, post_all2all_layout_transfer, pre_all2all_layout_transfer
)


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

def _repack_dicts(args: Dict[str, List[torch.Tensor]], num_splits: int):
    assert all(len(a) == num_splits for a in args.values())
    return [
        {k: a[i] for k, a in args.items()}
        for i in range(num_splits)
    ]

def _splits_all(tensors: List[torch.Tensor], num_splits: int):
    splits = [_split_tensor(t, num_splits) for t in tensors]
    return _repack_args(splits, num_splits)

def _split_all_dict(tensors: Dict[str, torch.Tensor], num_splits: int):
    splits = {k: _split_tensor(v, num_splits) for k, v in tensors.items()}
    return _repack_dicts(splits, num_splits)

def _gather_tensor(tensors: List[torch.Tensor], num_splits: int):
    assert len(tensors) == num_splits
    if any(t is None for t in tensors):
        assert all(t is None for t in tensors), "None tensors in gather_tensor"
        return None
    return torch.cat(tensors, dim=0)
####


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
        attn_out = attn_out.view(attn_out.shape[0], num_heads * head_dim)
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
        torch.cuda.nvtx.range_push("post_attn_to_mlp")
        attn_out = post_all2all_layout_transfer.apply(
            signal,
            packed_seq_params.attn_out_fwd_metadata,
            packed_seq_params.attn_out_bwd_metadata,
            packed_seq_params.dispatcher_id,
            False,  # is_qkv
        )
        attn_out = attn_out.reshape(
            attn_out.shape[0], 1, num_heads * head_dim,
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
        # TODO: write memcpy to nvshmem buffer kernel with stride support.
        query = query.reshape(query.shape[0], num_q_heads * head_dim)
        key = key.reshape(key.shape[0], num_kv_heads * head_dim)
        value = value.reshape(value.shape[0], num_kv_heads * head_dim)
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
        torch.cuda.nvtx.range_push("post_mlp_to_attn")
        query, key, value = post_all2all_layout_transfer.apply(
            signal,
            packed_seq_params.qkv_fwd_metadata,
            packed_seq_params.qkv_bwd_metadata,
            packed_seq_params.dispatcher_id,
            True,  # is_qkv
        )
        query = query.view(query.shape[0], num_q_heads, head_dim).contiguous()
        key = key.view(key.shape[0], num_kv_heads, head_dim).contiguous()
        value = value.view(value.shape[0], num_kv_heads, head_dim).contiguous()
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
        needs_gather = debug or packed_seq_params.do_gather   # NOTE: cannot infer if this is the last local layer or not.

        packed_seq_params_0 = packed_seq_params.seq_params[0]
        packed_seq_params_1 = packed_seq_params.seq_params[1]
        mlp_packed_seq_params_0 = packed_seq_params.mlp_layout_seq_params[0]
        mlp_packed_seq_params_1 = packed_seq_params.mlp_layout_seq_params[1]
        compute_stream = torch.cuda.current_stream()
        if debug:
            setattr(packed_seq_params_0, "stream", compute_stream)
            setattr(packed_seq_params_1, "stream", compute_stream)


        # NOTE: DO NOT REMOVE THIS DEBUG TENSOR! Torch seems to have some
        # liveness issue, that deallocates the tensors too early before
        # communication kernels on the customized stream is done.
        # This list help increasing the tensor's liveness a bit.
        debug_tensors = None
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
        # TODO: confirm this is equal to the stream of packed_seq_params_1
        comm_stream = packed_seq_params_0.stream

        # 2. pre-self-attention forward microbatch 0.
        # Ideally, this part should merge to the previous layer's post-self-attention to maximize
        # the communication-computation overlap.
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
        query_0, key_0, value_0, hidden_states_1 = TickSync.apply(
            compute_stream, comm_stream, query_0, key_0, value_0, hidden_states_1
        )

        # 3. pre-attention forward of microbatch 1, mlp2attn all2all of microbatch 0
        # NOTE: do not remove this debug tensor. see above.
        debug_tensors = (query_0, key_0, value_0)

        query_0, key_0, value_0 = _layout_mlp_to_attn(query_0, key_0, value_0, packed_seq_params_0)
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
        query_0, key_0, value_0, query_1, key_1, value_1 = TickSync.apply(
            compute_stream, comm_stream, query_0, key_0, value_0, query_1, key_1, value_1
        )
        # NOTE: do not remove this debug tensor. see above.
        debug_tensors = (query_1, key_1, value_1)

        # 4. self-attention forward of microbatch 0, mlp2attn all2all of microbatch 1
        query_1, key_1, value_1 = _layout_mlp_to_attn(query_1, key_1, value_1, packed_seq_params_1)
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
        query_1, key_1, value_1, core_attn_out_0 = TickSync.apply(
            compute_stream, comm_stream, query_1, key_1, value_1, core_attn_out_0
        )
        # NOTE: do not remove this debug tensor. see above.
        debug_tensors = core_attn_out_0

        # 5. post-self-attention forward of microbatch 0, mlp2attn all2all of microbatch 1
        core_attn_out_0 = _layout_attn_to_mlp(core_attn_out_0, packed_seq_params_0)
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
        core_attn_out_0, core_attn_out_1 = TickSync.apply(compute_stream, comm_stream, core_attn_out_0, core_attn_out_1)
        # NOTE: do not remove this debug tensor. see above.
        debug_tensors = core_attn_out_1

        # 6. mlp forward of microbatch 0, mlp2attn all2all of microbatch 1
        core_attn_out_1 = _layout_attn_to_mlp(core_attn_out_1, packed_seq_params_1)
        torch.cuda.nvtx.range_push("post_core_attn.0")
        mlp_output_0, context_0 = self._forward_post_core_attn(
            core_attn_out_0,
            residual_0,
            context_0,
            context_mask_0,
        )
        torch.cuda.nvtx.range_pop()
        core_attn_out_1, mlp_output_0 = TickSync.apply(compute_stream, comm_stream, core_attn_out_1, mlp_output_0)
        # NOTE: do not remove this debug tensor. see above.
        debug_tensors = None

        torch.cuda.nvtx.range_push("post_core_attn.1")
        mlp_output_1, context_1 = self._forward_post_core_attn(
            core_attn_out_1,
            residual_1,
            context_1,
            context_mask_1,
        )
        torch.cuda.nvtx.range_pop()
        # concatenate the two microbatches to one.
        if needs_gather:
            output = _gather_tensor([mlp_output_0, mlp_output_1], num_splits=2)
            context = _gather_tensor([context_0, context_1], num_splits=2)
        else:
            output = [mlp_output_0, mlp_output_1]
            context = [context_0, context_1]
        return output, context

    #### Debug use.
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

        # FIXME: support RoPE
        assert rotary_pos_emb is None, "RoPE needs the MLP layout packed seq params."
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

        signal = self._pre_attn_to_mlp(
            core_attn_out, packed_seq_params
        )
        signal = self._all_to_all(signal, packed_seq_params, is_qkv=False)
        core_attn_out = self._post_attn_to_mlp(signal, packed_seq_params)
        debug_tensors.append(core_attn_out)

        mlp_output, context = self._forward_post_core_attn(
            core_attn_out,
            residual,
            context,
            context_mask,
        )

        return mlp_output, context, debug_tensors


def add_ping_pang_forward(block: MegatronTransformerBlock):
    def init_ping_pang_communication_ctx(self):
        self.comm_stream = torch.cuda.Stream()
        self._ping_pang_debug = True

    def ping_pang_forward(
        self: MegatronTransformerBlock,
        hidden_states: Union[Tensor, WrappedTensor],
        attention_mask: Optional[Tensor],
        context: Optional[Tensor] = None,
        context_mask: Optional[Tensor] = None,
        rotary_pos_emb: Optional[Tensor] = None,
        rotary_pos_cos: Optional[Tensor] = None,
        rotary_pos_sin: Optional[Tensor] = None,
        attention_bias: Optional[Tensor] = None,
        inference_context: Optional[BaseInferenceContext] = None,
        packed_seq_params: Optional[PingPangPackedSeqParams] = None,
        sequence_len_offset: Optional[Tensor] = None,
        *,
        inference_params: Optional[BaseInferenceContext] = None,
    ):
        assert inference_context is None
        assert inference_params is None
        assert context is None
        assert context_mask is None
        assert sequence_len_offset is None
        if self.config.fp8:
            raise NotImplementedError("FP8 not supported yet")

        if isinstance(hidden_states, WrappedTensor):
            hidden_states = hidden_states.unwrap()

        if not self.pre_process:
            # See set_input_tensor()
            hidden_states = self.input_tensor

        hidden_states = make_viewless_tensor(inp=hidden_states, requires_grad=True, keep_graph=True)

        if self.config.sequence_parallel:
            rng_context = tensor_parallel.get_cuda_rng_tracker().fork()
        else:
            rng_context = nullcontext()
        outer_fp8_context = nullcontext()

        compute_stream = torch.cuda.current_stream()

        with rng_context, outer_fp8_context:
            # Forward pass.
            if self.config.recompute_granularity == 'full' and self.training:
                raise NotImplementedError("Full recompute not supported yet")
                hidden_states = self._checkpointed_forward(
                    hidden_states=hidden_states,
                    attention_mask=attention_mask,
                    context=context,
                    context_mask=context_mask,
                    rotary_pos_emb=rotary_pos_emb,
                    attention_bias=attention_bias,
                    packed_seq_params=packed_seq_params,
                )
            else:
                arg_group = {
                    "hidden_states": hidden_states,
                    "attention_mask": attention_mask,
                    "context": context,
                    "context_mask": context_mask,
                    "rotary_pos_emb": rotary_pos_emb,
                    "rotary_pos_cos": rotary_pos_cos,
                    "rotary_pos_sin": rotary_pos_sin,
                    "attention_bias": attention_bias,
                    "sequence_len_offset": sequence_len_offset,
                }
                arg_group_0, arg_group_1 = _split_all_dict(arg_group, 2)
                del arg_group

                packed_seq_params_0 = packed_seq_params.seq_params[0]
                packed_seq_params_1 = packed_seq_params.seq_params[1]
                setattr(packed_seq_params_0, "stream", self.comm_stream)
                setattr(packed_seq_params_1, "stream", self.comm_stream)
                arg_group_0["packed_seq_params"] = packed_seq_params_0
                arg_group_1["packed_seq_params"] = packed_seq_params_1
                arg_group_0["mlp_packed_seq_params"] = packed_seq_params.mlp_layout_seq_params[0]
                arg_group_1["mlp_packed_seq_params"] = packed_seq_params.mlp_layout_seq_params[1]

                for l_no in range(len(self.layers)):
                    inner_fp8_context = nullcontext()
                    with self.offload_context, inner_fp8_context:
                        arg_group_0, arg_group_1, hidden_states, context = self.forward_layers(
                            l_no, arg_group_0, arg_group_1, compute_stream
                        )

                    if (
                        torch.is_grad_enabled()
                        and self.config.cpu_offloading
                        and self.group_prefetch_offload_commit_async is not None
                    ):
                        hidden_states = self.group_prefetch_offload_commit_async(hidden_states)

        # Final layer norm.
        if self.final_layernorm is not None:
            hidden_states = self.final_layernorm(hidden_states)
            # TENorm produces a "viewed" tensor. This will result in schedule.py's
            # deallocate_output_tensor() throwing an error, so a viewless tensor is
            # created to prevent this.
            hidden_states = make_viewless_tensor(
                inp=hidden_states, requires_grad=True, keep_graph=True
            )

        return hidden_states

    def forward_layers(
        self,
        l_no: int,
        arg_group_0: Dict[str, Any],
        arg_group_1: Dict[str, Any],
        compute_stream: torch.cuda.Stream,
    ):
        layer = self.layers[l_no]
        prev_layer = self.layers[l_no - 1] if l_no > 0 else None
        # tick 0, second half
        arg_group_0 = _forward_pre_core_attn(layer, arg_group_0)
        if l_no > 0:
            _tick_sync(
                compute_stream, self.comm_stream,
                arg_group_0, "query",           # compute out
                arg_group_1, "core_attn_out",   # prev layer's comm out,
            )

        # tick 1
        # compute
        if l_no > 0:
            arg_group_1 = _forward_post_core_attn(prev_layer, arg_group_1)
        arg_group_1 = _forward_pre_core_attn(layer, arg_group_1)
        # communication
        arg_group_0 = _layout_mlp_to_attn(layer, arg_group_0)
        _tick_sync(
            compute_stream, self.comm_stream,
            arg_group_0, "query",   # comm out
            arg_group_1, "query",   # compute out
        )

        # tick 2
        # compute
        arg_group_0 = _forward_core_attn(layer, arg_group_0)
        # communication
        arg_group_1 = _layout_mlp_to_attn(layer, arg_group_1)
        _tick_sync(
            compute_stream, self.comm_stream,
            arg_group_0, "core_attn_out",   # compute out
            arg_group_1, "query",           # comm out.
        )

        # tick 3
        # compute
        arg_group_1 = _forward_core_attn(layer, arg_group_1)
        # communication
        arg_group_0 = _layout_attn_to_mlp(layer, arg_group_0)
        _tick_sync(
            compute_stream, self.comm_stream,
            arg_group_0, "core_attn_out",   # comm out
            arg_group_1, "core_attn_out",   # compute out
        )

        # tick 4, also the tick 0 of the next layer
        # compute
        arg_group_0 = _forward_post_core_attn(layer, arg_group_0)
        # communication
        arg_group_1 = _layout_attn_to_mlp(layer, arg_group_1)
        # NOTE: communication of this tick is at the next layer.

        # if the last layer, do the other half of tick 4 and tick 5
        if l_no == len(self.layers) - 1:
            # No next layer, do the sync here.
            _tick_sync(
                compute_stream, self.comm_stream,
                arg_group_0, "hidden_states",   # place holder
                arg_group_1, "core_attn_out",   # comm out
            )
            arg_group_1 = _forward_post_core_attn(layer, arg_group_1)
            # gathering the result
            hidden_states = _gather_tensor([
                arg_group_0["hidden_states"],
                arg_group_1["hidden_states"]
            ], 2)
            context = _gather_tensor([
                arg_group_0["context"],
                arg_group_1["context"]
            ], 2)
        else:
            hidden_states = None
            context = None
        return arg_group_0, arg_group_1, hidden_states, context

    def _forward_pre_core_attn(layer: TransformerLayer, args: Dict[str, Any]):
        hidden_states = args.pop("hidden_states")
        query, key, value, residual, attn_mask_type = layer._forward_pre_core_attn(
            hidden_states,
            args["rotary_pos_emb"],
            args["rotary_pos_cos"],
            args["rotary_pos_sin"],
            args["mlp_packed_seq_params"],
            args["sequence_len_offset"],
        )
        args["query"] = query
        args["key"] = key
        args["value"] = value
        args["residual"] = residual
        args["attn_mask_type"] = attn_mask_type
        return args

    def _TODOUPDATETHIS(layer: TransformerLayer, args: Dict[str, Any]):
        query = args.pop("query")
        key = args.pop("key")
        value = args.pop("value")
        query, key, value = _layout_mlp_to_attn(
            query, key, value,
            args["packed_seq_params"],
        )
        args["query"] = query
        args["key"] = key
        args["value"] = value
        return args

    def _forward_core_attn(layer: TransformerLayer, args: Dict[str, Any]):
        # pop out to make sure the tensor is freed
        query = args.pop("query")
        key = args.pop("key")
        value = args.pop("value")
        core_attn_out = layer._forward_core_attn(
            query,
            key,
            value,
            args["attention_mask"],
            args["attention_bias"],
            args["attn_mask_type"],
            args["packed_seq_params"],
        )
        args["core_attn_out"] = core_attn_out
        return args

    def _TODOUPDATE(layer: TransformerLayer, args: Dict[str, Any]):
        core_attn_out = args.pop("core_attn_out")
        core_attn_out = _layout_attn_to_mlp(
            core_attn_out, args["packed_seq_params"]
        )
        args["core_attn_out"] = core_attn_out
        return args

    def _forward_post_core_attn(layer: TransformerLayer, args: Dict[str, Any]):
        core_attn_out = args.pop("core_attn_out")
        residual = args.pop("residual")
        mlp_output, context = layer._forward_post_core_attn(
            core_attn_out,
            residual,
            args["context"],
            args["context_mask"],
        )
        args["hidden_states"] = mlp_output
        args["context"] = context
        return args

    def _tick_sync(compute_stream, comm_stream, arg_group_0, keys_0, arg_group_1, keys_1):
        if isinstance(keys_0, str):
            keys_0 = [keys_0]
        if isinstance(keys_1, str):
            keys_1 = [keys_1]
        tensors_0 = [arg_group_0[key] for key in keys_0]
        tensors_1 = [arg_group_1[key] for key in keys_1]
        tensors = tensors_0 + tensors_1
        out_tensors = TickSync.apply(
            compute_stream, comm_stream,
            *tensors
        )
        out_tensors_0 = out_tensors[:len(tensors_0)]
        out_tensors_1 = out_tensors[len(tensors_0):]
        for key, out_tensor in zip(keys_0, out_tensors_0):
            arg_group_0[key] = out_tensor
        for key, out_tensor in zip(keys_1, out_tensors_1):
            arg_group_1[key] = out_tensor

    def forward(self, *args, **kwargs):
        if self._ping_pang_debug:
            return self._normal_forward(*args, **kwargs)
        return self.ping_pang_forward(*args, **kwargs)

    block.forward_layers = types.MethodType(forward_layers, block)
    block.init_ping_pang_communication_ctx = types.MethodType(init_ping_pang_communication_ctx, block)
    block.ping_pang_forward = types.MethodType(ping_pang_forward, block)
    block._normal_forward = block.forward
    block.forward = types.MethodType(forward, block)
    block.init_ping_pang_communication_ctx()


class PingPangGPTModel(GPTModel):
    def __init__(self, *args, **kwargs):
        print("PingPangGPTModel init")
        super().__init__(*args, **kwargs)
        add_ping_pang_forward(self.decoder)

    def set_debug(self, debug: bool):
        self.decoder._ping_pang_debug = debug
