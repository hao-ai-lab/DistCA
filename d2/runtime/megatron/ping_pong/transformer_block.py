from contextlib import contextmanager, nullcontext
import functools
import os
from typing import Any, Dict, Optional, Union
import types

import torch
from torch import Tensor
from megatron.core import tensor_parallel
from megatron.core.inference.contexts import BaseInferenceContext
from megatron.core.models.gpt.gpt_model import GPTModel
from megatron.core.transformer.transformer_block import (
    TransformerBlock as MegatronTransformerBlock
)
from megatron.core.utils import WrappedTensor, make_viewless_tensor

from d2.runtime.attn_kernels.ops import DispatcherWrapper
from d2.runtime.megatron.ops.fused_comm_attn import dummy_backward
from d2.runtime.megatron.packed_seq_params import PingPangPackedSeqParams
from d2.runtime.megatron.ping_pong.tick_ops import (
    forward_core_attn, layout_mlp_to_attn, layout_attn_to_mlp, log_memory_usage, tick_nonca_compute, tick_nonca_compute_cuda_graph,
    tick_sync
)
from d2.runtime.megatron.ping_pong.transformer_layer import TransformerLayer
from d2.runtime.megatron.ping_pong.utils import split_all_dict, gather_tensor


class _debug_monkey_patch:
    """
    Context manager to monkey patch Transformer Layer forward in debugging.
    """
    def __init__(self, layer_forward_impl):
        self._layer_forward_impl = layer_forward_impl
    def __enter__(self):
        self.backup_forward = TransformerLayer.forward
        TransformerLayer.forward = self._layer_forward_impl
    def __exit__(self, exc_type, exc_value, traceback):
        TransformerLayer.forward = self.backup_forward


class PingPongTransformerBlockInterface(MegatronTransformerBlock):
    def __init__(self, *args, **kwargs):
        raise RuntimeError("Interface class should not be instantiated.")

    def init_value(self):
        self.ping_pong_comm_initialized: bool = False
        self.comm_stream: torch.cuda.Stream = None
        self._ping_pong_debug: bool = False
        self._debug_forward_impl: str = "orig"
        self.use_cuda_graph: bool = False
        self.tick_nonca_compute = tick_nonca_compute_cuda_graph if self.use_cuda_graph else tick_nonca_compute
    
    def init_layer_cuda_graphs(self):
        self.use_cuda_graph = True
        self.tick_nonca_compute = tick_nonca_compute_cuda_graph
        
        prev_layer = None
        seq_len = int(os.environ.get("D2_SEQ_LEN", -1))
        if seq_len == -1:
            raise ValueError("D2_SEQ_LEN is not set")
        
        # CUDA graph capture uses torch.autograd.grad() internally (via make_graphed_callables).
        # Megatron checkpoint/recompute hooks are not compatible with .grad() (they require
        # autograd.backward()), so we temporarily disable those recompute flags during capture.
        #
        # In particular, `recompute_mlp` uses `tensor_parallel.checkpoint()` (CheckpointFunction),
        # which hard-errors under autograd.grad().
        saved_recompute_flags = []
        for layer in self.layers:
            saved_recompute_flags.append(
                (
                    layer,
                    getattr(layer, "recompute_input_layernorm", False),
                    getattr(layer, "recompute_pre_mlp_layernorm", False),
                    getattr(layer, "recompute_mlp", False),
                )
            )
            if hasattr(layer, "recompute_input_layernorm"):
                layer.recompute_input_layernorm = False
            if hasattr(layer, "recompute_pre_mlp_layernorm"):
                layer.recompute_pre_mlp_layernorm = False
            if hasattr(layer, "recompute_mlp"):
                layer.recompute_mlp = False

        try:
            for layer in self.layers:
                layer: TransformerLayer
                layer.init_pre_attn_cuda_graph(
                    prev_layer,
                    seq_len=seq_len,
                    device=layer.self_attention.linear_qkv.weight.device,
                    dtype=layer.self_attention.linear_qkv.weight.dtype,
                )
                prev_layer = layer

            # This is the last layer
            layer.init_post_attn_cuda_graph(
                seq_len=seq_len,
                device=layer.self_attention.linear_qkv.weight.device,
                dtype=layer.self_attention.linear_qkv.weight.dtype,
            )
        finally:
            for layer, recompute_input_ln, recompute_pre_mlp_ln, recompute_mlp in saved_recompute_flags:
                if hasattr(layer, "recompute_input_layernorm"):
                    layer.recompute_input_layernorm = recompute_input_ln
                if hasattr(layer, "recompute_pre_mlp_layernorm"):
                    layer.recompute_pre_mlp_layernorm = recompute_pre_mlp_ln
                if hasattr(layer, "recompute_mlp"):
                    layer.recompute_mlp = recompute_mlp

    def init_ping_pong_communication_ctx(self, device: torch.device):
        assert not self.ping_pong_comm_initialized
        self.comm_stream = torch.cuda.Stream(device=device, priority=-1)
        self.ping_pong_comm_initialized = True
        if os.getenv("D2_SHOULD_USE_SAME_STREAM_FOR_COMM_AND_COMPUTE", "0") == "1":
            self.comm_stream = torch.cuda.current_stream(device=device)
        DispatcherWrapper.comm_stream = self.comm_stream

    def ping_pong_forward(
        self,
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
        # Remove this line to enable RoPE
        #rotary_pos_emb = None

        if self.config.sequence_parallel:
            rng_context = tensor_parallel.get_cuda_rng_tracker().fork()
        else:
            rng_context = nullcontext()
        outer_fp8_context = nullcontext()

        compute_stream = torch.cuda.current_stream()

        packed_seq_params_0 = packed_seq_params.seq_params[0]
        packed_seq_params_1 = packed_seq_params.seq_params[1]
        setattr(packed_seq_params_0, "stream", self.comm_stream)
        setattr(packed_seq_params_1, "stream", self.comm_stream)
        setattr(packed_seq_params_0, "dispatcher_id", 0)
        setattr(packed_seq_params_1, "dispatcher_id", 1)

        with rng_context, outer_fp8_context:
            # Forward pass.
            if self.config.recompute_granularity == 'full' and self.training:
                raise NotImplementedError("Full recompute full layer not supported in ping-pong forward yet.")
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
                arg_group_0, arg_group_1 = split_all_dict(arg_group, 2)
                del arg_group

                arg_group_0["packed_seq_params"] = packed_seq_params_0
                arg_group_1["packed_seq_params"] = packed_seq_params_1
                arg_group_0["mlp_packed_seq_params"] = packed_seq_params.mlp_layout_seq_params[0]
                arg_group_1["mlp_packed_seq_params"] = packed_seq_params.mlp_layout_seq_params[1]

                for l_no in range(len(self.layers)):
                    inner_fp8_context = nullcontext()
                    with self.offload_context, inner_fp8_context:
                        arg_group_0, arg_group_1, hidden_states, context = self.forward_layer_ping_pong(
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

    def forward_layer_ping_pong(
        self,
        l_no: int,
        arg_group_0: Dict[str, Any],
        arg_group_1: Dict[str, Any],
        compute_stream: torch.cuda.Stream,
    ):
        log_memory_usage(f"(L{l_no}) forward_layers:(start)")
        torch.cuda.nvtx.range_push(f"PingPong.forward_layer[{l_no}]")

        layer = self.layers[l_no]
        prev_layer = self.layers[l_no - 1] if l_no > 0 else None
        # Prepare name for nvtx.
        pre_ca_name = f"forward[{l_no}].pre_ca"
        prev_post_ca_name = f"forward[{l_no-1}].post_ca" if l_no > 0 else ""
        linear_name = prev_post_ca_name + "." + pre_ca_name

        # tick 0, compute part
        with torch.cuda.nvtx.range(linear_name + ".0"):
            arg_group_0 = self.tick_nonca_compute(layer, prev_layer, arg_group_0,
                                                  is_last_layer_post_attn=False)
        with torch.cuda.nvtx.range(f"forward[{l_no}].sync_tick.0"):
            tick_sync(
                compute_stream, self.comm_stream,
                arg_group_0, "signal",  # compute out
                # prev layer's comm out, or anything if it's the first layer
                arg_group_1, "signal" if l_no > 0 else "hidden_states",
                layer_info=f"L{l_no}", operation_info="tick_0"
            )

        # tick 1
        # communication
        with torch.cuda.nvtx.range(f"forward[{l_no}].all2all_mlp_to_attn.0"):
            arg_group_0 = layout_mlp_to_attn(layer, arg_group_0)
        # compute
        with torch.cuda.nvtx.range(linear_name + ".1"):
            arg_group_1 = self.tick_nonca_compute(layer, prev_layer, arg_group_1,
                                                  is_last_layer_post_attn=False)
        with torch.cuda.nvtx.range(f"forward[{l_no}].sync_tick.1"):
            tick_sync(
                compute_stream, self.comm_stream,
                arg_group_0, "signal",  # comm out
                arg_group_1, "signal",  # compute out
                layer_info=f"L{l_no}", operation_info="tick_1"
            )

        # tick 2
        # communication
        with torch.cuda.nvtx.range(f"forward[{l_no}].all2all_mlp_to_attn.1"):
            arg_group_1 = layout_mlp_to_attn(layer, arg_group_1)
        # compute
        with torch.cuda.nvtx.range(f"forward[{l_no}].core_attn.0"):
            arg_group_0 = forward_core_attn(layer, arg_group_0)
        with torch.cuda.nvtx.range(f"forward[{l_no}].sync_tick.2"):
            tick_sync(
                compute_stream, self.comm_stream,
                arg_group_0, "signal",  # compute out
                arg_group_1, "signal",  # comm out.
                layer_info=f"L{l_no}", operation_info="tick_2"
            )

        # tick 3
        # communication
        with torch.cuda.nvtx.range(f"forward[{l_no}].all2all_attn_to_mlp.0"):
            arg_group_0 = layout_attn_to_mlp(layer, arg_group_0)
        # compute
        with torch.cuda.nvtx.range(f"forward[{l_no}].core_attn_1"):
            arg_group_1 = forward_core_attn(layer, arg_group_1)
        with torch.cuda.nvtx.range(f"forward[{l_no}].sync_tick.3"):
            tick_sync(
                compute_stream, self.comm_stream,
                arg_group_0, "signal",  # comm out
                arg_group_1, "signal",  # compute out
                layer_info=f"L{l_no}", operation_info="tick_3"
            )

        # tick 4, communication. also the tick 0 of the next layer
        # communication
        with torch.cuda.nvtx.range(f"forward[{l_no}].all2all_attn_to_mlp.1"):
            arg_group_1 = layout_attn_to_mlp(layer, arg_group_1)
        # NOTE: sync of this tick is at the next layer.

        # if the last layer, do the other half of tick 4 and tick 5
        if l_no == len(self.layers) - 1:
            # No next layer, do the sync here.
            # compute
            with torch.cuda.nvtx.range(f"forward[{l_no}].post_core_attn.0"):
                arg_group_0 = self.tick_nonca_compute(layer, None, arg_group_0,
                                                      is_last_layer_post_attn=True)
            with torch.cuda.nvtx.range(f"forward[{l_no}].sync_tick.4"):
                tick_sync(
                    compute_stream, self.comm_stream,
                    arg_group_0, "hidden_states",   # place holder
                    arg_group_1, "signal",          # comm out
                    layer_info=f"L{l_no}", operation_info="tick_4"
                )
            with torch.cuda.nvtx.range(f"forward[{l_no}].post_core_attn.1"):
                arg_group_1 = self.tick_nonca_compute(layer, None, arg_group_1,
                                                      is_last_layer_post_attn=True)
            # gathering the result
            with torch.cuda.nvtx.range(f"forward[{l_no}].gather_ping_pong"):
                hidden_states = gather_tensor([arg_group_0["hidden_states"], arg_group_1["hidden_states"]], 2)
                context = gather_tensor([arg_group_0["context"],arg_group_1["context"]], 2)
        else:
            hidden_states = None
            context = None

        torch.cuda.nvtx.range_pop()

        log_memory_usage(f"(L{l_no}) forward_layers:(end)")
        return arg_group_0, arg_group_1, hidden_states, context

    def forward(self, *args, **kwargs):
        # print(f'{self._ping_pong_debug=}')
        """
        For Pipeline Parallel debugging, we use single-sided to ease debugging.
        """
        if self._ping_pong_debug:
            assert self._debug_forward_impl in ["orig", "single_sided", "orig_reimpl"], self._debug_forward_impl
            if self._debug_forward_impl == "single_sided":
                ctx = _debug_monkey_patch(TransformerLayer.forward_ping_pong_single_sided)
            elif self._debug_forward_impl == "orig_reimpl":
                ctx = _debug_monkey_patch(TransformerLayer.forward_orig_impl)
            else:
                ctx = nullcontext()
            with ctx:
                return self._normal_forward(*args, **kwargs)

        assert self.ping_pong_comm_initialized
        return self.ping_pong_forward(*args, **kwargs)


def add_ping_pong_forward(block: MegatronTransformerBlock) -> PingPongTransformerBlockInterface:

    block._normal_forward = block.forward
    block.init_value = types.MethodType(
        PingPongTransformerBlockInterface.init_value, block)
    block.init_ping_pong_communication_ctx = types.MethodType(
        PingPongTransformerBlockInterface.init_ping_pong_communication_ctx, block)
    block.ping_pong_forward = types.MethodType(
        PingPongTransformerBlockInterface.ping_pong_forward, block)
    block.forward_layer_ping_pong = types.MethodType(
        PingPongTransformerBlockInterface.forward_layer_ping_pong, block)
    block.forward = types.MethodType(
        PingPongTransformerBlockInterface.forward, block)
    block.init_layer_cuda_graphs = types.MethodType(
        PingPongTransformerBlockInterface.init_layer_cuda_graphs, block)

    return block


class PingPongGPTModel(GPTModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.decoder = add_ping_pong_forward(self.decoder)
        self.decoder.init_value()

    def set_debug(self, debug: bool, debug_fwd_impl: str = None):
        self.decoder._ping_pong_debug = debug
        if debug_fwd_impl:
            self.decoder._debug_forward_impl = debug_fwd_impl

    # very ugly hardcode here:
    # If the forward is a dummy forward, then:
    # 1. ensure to skip calculating the loss
    # 2. ensure to have a dummy hidden_states with correct shape
    #    - this prevents error in decoder.final_layernorm
    #    - this make layer.forward easier
    @contextmanager
    def _reset_stage_for_dummy_forward(self, reset):
        if not reset:
            yield
        else:
            decoder_pre_process = self.decoder.pre_process
            post_process = self.post_process
            self.decoder.pre_process = True
            self.post_process = False
            yield
            self.post_process = post_process
            self.decoder.pre_process = decoder_pre_process

    @functools.wraps(GPTModel.forward)
    def forward(self, input_ids, *args, **kwargs):
        # print(f'{len(self.decoder.layers)=}')
        is_dummy_forward = getattr(self.decoder.layers[0], "current_microbatch", 0) < 0
        if is_dummy_forward and not self.pre_process:
            # get dtype
            dtype = (
                torch.bfloat16 if self.config.bf16
                else torch.float16 if self.config.fp16
                else self.config.params_dtype
            )
            # create a dummy decoder_input
            sl = input_ids.shape[-1]
            hs = self.config.hidden_size
            if self.config.sequence_parallel:
                sl //= self.config.tensor_model_parallel_size
            kwargs['decoder_input'] = torch.zeros((sl, 1, hs), dtype=dtype, device='cuda')

        with self._reset_stage_for_dummy_forward(reset=is_dummy_forward):
            output = super().forward(input_ids, *args, **kwargs)

        return output

    def dummy_backward(self, packed_seq_params: PingPangPackedSeqParams):
        """
        A dummy backward that runs #layer times of decoder layer's backward.
        When the device is idle (at pipeline pre-fill or drain-out period),
        this makes it serve as a remote Core-Attention Server.
        """
        dtype = self.decoder.layers[0].self_attention.linear_qkv.weight.dtype
        device = self.decoder.layers[0].self_attention.linear_qkv.weight.device
        if isinstance(packed_seq_params, PingPangPackedSeqParams):
            packed_seq_params.seq_params[0].dispatcher_id = 0
            packed_seq_params.seq_params[1].dispatcher_id = 1
            packed_seq_params.seq_params[0].stream = self.decoder.comm_stream
            packed_seq_params.seq_params[1].stream = self.decoder.comm_stream
        with torch.cuda.nvtx.range(f"backward(with_dummy)"):
            for i, layer in enumerate(self.decoder.layers):
                with torch.cuda.nvtx.range(f"backward_layer[{i}]"):
                    dummy_backward(self.config, packed_seq_params, dtype, device)

    def init_ping_pong_communication_ctx(self, device: torch.device):
        self.decoder.init_ping_pong_communication_ctx(device)
