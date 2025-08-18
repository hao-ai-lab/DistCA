"""
Patch the forward method of WLBLLM into TransformerEngine's DotProductAttention.forward().

Copy from: TransformerEngine/transformer_engine/pytorch/attention/dot_product_attention/dot_product_attention.py
Class: DotProductAttention.forward()

"""

from contextlib import nullcontext
import math
import os
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import warnings
import logging

import rich
import torch

import transformer_engine_torch as tex
from transformer_engine.pytorch.utils import get_cudnn_version
from transformer_engine.pytorch.fp8 import get_fp8_te_dtype
from transformer_engine.pytorch.float8_tensor import Float8Tensor
from transformer_engine.pytorch.module.base import TransformerEngineBaseModule
from transformer_engine.pytorch.constants import (
    AttnMaskTypes,
    AttnTypes,
    dist_group_type,
)
from transformer_engine.pytorch.distributed import (
    get_distributed_world_size,
    checkpoint,
    set_all_rng_states,
    CudaRNGStatesTracker,
    graph_safe_rng_available,
)
from transformer_engine.pytorch.jit import no_torch_dynamo
from transformer_engine.pytorch.graph import is_graph_capturing
from transformer_engine.pytorch.attention.inference import InferenceParams

# Import attention utils
import transformer_engine.pytorch.attention.dot_product_attention.utils as dpa_utils
from transformer_engine.pytorch.attention.dot_product_attention.utils import (
    AttentionLogging as attn_log,
)

from transformer_engine.pytorch.attention.dot_product_attention.backends import (
    UnfusedDotProductAttention,
    FusedAttention,
    FlashAttention,
)

from transformer_engine.pytorch.utils import nvtx_range_push, nvtx_range_pop


@no_torch_dynamo(recursive=False)
def forward(
    self,
    query_layer: torch.Tensor,
    key_layer: torch.Tensor,
    value_layer: torch.Tensor,
    attention_mask: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]] = None,
    qkv_format: str = None,
    cu_seqlens_q: torch.Tensor = None,
    cu_seqlens_kv: torch.Tensor = None,
    cu_seqlens_q_padded: torch.Tensor = None,
    cu_seqlens_kv_padded: torch.Tensor = None,
    max_seqlen_q: int = None,
    max_seqlen_kv: int = None,
    attn_mask_type: Optional[str] = None,
    window_size: Optional[Tuple[int, int]] = None,
    checkpoint_core_attention: bool = False,
    core_attention_bias_type: str = "no_bias",
    core_attention_bias: Optional[torch.Tensor] = None,
    alibi_slopes: Optional[torch.Tensor] = None,
    fast_zero_fill: bool = True,
    inference_params: Optional[InferenceParams] = None,
    pad_between_seqs: Optional[bool] = None,
) -> torch.Tensor:
    with self.prepare_forward(
        query_layer,
        num_gemms=3,
        allow_non_contiguous=True,
    ) as query_layer:

        nvtx_range_push("te.DotProductAttention.assert_args")

        # checks for RNG
        if self.rng_states_tracker is not None and is_graph_capturing():
            assert isinstance(
                self.rng_states_tracker, CudaRNGStatesTracker
            ), "Unsupported RNG states tracker."
            assert (
                graph_safe_rng_available()
            ), "Upgrade PyTorch version to get RNG manipulation support for cuda graph capture."

        # checks for FP8
        if self.fp8:
            if self.fp8_meta["recipe"].fp8_mha:
                if not self.fp8_meta["recipe"].fp8_dpa:
                    self.fp8_meta["recipe"].fp8_dpa = True
                    self.logger.warning(
                        """Forcing fp8_meta["recipe"].fp8_dpa=True due to """
                        """fp8_meta["recipe"].fp8_mha=True"""
                    )
        if self.fp8 and self.fp8_meta["recipe"].fp8_dpa:
            forward_dtype = get_fp8_te_dtype(self.fp8_meta["recipe"], fprop_tensor=True)
            backward_dtype = get_fp8_te_dtype(self.fp8_meta["recipe"], fprop_tensor=False)
            assert forward_dtype in [
                tex.DType.kFloat8E4M3,
                tex.DType.kFloat8E5M2,
            ] and backward_dtype in [
                tex.DType.kFloat8E4M3,
                tex.DType.kFloat8E5M2,
            ], """DotProductAttention only supports "E4M3" and "E5M2" FP8 data types."""

        # checks for q/k/v shapes
        assert (
            query_layer.is_cuda and key_layer.is_cuda and value_layer.is_cuda
        ), "DotProductAttention only supports CUDA tensors."
        assert (
            query_layer.dtype == key_layer.dtype and query_layer.dtype == value_layer.dtype
        ), "Queries, keys and values must have the same data type!"
        assert (
            key_layer.shape[:-1] == value_layer.shape[:-1]
        ), "Keys and values must have the same batch size, sequence length and number of heads!"
        num_attention_heads = query_layer.shape[-2]
        num_gqa_groups = key_layer.shape[-2]
        assert (
            query_layer.shape[-1] == key_layer.shape[-1]
        ), "Queries and keys must have the same head dimension!"
        head_dim_qk, head_dim_v = query_layer.shape[-1], value_layer.shape[-1]
        assert (
            head_dim_qk == self.hidden_size_per_attention_head_k
        ), f"Keys have head_dim = {head_dim_qk}, "
        "but expected head_dim = {self.hidden_size_per_attention_head_k}!"
        assert (
            head_dim_v == self.hidden_size_per_attention_head_v
        ), f"Values have head_dim = {head_dim_v}, "
        "but expected head_dim = {self.hidden_size_per_attention_head_v}!"
        assert num_gqa_groups == self.num_gqa_groups_per_partition, (
            "Keys and values must have num_gqa_group ="
            f" {self.num_gqa_groups_per_partition} heads! Found {num_gqa_groups}."
        )

        nvtx_range_pop("te.DotProductAttention.assert_args")


        nvtx_range_push("te.DotProductAttention.check_attn_mask_type")

        # checks for attention mask
        if attn_mask_type is None:
            attn_mask_type = self.attn_mask_type
        else:
            attn_mask_type = attn_mask_type.replace(",", "_")
            if attn_mask_type == "causal_padding":
                attn_mask_type = "padding_causal"
        assert (
            attn_mask_type in AttnMaskTypes
        ), f"Attention mask type {attn_mask_type} is not supported!"

        # checks for sliding window
        if window_size is None:
            window_size = self.window_size
        window_size = dpa_utils.check_set_window_size(attn_mask_type, window_size)

        # checks for qkv_format
        if qkv_format is None:
            qkv_format = self.qkv_format
        assert qkv_format in [
            "sbhd",
            "bshd",
            "thd",
        ], "DotProductAttention only supports qkv_format = {'sbhd', 'bshd', 'thd'}!"
        batch_size = None

        nvtx_range_pop("te.DotProductAttention.check_attn_mask_type")

        nvtx_range_push("te.DotProductAttention.get_seqlen")

        if qkv_format in ["sbhd", "bshd"]:
            assert all(
                len(x.shape) == 4 for x in (query_layer, key_layer, value_layer)
            ), f"Queries, keys and values must be 4D tensors when {qkv_format=}!"
            if qkv_format == "sbhd":
                batch_size = query_layer.shape[1]
                max_seqlen_q = query_layer.shape[0] if max_seqlen_q is None else max_seqlen_q
                max_seqlen_kv = key_layer.shape[0] if max_seqlen_kv is None else max_seqlen_kv
            else:
                batch_size = query_layer.shape[0]
                max_seqlen_q = query_layer.shape[1] if max_seqlen_q is None else max_seqlen_q
                max_seqlen_kv = key_layer.shape[1] if max_seqlen_kv is None else max_seqlen_kv
        if qkv_format == "thd":
            assert all(
                len(x.shape) == 3 for x in (query_layer, key_layer, value_layer)
            ), "Queries, keys and values must be 3D tensors when qkv_format = thd!"
            assert (
                "padding" in attn_mask_type
            ), "Attention mask type must be padding or padding_causal for qkv_format=thd!"
            assert (
                cu_seqlens_q is not None and cu_seqlens_kv is not None
            ), "cu_seqlens_q and cu_seqlens_kv can not be None when qkv_format = thd!"
            assert (
                cu_seqlens_q.shape == cu_seqlens_kv.shape
                and len(cu_seqlens_q.shape) == 1
                and len(cu_seqlens_kv.shape) == 1
            ), "cu_seqlens_q and cu_seqlens_q must both have shape [batch_size + 1]!"
            assert (
                cu_seqlens_q.dtype == torch.int32 and cu_seqlens_kv.dtype == torch.int32
            ), "cu_seqlens_q and cu_seqlens_q must both be in dtype torch.int32!"
            batch_size = len(cu_seqlens_q) - 1
            if max_seqlen_q is None:
                if cu_seqlens_q_padded is not None:
                    seqlens_q = cu_seqlens_q_padded[1:] - cu_seqlens_q_padded[:-1]
                else:
                    seqlens_q = cu_seqlens_q[1:] - cu_seqlens_q[:-1]
                max_seqlen_q = int((seqlens_q.max().item() + 63) // 64 * 64)
            if max_seqlen_kv is None:
                if cu_seqlens_kv_padded is not None:
                    seqlens_kv = cu_seqlens_kv_padded[1:] - cu_seqlens_kv_padded[:-1]
                else:
                    seqlens_kv = cu_seqlens_kv[1:] - cu_seqlens_kv[:-1]
                max_seqlen_kv = int((seqlens_kv.max().item() + 63) // 64 * 64)

        nvtx_range_pop("te.DotProductAttention.get_seqlen")

        nvtx_range_push("te.DotProductAttention.inference")

        # update KV cache and retrieve saved tokens from cache for inference
        if inference_params is not None:
            assert self.layer_number is not None, "Layer number must be set!"

            # convert top-left causal to bottom-right causal due to KV caching
            # users can still use the same attention mask for inference as for training
            assert "padding" in attn_mask_type, "KV caching requires padding mask!"
            if attn_mask_type == "padding_causal":
                attn_mask_type = attn_mask_type + "_bottom_right"

            self.attention_type = "cross"
            self.flash_attention.attention_type = self.attention_type
            self.fused_attention.attention_type = self.attention_type
            self.unfused_attention.attention_type = self.attention_type

            query_layer, key_layer, value_layer = [
                x.contiguous() if not x.is_contiguous() else x
                for x in [query_layer, key_layer, value_layer]
            ]

            # get full K/V tensors from cache and adjust cu_seqlens, qkv_format based on the cache
            (
                key_layer,
                value_layer,
                cu_seqlens_q,
                cu_seqlens_kv,
                max_seqlen_kv,
                qkv_format,
            ) = inference_params.step(
                self.layer_number,
                key_layer,
                value_layer,
                qkv_format,
            )
            cu_seqlens_q_padded = None
            cu_seqlens_kv_padded = None

        # get qkv's memory layout
        if all(isinstance(x, Float8Tensor) for x in [query_layer, key_layer, value_layer]):
            (
                qkv_layout,
                query_layer._data,
                key_layer._data,
                value_layer._data,
                q_format,
                kv_format,
            ) = dpa_utils.get_qkv_layout(
                query_layer._data,
                key_layer._data,
                value_layer._data,
                qkv_format=qkv_format,
                inference_params=inference_params,
            )
        else:
            (
                qkv_layout,
                query_layer,
                key_layer,
                value_layer,
                q_format,
                kv_format,
            ) = dpa_utils.get_qkv_layout(
                query_layer,
                key_layer,
                value_layer,
                qkv_format=qkv_format,
                inference_params=inference_params,
            )

        nvtx_range_pop("te.DotProductAttention.inference")

        nvtx_range_push("te.DotProductAttention.adjust_max_seqlen_and_cu_seqlens")

        # adjust max_seqlen and cu_seqlens for CP
        cp_size = 1
        if isinstance(self.cp_group, dist_group_type):
            cp_size = get_distributed_world_size(self.cp_group)
        elif isinstance(self.cp_group, list):
            for group in self.cp_group:
                cp_size *= get_distributed_world_size(group)
        context_parallel = cp_size > 1
        if q_format in ["sbhd", "bshd"]:
            max_seqlen_q *= cp_size
            if cu_seqlens_q is None:
                if "padding" in attn_mask_type:
                    assert (
                        attention_mask is not None
                    ), "Please provide attention_mask for padding!"
                    if self.attention_type == "self":
                        cu_seqlens_q = dpa_utils.get_cu_seqlens(attention_mask)
                    else:
                        cu_seqlens_q = dpa_utils.get_cu_seqlens(attention_mask[0])
                else:
                    cu_seqlens_q = dpa_utils.get_full_cu_seqlens(
                        batch_size,
                        max_seqlen_q,
                        query_layer.device,
                    )
        if kv_format in ["sbhd", "bshd"]:
            max_seqlen_kv *= cp_size
            if cu_seqlens_kv is None:
                if "padding" in attn_mask_type:
                    assert (
                        attention_mask is not None
                    ), "Please provide attention_mask for padding!"
                    if self.attention_type == "self":
                        cu_seqlens_kv = dpa_utils.get_cu_seqlens(attention_mask)
                    else:
                        cu_seqlens_kv = dpa_utils.get_cu_seqlens(attention_mask[1])
                else:
                    cu_seqlens_kv = dpa_utils.get_full_cu_seqlens(
                        batch_size,
                        max_seqlen_kv,
                        key_layer.device,
                    )

        nvtx_range_pop("te.DotProductAttention.adjust_max_seqlen_and_cu_seqlens")

        nvtx_range_push("te.DotProductAttention.set_alibi_attributes")

        # set ALiBi attributes
        global _alibi_cache
        if alibi_slopes is not None:
            assert (
                core_attention_bias_type == "alibi"
            ), "core_attention_bias_type must be alibi in order to use alibi_slopes!"
            if self.layer_number == 1:
                _alibi_cache["_alibi_slopes_require_update"] = True
                _alibi_cache["_alibi_bias_require_update"] = True
        bottom_right_alignment = (attn_mask_type not in ["causal", "padding_causal"],)
        if core_attention_bias_type == "alibi":
            assert (
                core_attention_bias is None
            ), "core_attention_bias must be None when core_attention_bias_type is alibi!"
            if (
                _alibi_cache["_num_heads"] != query_layer.shape[-2]
                or _alibi_cache["_max_seqlen_q"] != max_seqlen_q
                or _alibi_cache["_max_seqlen_kv"] != max_seqlen_kv
                or _alibi_cache["_bottom_right_alignment"] != bottom_right_alignment
                or _alibi_cache["_alibi_slopes"] is None
            ):
                _alibi_cache["_alibi_slopes_require_update"] = True
                _alibi_cache["_alibi_bias_require_update"] = True

        # detect bias shape
        core_attention_bias_shape = None
        if core_attention_bias is not None:
            if (
                core_attention_bias.shape[0] == batch_size
                and core_attention_bias.shape[1] == query_layer.shape[-2]
            ):
                core_attention_bias_shape = "bhss"
            elif (
                core_attention_bias.shape[0] == 1
                and core_attention_bias.shape[1] == query_layer.shape[-2]
            ):
                core_attention_bias_shape = "1hss"
            elif (
                core_attention_bias.shape[0] == batch_size and core_attention_bias.shape[1] == 1
            ):
                core_attention_bias_shape = "b1ss"
            elif core_attention_bias.shape[0] == 1 and core_attention_bias.shape[1] == 1:
                core_attention_bias_shape = "11ss"
            else:
                assert (
                    False
                ), "core_attention_bias must be in one of {bhss, 1hss, b1ss, 11ss} shapes"

        nvtx_range_pop("te.DotProductAttention.set_alibi_attributes")

        nvtx_range_push("te.DotProductAttention.pad_between_seqs")

        if pad_between_seqs is None:
            if qkv_format == "thd":
                pad_between_seqs = (
                    cu_seqlens_q_padded is not None
                    and not torch.equal(cu_seqlens_q_padded[:-1], cu_seqlens_q[:-1])
                ) or (
                    cu_seqlens_kv_padded is not None
                    and not torch.equal(cu_seqlens_kv_padded[:-1], cu_seqlens_kv[:-1])
                )
            else:
                pad_between_seqs = False

        # gather attention params for get_attention_backend
        attention_params = dpa_utils.AttentionParams(
            qkv_type=type(query_layer),
            qkv_dtype=query_layer.dtype,
            qkv_layout=qkv_layout,
            batch_size=batch_size,
            num_heads=num_attention_heads,
            num_gqa_groups=num_gqa_groups,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_kv=max_seqlen_kv,
            head_dim_qk=head_dim_qk,
            head_dim_v=head_dim_v,
            attn_mask_type=attn_mask_type,
            window_size=window_size,
            alibi_slopes_shape=alibi_slopes.shape if alibi_slopes is not None else None,
            core_attention_bias_type=core_attention_bias_type,
            core_attention_bias_shape=core_attention_bias_shape,
            core_attention_bias_requires_grad=(
                core_attention_bias.requires_grad if core_attention_bias is not None else False
            ),
            pad_between_seqs=pad_between_seqs,
            attention_dropout=self.attention_dropout,
            context_parallel=context_parallel,
            deterministic=self.deterministic,
            is_training=self.training,
            fp8=self.fp8,
            fp8_meta=self.fp8_meta,
            inference_params=inference_params,
        )
        global _attention_backends
        if (
            _attention_backends["attention_params"] is None
            or attention_params != _attention_backends["attention_params"]
        ):
            _attention_backends["attention_params"] = attention_params
            _attention_backends["backend_selection_requires_update"] = True
        if _attention_backends["backend_selection_requires_update"]:
            (
                use_flash_attention,
                flash_attention_backend,
                use_fused_attention,
                fused_attention_backend,
                use_unfused_attention,
                _,
            ) = dpa_utils.get_attention_backend(attention_params)
            # Set global _attention_backends var using return value
            # from get_attention_backend()
            _attention_backends["use_flash_attention"] = use_flash_attention
            _attention_backends["flash_attention_backend"] = flash_attention_backend
            _attention_backends["use_fused_attention"] = use_fused_attention
            _attention_backends["fused_attention_backend"] = fused_attention_backend
            _attention_backends["use_unfused_attention"] = use_unfused_attention
            _attention_backends["backend_selection_requires_update"] = False
            if use_flash_attention:
                self.logger.info(
                    "Running with FlashAttention backend (version %s)",
                    flash_attention_backend,
                )
            elif use_fused_attention:
                self.logger.info(
                    "Running with FusedAttention backend (sub-backend %s)",
                    int(fused_attention_backend),
                )
            elif use_unfused_attention:
                self.logger.info("Running with UnfusedDotProductAttention backend")
        else:
            use_flash_attention = _attention_backends["use_flash_attention"]
            flash_attention_backend = _attention_backends["flash_attention_backend"]
            use_fused_attention = _attention_backends["use_fused_attention"]
            fused_attention_backend = _attention_backends["fused_attention_backend"]
            use_unfused_attention = _attention_backends["use_unfused_attention"]

        # raise exception if no backend is available
        if sum([use_flash_attention, use_fused_attention, use_unfused_attention]) == 0:
            raise ValueError(
                "No dot product attention backend is available for the provided inputs. Please"
                " run with NVTE_DEBUG=1 NVTE_DEBUG_LEVEL=2 to find out the reasons for"
                " disabling all backends."
            )

        nvtx_range_pop("te.DotProductAttention.pad_between_seqs")

        # run attention
        if use_flash_attention:
            nvtx_range_push("te.DotProductAttention.flash_attention.get_alibi")
            if core_attention_bias_type == "alibi":
                alibi_slopes, _ = dpa_utils.get_alibi(
                    _alibi_cache,
                    query_layer.shape[-2],
                    max_seqlen_q,
                    max_seqlen_kv,
                    alibi_slopes=alibi_slopes,
                )
            nvtx_range_pop("te.DotProductAttention.flash_attention.get_alibi")

            # debug_print(f"🔴 Using self.flash_attention: {self.flash_attention}")
            # import inspect
            # debug_print(f"🔴 Using self.flash_attention: {self.flash_attention.forward}")
            # debug_print(f"🔴 self.flash_attention.forward source file: {inspect.getfile(self.flash_attention.forward)}")
            # # debug_print(f"🔴 self.flash_attention.forward source code:")
            # debug_print(inspect.getsource(self.flash_attention.forward))
            
            nvtx_range_push("te.DotProductAttention.flash_attention.forward")
            ret = self.flash_attention(
                query_layer,
                key_layer,
                value_layer,
                attention_mask=attention_mask,
                qkv_layout=qkv_layout,
                cu_seqlens_q=cu_seqlens_q,
                cu_seqlens_kv=cu_seqlens_kv,
                attn_mask_type=attn_mask_type,
                window_size=window_size,
                alibi_slopes=alibi_slopes,
                cp_group=self.cp_group,
                cp_global_ranks=self.cp_global_ranks,
                cp_stream=self.cp_stream,
                cp_comm_type=self.cp_comm_type,
                max_seqlen_q=max_seqlen_q,
                max_seqlen_kv=max_seqlen_kv,
                fp8=self.fp8 and self.fp8_meta["recipe"].fp8_dpa,
                fp8_meta=self.fp8_meta,
                quantizers=self.quantizers,
                inference_params=inference_params,
                flash_attention_backend=flash_attention_backend,
            )
            nvtx_range_pop("te.DotProductAttention.flash_attention.forward")
            return ret

        if use_fused_attention:
            fu_core_attention_bias_type = core_attention_bias_type
            fu_core_attention_bias = core_attention_bias
            if core_attention_bias_type == "alibi" and (
                alibi_slopes is not None or max_seqlen_q != max_seqlen_kv
            ):
                fu_core_attention_bias_type = "post_scale_bias"
                _, fu_core_attention_bias = dpa_utils.get_alibi(
                    _alibi_cache,
                    query_layer.shape[-2],
                    max_seqlen_q,
                    max_seqlen_kv,
                    alibi_slopes=alibi_slopes,
                    bias_dtype=query_layer.dtype,
                    bottom_right_alignment=attn_mask_type not in ["causal", "padding_causal"],
                )
            # checkpoint_core_attention=False
            if checkpoint_core_attention:
                return self._checkpointed_attention_forward(
                    self.fused_attention,
                    query_layer,
                    key_layer,
                    value_layer,
                    qkv_layout=qkv_layout,
                    cu_seqlens_q=cu_seqlens_q,
                    cu_seqlens_kv=cu_seqlens_kv,
                    cu_seqlens_q_padded=cu_seqlens_q_padded,
                    cu_seqlens_kv_padded=cu_seqlens_kv_padded,
                    max_seqlen_q=max_seqlen_q,
                    max_seqlen_kv=max_seqlen_kv,
                    attn_mask_type=attn_mask_type,
                    attention_mask=attention_mask,
                    window_size=window_size,
                    fused_attention_backend=fused_attention_backend,
                    core_attention_bias_type=fu_core_attention_bias_type,
                    core_attention_bias=fu_core_attention_bias,
                    fast_zero_fill=fast_zero_fill,
                    cp_group=self.cp_group,
                    cp_global_ranks=self.cp_global_ranks,
                    cp_stream=self.cp_stream,
                    cp_comm_type=self.cp_comm_type,
                    fp8=self.fp8 and self.fp8_meta["recipe"].fp8_dpa,
                    fp8_meta=self.fp8_meta,
                    quantizers=self.quantizers,
                    pad_between_seqs=pad_between_seqs,
                    inference_params=inference_params,
                )
            return self.fused_attention(
                query_layer,
                key_layer,
                value_layer,
                qkv_layout=qkv_layout,
                cu_seqlens_q=cu_seqlens_q,
                cu_seqlens_kv=cu_seqlens_kv,
                cu_seqlens_q_padded=cu_seqlens_q_padded,
                cu_seqlens_kv_padded=cu_seqlens_kv_padded,
                max_seqlen_q=max_seqlen_q,
                max_seqlen_kv=max_seqlen_kv,
                attn_mask_type=attn_mask_type,
                attention_mask=attention_mask,
                window_size=window_size,
                fused_attention_backend=fused_attention_backend,
                core_attention_bias_type=fu_core_attention_bias_type,
                core_attention_bias=fu_core_attention_bias,
                fast_zero_fill=fast_zero_fill,
                cp_group=self.cp_group,
                cp_global_ranks=self.cp_global_ranks,
                cp_stream=self.cp_stream,
                cp_comm_type=self.cp_comm_type,
                fp8=self.fp8 and self.fp8_meta["recipe"].fp8_dpa,
                fp8_meta=self.fp8_meta,
                quantizers=self.quantizers,
                pad_between_seqs=pad_between_seqs,
                inference_params=inference_params,
            )

        from transformer_engine.pytorch.cpu_offload import CPUOffloadEnabled

        if CPUOffloadEnabled:
            warnings.warn(
                "Attention activation Offloading is only implemented"
                "with Flash Attention and Fused Attention!"
            )

        if use_unfused_attention:
            if checkpoint_core_attention:
                return self._checkpointed_attention_forward(
                    self.unfused_attention,
                    _alibi_cache,
                    query_layer,
                    key_layer,
                    value_layer,
                    qkv_layout=qkv_layout,
                    cu_seqlens_q=cu_seqlens_q,
                    cu_seqlens_kv=cu_seqlens_kv,
                    attn_mask_type=attn_mask_type,
                    attention_mask=attention_mask,
                    window_size=window_size,
                    core_attention_bias_type=core_attention_bias_type,
                    core_attention_bias=core_attention_bias,
                    alibi_slopes=alibi_slopes,
                    inference_params=inference_params,
                )
            return self.unfused_attention(
                _alibi_cache,
                query_layer,
                key_layer,
                value_layer,
                qkv_layout=qkv_layout,
                cu_seqlens_q=cu_seqlens_q,
                cu_seqlens_kv=cu_seqlens_kv,
                attn_mask_type=attn_mask_type,
                attention_mask=attention_mask,
                window_size=window_size,
                core_attention_bias_type=core_attention_bias_type,
                core_attention_bias=core_attention_bias,
                alibi_slopes=alibi_slopes,
                inference_params=inference_params,
            )
        return None

# ------------
# Monkey Patch
# ------------
def monkey_patch():
    import transformer_engine.pytorch.attention.dot_product_attention.dot_product_attention as dpa
    dpa.DotProductAttention.forward = forward
    print("Monkey patching wlbllm.megatron_patch.dot_product_attention.forward() into transformer_engine.pytorch.attention.dot_product_attention.dot_product_attention.DotProductAttention.forward()")