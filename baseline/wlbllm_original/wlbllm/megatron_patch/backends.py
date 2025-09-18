# Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Attention Backends."""
from contextlib import nullcontext
from importlib.metadata import version as get_pkg_version
from importlib.metadata import PackageNotFoundError
import os
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import warnings
import logging
from packaging.version import Version as PkgVersion
import rich
import time

import torch
import transformer_engine_torch as tex
from transformer_engine.pytorch.utils import (
    SplitAlongDim,
    get_device_compute_capability,
    combine_tensors,
    split_tensor_along_dim,
)
from transformer_engine.pytorch.utils import attention_mask_func
from transformer_engine.pytorch.tensor.quantized_tensor import (
    QuantizedTensor,
    prepare_for_saving,
    restore_from_saved,
)
from transformer_engine.pytorch.float8_tensor import Float8Tensor
from transformer_engine.pytorch.constants import (
    TE_DType,
    QKVLayouts,
    dist_group_type,
)
from transformer_engine.pytorch.cpp_extensions.fused_attn import (
    fused_attn_fwd,
    fused_attn_bwd,
    FusedAttnBackend,
    META_O,
    META_QKV,
)
from transformer_engine.pytorch.fp8 import get_fp8_torch_dtype
from transformer_engine.pytorch.distributed import get_distributed_world_size
from transformer_engine.pytorch.jit import no_torch_dynamo
from transformer_engine.pytorch.attention.dot_product_attention.context_parallel import (
    attn_forward_func_with_cp,
)
from transformer_engine.pytorch.attention.dot_product_attention.softmax import FusedScaleMaskSoftmax
from transformer_engine.pytorch.attention.inference import InferenceParams

# Import attention utils
import transformer_engine.pytorch.attention.dot_product_attention.utils as dpa_utils
from transformer_engine.pytorch.attention.dot_product_attention.utils import (
    FlashAttentionUtils as fa_utils,
)
from transformer_engine.pytorch.attention.dot_product_attention.utils import (
    AttentionLogging as attn_log,
)

# Global vars for flash attn v2 and v3 imports
flash_attn_cuda_bwd = None
flash_attn_func = None
flash_attn_varlen_func = None
_flash_attn_fwd = None
_flash_attn_bwd = None
_flash_attn_varlen_fwd = None
_flash_attn_varlen_bwd = None
try:
    fa_utils.version = PkgVersion(get_pkg_version("flash-attn"))
except PackageNotFoundError:
    pass  # only print warning if use_flash_attention_2 = True in get_attention_backend
else:
    if torch.cuda.is_available() and get_device_compute_capability() >= (10, 0):
        if fa_utils.version_required_blackwell <= fa_utils.version <= fa_utils.max_version:
            fa_utils.is_installed = True
    elif fa_utils.version_required <= fa_utils.version <= fa_utils.max_version:
        fa_utils.is_installed = True

    if fa_utils.is_installed:
        from flash_attn_2_cuda import varlen_bwd as flash_attn_cuda_bwd
        from flash_attn.flash_attn_interface import flash_attn_func, flash_attn_varlen_func
        from flash_attn.flash_attn_interface import _flash_attn_forward as _flash_attn_fwd
        from flash_attn.flash_attn_interface import _flash_attn_backward as _flash_attn_bwd
        from flash_attn.flash_attn_interface import (
            _flash_attn_varlen_forward as _flash_attn_varlen_fwd,
        )
        from flash_attn.flash_attn_interface import (
            _flash_attn_varlen_backward as _flash_attn_varlen_bwd,
        )

        # Setup Flash attention utils
        fa_utils.set_flash_attention_version()
    elif (
        torch.cuda.is_available()
        and get_device_compute_capability() >= (8, 0)
        and dpa_utils._NVTE_FLASH_ATTN
    ):
        attn_log.fa_logger.warning(
            "Supported flash-attn versions are %s. Found flash-attn %s.",
            dpa_utils._get_supported_versions(
                (
                    fa_utils.version_required
                    if get_device_compute_capability() < (10, 0)
                    else fa_utils.version_required_blackwell
                ),
                fa_utils.max_version,
            ),
            fa_utils.version,
        )
try:
    fa_utils.fa3_version = PkgVersion(get_pkg_version("flash-attn-3"))
except PackageNotFoundError:
    flash_attn_func_v3 = None
    flash_attn_varlen_func_v3 = None
    flash_attn_with_kvcache_v3 = None
    # pass  # only print warning if use_flash_attention_3 = True in get_attention_backend
else:
    from flash_attn_3.flash_attn_interface import flash_attn_func as flash_attn_func_v3
    from flash_attn_3.flash_attn_interface import (
        flash_attn_varlen_func as flash_attn_varlen_func_v3,
    )
    from flash_attn_3.flash_attn_interface import (
        flash_attn_with_kvcache as flash_attn_with_kvcache_v3,
    )
    from flash_attn_3.flash_attn_interface import _flash_attn_forward as _flash_attn_fwd_v3
    from flash_attn_3.flash_attn_interface import _flash_attn_backward as _flash_attn_bwd_v3

    fa_utils.set_flash_attention_3_params()


from transformer_engine.pytorch.utils import nvtx_range_push, nvtx_range_pop

attention_durations = []

def debug_print(*args, **kwargs):
    if os.getenv("D2_DEBUG_PRINT", "0") == "1":
        if torch.distributed.is_initialized():
            rank = torch.distributed.get_rank()
            rich.print(f"[Rank {rank}]", *args, **kwargs)
    return


# class FlashAttention(torch.nn.Module):
def forward(
    self,
    query_layer: torch.Tensor,
    key_layer: torch.Tensor,
    value_layer: torch.Tensor,
    attention_mask: Optional[Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]] = None,
    qkv_layout: str = "sbh3d",
    cu_seqlens_q: Optional[torch.Tensor] = None,
    cu_seqlens_kv: Optional[torch.Tensor] = None,
    max_seqlen_q: Optional[int] = None,
    max_seqlen_kv: Optional[int] = None,
    attn_mask_type: str = "causal",
    window_size: Optional[Tuple[int, int]] = None,
    alibi_slopes: Optional[torch.Tensor] = None,
    cp_group: Optional[Union[dist_group_type, List[dist_group_type]]] = None,
    cp_global_ranks: List[int] = None,
    cp_stream: torch.cuda.Stream = None,
    cp_comm_type: str = "p2p",
    fp8: bool = False,
    fp8_meta: Optional[Dict[str, Any]] = None,
    quantizers=None,
    inference_params: Optional[InferenceParams] = None,
    flash_attention_backend: Optional[PkgVersion] = PkgVersion("0"),
) -> torch.Tensor:
    """flash-attn fprop"""

    # debug_print("💜 Inside FlashAttention.forward")
    nvtx_range_push("te.FlashAttention.forward")

    nvtx_range_push("te.FlashAttention.asserts")


    assert all(
        x.dtype in [torch.float16, torch.bfloat16] or isinstance(x, Float8Tensor)
        for x in [query_layer, key_layer, value_layer]
    ), "FlashAttention only supports FP16 and BF16 data types, or Float8Tensors."
    assert (
        query_layer.is_cuda and key_layer.is_cuda and value_layer.is_cuda
    ), "FlashAttention currently only supports CUDA tensors."
    assert (
        qkv_layout in QKVLayouts
    ), f"FlashAttention does not support qkv_layout = {qkv_layout}!"

    cp_size = 1
    if isinstance(cp_group, dist_group_type):
        cp_size = get_distributed_world_size(cp_group)
    elif isinstance(cp_group, list):
        for group in cp_group:
            cp_size *= get_distributed_world_size(group)
    context_parallel = cp_size > 1

    # get q_format and kv_format for training and inference
    qkv_format, q_format, kv_format = dpa_utils.get_qkv_format(qkv_layout, inference_params)

    # convert q, k, v to bshd if they are in sbhd; qkv_format doesn't change
    if all(not isinstance(x, Float8Tensor) for x in [query_layer, key_layer, value_layer]):
        if qkv_format == "sbhd":
            # For now just 128, will make it more general in the future
            if (
                query_layer.shape[-1] == 128
                and query_layer.shape[0] * query_layer.shape[1] >= 512
                and qkv_layout == "sbh3d"
            ):
                query_layer, key_layer, value_layer = _PrepareQKVForFA.apply(
                    query_layer, key_layer, value_layer
                )
            else:
                query_layer, key_layer, value_layer = [
                    x.transpose(0, 1).contiguous()
                    for x in (query_layer, key_layer, value_layer)
                ]
        elif q_format == "sbhd" and kv_format == "bshd":
            query_layer = query_layer.transpose(0, 1).contiguous()
        if context_parallel:
            query_layer, key_layer, value_layer = [
                x.contiguous() for x in (query_layer, key_layer, value_layer)
            ]
    else:
        if qkv_format == "sbhd":
            query_layer._data, key_layer._data, value_layer._data = [
                x.transpose(0, 1).contiguous()
                for x in (query_layer._data, key_layer._data, value_layer._data)
            ]
            query_layer, key_layer, value_layer = [
                Float8Tensor.make_like(x, data=x._data, shape=x._data.shape)
                for x in (query_layer, key_layer, value_layer)
            ]
        elif q_format == "sbhd" and kv_format == "bshd":
            query_layer._data = query_layer._data.transpose(0, 1).contiguous()
            query_layer = Float8Tensor.make_like(
                query_layer, data=query_layer._data, shape=query_layer._data.shape
            )
        if context_parallel:
            query_layer._data, key_layer._data, value_layer._data = [
                x.contiguous() for x in (query_layer._data, key_layer._data, value_layer._data)
            ]

    nvtx_range_pop()

    
    nvtx_range_push("te.FlashAttention.get_cu_lens")
    # get batch_size, max_seqlen and cu_seqlens
    batch_size, context_len = None, None
    if inference_params is None:
        if qkv_format in ["sbhd", "bshd"]:
            batch_size = query_layer.shape[0]
            max_seqlen_q, max_seqlen_kv = query_layer.shape[1], key_layer.shape[1]
            max_seqlen_q *= cp_size
            max_seqlen_kv *= cp_size

            if "padding" in attn_mask_type:
                assert (
                    not context_parallel
                ), "Padding mask not supported with context parallelism!"

                # [b * s, h, d]
                query_layer, key_layer, value_layer = [
                    x.reshape(x.shape[0] * x.shape[1], *x.shape[2:])
                    for x in [query_layer, key_layer, value_layer]
                ]

                if self.attention_type == "self":
                    assert (
                        max_seqlen_q == max_seqlen_kv
                    ), "Maximum sequence length for Q and KV should be the same."
                    if cu_seqlens_q is None:
                        assert (
                            attention_mask is not None
                        ), "Please provide attention_mask for padding!"
                        cu_seqlens_q, indices_q = dpa_utils.get_cu_seqlens_and_indices(
                            attention_mask
                        )
                    else:
                        indices_q = dpa_utils.get_indices(max_seqlen_q, cu_seqlens_q)
                    cu_seqlens_kv = cu_seqlens_q
                    query_layer, key_layer, value_layer = dpa_utils.PackTensors.apply(
                        indices_q, query_layer, key_layer, value_layer
                    )
                else:
                    if cu_seqlens_q is None or cu_seqlens_kv is None:
                        assert (
                            attention_mask is not None
                        ), "Please provide attention_mask for padding!"
                        cu_seqlens_q, indices_q = dpa_utils.get_cu_seqlens_and_indices(
                            attention_mask[0]
                        )
                        cu_seqlens_kv, indices_kv = dpa_utils.get_cu_seqlens_and_indices(
                            attention_mask[1]
                        )
                    else:
                        indices_q = dpa_utils.get_indices(max_seqlen_q, cu_seqlens_q)
                        indices_kv = dpa_utils.get_indices(max_seqlen_kv, cu_seqlens_kv)
                    query_layer = dpa_utils.PackTensors.apply(indices_q, query_layer)
                    key_layer, value_layer = dpa_utils.PackTensors.apply(
                        indices_kv, key_layer, value_layer
                    )
            else:
                # Cumulative sequence lengths for unpadded data
                if cu_seqlens_q is None:
                    cu_seqlens_q = dpa_utils.get_full_cu_seqlens(
                        batch_size,
                        max_seqlen_q,
                        query_layer.device,
                    )
                if cu_seqlens_kv is None:
                    cu_seqlens_kv = dpa_utils.get_full_cu_seqlens(
                        batch_size,
                        max_seqlen_kv,
                        key_layer.device,
                    )
        elif qkv_format == "thd":
            assert (
                cu_seqlens_q is not None and cu_seqlens_kv is not None
            ), "cu_seqlens_q and cu_seqlens_kv can not be None when qkv_format = thd!"
            if max_seqlen_q is None:
                seqlens_q = cu_seqlens_q[1:] - cu_seqlens_q[:-1]
                max_seqlen_q = seqlens_q.max().item()
            if max_seqlen_kv is None:
                seqlens_kv = cu_seqlens_kv[1:] - cu_seqlens_kv[:-1]
                max_seqlen_kv = seqlens_kv.max().item()
    else:
        if qkv_format in ["sbhd_2bshd", "bshd"]:
            # q is in bshd in both cases from conversion above or the original input
            batch_size, context_len = query_layer.shape[:2]
            cu_seqlens_q = cu_seqlens_q[: batch_size + 1]
            cu_seqlens_kv = cu_seqlens_kv[: batch_size + 1]
            # convert from bshd to thd_2bshd for flash_attn_varlen_func/_with_kvcache;
            # kernel assumes tensor is contiguous
            if isinstance(query_layer, Float8Tensor):
                query_layer._data = tex.convert_bshd_to_thd(
                    query_layer._data,
                    cu_seqlens_q,
                    batch_size * context_len,
                )
                query_layer = Float8Tensor.make_like(
                    query_layer, data=query_layer._data, shape=query_layer._data.shape
                )
            else:
                query_layer = tex.convert_bshd_to_thd(
                    query_layer,
                    cu_seqlens_q,
                    batch_size * context_len,
                )


    nvtx_range_pop()


    nvtx_range_push("te.FlashAttention.run_attention")
    use_flash_attn_3 = False
    if flash_attention_backend is not None and flash_attention_backend > PkgVersion("3.0.0b"):
        use_flash_attn_3 = True
    if context_parallel and all(
        not isinstance(x, Float8Tensor) for x in [query_layer, key_layer, value_layer]
    ):
        assert (
            alibi_slopes is None
        ), "Alibi slope bias addition is not supported with context parallelism."
        with self.attention_dropout_ctx():
            # debug_print("💜 Inside FlashAttention.forward: attn_forward_func_with_cp")
            output = attn_forward_func_with_cp(
                self.training,
                query_layer,
                key_layer,
                value_layer,
                cu_seqlens_q,
                cu_seqlens_kv,
                max_seqlen_q,
                max_seqlen_kv,
                cu_seqlens_q if qkv_format == "thd" else None,
                cu_seqlens_kv if qkv_format == "thd" else None,
                self.attention_dropout if self.training else 0.0,
                cp_group,
                cp_global_ranks,
                cp_stream,
                cp_comm_type,
                softmax_scale=self.softmax_scale,
                qkv_format="bshd" if qkv_format == "sbhd" else qkv_format,
                attn_mask_type=attn_mask_type,
                deterministic=self.deterministic,
                window_size=window_size,
                quantizers=quantizers,
                pad_between_seqs=False,
                use_flash_attn_3=use_flash_attn_3,
            )
    else:
        from transformer_engine.pytorch.cpu_offload import (
            CPUOffloadEnabled,
            mark_activation_offload,
        )

        if CPUOffloadEnabled:
            mark_activation_offload(
                query_layer, key_layer, value_layer, cu_seqlens_q, cu_seqlens_kv
            )

        with self.attention_dropout_ctx():
            #       | API                     | use cases
            # ----------------------------------------------------------------------
            # FA v2 | flash_attn_func         | bshd/sbhd + not padding
            #       | flash_attn_varlen_func  | bshd/sbhd + padding
            #       |                         | thd + padding
            #       |                         | KV cache (not-paged/paged), i.e.
            #       |                         |     bshd/sbhd/thd + padding
            # FA v3 | flash_attn_func         | bshd/sbhd + not padding
            #       | flash_attn_varlen_func  | bshd/sbhd + padding
            #       |                         | thd + padding
            #       | flash_attn_with_kvcache | KV cache (not-paged/paged), i.e.
            #       |                         |     bshd/sbhd/thd + padding
            fa_optional_forward_args_thd = []
            if qkv_format in ["bshd", "sbhd"] and "padding" not in attn_mask_type:
                func = (
                    flash_attn_func if not use_flash_attn_3 else flash_attn_func_v3
                )  # pylint: disable=possibly-used-before-assignment
            else:
                if not use_flash_attn_3:
                    # debug_print("💜 Inside FlashAttention.forward: using func = flash_attn_varlen_func")
                    func = flash_attn_varlen_func
                elif inference_params is None:
                    # debug_print("💜 Inside FlashAttention.forward: using func = flash_attn_varlen_func_v3")
                    func = flash_attn_varlen_func_v3  # pylint: disable=possibly-used-before-assignment
                else:
                    # debug_print("💜 Inside FlashAttention.forward: using func = flash_attn_with_kvcache_v3")
                    func = flash_attn_with_kvcache_v3  # pylint: disable=possibly-used-before-assignment
                if not use_flash_attn_3 or inference_params is None:
                    fa_optional_forward_args_thd.append(cu_seqlens_q)
                    fa_optional_forward_args_thd.append(cu_seqlens_kv)
                    fa_optional_forward_args_thd.append(max_seqlen_q)
                    fa_optional_forward_args_thd.append(max_seqlen_kv)


            if getattr(self, "rank", None) is None:
                self.rank = torch.distributed.get_rank()


            my_func = func
            global attention_durations
            def baseline_func(*args, **kwargs):
                # debug_print(f"🤍 Calling inside FlashAttention: {func}.")
                # debug_print(f"args[0] = ", args[0].shape)
                # debug_print(f"args[1] = ", args[1].shape)
                # debug_print(f"args[2] = ", args[2].shape)
                # debug_print("args[3:]:", args[3:])
                # debug_print("kwargs:", kwargs)
                nvtx_range_push("te.FlashAttention.run_attention.func")
                torch.cuda.synchronize()
                start_time = time.time()
                r = my_func(*args, **kwargs)
                nvtx_range_pop()
                torch.cuda.synchronize()
                end_time = time.time()
                duration_in_ms = (end_time - start_time) * 1000
                if self.rank % 8 == 0:
                    print(f"💜 [Rank {self.rank}] FlashAttention.forward: {duration_in_ms} ms")
                    attention_durations.append(duration_in_ms)
                return r

            from wlbllm.megatron_patch.te_flash_attn import wlbllm_func
            
            # print(f"🟢 Calling inside the patched version of FlashAttention: {func}.")
            is_wlbllm_mode = (os.environ["WLBLLM_MODE"] == "1")
            if is_wlbllm_mode:
                def wlbllm_func_add_metadata(*args, **kwargs):
                    # metadata = ...
                    r = wlbllm_func(*args, **kwargs)
                    return r
                func = wlbllm_func_add_metadata
            else:
                func = baseline_func
            
            
            if not use_flash_attn_3:
                fa_optional_forward_kwargs = {}
                if fa_utils.v2_3_plus:
                    fa_optional_forward_kwargs["window_size"] = window_size
                if fa_utils.v2_4_plus:
                    fa_optional_forward_kwargs["alibi_slopes"] = alibi_slopes
                if fa_utils.v2_4_1_plus:
                    fa_optional_forward_kwargs["deterministic"] = self.deterministic
                if inference_params is not None:
                    # use block_table kwarg to support thd_2bshd for non-paged
                    fa_optional_forward_kwargs["block_table"] = (
                        inference_params.cache_manager.page_table[:batch_size]
                        if inference_params.is_paged
                        else inference_params.cache_manager.batch_indices_post_step.unsqueeze(
                            1
                        )[:batch_size]
                    )
                # debug_print(f"💜 Inside FlashAttention.forward: using func = {func}")
                output = func(
                    query_layer,
                    key_layer,
                    value_layer,
                    *fa_optional_forward_args_thd,
                    self.attention_dropout if self.training else 0.0,
                    softmax_scale=self.softmax_scale,
                    causal="causal" in attn_mask_type,
                    **fa_optional_forward_kwargs,
                )
            else:
                fa_3_optional_forward_kwargs = {}
                fa_3_optional_forward_kwargs["window_size"] = window_size
                if inference_params is None:
                    fa_3_optional_forward_kwargs["deterministic"] = self.deterministic
                else:
                    fa_3_optional_forward_kwargs["cu_seqlens_q"] = cu_seqlens_q
                    fa_3_optional_forward_kwargs["max_seqlen_q"] = max_seqlen_q
                    cache_seqlens = cu_seqlens_kv[1:] - cu_seqlens_kv[:-1]
                    fa_3_optional_forward_kwargs["cache_seqlens"] = cache_seqlens
                    # flash_attn_with_kvcache accepts thd_2bshd for non-paged
                    if inference_params.is_paged:
                        fa_3_optional_forward_kwargs["page_table"] = (
                            inference_params.cache_manager.page_table[:batch_size]
                        )
                if fp8:
                    QKV_quantizer = quantizers["scaling_fwd"][META_QKV]
                    torch_dtype = get_fp8_torch_dtype(fp8_meta["recipe"], fprop_tensor=True)
                    torch_orig_dtype = query_layer.dtype

                    def convert_to_torch_float8(tensor, dtype):
                        out = torch.Tensor().to(device=tensor.device, dtype=dtype)
                        out.set_(
                            tensor._data.untyped_storage(),
                            tensor._data.storage_offset(),
                            tensor._data.shape,
                            tensor._data.stride(),
                        )
                        return out

                    # "fp8_mha" decides outputs in fp8, while inputs are inferred from
                    # the real dtype
                    assert isinstance(key_layer, query_layer.__class__) and isinstance(
                        value_layer, query_layer.__class__
                    ), "q, k, and v must have the same type."
                    if not isinstance(query_layer, Float8Tensor):
                        query_layer, key_layer, value_layer = (
                            QKV_quantizer(x) for x in [query_layer, key_layer, value_layer]
                        )
                    batch_size = cu_seqlens_q.shape[0] - 1
                    num_heads_k = key_layer.shape[-2]
                    fa_3_optional_forward_kwargs["q_descale"] = (
                        query_layer._scale_inv.unsqueeze(0).repeat(batch_size, num_heads_k)
                    )
                    fa_3_optional_forward_kwargs["k_descale"] = key_layer._scale_inv.unsqueeze(
                        0
                    ).repeat(batch_size, num_heads_k)
                    fa_3_optional_forward_kwargs["v_descale"] = (
                        value_layer._scale_inv.unsqueeze(0).repeat(batch_size, num_heads_k)
                    )
                    query_layer, key_layer, value_layer = (
                        convert_to_torch_float8(x, torch_dtype)
                        for x in [query_layer, key_layer, value_layer]
                    )
                try:
                    # debug_print(f"💜 Inside FlashAttention.forward: using func = {func}")
                    output = func(
                        query_layer,
                        key_layer,
                        value_layer,
                        *fa_optional_forward_args_thd,
                        softmax_scale=self.softmax_scale,
                        causal="causal" in attn_mask_type,
                        **fa_3_optional_forward_kwargs,
                    )
                    if isinstance(output, (List, Tuple)):
                        output = output[0]
                except TypeError as e:
                    if fa_utils.v3_0_0_beta:
                        e.args = (
                            e.args[0]
                            + ". Please update your flash-attn v3 (beta) installation as it "
                            + "may have added more supported arguments to its API. \n"
                            + fa_utils.v3_installation_steps,
                        ) + e.args[1:]
                    raise

                if fp8:
                    output = output.to(dtype=torch_orig_dtype)
                if fp8 and fp8_meta["recipe"].fp8_mha:
                    O_quantizer = quantizers["scaling_fwd"][META_O]
                    output = O_quantizer(output)

    nvtx_range_pop()


    nvtx_range_push("te.FlashAttention.convert_output_layout")
    if inference_params is None:
        if qkv_format in ["sbhd", "bshd"] and "padding" in attn_mask_type:
            output = dpa_utils.UnpackTensor.apply(indices_q, batch_size * max_seqlen_q, output)
    elif qkv_format in ["bshd", "sbhd_2bshd"]:
        # all KV caching cases use thd_2bshd for calculation
        # convert results back to bshd from thd_2bshd
        if isinstance(query_layer, Float8Tensor):
            output._data = tex.convert_thd_to_bshd(
                output._data,
                cu_seqlens_q,
                batch_size,
                context_len,
            )
            output = Float8Tensor.make_like(output, data=output._data, shape=output._data.shape)
        else:
            output = tex.convert_thd_to_bshd(
                output,
                cu_seqlens_q,
                batch_size,
                context_len,
            )

    if q_format == "sbhd":
        # (bs)hd -> bs(hd) -> sb(hd)
        if fp8 and fp8_meta["recipe"].fp8_mha:
            output_data = (
                output._data.reshape(batch_size, max_seqlen_q // cp_size, -1)
                .transpose(0, 1)
                .contiguous()
            )
            output = Float8Tensor.make_like(
                output,
                data=output_data,
                shape=output_data.shape,
            )
        else:
            output = output.view(batch_size, max_seqlen_q // cp_size, -1).transpose(0, 1)
    elif q_format == "bshd":
        # (bs)hd -> bs(hd)
        output = output.reshape(batch_size, max_seqlen_q // cp_size, -1)
    elif q_format == "thd":
        # thd -> t(hd)
        output = output.reshape(output.shape[0], -1)

    nvtx_range_pop()
    
    nvtx_range_pop()
    return output.contiguous()


# ------------
# Monkey Patch
# ------------
def monkey_patch():
    import transformer_engine.pytorch.attention.dot_product_attention.backends as tex_backends
    tex_backends.FlashAttention.forward = forward

    print("Monkey patching wlbllm.megatron_patch.backends.forward() into transformer_engine.pytorch.attention.dot_product_attention.backends.FlashAttention.forward()")

from contextlib import contextmanager
@contextmanager
def monkey_patch_context():
    import transformer_engine.pytorch.attention.dot_product_attention.backends as tex_backends
    old_forward = tex_backends.FlashAttention.forward
    tex_backends.FlashAttention.forward = forward
    print("Monkey patching wlbllm.megatron_patch.backends.forward() into transformer_engine.pytorch.attention.dot_product_attention.backends.FlashAttention.forward()")
    yield
    tex_backends.FlashAttention.forward = old_forward
    print("Unmonkey patching wlbllm.megatron_patch.backends.forward() from transformer_engine.pytorch.attention.dot_product_attention.backends.FlashAttention.forward()")