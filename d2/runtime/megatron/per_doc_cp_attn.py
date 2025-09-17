from megatron.core.extensions.transformer_engine import TEDotProductAttention
import torch
import transformer_engine.pytorch.attention.dot_product_attention.utils as dpa_utils

"""
Copied from transformer_engine.pytorch.attention.dot_product_attention.utils.
This is to monkey patch the get_attention_backend function to enable the per doc CP case.
"""
import os
import logging

from dataclasses import fields
from packaging.version import Version as PkgVersion

import torch
import transformer_engine as te
from transformer_engine.pytorch.cpp_extensions.fused_attn import (
    QKVLayout,
    AttnBiasType,
    AttnMaskType,
    FusedAttnBackend,
)
from transformer_engine.pytorch.float8_tensor import Float8Tensor
from transformer_engine.pytorch.fp8 import get_fp8_te_dtype
from transformer_engine.pytorch.constants import TE_DType


from transformer_engine.pytorch.utils import (
    get_device_compute_capability,
    get_cudnn_version,
)
from transformer_engine.pytorch.attention.dot_product_attention.utils import (
    AttentionLogging, AttentionParams, FlashAttentionUtils,
    check_set_window_size, get_qkv_format, tex, _NVTE_FLASH_ATTN, _get_supported_versions
)

__already_called_get_attention_backend = False
__ret_value = None
def get_attention_backend(attention_params):
    global __already_called_get_attention_backend
    global __ret_value

    # if __already_called_get_attention_backend:
    #     pass
    if os.getenv("EXPERIMENT_D2_FLASH_ATTN_SKIP_GET_BACKEND", "1") == "1" and __already_called_get_attention_backend:
        return __ret_value

    
    # NOTE: As part of refactoring attention.py, populating the _attention_backends cache in attention
    # is no longer performed at the end of get_attention_backend(), but the responsibility of doing so
    # is shifted over to the caller of this function
    qkv_type = attention_params.qkv_type
    qkv_dtype = attention_params.qkv_dtype
    qkv_layout = attention_params.qkv_layout
    batch_size = attention_params.batch_size
    num_heads = attention_params.num_heads
    num_gqa_groups = attention_params.num_gqa_groups
    max_seqlen_q = attention_params.max_seqlen_q
    max_seqlen_kv = attention_params.max_seqlen_kv
    head_dim_qk = attention_params.head_dim_qk
    head_dim_v = attention_params.head_dim_v
    attn_mask_type = attention_params.attn_mask_type
    window_size = attention_params.window_size
    alibi_slopes_shape = attention_params.alibi_slopes_shape
    core_attention_bias_type = attention_params.core_attention_bias_type
    core_attention_bias_shape = attention_params.core_attention_bias_shape
    core_attention_bias_requires_grad = attention_params.core_attention_bias_requires_grad
    pad_between_seqs = attention_params.pad_between_seqs
    attention_dropout = attention_params.attention_dropout
    context_parallel = attention_params.context_parallel
    deterministic = attention_params.deterministic
    is_training = attention_params.is_training
    fp8 = attention_params.fp8
    fp8_meta = attention_params.fp8_meta
    inference_params = attention_params.inference_params

    # Run config
    logger = logging.getLogger("DotProductAttention")
    logger.setLevel(AttentionLogging._log_level)
    if not logger.hasHandlers():
        logger.addHandler(AttentionLogging._stream_handler)
    device_compute_capability = get_device_compute_capability()
    cudnn_version = get_cudnn_version()
    run_config = {
        "transformer_engine_version": te.__version__,
        "compute_capability": "sm"
        + str(10 * device_compute_capability[0] + device_compute_capability[1]),
        "flash_attn_version": (
            str(FlashAttentionUtils.version)
            if FlashAttentionUtils.is_installed
            else "not installed"
        ),
        "flash_attn_3_version": (
            str(FlashAttentionUtils.fa3_version)
            if FlashAttentionUtils.v3_is_installed
            else "not installed"
        ),
        "cudnn_version": ".".join([str(i) for i in cudnn_version]),
    }
    attention_params_dict = {
        field.name: getattr(attention_params, field.name) for field in fields(attention_params)
    }
    run_config.update(attention_params_dict)
    if fp8:
        run_config["NVTE_FP8_DPA_BWD"] = int(os.getenv("NVTE_FP8_DPA_BWD", "1"))
    logger.debug("Running with config=%s", run_config)

    # The following sections check if `FlashAttention` supports the provided attention params,
    # regardless of whether FA2 or FA3 is installed. If FA2 or FA3 is not installed but is
    # necessary for performance/functionality, a warning will be issued to prompt users to
    # install an appropriate FA version.
    qkv_format, q_format, _ = get_qkv_format(qkv_layout, inference_params)

    # Filter: Environment variables
    use_flash_attention = int(os.getenv("NVTE_FLASH_ATTN", "1"))
    use_flash_attention_2 = use_flash_attention
    use_flash_attention_3 = use_flash_attention
    flash_attention_backend = None
    use_fused_attention = int(os.getenv("NVTE_FUSED_ATTN", "1"))
    use_unfused_attention = int(os.getenv("NVTE_UNFUSED_ATTN", "1"))
    if not use_flash_attention_2 and FlashAttentionUtils.is_installed:
        logger.debug("Disabling FlashAttention 2 due to NVTE_FLASH_ATTN=0")
    if not use_flash_attention_3 and FlashAttentionUtils.v3_is_installed:
        logger.debug("Disabling FlashAttention 3 due to NVTE_FLASH_ATTN=0")
    if not use_fused_attention:
        logger.debug("Disabling FusedAttention due to NVTE_FUSED_ATTN=0")
    if not use_unfused_attention:
        logger.debug("Disabling UnfusedDotProductAttention due to NVTE_UNFUSED_ATTN=0")

    # Filter: Compute capability
    if device_compute_capability < (8, 0):
        if use_flash_attention_2 and FlashAttentionUtils.is_installed:
            logger.debug("Disabling FlashAttention 2 for compute capability < sm80")
        use_flash_attention_2 = False
        if use_fused_attention:
            logger.debug("Disabling FusedAttention for compute capability < sm80")
            use_fused_attention = False
    if device_compute_capability != (9, 0):
        if use_flash_attention_3 and FlashAttentionUtils.v3_is_installed:
            logger.debug("Disabling FlashAttention 3 for compute capability != sm90")
        use_flash_attention_3 = False

    # Filter: Data type
    if qkv_dtype not in [torch.bfloat16, torch.float16]:
        if use_flash_attention_2 and FlashAttentionUtils.is_installed:
            logger.debug(
                "Disabling FlashAttention 2 for unsupported qkv_dtype = %s. "
                "Supported: qkv_dtype = {torch.bfloat16, torch.float16}. ",
                qkv_dtype,
            )
        use_flash_attention_2 = False
    if qkv_dtype not in [torch.bfloat16, torch.float16, torch.float8_e4m3fn] or qkv_type not in [
        torch.Tensor,
        Float8Tensor,
    ]:
        if use_flash_attention_3 and FlashAttentionUtils.v3_is_installed:
            logger.debug(
                "Disabling FlashAttention 3 for unsupported qkv_dtype = %s, qkv_type = %s. "
                "Supported: qkv_dtype = {torch.bfloat16, torch.float16, torch.float8_e4m3fn}, "
                "qkv_type = {torch.Tensor, Float8Tensor}. ",
                qkv_dtype,
                qkv_type,
            )
        use_flash_attention_3 = False
        if use_fused_attention:
            logger.debug(
                "Disabling FusedAttention for unsupported qkv_dtype = %s, qkv_type = %s. "
                "Supported: qkv_dtype = {torch.bfloat16, torch.float16, torch.float8_e4m3fn}, "
                "qkv_type = {torch.Tensor, Float8Tensor}. ",
                qkv_dtype,
                qkv_type,
            )
            use_fused_attention = False

    # Filter: Execution type
    if fp8 and fp8_meta["recipe"].fp8_dpa:
        if use_flash_attention_2 and FlashAttentionUtils.is_installed:
            logger.debug("Disabling FlashAttention 2 for FP8 attention")
        use_flash_attention_2 = False
        if use_flash_attention_3 and is_training:
            if FlashAttentionUtils.v3_is_installed:
                logger.debug("Disabling FlashAttention 3 for FP8 training")
            use_flash_attention_3 = False
        if use_unfused_attention:
            logger.debug("Disabling UnfusedDotProductAttention for FP8 attention")
            use_unfused_attention = False

    # Filter: KV cache
    # backend  | precision      |    KV cache     | architecture | qkv_format    | page_size
    # ---------------------------------------------------------------------------------------
    # Fused    | FP16/BF16      | non-paged/paged | sm80+        | bshd,sbhd,thd | >= 1
    # Flash v2 | FP16/BF16      | non-paged/paged | sm80+        | bshd,sbhd,thd | >= 256
    # Flash v3 | FP16/BF16      | non-paged/paged | sm90         | bshd,sbhd,thd | >= 1
    #          | FP8            | non-paged/paged | sm90         | thd           | >= 1
    # Unfused  | FP32/FP16/BF16 | non-paged/paged | all          | bshd,sbhd,thd | >= 1
    if inference_params is not None:
        if device_compute_capability == (8, 9) and cudnn_version < (9, 11, 0):
            logger.debug("Disabling FusedAttention for KV caching for sm89 and cuDNN < 9.11")
            use_fused_attention = False
        if context_parallel:
            logger.debug("Disabling all backends for KV caching with context parallelism")
            use_flash_attention = False
            use_fused_attention = False
            use_unfused_attention = False
        if fp8 and fp8_meta["recipe"].fp8_dpa:
            if fp8_meta["recipe"].fp8_mha:
                logger.debug("Disabling all backends for KV caching with FP8 MHA")
                use_flash_attention = False
                use_fused_attention = False
                use_unfused_attention = False
            if use_flash_attention_3 and q_format != "thd":
                if FlashAttentionUtils.v3_is_installed:
                    logger.debug("Disabling FlashAttention 3 for FP8 KV caching and non-THD")
                use_flash_attention_3 = False
            if use_fused_attention:
                logger.debug("Disabling FusedAttention for FP8 KV caching")
                use_fused_attention = False
        else:
            if q_format == "thd" and pad_between_seqs:
                logger.debug("Disabling all backends for pad_between_seqs = True and KV caching")
                use_flash_attention = False
                use_fused_attention = False
                use_unfused_attention = False
        if inference_params.is_paged:
            if use_flash_attention_2 and inference_params.page_size < 256:
                if FlashAttentionUtils.is_installed:
                    logger.debug("Disabling FlashAttention 2 for page size < 256")
                use_flash_attention_2 = False
            if use_flash_attention_2:
                if not FlashAttentionUtils.is_installed:
                    FlashAttentionUtils.version_required = PkgVersion("2.5")
                elif not FlashAttentionUtils.v2_5_plus:
                    logger.debug(
                        "Disabling FlashAttention 2 as paged attention requires flash-attn 2.5+"
                    )
                    use_flash_attention_2 = False

    # Filter: Head dimension
    if head_dim_qk != head_dim_v:
        if (use_flash_attention_2 and FlashAttentionUtils.is_installed) or (
            use_flash_attention_3 and FlashAttentionUtils.v3_is_installed
        ):
            logger.debug("Disabling FlashAttention as it does not support MLA.")
        use_flash_attention = False
        qkv_layout_group = qkv_layout.replace("b", "").replace("s", "").replace("t", "")
        if use_fused_attention and qkv_layout_group != "hd_hd_hd":
            logger.debug(
                "Disabling FusedAttention as MLA is not supported with qkv_layout = %s",
                qkv_layout,
            )
            use_fused_attention = False
    if use_flash_attention_2 and (
        head_dim_qk > 256
        or head_dim_qk % 8 != 0
        or (
            head_dim_qk > 192
            and device_compute_capability not in ((8, 0), (9, 0), (10, 0), (12, 0))
        )
    ):
        if FlashAttentionUtils.is_installed:
            logger.debug(
                "Disabling FlashAttention 2 due to unsupported head_dim_qk and head_dim_v. "
                "Supported: head_dim_qk = head_dim_v, head_dim_qk %%8 = 0, "
                "head_dim_qk <= 256 (>192 requires sm80/90/100+). "
                "Found: head_dim_qk = %s, head_dim_v = %s, on sm%s.",
                head_dim_qk,
                head_dim_v,
                ".".join([str(i) for i in device_compute_capability]),
            )
        use_flash_attention_2 = False
    if use_flash_attention_3 and (head_dim_qk > 128 or head_dim_v > 128):
        if FlashAttentionUtils.v3_is_installed:
            logger.debug("Disabling FlashAttention 3 for head_dim > 128")
        use_flash_attention_3 = False

    # Filter: QKV layout
    if qkv_format == "thd":
        if use_unfused_attention:
            logger.debug("Disabling UnfusedDotProductAttention for qkv_format = thd")
            use_unfused_attention = False
        if pad_between_seqs:
            if (use_flash_attention_2 and FlashAttentionUtils.is_installed) or (
                use_flash_attention_3 and FlashAttentionUtils.v3_is_installed
            ):
                logger.debug(
                    "Disabling FlashAttention for qkv_format = thd when there is "
                    "padding between sequences, i.e. [a, a, PAD, b, b, b, PAD, c, PAD]"
                )
            use_flash_attention = False

    # Filter: Dropout
    if attention_dropout != 0.0 and use_flash_attention_3:
        logger.debug("Disabling FlashAttention 3 for dropout")
        use_flash_attention_3 = False

    # Filter: Context parallelism
    # qkv_format | attn_mask_type              | attn_bias_type           | supported backends
    # ----------------------------------------------------------------------------------------------------
    # bshd, sbhd | self-attention:             | no_bias, post_scale_bias | FlashAttention, FusedAttention
    #            |     no_mask, causal         |                          |
    #            | cross-attention:            |                          |
    #            |     no_mask                 |                          |
    # thd        | self-attention:             | no_bias                  | FlashAttention, FusedAttention
    #            |     padding, padding_causal |                          | if no padding between sequences,
    #            | cross-attention:            |                          | FusedAttention
    #            |     padding                 |                          | if there is padding between sequences
    # Note: context parallelism requires seq_len % (cp_size * 2) == 0 for each sequence in q, k, v.
    if context_parallel and use_unfused_attention:
        logger.debug(
            "Disabling UnfusedDotProductAttention as it does not support context parallelism"
        )
        use_unfused_attention = False
    if context_parallel and (use_flash_attention_2 or use_flash_attention_3):
        if FlashAttentionUtils.is_installed or FlashAttentionUtils.v3_is_installed:
            if fp8 and fp8_meta["recipe"].fp8_dpa:
                logger.debug(
                    "Disabling FlashAttention as it does not support context parallelism with FP8"
                )
                use_flash_attention = False
            if "bottom_right" in attn_mask_type:
                logger.debug(
                    "Disabling FlashAttention as it does not support context parallelism with"
                    " causal_bottom_right masking"
                )
                use_flash_attention = False
            # NOTE(yonghao): this is disabled for per doc CP attention.
            elif "causal" in attn_mask_type and max_seqlen_q != max_seqlen_kv and False:
                logger.debug(
                    "Disabling FlashAttention as it does not support context parallelism with"
                    " causal masking for cross-attention"
                )
                use_flash_attention = False
            elif core_attention_bias_type not in ["no_bias", "post_scale_bias"]:
                logger.debug(
                    "Disabling FlashAttention as it does not support context parallelism with bias"
                    " type of %s",
                    core_attention_bias_type,
                )
                use_flash_attention = False
            elif qkv_format == "thd" and core_attention_bias_type != "no_bias":
                logger.debug(
                    "Disabling FlashAttention as it does not support context parallelism with"
                    " attention bias for THD format"
                )
                use_flash_attention = False

    if context_parallel and use_fused_attention:
        if "bottom_right" in attn_mask_type:
            logger.debug(
                "Disabling FusedAttention as it does not support context parallelism with"
                " causal_bottom_right masking"
            )
            use_fused_attention = False
        elif "causal" in attn_mask_type and max_seqlen_q != max_seqlen_kv:
            logger.debug(
                "Disabling FusedAttention as it does not support context parallelism with causal"
                " masking for cross-attention"
            )
            use_fused_attention = False
        elif core_attention_bias_type not in ["no_bias", "post_scale_bias"]:
            logger.debug(
                "Disabling FusedAttention as it does not support context parallelism with bias type"
                " of %s",
                core_attention_bias_type,
            )
            use_fused_attention = False
        elif qkv_format == "thd" and core_attention_bias_type != "no_bias":
            logger.debug(
                "Disabling FusedAttention as it does not support context parallelism with attention"
                " bias for THD format"
            )
            use_fused_attention = False
        elif head_dim_qk != head_dim_v:
            logger.debug(
                "Disabling FusedAttention as it does not support context parallelism with MLA"
            )
            use_fused_attention = False

    # Filter: Attention mask
    # attn_mask_type              | attention_mask                       | supported backends
    # ----------------------------------------------------------------------------------------
    # no_mask                     | None                                 | All
    # padding                     |                                      | All
    #     self-attention          | One tensor in shape [b, 1, 1, sq]    |
    #     cross-attention         | Tuple of two tensors in shapes       |
    #                             | [b, 1, 1, sq] and [b, 1, 1, skv]     |
    # causal                      | None                                 |
    #     self-attention          |                                      | All
    #     cross-attention         |                                      | FusedAttention, UnfusedDotProductAttention
    # padding_causal              | Same as "padding"                    |
    #     self-attention          |                                      | All
    #     cross-attention         |                                      | FusedAttention, UnfusedDotProductAttention
    # causal_bottom_right         | None                                 | All
    # padding_causal_bottom_right | Same as "padding"                    | All
    # arbitrary                   | One tensor in shape broadcastable to | UnfusedDotProductAttention
    #                             | [b, h, sq, skv]                      |
    if attn_mask_type == "arbitrary":
        if (use_flash_attention_2 and FlashAttentionUtils.is_installed) or (
            use_flash_attention_3 and FlashAttentionUtils.v3_is_installed
        ):
            logger.debug("Disabling FlashAttention for arbitrary mask")
        use_flash_attention = False
        if use_fused_attention:
            logger.debug("Disabling FusedAttention for arbitrary mask")
        use_fused_attention = False
    # NOTE(yonghao): this is disabled for per doc CP attention.
    if (
        (use_flash_attention_2 or use_flash_attention_3)
        and attn_mask_type in ["causal", "padding_causal"]
        and max_seqlen_q != max_seqlen_kv
        and False
    ):
        logger.warning(
            "Disabling FlashAttention as it only supports bottom-right-diagonal "
            "causal mask since flash-attn 2.1 (our minimum supported version). See "
            "https://github.com/Dao-AILab/flash-attention#21-change-behavior-of-causal-flag"
        )
        use_flash_attention = False

    # Filter: Sliding window attention
    #    backend                 |      window_size       | diagonal alignment
    # ---------------------------------------------------------------------------------
    # FlashAttention             | (-1, -1) or (>=0, >=0) | bottom right
    # FusedAttention             | (-1,  0) or (>=0, 0)   | top left
    # UnfusedDotProductAttention | (-1, -1) or (>=0, >=0) | both;
    #                            |                        | converts window_size to an 'arbitrary' mask
    if window_size is None:
        window_size = check_set_window_size(attn_mask_type, window_size)
    else:
        if use_fused_attention and (window_size[0] != -1 or window_size[1] not in [-1, 0]):
            if fp8 and (fp8_meta["recipe"].fp8_dpa or fp8_meta["recipe"].fp8_mha):
                logger.debug(
                    "Disabling FusedAttention as it does not support sliding window attention"
                    " for FP8"
                )
                use_fused_attention = False
            elif window_size[1] != 0 or attention_dropout != 0.0:
                logger.debug(
                    "Disabling FusedAttention as it only supports sliding window attention "
                    "with (left, 0) and no dropout"
                )
                use_fused_attention = False
            elif max_seqlen_q > max_seqlen_kv:
                logger.debug(
                    "Disabling FusedAttention as it does not support sliding window attention "
                    "with s_q > s_kv for cross-attention"
                )
                use_fused_attention = False
        if use_flash_attention_2 and (window_size[0] != -1 or window_size[1] not in [-1, 0]):
            if not FlashAttentionUtils.is_installed:
                FlashAttentionUtils.version_required = PkgVersion("2.3")
            elif not FlashAttentionUtils.v2_3_plus:
                logger.debug(
                    "Disabling FlashAttention as sliding window attention requires flash-attn 2.3+"
                )
                use_flash_attention_2 = False

    # Filter: Attention bias
    #    backend                 |      bias types              | ALiBi diagonal alignment
    # ---------------------------------------------------------------------------------
    # FlashAttention             | no_bias, alibi/alibi_slopes  | bottom right
    # FusedAttention             | no_bias, post_scale_bias     |
    #                            | alibi/alibi_slopes           | top left,
    #                            |                              | bottom_right (converts to a 'post_scale_bias' bias)
    # UnfusedDotProductAttention | no_bias, pre/post_scale_bias |
    #                            | alibi/alibi_slopes           | both; converts to a 'post_scale_bias' bias
    if core_attention_bias_type == "alibi":
        if use_flash_attention_3:
            if FlashAttentionUtils.v3_is_installed:
                logger.debug("Disabling FlashAttention 3 for ALiBi")
            use_flash_attention_3 = False
        if use_flash_attention_2:
            if not FlashAttentionUtils.is_installed:
                FlashAttentionUtils.version_required = PkgVersion("2.4")
            elif not FlashAttentionUtils.v2_4_plus:
                logger.debug("Disabling FlashAttention as ALiBi requires flash-attn 2.4+")
                use_flash_attention_2 = False

    if (
        core_attention_bias_type not in ["no_bias", "alibi"]
        or core_attention_bias_shape is not None
    ):
        if (use_flash_attention_2 and FlashAttentionUtils.is_installed) or (
            use_flash_attention_3 and FlashAttentionUtils.v3_is_installed
        ):
            logger.debug("Disabling FlashAttention for pre/post_scale_bias")
        use_flash_attention = False

    fu_core_attention_bias_type = core_attention_bias_type
    fu_core_attention_bias_shape = core_attention_bias_shape
    fu_core_attention_bias_requires_grad = core_attention_bias_requires_grad
    if (
        use_fused_attention
        and core_attention_bias_type == "alibi"
        and (alibi_slopes_shape is not None or max_seqlen_q != max_seqlen_kv)
    ):
        fu_core_attention_bias_type = "post_scale_bias"
        fu_core_attention_bias_requires_grad = False
        if alibi_slopes_shape is None:
            fu_core_attention_bias_shape = "1hss"
        elif len(alibi_slopes_shape) == 1 and alibi_slopes_shape[0] == num_heads:
            fu_core_attention_bias_shape = "1hss"
        elif (
            len(alibi_slopes_shape) == 2
            and alibi_slopes_shape[0] == batch_size
            and alibi_slopes_shape[1] == num_heads
        ):
            fu_core_attention_bias_shape = "bhss"

    if (
        use_fused_attention
        and fu_core_attention_bias_type == "post_scale_bias"
        and fu_core_attention_bias_shape != "1hss"
    ):
        if fu_core_attention_bias_requires_grad:
            # remove this line when cuDNN adds bwd support for
            # [1, 1, s, s], [b, 1, s, s] and [b, h, s, s]
            logger.debug("Disabling FusedAttention for dBias in [1, H, S, S] shape")
            use_fused_attention = False
        else:
            # max512 backend will only support [1, h, s, s]
            os.environ["NVTE_FUSED_ATTN_BACKEND"] = "1"

    # Filter: cuDNN support
    fused_attention_backend = None
    if use_fused_attention:
        q_type = TE_DType[qkv_dtype]
        kv_type = q_type
        if fp8 and fp8_meta["recipe"].fp8_dpa:
            q_type = get_fp8_te_dtype(fp8_meta["recipe"], fprop_tensor=True)
            kv_type = q_type
        fused_attention_backend = tex.get_fused_attn_backend(
            q_type,
            kv_type,
            QKVLayout[qkv_layout],
            AttnBiasType[fu_core_attention_bias_type],
            AttnMaskType[attn_mask_type],
            attention_dropout,
            num_heads,
            num_gqa_groups,
            max_seqlen_q,
            max_seqlen_kv,
            head_dim_qk,
            head_dim_v,
            window_size[0],
            window_size[1],
        )
        if fused_attention_backend == FusedAttnBackend["No_Backend"]:
            logger.debug("Disabling FusedAttention as no backend supports the provided input")
            use_fused_attention = False
            fused_attention_backend = None
        if (
            use_fused_attention
            and window_size is not None
            and window_size[0] != -1
            and fused_attention_backend != FusedAttnBackend["F16_arbitrary_seqlen"]
        ):
            logger.debug(
                "Disabling FusedAttention as only sub-backend %s does not support "
                "slidng window attention",
                int(fused_attention_backend),
            )
            use_fused_attention = False
            fused_attention_backend = None
        if (
            use_fused_attention
            and fused_attention_backend == FusedAttnBackend["F16_max512_seqlen"]
            and fu_core_attention_bias_type == "post_scale_bias"
            and fu_core_attention_bias_shape != "1hss"
        ):
            logger.debug(
                "Disabling FusedAttention as cuDNN sub-backend 0 only supports post_scale_bias in"
                " [1, H, S, S] shape"
            )
            use_fused_attention = False
            fused_attention_backend = None

    # Filter: Determinism
    # backend                      | deterministic
    # ---------------------------------------------
    # FlashAttention               |
    #     flash-attn >=2.0, <2.4.1 | no
    #     flash-attn >=2.4.1       | yes
    # FusedAttention               |
    #     sub-backend 0            | yes
    #     sub-backend 1            | workspace optimization path and sm90+: yes;
    #                              | otherwise: no
    #     sub-backend 2            | no
    # UnfusedDotProductAttention   | yes
    if use_flash_attention_2 and deterministic:
        if not FlashAttentionUtils.is_installed:
            FlashAttentionUtils.version_required = PkgVersion("2.4.1")
        elif not FlashAttentionUtils.v2_4_1_plus:
            logger.warning(
                "Disabling FlashAttention as version <2.4.1 does not support deterministic "
                "execution. To use FlashAttention with deterministic behavior, "
                "please install flash-attn >= 2.4.1."
            )
            use_flash_attention_2 = False
    if use_fused_attention and deterministic:
        if fused_attention_backend == FusedAttnBackend["FP8"] and is_training:
            logger.debug("Disabling FusedAttention for determinism reasons")
            use_fused_attention = False
        if (
            fused_attention_backend == FusedAttnBackend["F16_arbitrary_seqlen"]
            and is_training
            and (
                device_compute_capability < (9, 0)
                or core_attention_bias_requires_grad
                or cudnn_version < (8, 9, 5)
            )
        ):
            logger.debug("Disabling FusedAttention for determinism reasons")
            use_fused_attention = False

    # use_flash_attention may have been set above
    use_flash_attention_2 = use_flash_attention and use_flash_attention_2
    use_flash_attention_3 = use_flash_attention and use_flash_attention_3

    # `FusedAttention` and `FlashAttention` are faster backends than `UnfusedDotProductAttention`.
    # When `FusedAttention` does not support the provided attention params, and `FlashAttention`
    # does, we recommend users to install flash-attn if not installed already.
    if not use_fused_attention and _NVTE_FLASH_ATTN:
        if (
            use_flash_attention_3
            and not FlashAttentionUtils.v3_is_installed
            and not FlashAttentionUtils.v3_warning_printed
            and torch.cuda.current_device() == 0
        ):
            logger.warning(
                "flash-attn v3 may provide important feature support or performance improvement."
                " Please install flash-attn v3 by \n%s",
                FlashAttentionUtils.v3_installation_steps,
            )
            FlashAttentionUtils.v3_warning_printed = True
        elif (
            use_flash_attention_2
            and not FlashAttentionUtils.is_installed
            and not FlashAttentionUtils.warning_printed
            and torch.cuda.current_device() == 0
        ):
            logger.warning(
                "flash-attn may provide important feature support or performance improvement."
                " Please install flash-attn %s by pip3 install flash-attn==<version>.",
                _get_supported_versions(
                    FlashAttentionUtils.version_required,
                    FlashAttentionUtils.max_version,
                ),
            )
            FlashAttentionUtils.warning_printed = True
    # All available backends
    if use_flash_attention_2 and not FlashAttentionUtils.is_installed:
        use_flash_attention_2 = False
    if use_flash_attention_3 and not FlashAttentionUtils.v3_is_installed:
        use_flash_attention_3 = False
    use_flash_attention = use_flash_attention_2 or use_flash_attention_3
    available_backends = [use_flash_attention, use_fused_attention, use_unfused_attention]
    if use_flash_attention_2:
        flash_attention_backend = FlashAttentionUtils.version
    if use_flash_attention_3:
        flash_attention_backend = FlashAttentionUtils.fa3_version

    logger.debug(
        "Available backends = {FlashAttention=%s%s, FusedAttention=%s%s,"
        " UnfusedDotProductAttention=%s}",
        bool(available_backends[0]),
        (f" ({str(flash_attention_backend)})" if flash_attention_backend is not None else ""),
        bool(available_backends[1]),
        (
            f" (sub-backend {int(fused_attention_backend)})"
            if fused_attention_backend is not None
            else ""
        ),
        bool(available_backends[2]),
    )

    # Select FusedAttention for performance
    # TODO(yonghao): disable this perference because we found numerical error in this case.
    # For the best performance, we should figure out how to solve this.
    # if use_flash_attention and use_fused_attention and device_compute_capability >= (9, 0):
    #     logger.debug(
    #         "Disabling FlashAttention to give FusedAttention preference on Hopper+ "
    #         "for performance reasons"
    #     )
    #     use_flash_attention = False

    # Selected backend
    if use_flash_attention:
        use_fused_attention = False
        use_unfused_attention = False
    elif use_fused_attention:
        use_unfused_attention = False
    selected_backend = "NoBackend"
    if use_flash_attention:
        selected_backend = f"FlashAttention ({str(flash_attention_backend)})"
    elif use_fused_attention:
        selected_backend = f"FusedAttention (sub-backend {int(fused_attention_backend)})"
    elif use_unfused_attention:
        selected_backend = "UnfusedDotProductAttention"
    logger.debug("Selected backend = %s", selected_backend)

    __ret_value = (
        use_flash_attention,
        flash_attention_backend,
        use_fused_attention,
        fused_attention_backend,
        use_unfused_attention,
        available_backends,
    )
    return __ret_value


@torch.no_grad()
def get_padding_mask(
    batch_size: int,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_kv: torch.Tensor,
    max_seqlen_q: int,
    max_seqlen_kv: int,
):
    """Convert cu_seqlens to attention_mask"""
    seqlens_q = cu_seqlens_q[1:] - cu_seqlens_q[:-1]
    seqlens_kv = cu_seqlens_kv[1:] - cu_seqlens_kv[:-1]
    attention_mask_q = torch.Tensor([]).to(dtype=torch.bool)
    attention_mask_kv = torch.Tensor([]).to(dtype=torch.bool)
    for i in range(batch_size):
        attention_mask_q = torch.cat(
            [
                attention_mask_q,
                torch.Tensor([False] * seqlens_q[i] + [True] * (max_seqlen_q - seqlens_q[i]))
                .to(dtype=torch.bool)
                .unsqueeze(0)
                .unsqueeze(0)
                .unsqueeze(0),
            ],
            dim=0,
        )
        attention_mask_kv = torch.cat(
            [
                attention_mask_kv,
                torch.Tensor([False] * seqlens_kv[i] + [True] * (max_seqlen_kv - seqlens_kv[i]))
                .to(dtype=torch.bool)
                .unsqueeze(0)
                .unsqueeze(0)
                .unsqueeze(0),
            ],
            dim=0,
        )
    attention_mask = (
        attention_mask_q.to(device="cuda"),
        attention_mask_kv.to(device="cuda"),
    )
    return attention_mask


def attn_params_eq(self, other):
    """
    Overwrite the original TE's __eq__, otherwise for ping-pong the
    batch size, max seqlen q, and max seqlen kv keeps bouncing between the
    two ping-pong splits.

    Overwrite dataclass.__eq__ so that only fp8_meta["recipe"] is compared,
    since all other entries of fp8_meta are unused in get_attention_backend.
    """
    if not isinstance(other, self.__class__):
        return NotImplemented
    for field in fields(self):
        fname = field.name
        sf = getattr(self, fname)
        of = getattr(other, fname)
        #### Our change
        # batch_size is only used to compare with alibi_slopes_shape
        if fname == "batch_size" and self.alibi_slopes_shape is None:
            continue
        # TE does not understand that q and kv can have different max seqlen
        if fname in ["max_seqlen_q", "max_seqlen_kv"]:
            continue
        #### End our change

        if fname != "fp8_meta":
            if sf != of:
                return False
        elif sf.get("recipe", None) != of.get("recipe", None):
            return False
    return True


class MonkeyPatch:
    def __enter__(self):
        self.backup_get_attention_backend = dpa_utils.get_attention_backend
        self.backup_attn_param_eq = AttentionParams.__eq__
        dpa_utils.get_attention_backend = get_attention_backend
        # TODO(yonghao): as now max_seqlen is an int, this is cheap, maybe
        # no need to do the monkey patching
        AttentionParams.__eq__ = attn_params_eq
    
    def __exit__(self, exc_type, exc_value, traceback):
        dpa_utils.get_attention_backend = self.backup_get_attention_backend
        AttentionParams.__eq__ = self.backup_attn_param_eq


class PerDocCPAttention(TEDotProductAttention):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.cp_group = None
        self.cp_global_ranks = None
        self.cp_comm_type = "p2p"

    def forward(self, *args, **kwargs):
        with MonkeyPatch():
            return super().forward(*args, **kwargs)
    
