import warnings
from typing import Optional

import torch

from megatron.core.extensions.transformer_engine import (
    TENorm,
)
from megatron.core.fusions.fused_bias_dropout import get_bias_dropout_add
from megatron.core.models.gpt.gpt_layer_specs import (
    get_gpt_layer_with_transformer_engine_spec,
    get_gpt_mtp_block_spec,
    get_mlp_module_spec,
)
from megatron.core.transformer.attention import SelfAttention, SelfAttentionSubmodules
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.transformer.identity_op import IdentityOp
from megatron.core.transformer.spec_utils import ModuleSpec
from megatron.core.transformer.torch_norm import L2Norm
from megatron.core.transformer.transformer_block import (
    TransformerBlockSubmodules,
    get_num_layers_to_build,
)
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.transformer.transformer_layer import (
    TransformerLayerSubmodules,
    get_transformer_layer_offset,
)
from megatron.core.utils import is_te_min_version

import apex  # pylint: disable=unused-import
from megatron.core.fusions.fused_layer_norm import FusedLayerNorm
from megatron.core.extensions.transformer_engine import (
    TELayerNormColumnParallelLinear,
    TENorm,
    TERowParallelLinear,
)
LNImpl = FusedLayerNorm

from d2.runtime.megatron_patch.transformer_layer import (
    PingPangGPTModel, TransformerLayer as PingPangTransformerLayer
)
from d2.runtime.megatron_patch.per_doc_cp_attn import PerDocCPAttention


def get_gpt_layer_with_transformer_engine_spec(
    num_experts: Optional[int] = None,
    moe_grouped_gemm: Optional[bool] = False,
    qk_layernorm: Optional[bool] = False,
    multi_latent_attention: Optional[bool] = False,
    fp8: Optional[str] = None,  # pylint: disable=unused-arguments
    moe_use_legacy_grouped_gemm: Optional[bool] = False,
    qk_l2_norm: Optional[bool] = False,
) -> ModuleSpec:
    """Use this spec to use lower-level Transformer Engine modules (required for fp8 training).


    Args:
        num_experts (int, optional): Number of experts. Defaults to None.
        moe_grouped_gemm (bool, optional): To use Grouped GEMM. Defaults to False.
        qk_layernorm (bool, optional): To use layernorm for queries/keys. Defaults to False.
        fp8 (str, optional): Deprecated. For temporary Nemo compatibility.
        moe_use_legacy_grouped_gemm (bool, optional): Force use the legacy GroupedMLP.
                                                      Defaults to False.
        qk_l2_norm (bool, optional): To use l2 norm for queries/keys. Defaults to False.

    Returns:
        ModuleSpec: Module specification with TE modules
    """
    if fp8 is not None:
        warnings.warn(
            'The fp8 argument in "get_gpt_layer_with_transformer_engine_spec" has been deprecated'
            ' and will be removed soon. Please update your code accordingly.'
        )

    mlp = get_mlp_module_spec(
        use_te=True,
        num_experts=num_experts,
        moe_grouped_gemm=moe_grouped_gemm,
        moe_use_legacy_grouped_gemm=moe_use_legacy_grouped_gemm,
    )
    assert not multi_latent_attention

    # TENorm significantly harms convergence when used
    # for QKLayerNorm if TE Version < 1.9;
    # we instead use the Apex implementation.
    qk_norm = TENorm if is_te_min_version("1.9.0") else FusedLayerNorm
    return ModuleSpec(
        module=PingPangTransformerLayer,    # NOTE: monkey patch the transformer layer for ping-pang parallel
        submodules=TransformerLayerSubmodules(
            self_attention=ModuleSpec(
                module=SelfAttention,
                params={"attn_mask_type": AttnMaskType.causal},
                submodules=SelfAttentionSubmodules(
                    linear_qkv=TELayerNormColumnParallelLinear,
                    core_attention=PerDocCPAttention,
                    linear_proj=TERowParallelLinear,
                    q_layernorm=(
                        L2Norm if qk_l2_norm else (qk_norm if qk_layernorm else IdentityOp)
                    ),
                    k_layernorm=(
                        L2Norm if qk_l2_norm else (qk_norm if qk_layernorm else IdentityOp)
                    ),
                ),
            ),
            self_attn_bda=get_bias_dropout_add,
            pre_mlp_layernorm=TENorm if num_experts else IdentityOp,
            mlp=mlp,
            mlp_bda=get_bias_dropout_add,
        ),
    )


def get_gpt_decoder_block_spec(
    config: TransformerConfig,
    use_transformer_engine: bool,
    normalization: Optional[str] = None,
    qk_l2_norm: Optional[bool] = False,
) -> TransformerBlockSubmodules:
    """GPT block spec."""
    assert use_transformer_engine
    layer_norm_impl = TENorm

    # Layer specs.
    dense_layer_spec = get_gpt_layer_with_transformer_engine_spec(
        num_experts=None,
        moe_grouped_gemm=False,
        qk_layernorm=config.qk_layernorm,
        multi_latent_attention=config.multi_latent_attention,
        moe_use_legacy_grouped_gemm=config.moe_use_legacy_grouped_gemm,
        qk_l2_norm=qk_l2_norm,
    )
    moe_layer_spec = get_gpt_layer_with_transformer_engine_spec(
        num_experts=config.num_moe_experts,
        moe_grouped_gemm=config.moe_grouped_gemm,
        qk_layernorm=config.qk_layernorm,
        multi_latent_attention=config.multi_latent_attention,
        moe_use_legacy_grouped_gemm=config.moe_use_legacy_grouped_gemm,
        qk_l2_norm=qk_l2_norm,
    )

    # Parse config.moe_layer_freq to determine the pattern of expert/dense layers.
    # 0 stands for dense layers, 1 stands for expert layers.
    # For integer N: Creates a pattern with one expert layer every N layers.
    # For string pattern: Evaluates the str directly (e.g. "[1,0,1]" for alternating expert/dense).
    if isinstance(config.moe_layer_freq, int):
        moe_layer_pattern = [
            1 if (i % config.moe_layer_freq == 0) else 0 for i in range(config.num_layers)
        ]
    elif isinstance(config.moe_layer_freq, list):
        moe_layer_pattern = config.moe_layer_freq
        assert len(moe_layer_pattern) == config.num_layers, (
            f"Invalid length of moe_layer_pattern: {len(moe_layer_pattern)}, "
            f"expected {config.num_layers}, "
            f"current moe layer pattern: {config.moe_layer_freq}"
        )
    else:
        raise ValueError(
            f"Invalid moe_layer_freq: {type(config.moe_layer_freq)}, {config.moe_layer_freq}"
        )

    # Create the layer specs for the model.
    layer_specs = []
    for layer_number in range(config.num_layers):
        if moe_layer_pattern[layer_number] == 1:
            layer_specs.append(moe_layer_spec)
        elif moe_layer_pattern[layer_number] == 0:
            layer_specs.append(dense_layer_spec)
        else:
            raise ValueError(f"Invalid layer pattern: {moe_layer_pattern}")

    # Slice the layer specs to only include the layers that are built in this pipeline stage.
    # Note: MCore layer_number starts at 1
    offset = get_transformer_layer_offset(config)
    num_layers_to_build = get_num_layers_to_build(config)
    layer_specs = layer_specs[offset : offset + num_layers_to_build]

    # Block spec.
    block_spec = TransformerBlockSubmodules(layer_specs=layer_specs, layer_norm=layer_norm_impl)

    return block_spec


def model_provider(pre_process=True, post_process=True) -> PingPangGPTModel:
    """Builds the model.

    Args:
        pre_process (bool, optional): Set to true if you need to compute embedings. Defaults to True.
        post_process (bool, optional): Set to true if you need to want to compute output logits/loss. Defaults to True.

    Returns:
        GPTModel: The returned model
    """
    from megatron.training import get_args, print_rank_0
    from megatron.training.arguments import core_transformer_config_from_args

    args = get_args()
    use_te = args.transformer_impl == "transformer_engine"
    assert use_te

    if args.record_memory_history:
        torch.cuda.memory._record_memory_history(True,
            trace_alloc_max_entries=100000,
            trace_alloc_record_context=True)

        def oom_observer(device, alloc, device_alloc, device_free):
            print('saving allocated state during OOM')
            snapshot = torch.cuda.memory._snapshot()
            from pickle import dump
            dump(snapshot, open(f"oom_rank-{torch.distributed.get_rank()}_{args.memory_snapshot_path}", 'wb'))

        torch._C._cuda_attach_out_of_memory_observer(oom_observer)

    print_rank_0('building GPT model ...')
    # Experimental loading arguments from yaml
    config = core_transformer_config_from_args(args)

    transformer_layer_spec = get_gpt_decoder_block_spec(
        config, use_transformer_engine=True, normalization=args.normalization
    )

    mtp_block_spec = None
    if args.mtp_num_layers is not None:
        mtp_block_spec = get_gpt_mtp_block_spec(config, transformer_layer_spec, use_transformer_engine=use_te)

    model = PingPangGPTModel(
        config=config,
        transformer_layer_spec=transformer_layer_spec,
        vocab_size=args.padded_vocab_size,
        max_sequence_length=args.max_position_embeddings,
        pre_process=pre_process,
        post_process=post_process,
        fp16_lm_cross_entropy=args.fp16_lm_cross_entropy,
        parallel_output=True,
        share_embeddings_and_output_weights=not args.untie_embeddings_and_output_weights,
        position_embedding_type=args.position_embedding_type,
        rotary_percent=args.rotary_percent,
        rotary_base=args.rotary_base,
        rope_scaling=args.use_rope_scaling,
        mtp_block_spec=mtp_block_spec,
    )

    return model


def get_gpt_config(
    num_layers: int,
    hidden_size: int,
    num_attention_heads: int,
    ffn_hidden_size: int,
    hidden_dropout: float = 0.,
    attention_dropout: float = 0.,
    normalization: str = "RMSNorm",
    fp16: bool = False,
    bf16: bool = False,
    num_query_groups: int=None,
    tensor_model_parallel_size: int = 1,
    **kwargs,
):
    return TransformerConfig(
        num_layers=num_layers,
        hidden_size=hidden_size,
        num_attention_heads=num_attention_heads,
        ffn_hidden_size=ffn_hidden_size,
        hidden_dropout=hidden_dropout,
        attention_dropout=attention_dropout,
        normalization=normalization,
        fp16=fp16,
        bf16=bf16,
        num_query_groups=num_query_groups,
        tensor_model_parallel_size=tensor_model_parallel_size,
        **kwargs,
    )
