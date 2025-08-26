"""Mainly learned from verl."""
from abc import ABC, abstractmethod
from enum import Enum
import logging
from typing import Callable
import rich
import os

from megatron.core import mpu, tensor_parallel
from megatron.core.transformer.module import Float16Module
from megatron.core.distributed import DistributedDataParallel as DDP
from megatron.core.distributed import DistributedDataParallelConfig
from megatron.core.enums import ModelType
from megatron.core.optimizer.optimizer import OptimizerConfig
from megatron.core.optimizer_param_scheduler import OptimizerParamScheduler
from megatron.core.transformer import TransformerConfig
from megatron.core.utils import get_attr_wrapped_model
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PretrainedConfig

from d2.runtime.megatron_patch.model_patch import get_gpt_decoder_block_spec, PingPangGPTModel

from test_util import get_torch_device, get_device_name

logger = logging.getLogger(__name__)


def get_device_id() -> int:
    """Return current device id based on the device type.
    Returns:
        device index
    """
    return get_torch_device().current_device()


def make_batch_generator(batches, vpp_size):
    """
    Creates a batch generator suitable for Megatron pipeline parallelism,
    handling virtual pipeline parallelism (VPP).

    If VPP is used (vpp_size > 1), it duplicates the batch iterator for each
    virtual pipeline stage. Otherwise, it returns a single iterator.

    Args:
        batches: An iterable (e.g., list) of micro-batches.
        vpp_size (int): The virtual pipeline model parallel size.

    Returns:
        An iterator or a list of iterators over the micro-batches.
    """
    if vpp_size > 1:
        # has vpp
        batch_generator = [batches] * vpp_size  # number of vpp chunks
        batch_generator = [iter(b) for b in batch_generator]
    else:
        # no vpp
        batch_generator = iter(batches)
    return batch_generator


def get_megatron_optimizer_param_scheduler(
    optimizer,
    config,
):
    """
    Get the optimizer parameter scheduler for Megatron.
    """
    if config.get("lr_decay_steps", None) is None:
        config.lr_decay_steps = config.total_training_steps
    wsd_decay_steps = None
    if config.get("lr_wsd_decay_steps", None) is not None:
        wsd_decay_steps = config.lr_wsd_decay_steps
    if config.get("lr_warmup_steps_ratio", None) is not None and (
        config.get("lr_warmup_steps", None) is None or config.lr_warmup_steps <= 0
    ):
        config.lr_warmup_steps = int(config.lr_warmup_steps_ratio * config.lr_decay_steps)

    opt_param_scheduler = OptimizerParamScheduler(
        optimizer,
        init_lr=config.lr_warmup_init,
        max_lr=config.lr,
        min_lr=config.min_lr,
        lr_warmup_steps=config.lr_warmup_steps,
        lr_decay_steps=config.lr_decay_steps,
        lr_decay_style=config.lr_decay_style,
        start_wd=config.weight_decay,
        end_wd=config.weight_decay,
        wd_incr_steps=config.total_training_steps,
        wd_incr_style=config.weight_decay_incr_style,
        use_checkpoint_opt_param_scheduler=config.use_checkpoint_opt_param_scheduler,
        override_opt_param_scheduler=(not config.use_checkpoint_opt_param_scheduler),
        wsd_decay_steps=wsd_decay_steps,
        lr_wsd_decay_style=config.lr_wsd_decay_style,
    )

    return opt_param_scheduler


def init_megatron_optim_config(optim_config: dict) -> OptimizerConfig:
    config = OptimizerConfig(
        optimizer=optim_config.get("optimizer", "adam"),
        lr=optim_config.get("lr"),
        min_lr=optim_config.get("min_lr", None),
        clip_grad=optim_config.get("clip_grad", 1.0),
        weight_decay=optim_config.get("weight_decay", 0.01),
        bf16=True,
        params_dtype=torch.bfloat16,
        use_distributed_optimizer=True,
    )
    return config


def get_model_size(model: nn.Module, scale="auto"):
    n_params = sum(p.numel() for p in model.parameters())

    if scale == "auto":
        if n_params > 1e9:
            scale = "B"
        elif n_params > 1e6:
            scale = "M"
        elif n_params > 1e3:
            scale = "K"
        else:
            scale = ""

    if scale == "B":
        n_params = n_params / 1e9
    elif scale == "M":
        n_params = n_params / 1e6
    elif scale == "K":
        n_params = n_params / 1e3
    elif scale == "":
        pass
    else:
        raise NotImplementedError(f"Unknown scale {scale}")

    return n_params, scale


def print_model_size(model: nn.Module, name: str = None):
    n_params, scale = get_model_size(model, scale="auto")
    if name is None:
        name = model.__class__.__name__
    print(f"{name} contains {n_params:.2f}{scale} parameters")


def get_model(
    model_provider_func,
    model_type=ModelType.encoder_or_decoder,
    wrap_with_ddp=True,
    use_distributed_optimizer=True,
    transformer_config=None,
    override_ddp_config=None,
):
    """Build the model."""
    # Build model.
    if (
        mpu.get_pipeline_model_parallel_world_size() > 1
        and mpu.get_virtual_pipeline_model_parallel_world_size() is not None
    ):
        assert model_type != ModelType.encoder_and_decoder, (
            "Interleaved schedule not supported for model with both encoder and decoder"
        )
        model = []
        for i in range(mpu.get_virtual_pipeline_model_parallel_world_size()):
            mpu.set_virtual_pipeline_model_parallel_rank(i)
            # Set pre_process and post_process only after virtual rank is set.
            pre_process = mpu.is_pipeline_first_stage()
            post_process = mpu.is_pipeline_last_stage()
            this_model = model_provider_func(pre_process=pre_process, post_process=post_process)
            this_model.model_type = model_type
            model.append(this_model)
        mpu.set_virtual_pipeline_model_parallel_rank(0)
    else:
        pre_process = mpu.is_pipeline_first_stage()
        post_process = mpu.is_pipeline_last_stage()
        add_encoder = True
        add_decoder = True
        if model_type == ModelType.encoder_and_decoder:
            if mpu.get_pipeline_model_parallel_world_size() > 1:
                assert mpu.get_pipeline_model_parallel_split_rank() is not None, (
                    "Split rank needs to be specified for model with both encoder and decoder"
                )
                rank = mpu.get_pipeline_model_parallel_rank()
                split_rank = mpu.get_pipeline_model_parallel_split_rank()
                world_size = mpu.get_pipeline_model_parallel_world_size()
                pre_process = rank == 0 or rank == split_rank
                post_process = (rank == (split_rank - 1)) or (rank == (world_size - 1))
                add_encoder = mpu.is_pipeline_stage_before_split()
                add_decoder = mpu.is_pipeline_stage_after_split()
            model = model_provider_func(
                pre_process=pre_process, post_process=post_process, add_encoder=add_encoder, add_decoder=add_decoder
            )
        else:
            model = model_provider_func(pre_process=pre_process, post_process=post_process)
        model.model_type = model_type

    if not isinstance(model, list):
        model = [model]

    # Set tensor model parallel attributes if not set.
    # Only parameters that are already tensor model parallel have these
    # attributes set for them. We should make sure the default attributes
    # are set for all params so the optimizer can use them.
    for model_module in model:
        for param in model_module.parameters():
            tensor_parallel.set_defaults_if_not_set_tensor_model_parallel_attributes(param)

    # Print number of parameters.
    if mpu.get_data_parallel_rank() == 0:
        print(
            " > number of parameters on (tensor, pipeline) model parallel rank ({}, {}): {}".format(
                mpu.get_tensor_model_parallel_rank(),
                mpu.get_pipeline_model_parallel_rank(),
                sum([sum([p.nelement() for p in model_module.parameters()]) for model_module in model]),
            ),
            flush=True,
        )

    # GPU allocation.
    if transformer_config is None or (not transformer_config.use_cpu_initialization):
        for model_module in model:
            model_module.to(f"{get_device_name()}:{get_device_id()}")

    # Fp16 conversion.
    config: TransformerConfig = get_attr_wrapped_model(model[0], "config", allow_none=False)
    config.fp8 = None
    tfconfig: TransformerConfig = model[0].config
    if config.fp16 or config.bf16:  # the ModelParallelConfig in GPTModel
        model = [Float16Module(config, model_module) for model_module in model]

    if wrap_with_ddp:
        ddp_models = []
        ddp_config_dict = {
            "use_distributed_optimizer": use_distributed_optimizer,
            "grad_reduce_in_fp32": True,
            "overlap_grad_reduce": False,
        }
        if override_ddp_config is not None:
            ddp_config_dict.update(override_ddp_config)
        ddp_config = DistributedDataParallelConfig(**ddp_config_dict)
        for model_chunk_idx, model_chunk in enumerate(model):
            ddp_model = DDP(
                config=tfconfig,
                module=model_chunk,
                disable_bucketing=(model_chunk_idx > 0),
                ddp_config=ddp_config,
            )
            ddp_models.append(ddp_model)
        model = ddp_models
        # # Broadcast params from data parallel src rank to other data parallel ranks.
        # # if args.data_parallel_random_init:
        for model_module in model:
            model_module.broadcast_params()
    return model


def update_model_config(module_config, override_config_kwargs):
    """Update the module config with the override_config_kwargs.
    Args:
        module_config: The module config from Huggingface Transformers.
        override_config_kwargs: The kwargs to override the module config.
    """
    for key, val in override_config_kwargs.items():
        if isinstance(val, dict):
            update_model_config(getattr(module_config, key), val)
        else:
            setattr(module_config, key, val)


class SupportedModel(Enum):
    LLAMA = "LlamaForCausalLM"  # tested
    QWEN2 = "Qwen2ForCausalLM"  # tested
    # MIXTRAL = "MixtralForCausalLM"  # tested
    # QWEN3 = "Qwen3ForCausalLM"  # tested


def _get_base_transformer_config(
    hf_config: PretrainedConfig, dtype: torch.dtype, **override_transformer_config_kwargs
) -> dict:
    """
    Create a base TransformerConfig with common parameters across different model architectures.
    TODO: (ycl) use dataclass or converter config?

    Args:
        hf_config: HuggingFace model configuration
        dtype: Data type for the model
        override_transformer_config_kwargs: Additional parameters to override defaults

    Returns:
        TransformerConfig with common parameters
    """

    # Common parallel state parameters
    overlap_p2p_comm = (
        mpu.get_virtual_pipeline_model_parallel_world_size() is not None
        and mpu.get_virtual_pipeline_model_parallel_world_size() > 1
    )
    batch_p2p_comm = False

    # Base configuration with common parameters
    base_config = {
        # Model architecture parameters
        "num_layers": hf_config.num_hidden_layers,
        "hidden_size": hf_config.hidden_size,
        "num_attention_heads": hf_config.num_attention_heads,
        "num_query_groups": hf_config.num_key_value_heads,
        "ffn_hidden_size": hf_config.intermediate_size,
        "attention_dropout": hf_config.attention_dropout,
        "hidden_dropout": getattr(hf_config, "hidden_dropout", 0.0),
        "kv_channels": getattr(hf_config, "head_dim", None),
        "layernorm_epsilon": hf_config.rms_norm_eps,
        "add_bias_linear": True,
        # Activation and normalization
        "activation_func": F.silu,
        "normalization": "RMSNorm",
        "gated_linear_unit": True,
        # Data types
        "pipeline_dtype": dtype,
        "params_dtype": dtype,
        "bf16": dtype is torch.bfloat16,
        # Parallel configuration
        "tensor_model_parallel_size": mpu.get_tensor_model_parallel_world_size(),
        "pipeline_model_parallel_size": mpu.get_pipeline_model_parallel_world_size(),
        "expert_model_parallel_size": mpu.get_expert_model_parallel_world_size(),
        "expert_tensor_parallel_size": mpu.get_expert_tensor_parallel_world_size(),
        "virtual_pipeline_model_parallel_size": mpu.get_virtual_pipeline_model_parallel_world_size(),
        "context_parallel_size": mpu.get_context_parallel_world_size(),
        "overlap_p2p_comm": overlap_p2p_comm,
        "batch_p2p_comm": batch_p2p_comm,
        "sequence_parallel": mpu.get_tensor_model_parallel_world_size() > 1,
        # Common settings
        "variable_seq_lengths": True,
        "masked_softmax_fusion": True,
        "moe_token_dispatcher_type": "alltoall",
    }

    # Update with any provided overrides
    # override_transformer_config_kwargs as kwargs shall never be none
    base_config.update(override_transformer_config_kwargs)

    return base_config


def _override_hf_config_num_layers_with_env_var(hf_config: PretrainedConfig):
    # TODO:(Refactor) Override the number of layers in the HuggingFace config based on environment variable.
    # This is a temporary workaround to modify the number of layers for testing purposes.
    # Note: There appears to be an issue with the override config mechanism not properly
    # updating the HF config in some cases.

    try:
        rank = int(os.environ.get("RANK", 0))
    except ValueError:
        rank = 0
    
    
    num_layers = os.environ.get("NUM_LAYERS", None)
    
    if num_layers is not None:
        num_layers = int(num_layers)
        if hasattr(hf_config, 'num_hidden_layers'):
            hf_config.num_hidden_layers = num_layers
        elif hasattr(hf_config, 'num_layers'):
            hf_config.num_layers = num_layers
        elif hasattr(hf_config, 'n_layer'):
            hf_config.n_layer = num_layers
        else:
            # Fallback - try to set the most common attribute name
            hf_config.num_hidden_layers = num_layers
        if rank == 0:
            rich.print(f"Override HF Config: Set number of layers: {num_layers}. Now HF Config: ", str(hf_config))
    return hf_config
    

def hf_to_mcore_config_dense(
    hf_config: PretrainedConfig, dtype: torch.dtype, **override_transformer_config_kwargs
) -> TransformerConfig:
    
    hf_config = _override_hf_config_num_layers_with_env_var(hf_config)

    # for LlamaForCausalLM or Qwen2ForCausalLM
    qkv_bias = True if "Qwen2ForCausalLM" in hf_config.architectures else getattr(hf_config, "attention_bias", False)
    qk_layernorm = True if "Qwen3ForCausalLM" in hf_config.architectures else False

    args: dict = _get_base_transformer_config(
        hf_config=hf_config,
        dtype=dtype,
        use_cpu_initialization=False,
        add_bias_linear=False,
        add_qkv_bias=qkv_bias,
        qk_layernorm=qk_layernorm,
    )
    # override_transformer_config_kwargs as kwargs shall never be none
    args.update(override_transformer_config_kwargs)
    return TransformerConfig(**args)


class BaseModelInitializer(ABC):
    """Base class for model initializers."""

    def __init__(self, tfconfig: TransformerConfig, hf_config: PretrainedConfig):
        self.tfconfig = tfconfig
        self.hf_config = hf_config

    @abstractmethod
    def get_transformer_layer_spec(self):
        """Get the transformer layer specification.
        https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/core/models/gpt/gpt_layer_specs.py"""
        pass

    def get_rope_scaling_args(self) -> dict:
        """Get rope scaling args."""
        rope_scaling_args = {}
        if "rope_scaling" in self.hf_config:
            if self.hf_config.rope_scaling is not None:
                # assert self.hf_config.rope_scaling["type"] == "linear", "only linear scaling is supported for now"
                rope_scaling_args["seq_len_interpolation_factor"] = self.hf_config.rope_scaling["factor"]
        return rope_scaling_args

    def initialize(
        self,
        pre_process: bool = True,
        post_process: bool = True,
        share_embeddings_and_output_weights: bool = False,
        value: bool = False,
        **extra_kwargs,
    ) -> PingPangGPTModel:
        """Initialize a GPT model with the given configuration.
        https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/core/models/gpt/gpt_model.py

        Args:
            pre_process (bool): include embedding layer.
            post_process (bool): including an output layer.
            share_embeddings_and_output_weights (bool): input embeddings and output logit weights are shared.
            value (bool): add an extra linear layer for classification or regression.

        Returns:
            GPTModel: An initialized GPT model instance
        """
        transformer_layer_spec = self.get_transformer_layer_spec()
        rope_scaling_args = self.get_rope_scaling_args()
        mtp_block_spec = extra_kwargs.get("mtp_block_spec", None)
        model = PingPangGPTModel(
            config=self.tfconfig,
            transformer_layer_spec=transformer_layer_spec,
            vocab_size=self.hf_config.vocab_size,
            max_sequence_length=self.hf_config.max_position_embeddings,
            pre_process=pre_process,
            post_process=post_process,
            share_embeddings_and_output_weights=share_embeddings_and_output_weights,
            position_embedding_type="rope",
            rotary_base=self.hf_config.rope_theta,
            **rope_scaling_args,
            mtp_block_spec=mtp_block_spec,
        )

        if post_process and value:
            assert False, "classification is not supported"

        return model


class DenseModel(BaseModelInitializer):
    """Initializer for dense models like Llama and Qwen2."""

    def get_transformer_layer_spec(self):
        assert self.tfconfig.normalization == "RMSNorm", "only RMSNorm is supported for now"
        return get_gpt_decoder_block_spec(self.tfconfig, use_transformer_engine=True)


MODEL_CONFIG_CONVERTER_REGISTRY: dict[SupportedModel, Callable[[PretrainedConfig, torch.dtype], TransformerConfig]] = {
    SupportedModel.LLAMA: hf_to_mcore_config_dense,
    SupportedModel.QWEN2: hf_to_mcore_config_dense,
}


MODEL_INITIALIZER_REGISTRY: dict[SupportedModel, type[BaseModelInitializer]] = {
    SupportedModel.LLAMA: DenseModel,
    SupportedModel.QWEN2: DenseModel,
}


def get_supported_model(model_type: str) -> SupportedModel:
    try:
        return SupportedModel(model_type)
    except ValueError as err:
        supported_models = [e.value for e in SupportedModel]
        raise NotImplementedError(
            f"Model Type: {model_type} not supported. Supported models: {supported_models}"
        ) from err


def hf_to_mcore_config(
    hf_config: PretrainedConfig, dtype: torch.dtype, **override_transformer_config_kwargs
) -> TransformerConfig:
    """Convert huggingface PretrainedConfig to mcore TransformerConfig.

    Args:
        hf_config: The huggingface PretrainedConfig.
        dtype: The dtype of the model.
        **override_transformer_config_kwargs: The kwargs to override the transformer config.

    Returns:
        The mcore TransformerConfig.
    """
    assert len(hf_config.architectures) == 1, "Only one architecture is supported for now"
    model = get_supported_model(hf_config.architectures[0])
    return MODEL_CONFIG_CONVERTER_REGISTRY[model](hf_config, dtype, **override_transformer_config_kwargs)


def init_mcore_model(
    tfconfig: TransformerConfig,
    hf_config: PretrainedConfig,
    pre_process: bool = True,
    post_process: bool = None,
    *,
    share_embeddings_and_output_weights: bool = False,
    value: bool = False,
    **extra_kwargs,  # may be used for vlm and moe
) -> nn.Module:
    """
    Initialize a Mcore model.

    Args:
        tfconfig: The transformer config.
        hf_config: The HuggingFace config.
        pre_process: Optional pre-processing function.
        post_process: Optional post-processing function.
        share_embeddings_and_output_weights: Whether to share embeddings and output weights.
        value: Whether to use value.
        **extra_kwargs: Additional keyword arguments.

    Returns:
        The initialized model.
    """
    assert len(hf_config.architectures) == 1, "Only one architecture is supported for now"
    model = get_supported_model(hf_config.architectures[0])
    initializer_cls = MODEL_INITIALIZER_REGISTRY[model]
    initializer = initializer_cls(tfconfig, hf_config)
    return initializer.initialize(
        pre_process=pre_process,
        post_process=post_process,
        share_embeddings_and_output_weights=share_embeddings_and_output_weights,
        value=value,
        **extra_kwargs,
    )

ALL_MODULE_WRAPPER_CLASSNAMES = (DDP, Float16Module)

def unwrap_model(model, module_instances=ALL_MODULE_WRAPPER_CLASSNAMES):
    return_list = True
    if not isinstance(model, list):
        model = [model]
        return_list = False
    unwrapped_model = []
    for model_module in model:
        while isinstance(model_module, module_instances):
            model_module = model_module.module
        unwrapped_model.append(model_module)
    if not return_list:
        return unwrapped_model[0]
    return unwrapped_model


def gptmodel_forward(
    model, input_ids, attention_mask, position_ids, sequence_parallel, packed_seq_params,
    labels=None, logits_processor=None, logits_processor_kwargs=None,
):
    pre_process = unwrap_model(model).pre_process
    if pre_process:
        input_ids = input_ids.unsqueeze(0)
    post_process = unwrap_model(model).post_process
    output_orig = model(
        input_ids=input_ids,
        labels=labels,
        attention_mask=None,
        position_ids=position_ids,
        packed_seq_params=packed_seq_params,
    )
    if post_process and logits_processor is not None:
        # FIXME(yonghao): add logits processor logic, if needed
        pass
    return output_orig
