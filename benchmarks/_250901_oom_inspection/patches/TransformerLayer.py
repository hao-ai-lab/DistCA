"""
Partial patch to add `log_memory_usage` to the forward method to log the memory usage.
    Megatron-LM/megatron/core/transformer/transformer_layer.py
"""

def log_memory_usage(message: str):
    import d2.mem
    d2.mem.log_memory_usage(message)
    return 



class TransformerLayer(MegatronModule, BaseTransformerLayer):
    """A single transformer layer.

    Transformer layer takes input with size [s, b, h] and returns an
    output of the same size.
    """

    def __init__(
        self,
        config: TransformerConfig,
        submodules: TransformerLayerSubmodules,
        layer_number: int = 1,
        hidden_dropout: Optional[float] = None,
    ):
        super().__init__(config=config)

        # Enable cuda graphs.
        if config.enable_cuda_graph or config.external_cuda_graph:
            assert not (
                config.enable_cuda_graph and config.external_cuda_graph
            ), "Cudagraphs and external cudagraphs cannot be enabled at the same time"
            if config.enable_cuda_graph:
                if not self.training:
                    # Cudagraphs for inference are only enabled with the flash decoding kernel
                    assert (
                        self.config.flash_decode
                    ), "--flash-decode is required to use CUDA graphs during inference"
                self.cudagraph_manager = CudaGraphManager(config)
            else:
                # List to store CUDA graphs. A list of `N` CUDA graphs for this layer where N is
                # the number of microbatches. Multiple CUDA graphs per layer is required to support
                # pipelining which requires running FWD graph of multiple microbatches before BWD
                # graph. To enable CUDA graph, this list should be populated in the model training
                # script with the graphs returned by make_graphed_callables API before the first
                # training step.
                self.cuda_graphs = []
                # List to store forward pre-hooks. Forward pre-hooks are not captured into CUDA
                # graphs. Those hooks and args are collected in this list and should be manually
                # triggered before CUDA Graph running. This is required to ensure the correct param
                # all-gather overlap with forward compute.
                self.cuda_graph_manual_hooks = []
                self.current_microbatch = -1

        self.submodules_config = submodules
        self.layer_number = layer_number + get_transformer_layer_offset(self.config)
        self.hidden_dropout = config.hidden_dropout if hidden_dropout is None else hidden_dropout

        # [Module 1: Input Layernorm] Optional Layernorm on the input data
        # TODO: add pytorch only layernorm
        self.input_layernorm = build_module(
            submodules.input_layernorm,
            config=self.config,
            hidden_size=self.config.hidden_size,
            eps=self.config.layernorm_epsilon,
        )

        attention_optional_kwargs = {}
        if config.context_parallel_size > 1 and config.cp_comm_type is not None:
            if isinstance(config.cp_comm_type, list):
                attention_optional_kwargs["cp_comm_type"] = config.cp_comm_type[self.layer_number]
            else:
                attention_optional_kwargs["cp_comm_type"] = config.cp_comm_type

        # [Module 2: SelfAttention]
        self.self_attention = build_module(
            submodules.self_attention,
            config=self.config,
            layer_number=layer_number,
            **attention_optional_kwargs,
        )

        # [Module 3: BiasDropoutFusion]
        self.self_attn_bda = build_module(submodules.self_attn_bda)

        # [Module 4: Post SelfAttention] Optional Layernorm after self-attn
        self.pre_cross_attn_layernorm = build_module(
            submodules.pre_cross_attn_layernorm,
            config=self.config,
            hidden_size=self.config.hidden_size,
            eps=self.config.layernorm_epsilon,
        )

        # [Module 5: CrossAttention]
        self.cross_attention = build_module(
            submodules.cross_attention,
            config=self.config,
            layer_number=layer_number,
            **attention_optional_kwargs,
        )

        # [Module 6: BiasDropoutFusion]
        self.cross_attn_bda = build_module(submodules.cross_attn_bda, config=self.config)

        # [Module 7: Pre MLP] Optional Layernorm before MLP
        self.pre_mlp_layernorm = build_module(
            submodules.pre_mlp_layernorm,
            config=self.config,
            hidden_size=self.config.hidden_size,
            eps=self.config.layernorm_epsilon,
        )
        # [Module 8: MLP block]
        self.mlp = build_module(submodules.mlp, config=self.config)
        if hasattr(self.mlp, 'set_layer_number'):
            self.mlp.set_layer_number(self.layer_number)

        # [Module 9: BiasDropoutFusion]
        self.mlp_bda = build_module(submodules.mlp_bda)

        self.recompute_input_layernorm = False
        self.recompute_pre_mlp_layernorm = False
        self.recompute_mlp = False
        if self.config.recompute_granularity == 'selective':
            if "layernorm" in self.config.recompute_modules:
                if not isinstance(self.input_layernorm, IdentityOp):
                    self.recompute_input_layernorm = True
                if not isinstance(self.pre_mlp_layernorm, IdentityOp):
                    self.recompute_pre_mlp_layernorm = True
            if "mlp" in self.config.recompute_modules:
                from megatron.core.transformer.moe.moe_layer import MoELayer

                if not isinstance(self.mlp, MoELayer):
                    self.recompute_mlp = True

        # @jcasper how should we handle nvfuser?
        # Set bias+dropout+add fusion grad_enable execution handler.
        # TORCH_MAJOR = int(torch.__version__.split('.')[0])
        # TORCH_MINOR = int(torch.__version__.split('.')[1])
        # use_nvfuser = TORCH_MAJOR > 1 or (TORCH_MAJOR == 1 and TORCH_MINOR >= 10)
        # self.bias_dropout_add_exec_handler = nullcontext if use_nvfuser else torch.enable_grad
        self.bias_dropout_add_exec_handler = torch.enable_grad

    @staticmethod
    def _get_layer_offset(config: TransformerConfig):
        """
        Get the layer offset for the current pipeline stage.

        Deprecated: please use `get_transformer_layer_offset` instead.
        """

        warnings.warn(
            "TransformerLayer._get_layer_offset is deprecated."
            "Please use get_transformer_layer_offset instead."
        )
        return get_transformer_layer_offset(config)

    def forward(self, *args, **kwargs):
        """
        Perform a forward pass through the transformer layer.

        This method calls the core computation of a transformer layer, including
        self-attention, cross-attention (if applicable), and feed-forward operations.
        """
        log_memory_usage(f"(L{self.layer_number}) forward(init, before attention)")
        pre_mlp_layernorm_output, residual, context = self._forward_attention(*args, **kwargs)
        log_memory_usage(f"(L{self.layer_number}) forward(after attention, before mlp)")
        output = self._forward_mlp(pre_mlp_layernorm_output, residual)
        log_memory_usage(f"(L{self.layer_number}) forward(end, after mlp)")
        return output, context

    def _forward_attention(
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
        packed_seq_params: Optional[PackedSeqParams] = None,
        sequence_len_offset: Optional[Tensor] = None,
        *,
        inference_params: Optional[Any] = None,
    ):
        """
        Perform a forward pass through the attention layer and the layernorms before and after
        the attention operations.

        Args:
            hidden_states (Tensor): Input tensor of shape [s, b, h] where s is sequence length,
                b is batch size, and h is hidden size.
            attention_mask (Tensor): Mask tensor for self-attention.
            context (Tensor, optional): Context tensor for cross-attention.
            context_mask (Tensor, optional): Mask tensor for cross-attention.
            rotary_pos_emb (Tensor, optional): Rotary positional embeddings.
            attention_bias (Tensor, optional): Bias tensor for Q * K.T.
            inference_context (object, optional): Parameters for inference-time optimizations.
            packed_seq_params (object, optional): Parameters for packed sequence processing.
            sequence_len_offset (Tensor, optional): Offset along sequence dimension
                during inference.

        Returns:
            Tuple[Tensor, Tensor, Tensor]: A tuple containing:
                pre_mlp_layernorm_output (Tensor): Transformed hidden states before the MLP.
                residual (Tensor): Residual connection.
                context (Tensor): Updated context tensor if cross-attention is used,
                otherwise None.
        """

        log_memory_usage(f"(L{self.layer_number}) _forward_attention(init, before input layernorm)")

        inference_context = deprecate_inference_params(inference_context, inference_params)

        # Residual connection.
        residual = hidden_states

        
        # Optional Input Layer norm
        if self.recompute_input_layernorm:
            self.input_layernorm_checkpoint = tensor_parallel.CheckpointWithoutOutput()
            input_layernorm_output = self.input_layernorm_checkpoint.checkpoint(
                self.input_layernorm, hidden_states
            )
        else:
            input_layernorm_output = self.input_layernorm(hidden_states)

        log_memory_usage(f"(L{self.layer_number}) _forward_attention(after input layernorm, before self attention)")

        # Self attention.
        attention_output_with_bias = self.self_attention(
            input_layernorm_output,
            attention_mask=attention_mask,
            inference_context=inference_context,
            rotary_pos_emb=rotary_pos_emb,
            rotary_pos_cos=rotary_pos_cos,
            rotary_pos_sin=rotary_pos_sin,
            attention_bias=attention_bias,
            packed_seq_params=packed_seq_params,
            sequence_len_offset=sequence_len_offset,
        )

        log_memory_usage(f"(L{self.layer_number}) _forward_attention(after self attention, before self attn bda)")

        if self.recompute_input_layernorm:
            # discard the output of the input layernorm and register the recompute
            # as a gradient hook of attention_output_with_bias[0]
            self.input_layernorm_checkpoint.discard_output_and_register_recompute(
                attention_output_with_bias[0]
            )

        log_memory_usage(f"(L{self.layer_number}) _forward_attention(after self attn bda, before cross attention)")

        # TODO: could we move `bias_dropout_add_exec_handler` itself
        # inside the module provided in the `bias_dropout_add_spec` module?
        with self.bias_dropout_add_exec_handler():
            hidden_states = self.self_attn_bda(self.training, self.config.bias_dropout_fusion)(
                attention_output_with_bias, residual, self.hidden_dropout
            )

        log_memory_usage(f"(L{self.layer_number}) _forward_attention(after cross attn bda, before cross attn layernorm)")

        # Residual connection.
        residual = hidden_states

        # Optional Layer norm after self-attention
        log_memory_usage(f"(L{self.layer_number}) _forward_attention(after self attn bda, before pre cross attn layernorm)")
        pre_cross_attn_layernorm_output = self.pre_cross_attn_layernorm(hidden_states)

        log_memory_usage(f"(L{self.layer_number}) _forward_attention(after pre cross attn layernorm, before cross attention)")

        # Cross attention.
        attention_output_with_bias = self.cross_attention(
            pre_cross_attn_layernorm_output,
            attention_mask=context_mask,
            key_value_states=context,
            inference_context=inference_context,
        )

        log_memory_usage(f"(L{self.layer_number}) _forward_attention(after cross attention, before cross attn bda, before context)")

        if isinstance(attention_output_with_bias, dict) and "context" in attention_output_with_bias:
            context = attention_output_with_bias["context"]

        # TODO: could we move `bias_dropout_add_exec_handler` itself
        # inside the module provided in the `bias_dropout_add_spec` module?
        with self.bias_dropout_add_exec_handler():
            hidden_states = self.cross_attn_bda(self.training, self.config.bias_dropout_fusion)(
                attention_output_with_bias, residual, self.hidden_dropout
            )

        log_memory_usage(f"(L{self.layer_number}) _forward_attention(after cross attn bda, before pre mlp layernorm)")

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

        log_memory_usage(f"(L{self.layer_number}) _forward_attention(end)")


        
        return pre_mlp_layernorm_output, residual, context