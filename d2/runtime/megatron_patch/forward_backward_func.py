import contextlib
from typing import Union, Iterator, List

import torch
from megatron.core import parallel_state
from megatron.core.enums import ModelType
from megatron.core.pipeline_parallel import p2p_communication
from megatron.core.transformer.cuda_graphs import create_cudagraphs
from megatron.core.utils import (
    get_model_config,
    get_model_type,
    get_model_config,
    get_model_xattn,
)
from megatron.core.pipeline_parallel.schedules import (
    clear_embedding_activation_buffer,
    get_tensor_shapes,
    recv_forward,
    send_forward,
    recv_backward,
    send_backward,
    send_forward_recv_backward,
    send_backward_recv_forward,
    forward_step,
    custom_backward,
    # backward_step,
    check_first_val_step,
    deallocate_output_tensor,
    finish_embedding_wgrad_compute,
)


from torch.cuda.nvtx import range_push, range_pop

def send_forward_recv_forward(output_tensors, recv_prev, tensor_shapes, config):
    """Wrapper for p2p_communication.send_backward_recv_forward used
    with non-interleaving schedule."""
    if not isinstance(output_tensors, list):
        output_tensors = [output_tensors]
    input_tensors = []
    for output_tensor, tensor_shape in zip(output_tensors, tensor_shapes):
        if tensor_shape is None:
            input_tensors.append(None)
            continue
        input_tensor = p2p_communication.send_forward_recv_forward(
            output_tensor, recv_prev, tensor_shape, config
        )
        input_tensors.append(input_tensor)
    return input_tensors


def send_backward_recv_backward(input_tensor_grads, recv_next, tensor_shapes, config):
    """Wrapper for p2p_communication.send_backward_recv_forward used
    with non-interleaving schedule."""
    if not isinstance(input_tensor_grads, list):
        input_tensor_grads = [input_tensor_grads]
    output_tensor_grads = []
    for input_tensor_grad, tensor_shape in zip(input_tensor_grads, tensor_shapes):
        if tensor_shape is None:
            output_tensor_grads.append(None)
            continue
        output_tensor_grad = p2p_communication.send_backward_recv_backward(
            input_tensor_grad, recv_next, tensor_shape, config
        )
        output_tensor_grads.append(output_tensor_grad)
    return output_tensor_grads


def send_forward_backward_recv_forward_backward(output_tensors, input_tensor_grads, recv_prev, recv_next, tensor_shapes, config):
    """Wrapper for p2p_communication.send_backward_recv_forward used
    with non-interleaving schedule."""
    if not isinstance(output_tensors, list):
        output_tensors = [output_tensors]
    if not isinstance(input_tensor_grads, list):
        input_tensor_grads = [input_tensor_grads]
    input_tensors = []
    output_tensor_grads = []
    for output_tensor, input_tensor_grad, tensor_shape in zip(output_tensors, input_tensor_grads, tensor_shapes):
        if tensor_shape is None:
            input_tensors.append(None)
            continue
        input_tensor, output_tensor_grad = p2p_communication.send_forward_backward_recv_forward_backward(
            output_tensor, input_tensor_grad, recv_prev, recv_next, tensor_shape, config
        )
        input_tensors.append(input_tensor)
        output_tensor_grads.append(output_tensor_grad)
    return input_tensors, output_tensor_grads


forward_backward_pipelining_without_interleaving_first_run = True


def forward_backward_pipelining_without_interleaving(
    *,
    forward_step_func,
    data_iterator: Union[Iterator, List[Iterator]],
    model: Union[torch.nn.Module, List[torch.nn.Module]],
    num_microbatches: int,
    seq_length: int,
    micro_batch_size: int,
    decoder_seq_length: int = None,
    forward_only: bool = False,
    collect_non_loss_data: bool = False,
    first_val_step: bool = None,
    dummy_bwd_func=None,
):
    """Run non-interleaved 1F1B schedule, with communication between pipeline
    stages. Returns dictionary with losses if the last stage, empty dict otherwise."""

    if isinstance(model, list):
        assert (
            len(model) == 1
        ), "non-interleaved pipeline-parallel schedule does not support model chunking"
        model = model[0]
    if isinstance(data_iterator, list):
        assert (
            len(data_iterator) == 1
        ), "non-interleaved pipeline-parallel schedule does not support model chunking"
        data_iterator = data_iterator[0]

    config = get_model_config(model)
    if config.overlap_p2p_comm:
        raise ValueError(
            "Non-interleaved pipeline parallelism does not support overlapping p2p communication"
        )

    # Needed only when gradients are finalized in M-Core
    if config.finalize_model_grads_func is not None and not forward_only:
        embedding_module = clear_embedding_activation_buffer(config, model)

    if config.timers is not None:
        config.timers('forward-backward', log_level=1).start(barrier=config.barrier_with_L1_time)

    # Disable async grad reductions
    no_sync_func = config.no_sync_func
    if no_sync_func is None:
        no_sync_func = contextlib.nullcontext
    no_sync_context = None

    def disable_grad_sync():
        """Disable asynchronous grad reductions"""
        nonlocal no_sync_context
        if no_sync_context is None:
            no_sync_context = no_sync_func()
            no_sync_context.__enter__()

    def enable_grad_sync():
        """Enable asynchronous grad reductions"""
        nonlocal no_sync_context
        if no_sync_context is not None:
            no_sync_context.__exit__(None, None, None)
            no_sync_context = None

    disable_grad_sync()

    # Compute number of warmup microbatches.
    num_warmup_microbatches = (
        parallel_state.get_pipeline_model_parallel_world_size()
        - parallel_state.get_pipeline_model_parallel_rank()
        - 1
    )
    num_warmup_microbatches = min(num_warmup_microbatches, num_microbatches)
    num_microbatches_remaining = num_microbatches - num_warmup_microbatches
    # NOTE: this function can only works for num_microbatches >= PP size.

    # Checkpoint the activations of partial Transformer layers in a number of micro-batches
    # within the maximum outstanding micro-batch backpropagations.
    # Micro-batches with the ids less than 'num_microbatches_with_partial_activation_checkpoints'
    # checkpoint partial Transformer layers (or skip checkpointing) and
    # the rest of micro-batches within a window of micro-batches checkpoint
    # all Transformer layers. The window of micro-batches is set by the maximum
    # outstanding backpropagations and becomes smaller at later pipeline stages.
    # Please refer the appendix C in https://arxiv.org/pdf/2205.05198.pdf
    max_outstanding_backprops = None
    if config.num_microbatches_with_partial_activation_checkpoints is not None:
        max_outstanding_backprops = num_warmup_microbatches + 1

    model_type = get_model_type(model)
    encoder_decoder_xattn = get_model_xattn(model)

    rank = parallel_state.get_pipeline_model_parallel_rank()
    recv_tensor_shapes = get_tensor_shapes(
        rank=rank - 1,
        model_type=model_type,
        seq_length=seq_length,
        micro_batch_size=micro_batch_size,
        decoder_seq_length=decoder_seq_length,
        config=config,
        encoder_decoder_xattn=encoder_decoder_xattn,
    )
    send_tensor_shapes = get_tensor_shapes(
        rank=rank,
        model_type=model_type,
        seq_length=seq_length,
        micro_batch_size=micro_batch_size,
        decoder_seq_length=decoder_seq_length,
        config=config,
        encoder_decoder_xattn=encoder_decoder_xattn,
    )

    # Input, output tensors only need to be saved when doing backward passes
    input_tensors = None
    output_tensors = None
    total_num_tokens = torch.tensor(0, dtype=torch.int).cuda()

    if not forward_only:
        input_tensors = []
        output_tensors = []
    forward_data_store = []
    dummy_data_store = []

    global forward_backward_pipelining_without_interleaving_first_run
    if forward_backward_pipelining_without_interleaving_first_run:
        # NOTE: send/recv something to ensure comm is initialized.
        send_forward([torch.zeros(*send_tensor_shapes, dtype=torch.bfloat16).cuda()], send_tensor_shapes, config)
        recv_forward(recv_tensor_shapes, config)
        forward_backward_pipelining_without_interleaving_first_run = False

    num_warmup_microbatches_including_dummy = parallel_state.get_pipeline_model_parallel_world_size() - 1

    # Run warmup forward passes.
    for i in range(num_warmup_microbatches_including_dummy):
        if i < rank:
            with torch.no_grad():
                _ = forward_step(
                    forward_step_func,
                    data_iterator,
                    model,
                    num_microbatches,
                    [torch.tensor([]).cuda()],
                    dummy_data_store,
                    config,
                    collect_non_loss_data,
                    None,
                    False,
                    current_microbatch=i - rank,
                    encoder_decoder_xattn=encoder_decoder_xattn,
                )
        else:
            # Decide to checkpoint all layers' activations of the current micro-batch
            if max_outstanding_backprops is not None:
                checkpoint_activations_microbatch = (
                    i % max_outstanding_backprops
                    >= config.num_microbatches_with_partial_activation_checkpoints
                )
            else:
                checkpoint_activations_microbatch = None

            if i == rank:
                input_tensor = recv_forward(recv_tensor_shapes, config)

            output_tensor, num_tokens = forward_step(
                forward_step_func,
                data_iterator,
                model,
                num_microbatches,
                input_tensor,
                forward_data_store,
                config,
                collect_non_loss_data,
                checkpoint_activations_microbatch,
                check_first_val_step(first_val_step, forward_only, i == 0),
                current_microbatch=i - rank,
                encoder_decoder_xattn=encoder_decoder_xattn,
            )
            if not forward_only:
                input_tensors.append(input_tensor)
                output_tensors.append(output_tensor)

            input_tensor = send_forward_recv_forward(
                output_tensor, not parallel_state.is_pipeline_first_stage(), send_tensor_shapes, config
            )

            if not forward_only:
                deallocate_output_tensor(output_tensor[0], config.deallocate_pipeline_outputs)

            total_num_tokens += num_tokens


    # Before running 1F1B, need to receive first forward tensor.
    # If all microbatches are run in warmup / cooldown phase, then no need to
    # receive this tensor here.
    # if num_microbatches_remaining > 0:
    #     input_tensor = recv_forward(recv_tensor_shapes, config)

    # Run 1F1B in steady state.
    for i in range(num_microbatches):
        forward_dummy = i >= num_microbatches_remaining
        backward_dummy = i < num_warmup_microbatches
        next_forward_dummy = i + 1 >= num_microbatches_remaining
        next_backward_dummy = i + 1 < num_warmup_microbatches
        if i == 0 == num_warmup_microbatches:
            output_tensor_grad = recv_backward(send_tensor_shapes, config)
        if i == 0 == num_warmup_microbatches:
            input_tensor = recv_forward(recv_tensor_shapes, config)

        if forward_dummy:
            with torch.no_grad():
                _ = forward_step(
                    forward_step_func,
                    data_iterator,
                    model,
                    num_microbatches,
                    [torch.tensor([]).cuda()],
                    dummy_data_store,
                    config,
                    collect_non_loss_data,
                    None,
                    False,
                    current_microbatch=i - num_microbatches_remaining - parallel_state.get_pipeline_model_parallel_world_size(),
                    encoder_decoder_xattn=encoder_decoder_xattn,
                )
        else:
            last_iteration = i == (num_microbatches_remaining - 1)

            # Decide to checkpoint all layers' activations of the current micro-batch
            if max_outstanding_backprops is not None:
                checkpoint_activations_microbatch = (
                    (i + num_warmup_microbatches) % max_outstanding_backprops
                ) >= config.num_microbatches_with_partial_activation_checkpoints
            else:
                checkpoint_activations_microbatch = None

            output_tensor, num_tokens = forward_step(
                forward_step_func,
                data_iterator,
                model,
                num_microbatches,
                input_tensor,
                forward_data_store,
                config,
                collect_non_loss_data,
                checkpoint_activations_microbatch,
                check_first_val_step(
                    first_val_step, forward_only, (i == 0) and (num_warmup_microbatches == 0)
                ),
                current_microbatch=i + num_warmup_microbatches,
                encoder_decoder_xattn=encoder_decoder_xattn,
            )
            total_num_tokens += num_tokens

            if forward_only:
                if parallel_state.is_pipeline_last_stage():
                    output_tensor = [None]
                input_tensor = send_forward_recv_forward(
                    output_tensor, not parallel_state.is_pipeline_first_stage() and not next_forward_dummy, send_tensor_shapes, config
                )

            if not forward_only:
                # Add input_tensor and output_tensor to end of list.
                input_tensors.append(input_tensor)
                output_tensors.append(output_tensor)
                save_output_tensor = output_tensor

        if forward_only:
            pass
        else:
            if backward_dummy:
                dummy_bwd_func(model)

            else:

                # Pop input_tensor and output_tensor from the start of the list for
                # the backward pass.
                input_tensor = input_tensors.pop(0)
                output_tensor = output_tensors.pop(0)

                # Enable grad sync for the last microbatch in the batch if the full
                # backward pass completes in the 1F1B stage.
                if num_warmup_microbatches == 0 and last_iteration:
                    if config.grad_sync_func is None or rank == 0:
                        enable_grad_sync()

                input_tensor_grad = backward_step(
                    input_tensor, output_tensor, output_tensor_grad, model_type, config
                )

            if parallel_state.is_pipeline_first_stage() or backward_dummy:
                input_tensor_grad = [None]
            if parallel_state.is_pipeline_last_stage() or forward_dummy:
                save_output_tensor = [None]
            torch.cuda.nvtx.range_push(f'send forward backward {rank=}, {i=}')
            input_tensor, output_tensor_grad = send_forward_backward_recv_forward_backward(
                save_output_tensor, input_tensor_grad,
                not parallel_state.is_pipeline_first_stage() and not next_forward_dummy,
                not parallel_state.is_pipeline_last_stage() and not next_backward_dummy,
                send_tensor_shapes, config
            )
            deallocate_output_tensor(save_output_tensor[0], config.deallocate_pipeline_outputs)
            torch.cuda.nvtx.range_pop()

    # Run cooldown backward passes.
    if not forward_only:
        for i in range(num_warmup_microbatches_including_dummy):
            dummy = i + rank >= num_warmup_microbatches_including_dummy
            next_dummy = i + 1 + rank >= num_warmup_microbatches_including_dummy
            if dummy:
                dummy_bwd_func(model)
            else:
                # Enable async grad reduction in the last backward pass
                # Note: If grad sync function is provided, only enable
                # async grad reduction in first pipeline stage. Other
                # pipeline stages do grad reduction during pipeline
                # bubble.
                if i == num_warmup_microbatches - 1:
                    if config.grad_sync_func is None or rank == 0:
                        enable_grad_sync()

                input_tensor = input_tensors.pop(0)
                output_tensor = output_tensors.pop(0)

                input_tensor_grad = backward_step(
                    input_tensor, output_tensor, output_tensor_grad, model_type, config
                )

                output_tensor_grad = send_backward_recv_backward(
                    input_tensor_grad, not parallel_state.is_pipeline_last_stage() and not next_dummy, recv_tensor_shapes, config
                )

        # Launch any remaining grad reductions.
        if no_sync_context is not None:
            enable_grad_sync()
            if config.grad_sync_func is not None:
                config.grad_sync_func(model.parameters())

    if config.finalize_model_grads_func is not None and not forward_only:

        # If defer_embedding_wgrad_compute is enabled we need to do the
        # weight gradient GEMM's here.
        finish_embedding_wgrad_compute(config, embedding_module)

        # Finalize model grads (perform full grad all-reduce / reduce-scatter for
        # data parallelism, layernorm all-reduce for sequence parallelism, and
        # embedding all-reduce for pipeline parallelism).
        config.finalize_model_grads_func(
            [model], total_num_tokens if config.calculate_per_token_loss else None
        )

    if config.timers is not None:
        config.timers('forward-backward').stop()

    if hasattr(config, 'enable_cuda_graph') and config.enable_cuda_graph:
        create_cudagraphs()

    return forward_data_store


def backward_step(input_tensor, output_tensor, output_tensor_grad, model_type, config, retain_graph=False):
    """Backward step through passed-in output tensor.

    If last stage, output_tensor_grad is None, otherwise gradient of loss
    with respect to stage's output tensor.

    Returns gradient of loss with respect to input tensor (None if first
    stage)."""
    range_push("backward_step")

    # NOTE: This code currently can handle at most one skip connection. It
    # needs to be modified slightly to support arbitrary numbers of skip
    # connections.

    if config.timers is not None:
        config.timers('backward-compute', log_level=2).start()

    # Retain the grad on the input_tensor.
    unwrap_input_tensor_grad = False
    if not isinstance(input_tensor, list):
        input_tensor = [input_tensor]
        unwrap_input_tensor_grad = True
    for x in input_tensor:
        if x is not None:
            x.retain_grad()

    if not isinstance(output_tensor, list):
        output_tensor = [output_tensor]
    if not isinstance(output_tensor_grad, list):
        output_tensor_grad = [output_tensor_grad]

    # Backward pass.
    if output_tensor_grad[0] is None and config.grad_scale_func is not None:
        output_tensor[0] = config.grad_scale_func(output_tensor[0])

    # In multi-modal models like VLM, some batches may not have images.
    # When no image is present, the vision encoder (as a separate pipeline stage)
    # will not participate in the computation.
    # This results in a tensor that does not require gradients.
    # In such cases, we intentionally skip the backward pass while preserving zero gradients.
    if output_tensor[0].requires_grad:
        if config.deallocate_pipeline_outputs:
            custom_backward(output_tensor[0], output_tensor_grad[0])
        else:
            torch.autograd.backward(output_tensor[0], grad_tensors=output_tensor_grad[0], retain_graph=retain_graph)

    # Collect the grad of the input_tensor.
    input_tensor_grad = [None]
    if input_tensor is not None:
        input_tensor_grad = []
        for x in input_tensor:
            if x is None:
                input_tensor_grad.append(None)
            else:
                input_tensor_grad.append(x.grad)

    # Handle single skip connection if it exists (encoder_hidden_state in
    # model with encoder and decoder).
    if (
        parallel_state.get_pipeline_model_parallel_world_size() > 1
        and model_type == ModelType.encoder_and_decoder
        and len(output_tensor_grad) > 1  # excludes models that lack a skip connection.
    ):
        if output_tensor_grad[1] is not None:
            assert input_tensor_grad[-1] is not None
            input_tensor_grad[-1].add_(output_tensor_grad[1])
    if unwrap_input_tensor_grad:
        input_tensor_grad = input_tensor_grad[0]

    if config.timers is not None:
        config.timers('backward-compute').stop()

    range_pop()

    return input_tensor_grad
