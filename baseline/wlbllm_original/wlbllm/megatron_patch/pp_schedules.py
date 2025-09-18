import contextlib
from typing import Iterator, List, Union

import torch
from torch.autograd.variable import Variable

from megatron.core import parallel_state
from megatron.core.enums import ModelType
from megatron.core.pipeline_parallel.schedules import (
    p2p_communication, 
    clear_embedding_activation_buffer,
    finish_embedding_wgrad_compute,
    get_tensor_shapes, 
    recv_forward,
    send_forward,
    recv_backward,
    send_backward,
    send_forward_recv_backward,
    send_backward_recv_forward,
    deallocate_output_tensor,
    forward_step, backward_step,
    check_first_val_step, backward_step,
)
import megatron.core.pipeline_parallel.schedules

from megatron.core.transformer.cuda_graphs import create_cudagraphs
from megatron.core.transformer.moe.router import MoEAuxLossAutoScaler
from megatron.core.transformer.multi_token_prediction import MTPLossAutoScaler
from megatron.core.utils import (
    drain_embedding_wgrad_compute,
    get_attr_wrapped_model,
    get_model_config,
    get_model_type,
    get_model_xattn,
)

import os
import wlbllm.registry
from typing import Union, Iterator, List
import torch
import contextlib




def wlb_swap_next_forward_metadata():
    # Call this function before entering forward step.
    swap_metadata_fn = wlbllm.registry.get("swap_metadata_fn")
    forward_cnt = wlbllm.registry.get("forward_cnt")
    print(f"🟡 wlb_swap_next_forward_metadata[{forward_cnt}]: start swapping forward metadata")
    swap_metadata_fn(forward_cnt)
    print(f"🟡 wlb_swap_next_forward_metadata[{forward_cnt}]: end swapping forward metadata")
    wlbllm.registry.set("forward_cnt", forward_cnt + 1)
    return

def wlb_swap_next_backward_metadata():
    # Call this function before entering backward step.
    swap_metadata_fn = wlbllm.registry.get("swap_metadata_fn")
    backward_cnt = wlbllm.registry.get("backward_cnt")
    print(f"🟡 wlb_swap_next_backward_metadata[{backward_cnt}]: start swapping backward metadata")
    swap_metadata_fn(backward_cnt)
    print(f"🟡 wlb_swap_next_backward_metadata[{backward_cnt}]: end swapping backward metadata")
    wlbllm.registry.set("backward_cnt", backward_cnt + 1)
    return

def is_wlb_func():
    return os.environ["WLBLLM_MODE"] == "1"



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
):
    """Run non-interleaved 1F1B schedule, with communication between pipeline
    stages. Returns dictionary with losses if the last stage, empty dict otherwise."""
    if is_wlb_func():
        print(f"🟡 forward_backward_pipelining_without_interleaving: is_wlb_func()")
        def forward_step_func_with_wlb_metadata(*args, **kwargs):
            wlb_swap_next_forward_metadata()
            return forward_step(*args, **kwargs)
        
        def backward_step_func_with_wlb_metadata(*args, **kwargs):
            wlb_swap_next_backward_metadata()
            return backward_step(*args, **kwargs)
        
        forward_step__func = forward_step_func_with_wlb_metadata
        backward_step__func = backward_step_func_with_wlb_metadata
    else:
        forward_step__func = forward_step
        backward_step__func = backward_step
        pass


    # Add the counter of forward and backward steps.
    forward_batch_id = 0
    backward_batch_id = 0

    # Add nvtx around forward/backward step
    def forward_step__with_nvtx(*args, **kwargs):
        nonlocal forward_batch_id
        with torch.cuda.nvtx.range(f"forward_step[{forward_batch_id}]"):
            ret = forward_step__func(*args, **kwargs)
        forward_batch_id += 1
        return ret
    
    def backward_step__with_nvtx(*args, **kwargs):
        nonlocal backward_batch_id
        with torch.cuda.nvtx.range(f"backward_step[{backward_batch_id}]"):
            ret = backward_step__func(*args, **kwargs)
        backward_batch_id += 1
        return ret
    
    modified_forward_step = forward_step__with_nvtx
    modified_backward_step = backward_step__with_nvtx



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

    # Run warmup forward passes.
    for i in range(num_warmup_microbatches):
        # Decide to checkpoint all layers' activations of the current micro-batch
        if max_outstanding_backprops is not None:
            checkpoint_activations_microbatch = (
                i % max_outstanding_backprops
                >= config.num_microbatches_with_partial_activation_checkpoints
            )
        else:
            checkpoint_activations_microbatch = None

        input_tensor = recv_forward(recv_tensor_shapes, config)
        print(f"🟡 forward_backward_pipelining_without_interleaving[{i}]: start forward step")
        output_tensor, num_tokens = modified_forward_step(
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
            current_microbatch=i,
            encoder_decoder_xattn=encoder_decoder_xattn,
        )
        print(f"🟡 forward_backward_pipelining_without_interleaving[{i}]: end forward step")
        print(f"🟡 forward_backward_pipelining_without_interleaving[{i}]: start send forward")
        send_forward(output_tensor, send_tensor_shapes, config)
        print(f"🟡 forward_backward_pipelining_without_interleaving[{i}]: end send forward")
        total_num_tokens += num_tokens

        if not forward_only:
            input_tensors.append(input_tensor)
            output_tensors.append(output_tensor)
            deallocate_output_tensor(output_tensor[0], config.deallocate_pipeline_outputs)

    # Before running 1F1B, need to receive first forward tensor.
    # If all microbatches are run in warmup / cooldown phase, then no need to
    # receive this tensor here.
    if num_microbatches_remaining > 0:
        print(f"🟡 forward_backward_pipelining_without_interleaving[remaining]: start recv forward")
        input_tensor = recv_forward(recv_tensor_shapes, config)
        print(f"🟡 forward_backward_pipelining_without_interleaving[remaining]: end recv forward")

    # Run 1F1B in steady state.
    for i in range(num_microbatches_remaining):
        last_iteration = i == (num_microbatches_remaining - 1)

        # Decide to checkpoint all layers' activations of the current micro-batch
        if max_outstanding_backprops is not None:
            checkpoint_activations_microbatch = (
                (i + num_warmup_microbatches) % max_outstanding_backprops
            ) >= config.num_microbatches_with_partial_activation_checkpoints
        else:
            checkpoint_activations_microbatch = None

        print(f"🟡 forward_backward_pipelining_without_interleaving[1F1B{i}]: start forward step")
        output_tensor, num_tokens = modified_forward_step(
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
            print(f"🟡 forward_backward_pipelining_without_interleaving[1F1B{i}]: start send forward")
            send_forward(output_tensor, send_tensor_shapes, config)
            print(f"🟡 forward_backward_pipelining_without_interleaving[1F1B{i}]: end send forward")

            if not last_iteration:
                print(f"🟡 forward_backward_pipelining_without_interleaving[1F1B{i}]: start recv forward")
                input_tensor = recv_forward(recv_tensor_shapes, config)
                print(f"🟡 forward_backward_pipelining_without_interleaving[1F1B{i}]: end recv forward")

        else:
            print(f"🟡 forward_backward_pipelining_without_interleaving[1F1B{i}]: start send forward recv backward")
            output_tensor_grad = send_forward_recv_backward(
                output_tensor, send_tensor_shapes, config
            )
            print(f"🟡 forward_backward_pipelining_without_interleaving[1F1B{i}]: end send forward recv backward")

            # Add input_tensor and output_tensor to end of list.
            input_tensors.append(input_tensor)
            output_tensors.append(output_tensor)
            deallocate_output_tensor(output_tensor[0], config.deallocate_pipeline_outputs)

            # Pop input_tensor and output_tensor from the start of the list for
            # the backward pass.
            input_tensor = input_tensors.pop(0)
            output_tensor = output_tensors.pop(0)

            # Enable grad sync for the last microbatch in the batch if the full
            # backward pass completes in the 1F1B stage.
            if num_warmup_microbatches == 0 and last_iteration:
                if config.grad_sync_func is None or rank == 0:
                    enable_grad_sync()

            print(f"🟡 forward_backward_pipelining_without_interleaving[1F1B{i}]: start backward step")
            input_tensor_grad = modified_backward_step(
                input_tensor, output_tensor, output_tensor_grad, model_type, config
            )
            print(f"🟡 forward_backward_pipelining_without_interleaving[1F1B{i}]: end backward step")

            if last_iteration:
                input_tensor = None
                print(f"🟡 forward_backward_pipelining_without_interleaving[1F1B{i}]: start send backward")
                send_backward(input_tensor_grad, recv_tensor_shapes, config)
                print(f"🟡 forward_backward_pipelining_without_interleaving[1F1B{i}]: end send backward")
            else:
                print(f"🟡 forward_backward_pipelining_without_interleaving[1F1B{i}]: start send backward recv forward")
                input_tensor = send_backward_recv_forward(
                    input_tensor_grad, recv_tensor_shapes, config
                )
                print(f"🟡 forward_backward_pipelining_without_interleaving[1F1B{i}]: end send backward recv forward")

    # Run cooldown backward passes.
    if not forward_only:
        for i in range(num_warmup_microbatches):

            # Enable async grad reduction in the last backward pass
            # Note: If grad sync function is provided, only enable
            # async grad reduction in first pipeline stage. Other
            # pipeline stages do grad reduction during pipeline
            # bubble.
            if i == num_warmup_microbatches - 1:
                if config.grad_sync_func is None or rank == 0:
                    enable_grad_sync()

            print(f"🟡 forward_backward_pipelining_without_interleaving[cooldown{i}]: start backward step")
            input_tensor = input_tensors.pop(0)
            output_tensor = output_tensors.pop(0)

            output_tensor_grad = recv_backward(send_tensor_shapes, config)

            input_tensor_grad = modified_backward_step(
                input_tensor, output_tensor, output_tensor_grad, model_type, config
            )
            print(f"🟡 forward_backward_pipelining_without_interleaving[cooldown{i}]: end backward step")
            
            print(f"🟡 forward_backward_pipelining_without_interleaving[cooldown{i}]: start send backward")
            send_backward(input_tensor_grad, recv_tensor_shapes, config)
            print(f"🟡 forward_backward_pipelining_without_interleaving[cooldown{i}]: end send backward")

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
