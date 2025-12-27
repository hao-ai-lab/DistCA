"""
Modified version of megatron.training.training.py to support DistCA.
- train(): main training loop
- train_step(): single training step
"""
import dataclasses
from datetime import datetime
import functools
import gc
import logging
import math
import os
import sys
from typing import List

import torch.distributed
from megatron.training.log_handler import CustomHandler
# Make default logging level INFO, but filter out all log messages not from MCore.
logging.basicConfig(handlers=[CustomHandler()], level=logging.INFO)
from megatron.training.theoretical_memory_usage import report_theoretical_memory
import time
# The earliest we can measure the start time.
_TRAIN_START_TIME = time.time()
import torch

from megatron.core import mpu, tensor_parallel
from megatron.core.utils import (
    check_param_hashes_across_dp_replicas,
    get_model_config,
    StragglerDetector,
    is_te_min_version,
)
from megatron.core.fp8_utils import correct_amax_history_if_needed
from megatron.training.checkpointing import load_checkpoint
from megatron.training.checkpointing import save_checkpoint
from megatron.training.checkpointing import checkpoint_exists
from megatron.core.transformer.module import Float16Module
from megatron.core.distributed import DistributedDataParallelConfig
from megatron.core.distributed import DistributedDataParallel as DDP
from megatron.core.distributed.custom_fsdp import FullyShardedDataParallel as custom_FSDP
try:
    from megatron.core.distributed import TorchFullyShardedDataParallel as torch_FSDP

    HAVE_FSDP2 = True
except ImportError:
    HAVE_FSDP2 = False

from megatron.core.distributed import finalize_model_grads
from megatron.core.enums import ModelType
from megatron.core.optimizer import get_megatron_optimizer, OptimizerConfig
from megatron.core.rerun_state_machine import (
    get_rerun_state_machine,
    destroy_rerun_state_machine,
    RerunDataIterator,
    RerunMode,
)
from megatron.training.initialize import initialize_megatron
from megatron.training.initialize import write_args_to_tensorboard
from megatron.training.initialize import set_jit_fusion_options
from megatron.training.utils import (
    get_batch_on_this_cp_rank,
    get_batch_on_this_tp_rank,
)
from megatron.legacy.data.data_samplers import build_pretraining_data_loader
from megatron.core.optimizer_param_scheduler import OptimizerParamScheduler
from megatron.core.transformer.moe import upcycling_utils
from megatron.core.transformer.moe.moe_utils import track_moe_metrics
from megatron.core.transformer.multi_token_prediction import MTPLossLoggingHelper
from megatron.core.parallel_state import (
    destroy_global_memory_buffer,
    destroy_model_parallel,
    get_amax_reduction_group,
    model_parallel_is_initialized,
)
from megatron.core.pipeline_parallel import get_forward_backward_func as get_original_forward_backward_func
from megatron.core.pipeline_parallel.schedules import (
    convert_schedule_table_to_order,
    get_pp_rank_microbatches,
    get_schedule_table,
)
from megatron.core.num_microbatches_calculator import (
    destroy_num_microbatches_calculator,
    get_current_global_batch_size,
    get_current_running_global_batch_size,
    get_num_microbatches,
    update_num_microbatches)

from megatron.training.async_utils import maybe_finalize_async_save
from megatron.training.utils import (
    append_to_progress_log,
    calc_params_l2_norm,
    check_adlr_autoresume_termination,
    logical_and_across_model_parallel_group,
    reduce_max_stat_across_model_parallel_group,
    is_last_rank,
    print_rank_0,
    print_rank_last,
    report_memory,
    unwrap_model,
    update_use_dist_ckpt,
)
from megatron.training.global_vars import (
    destroy_global_vars,
    get_args,
    get_signal_handler,
    get_timers,
    get_tensorboard_writer,
    get_wandb_writer,
    get_one_logger,
)
from megatron.training import one_logger_utils
from megatron.training import ft_integration


# ======================
# Defined in training.py
# ======================
from megatron.training.training import (
    print_datetime, report_memory, should_disable_forward_pre_hook, disable_forward_pre_hook, enable_forward_pre_hook, checkpoint_and_decide_exit, evaluate_and_print_results, post_training_step_callbacks, save_checkpoint_and_time, dummy_train_step, num_floating_point_operations, training_log, cuda_graph_capture, cuda_graph_set_manual_hooks
)

stimer = StragglerDetector()

# ======================
# DistCA patch the batch size calculator
# ======================

from megatron.core.num_microbatches_calculator import get_num_microbatches as get_original_num_microbatches
# return get_original_num_microbatches()

def get_distca_num_microbatches() -> int:
    args = get_args()
    return args.micro_batch_size

def get_num_microbatches() -> int:
    return get_distca_num_microbatches()

def get_original_batch_size(args) -> int:
    return mpu.get_data_parallel_world_size() * \
                         args.micro_batch_size * \
                         get_num_microbatches()

def get_distca_batch_size() -> int:
    return get_num_microbatches()

def get_batch_size() -> int:
    # if not args.use_distca: return get_original_batch_size(args)
    return get_distca_batch_size() 


def train(forward_step_func, model, optimizer, opt_param_scheduler,
          train_data_iterator, valid_data_iterator,
          process_non_loss_data_func, config, checkpointing_context, non_loss_data_func,
          distca_kwargs: dict = None,
          ):
    """Training function: run train_step desired number of times, run validation, checkpoint.
    
    Find `train_step()`.
    """
    args = get_args()
    timers = get_timers()
    one_logger = get_one_logger()
    distca_kwargs = distca_kwargs or {}

    if args.run_workload_inspector_server:
        try:
            from workload_inspector.utils.webserver import run_server
            import threading
            threading.Thread(target=run_server, daemon=True, args=(torch.distributed.get_rank(), )).start()
        except ModuleNotFoundError:
            print_rank_0("workload inspector module not found.")

    # Write args to tensorboard
    write_args_to_tensorboard()

    # Turn on training mode which enables dropout.
    for model_module in model:
        model_module.train()

    # Tracking loss.
    total_loss_dict = {}

    # Iterations.
    iteration = args.iteration
    # Make sure rerun_state_machine has the right iteration loaded from checkpoint.
    rerun_state_machine = get_rerun_state_machine()
    if rerun_state_machine.current_iteration != iteration:
        print_rank_0(f"Setting rerun_state_machine.current_iteration to {iteration}...")
        rerun_state_machine.current_iteration = iteration

    # Track E2E metrics at the start of training.
    one_logger_utils.on_train_start(iteration=iteration, consumed_train_samples=args.consumed_train_samples,
                                    train_samples=args.train_samples, seq_length=args.seq_length,
                                    train_iters=args.train_iters, save=args.save, async_save=args.async_save,
                                    log_throughput=args.log_throughput,
                                    num_floating_point_operations_so_far=args.num_floating_point_operations_so_far)

    num_floating_point_operations_so_far = args.num_floating_point_operations_so_far

    # Setup some training config params.
    config.grad_scale_func = optimizer.scale_loss
    config.timers = timers
    if isinstance(model[0], (custom_FSDP, DDP)) and args.overlap_grad_reduce:
        assert config.no_sync_func is None, \
            ('When overlap_grad_reduce is True, config.no_sync_func must be None; '
             'a custom no_sync_func is not supported when overlapping grad-reduce')
        config.no_sync_func = [model_chunk.no_sync for model_chunk in model]
        if len(model) == 1:
            config.no_sync_func = config.no_sync_func[0]
        if args.align_grad_reduce:
            config.grad_sync_func = [model_chunk.start_grad_sync for model_chunk in model]
            if len(model) == 1:
                config.grad_sync_func = config.grad_sync_func[0]
    if args.overlap_param_gather and args.align_param_gather:
        config.param_sync_func = [model_chunk.start_param_sync for model_chunk in model]
        if len(model) == 1:
            config.param_sync_func = config.param_sync_func[0]
    config.finalize_model_grads_func = finalize_model_grads

    timers('interval-time', log_level=0).start(barrier=True)
    print_datetime('before the start of training step')
    report_memory_flag = True
    pre_hook_enabled = False
    should_exit = False
    exit_code = 0

    if args.manual_gc:
        # Disable the default garbage collector and perform the collection manually.
        # This is to align the timing of garbage collection across ranks.
        assert args.manual_gc_interval >= 0, \
            'Manual garbage collection interval should be larger than or equal to 0'
        gc.disable()
        gc.collect()

    # Singleton initialization of straggler detector.
    if args.log_straggler:
        global stimer
        world = torch.distributed.get_world_size()
        rank = torch.distributed.get_rank()
        mmcnt = args.straggler_minmax_count
        stimer.configure(world, rank,
                mmcnt = mmcnt,
                enabled = not args.disable_straggler_on_startup,
                port = args.straggler_ctrlr_port)
    num_floating_point_operations_since_last_log_event = 0.0

    num_microbatches = get_num_microbatches()
    eval_duration = 0.0
    eval_iterations = 0

    def get_e2e_base_metrics():
        """Get base metrics values for one-logger to calculate E2E tracking metrics.
        """
        num_floating_point_operations_since_current_train_start = \
            num_floating_point_operations_so_far - args.num_floating_point_operations_so_far
        return {
            'iteration': iteration,
            'train_duration': timers('interval-time').active_time(),
            'eval_duration': eval_duration,
            'eval_iterations': eval_iterations,
            'total_flops_since_current_train_start': num_floating_point_operations_since_current_train_start,
            'num_floating_point_operations_so_far': num_floating_point_operations_so_far,
            'consumed_train_samples': args.consumed_train_samples,
            'world_size': args.world_size,
            'seq_length': args.seq_length
        }
    # Cache into one-logger for callback.
    if one_logger:
        with one_logger.get_context_manager():
            one_logger.store_set('get_e2e_base_metrics', get_e2e_base_metrics)

    prof = None
    if args.profile and torch.distributed.get_rank() in args.profile_ranks and args.use_pytorch_profiler:
        prof = torch.profiler.profile(
        schedule=torch.profiler.schedule(
            wait=max(args.profile_step_start-1, 0),
            warmup=1 if args.profile_step_start > 0 else 0,
            active=args.profile_step_end-args.profile_step_start,
            repeat=1),
        on_trace_ready=torch.profiler.tensorboard_trace_handler(args.tensorboard_dir),
        record_shapes=True,
        with_stack=True)
        prof.start()

    start_iteration = iteration
    # Disable forward pre-hook to start training to ensure that errors in checkpoint loading
    # or random initialization don't propagate to all ranks in first all-gather (which is a
    # no-op if things work correctly).
    if should_disable_forward_pre_hook(args):
        disable_forward_pre_hook(model, param_sync=False)
        # Also remove param_sync_func temporarily so that sync calls made in
        # `forward_backward_func` are no-ops.
        param_sync_func = config.param_sync_func
        config.param_sync_func = None
        pre_hook_enabled = False
    # Also, check weight hash across DP replicas to be very pedantic.
    if args.check_weight_hash_across_dp_replicas_interval is not None:
        assert check_param_hashes_across_dp_replicas(model, cross_check=True), \
            "Parameter hashes not matching across DP replicas"
        torch.distributed.barrier()
        print_rank_0(f">>> Weight hashes match after {iteration} iterations...")

    # Run training iterations till done.
    while iteration < args.train_iters:
        print_rank_0(f"<< Training Step Iteration {iteration} of {args.train_iters}")
        if args.profile and torch.distributed.get_rank() in args.profile_ranks:
            if args.use_pytorch_profiler:
                prof.step()
            elif iteration == args.profile_step_start:
                torch.cuda.cudart().cudaProfilerStart()
                torch.autograd.profiler.emit_nvtx(record_shapes=True).__enter__()

        ft_integration.on_checkpointing_start()
        maybe_finalize_async_save(blocking=False)
        ft_integration.on_checkpointing_end(is_async_finalization=True)

        
        # Update number of microbatches first without consistency check to decide if a
        # checkpoint should be saved. If the number of microbatches is different
        # from the previous iteration, save a checkpoint. Then run consistency check
        # to make sure training configuration is still valid.
        # TODO: <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
        # TODO: DistCA: update number of microbatches
        # TODO: What is the original code even doing here?
        update_num_microbatches(args.consumed_train_samples, consistency_check=False, verbose=True)
        if get_num_microbatches() != num_microbatches and iteration != 0:
            assert get_num_microbatches() > num_microbatches, \
                (f"Number of microbatches should be increasing due to batch size rampup; "
                 f"instead going from {num_microbatches} to {get_num_microbatches()}")
            if args.save is not None:
                save_checkpoint_and_time(iteration, model, optimizer,
                                         opt_param_scheduler,
                                         num_floating_point_operations_so_far,
                                         checkpointing_context, train_data_iterator=train_data_iterator)
        num_microbatches = get_num_microbatches()
        print_rank_0(f"<< Number of microbatches: {num_microbatches}")
        update_num_microbatches(args.consumed_train_samples, consistency_check=True, verbose=True)
        # TODO: >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

        

        # Completely skip iteration if needed.
        if iteration in args.iterations_to_skip:
            # Dummy train_step to fast forward train_data_iterator.
            dummy_train_step(train_data_iterator)
            iteration += 1
            # TODO: <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< distca patch
            batch_size = get_batch_size()
            # TODO: ================================
            # batch_size = mpu.get_data_parallel_world_size() * \
            #              args.micro_batch_size * \
            #              get_num_microbatches()
            # TODO: >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> megatron original code
            args.consumed_train_samples += batch_size
            args.skipped_train_samples += batch_size
            continue

        # Run training step.
        args.curr_iteration = iteration
        ft_integration.on_training_step_start()
        print_rank_0(f"<< Training Step Start")
        loss_dict, skipped_iter, should_checkpoint, should_exit, exit_code, grad_norm, num_zeros_in_grad = \
            train_step(forward_step_func,
                       train_data_iterator,
                       model,
                       optimizer,
                       opt_param_scheduler,
                       config,
                       distca_kwargs=distca_kwargs,
                       )
        print_rank_0(f"<< Training Step End")
        ft_integration.on_training_step_end()
        if should_checkpoint:
            save_checkpoint_and_time(iteration, model, optimizer,
                                     opt_param_scheduler,
                                     num_floating_point_operations_so_far,
                                     checkpointing_context, train_data_iterator=train_data_iterator)
        if should_exit:
            break

        # Enable forward pre-hooks after first set of forward and backward passes.
        # When running in fp16, skip all NaN iterations until steady-state loss scaling value
        # is reached.
        if iteration == start_iteration:
            if skipped_iter:
                # Only enable forward pre-hook after a training step has successfully run. Relevant
                # for fp16 codepath where first XX iterations are skipped until steady-state loss
                # scale value is reached.
                start_iteration = iteration + 1
            else:
                # Enable forward pre-hook after training step has successfully run. All subsequent
                # forward passes will use the forward pre-hook / `param_sync_func` in
                # `forward_backward_func`.
                if should_disable_forward_pre_hook(args):
                    enable_forward_pre_hook(model)
                    config.param_sync_func = param_sync_func
                    pre_hook_enabled = True

        iteration += 1
        print_rank_0(f"<< Iteration {iteration} of {args.train_iters}")
        # TODO: <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< distca patch
        batch_size = get_batch_size()
        # TODO: ================================
        # batch_size = mpu.get_data_parallel_world_size() * \
        #              args.micro_batch_size * \
        #              get_num_microbatches()
        # TODO: >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> megatron original code
        
        args.consumed_train_samples += batch_size
        
        # TODO: <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< distca patch
        # In DistCA: get_batch_size() = get_num_microbatches() = args.micro_batch_size (constant)
        # The calculator still tracks:
        #   - get_current_global_batch_size(): desired global batch size (samples)
        #   - get_current_running_global_batch_size(): actual global batch size that can run (samples)
        # The difference represents skipped samples when decrease_batch_size_if_needed rounds down.
        # In DistCA, since batch_size is constant, we still use the calculator's values to track
        # skipped samples during rampup or when batch size needs to be decreased.
        # num_skipped_samples_in_batch = (get_current_global_batch_size() -
        #                                 get_current_running_global_batch_size())
        # TODO: We should properly patch the 
        # `get_current_global_batch_size`, `get_current_running_global_batch_size` 
        # functions etc and then use them to calculate the number of skipped samples.
        num_skipped_samples_in_batch = 0
        # TODO: ================================
        # num_skipped_samples_in_batch = (get_current_global_batch_size() -
        #                                 get_current_running_global_batch_size())
        # TODO: >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> megatron original code
        
        if args.decrease_batch_size_if_needed:
            assert num_skipped_samples_in_batch >= 0
        else:
            assert num_skipped_samples_in_batch == 0
        args.skipped_train_samples += num_skipped_samples_in_batch
        num_floating_point_operations_in_batch = num_floating_point_operations(args, batch_size)
        num_floating_point_operations_so_far += num_floating_point_operations_in_batch
        num_floating_point_operations_since_last_log_event += num_floating_point_operations_in_batch

        # Logging.
        if not optimizer.is_stub_optimizer:
            loss_scale = optimizer.get_loss_scale().item()
        else:
            loss_scale = 1.0
        params_norm = None

        if args.log_params_norm:
            params_norm = calc_params_l2_norm(model)
        learning_rate = None
        decoupled_learning_rate = None
        for param_group in optimizer.param_groups:
            if param_group['is_decoupled_lr']:
                decoupled_learning_rate = param_group['lr']
            else:
                learning_rate = param_group['lr']
        report_memory_flag = training_log(loss_dict, total_loss_dict,
                                          learning_rate,
                                          decoupled_learning_rate,
                                          iteration, loss_scale,
                                          report_memory_flag, skipped_iter,
                                          grad_norm, params_norm, num_zeros_in_grad)

        # Evaluation.
        if args.eval_interval and iteration % args.eval_interval == 0 and \
            args.do_valid:
            print_rank_0(f"<< Evaluation Start")
            timers('interval-time').stop()
            if should_disable_forward_pre_hook(args):
                disable_forward_pre_hook(model)
                pre_hook_enabled = False
            if args.manual_gc and args.manual_gc_eval:
                # Collect all objects.
                gc.collect()
            prefix = f'iteration {iteration}'
            timers('eval-time', log_level=0).start(barrier=True)
            evaluate_and_print_results(prefix, forward_step_func,
                                       valid_data_iterator, model,
                                       iteration, process_non_loss_data_func,
                                       config, verbose=False, write_to_tensorboard=True,
                                       non_loss_data_func=non_loss_data_func)
            print_rank_0(f"<< Evaluation End")
            eval_duration += timers('eval-time').elapsed()
            eval_iterations += args.eval_iters
            timers('eval-time').stop()
            one_logger_utils.track_e2e_metrics()

            if args.manual_gc and args.manual_gc_eval:
                # Collect only the objects created and used in evaluation.
                gc.collect(generation=0)
            if should_disable_forward_pre_hook(args):
                enable_forward_pre_hook(model)
                pre_hook_enabled = True
            timers('interval-time', log_level=0).start(barrier=True)

        # Miscellaneous post-training-step functions (e.g., FT heartbeats, GC).
        # Some of these only happen at specific iterations.
        post_training_step_callbacks(model, optimizer, opt_param_scheduler, iteration, prof,
                                     num_floating_point_operations_since_last_log_event)

        # Checkpoint and decide whether to exit.
        should_exit = checkpoint_and_decide_exit(model, optimizer, opt_param_scheduler, iteration,
                                                 num_floating_point_operations_so_far,
                                                 checkpointing_context, train_data_iterator)
        if should_exit:
            break

    one_logger_utils.track_e2e_metrics()

    # Flush TensorBoard, WandB writers and one-logger.
    writer = get_tensorboard_writer()
    if writer:
        writer.flush()

    # Close out pre-hooks if using distributed optimizer and overlapped param gather.
    if pre_hook_enabled:
        disable_forward_pre_hook(model)

    ft_integration.on_checkpointing_start()
    # This will finalize all unfinalized async request and terminate
    # a persistent async worker if persistent ckpt worker is enabled
    maybe_finalize_async_save(blocking=True, terminate=True)
    ft_integration.on_checkpointing_end(is_async_finalization=True)
    if args.enable_ft_package and ft_integration.get_rank_monitor_client() is not None:
        ft_integration.get_rank_monitor_client().shutdown_workload_monitoring()

    # If any exit conditions (signal handler, duration, iterations) have been reached, exit.
    if should_exit:
        wandb_writer = get_wandb_writer()
        if wandb_writer:
            wandb_writer.finish()
        ft_integration.shutdown()
        sys.exit(exit_code)

    return iteration, num_floating_point_operations_so_far



def get_distca_forward_backward_func(distca_kwargs=None):
    """Adapt from megatron.core.pipeline_parallel.schedules.get_forward_backward_func() to support DistCA.
    This function gets called every training iteration. 
    """
    distca_kwargs = distca_kwargs or {}
    with_dummy = distca_kwargs.get("with_dummy", False)
    if with_dummy:
        from distca.runtime.megatron.forward_backward_func import forward_backward_pipelining_without_interleaving
        func = forward_backward_pipelining_without_interleaving
        # if "dummy_bwd_func" in distca_kwargs:
        #     func = partial(func, dummy_bwd_func=distca_kwargs["dummy_bwd_func"])
        return func

    # Otherwise, get the megatron original forward-backward function
    from megatron.core.pipeline_parallel import get_forward_backward_func as get_original_forward_backward_func
    return get_original_forward_backward_func()


def train_step(forward_step_func, data_iterator,
               model, optimizer, opt_param_scheduler, config,
               distca_kwargs: dict = None,
               ):
    """Single training step."""
    args = get_args()
    timers = get_timers()
    distca_kwargs = distca_kwargs or {}

    # CUDA Graph capturing only executes once, when it's the first training iteration.
    if args.curr_iteration == args.iteration and args.external_cuda_graph:
        cuda_graph_capture(model, config, args)

        # Set grad to zero.
        for model_chunk in model:
            model_chunk.zero_grad_buffer()
        optimizer.zero_grad()

        # Collect garbage and empty unused memory.
        gc.collect()
        torch.cuda.empty_cache()

    rerun_state_machine = get_rerun_state_machine()
    while rerun_state_machine.should_run_forward_backward(data_iterator):
        # Set grad to zero.
        for model_chunk in model:
            model_chunk.zero_grad_buffer()
        optimizer.zero_grad()

        # Forward pass.
        # TODO: <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< distca patch
        forward_backward_func = get_distca_forward_backward_func(distca_kwargs=distca_kwargs)
        # TODO: ================================
        # forward_backward_func = get_original_forward_backward_func()
        # TODO: >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> megatron original code
        losses_reduced = forward_backward_func(
            forward_step_func=forward_step_func,
            data_iterator=data_iterator,
            model=model,
            num_microbatches=get_num_microbatches(),
            seq_length=args.seq_length,
            micro_batch_size=args.micro_batch_size,
            decoder_seq_length=args.decoder_seq_length,
            forward_only=False)
    should_checkpoint, should_exit, exit_code = rerun_state_machine.should_checkpoint_and_exit()
    if should_exit:
        return {}, True, should_checkpoint, should_exit, exit_code, None, None

    # Empty unused memory.
    if args.empty_unused_memory_level >= 1:
        torch.cuda.empty_cache()

    # Vision gradients.
    if args.vision_pretraining and args.vision_pretraining_type == "dino":
        unwrapped_model = unwrap_model(model[0])
        unwrapped_model.cancel_gradients_last_layer(args.curr_iteration)

    # Update parameters.

    timers('optimizer', log_level=1).start(barrier=args.barrier_with_L1_time)
    update_successful, grad_norm, num_zeros_in_grad = optimizer.step()
    timers('optimizer').stop()

    # when freezing sub-models we may have a mixture of successful and unsucessful ranks,
    # so we must gather across mp ranks
    update_successful = logical_and_across_model_parallel_group(update_successful)
    # grad_norm and num_zeros_in_grad will be None on ranks without trainable params,
    # so we must gather across mp ranks
    grad_norm = reduce_max_stat_across_model_parallel_group(grad_norm)
    if args.log_num_zeros_in_grad:
        num_zeros_in_grad = reduce_max_stat_across_model_parallel_group(num_zeros_in_grad)

    # Vision momentum.
    if args.vision_pretraining and args.vision_pretraining_type == "dino":
        unwrapped_model = unwrap_model(model[0])
        unwrapped_model.update_momentum(args.curr_iteration)

    # Update learning rate.
    if update_successful:
        increment = get_num_microbatches() * \
                    args.micro_batch_size * \
                    args.data_parallel_size
        opt_param_scheduler.step(increment=increment)
        skipped_iter = 0
    else:
        skipped_iter = 1

    # Empty unused memory.
    if args.empty_unused_memory_level >= 2:
        torch.cuda.empty_cache()

    # Set the manual hooks when CUDA Graphs are enabled.
    if args.curr_iteration == args.iteration and args.external_cuda_graph:
        if args.use_distributed_optimizer and args.overlap_param_gather:
            cuda_graph_set_manual_hooks(model)

    if mpu.is_pipeline_last_stage(ignore_virtual=True):
        # Average loss across microbatches.
        loss_reduced = {}
        for key in losses_reduced[0].keys():
            numerator = 0
            denominator = 0
            for x in losses_reduced:
                val = x[key]
                # there is one dict per microbatch. in new reporting, we average
                # over the total number of tokens across the global batch.
                if isinstance(val, tuple) or isinstance(val, list):
                    numerator += val[0]
                    denominator += val[1]
                else:
                    # legacy behavior. we average over the number of microbatches,
                    # and so the denominator is 1.
                    numerator += val
                    denominator += 1
            loss_reduced[key] = numerator / denominator
        return loss_reduced, skipped_iter, should_checkpoint, should_exit, exit_code, grad_norm, num_zeros_in_grad
    return {}, skipped_iter, should_checkpoint, should_exit, exit_code, grad_norm, num_zeros_in_grad

