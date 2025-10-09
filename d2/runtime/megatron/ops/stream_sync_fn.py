"""
Wrapper torch function of the n2n communication to orchestrate the ping-pang parallel.
"""
import torch
import time
import os
from typing import Dict, List


class TickSync(torch.autograd.Function):
    """
    Synchronize compute and communication streams. This enables a sync at the backward stage.
    Timeline:
    ------------------------------------------->
    Compute i | TickSync i | Compute | ...
    Comm    i | TickSync i | Comm    | ...

    Backward
    <-------------------------------------------
    Compute_grad | TickSync | Compute_grad | ...
    Comm_grad    | TickSync | Comm_grad    | ...
    """
    
    # Class variables for blocking detection
    _blocking_events: List[Dict] = []
    _pending_events: List = []  # List of (start_event, end_event, phase, layer_info, operation_info) tuples
    _enabled: bool = False
    _threshold_ms: float = 1.0  # 1ms threshold
    
    @classmethod
    def enable_blocking_detection(cls, enabled: bool = True, threshold_ms: float = 1.0):
        """Enable or disable blocking detection."""
        cls._enabled = enabled
        cls._threshold_ms = threshold_ms
    
    @classmethod
    def get_blocking_events(cls) -> List[Dict]:
        """Get all blocking events."""
        return cls._blocking_events.copy()
    
    @classmethod
    def clear_blocking_events(cls):
        """Clear all blocking events."""
        cls._blocking_events = []
    
    @classmethod
    def process_pending_events(cls):
        """Process all pending CUDA events and detect blocking."""
        if not cls._pending_events:
            return
        
        # Synchronize to ensure all events are complete
        torch.cuda.synchronize()
        
        for start_event, end_event, phase, layer_info, operation_info in cls._pending_events:
            # Calculate elapsed time in milliseconds
            wait_time_ms = start_event.elapsed_time(end_event)
            
            # Create blocking event
            blocking_event = {
                'timestamp': time.time(),
                'wait_time_ms': wait_time_ms,
                'threshold_ms': cls._threshold_ms,
                'phase': phase,
                'layer_info': layer_info,
                'operation_info': operation_info
            }
            cls._blocking_events.append(blocking_event)
                
            if wait_time_ms > cls._threshold_ms:
                # Print warning if detailed logging is enabled
                if os.getenv("D2_TICKSYNC_DETAILED_LOGGING", "0") == "1":
                    print(f"⚠️  TICKSYNC BLOCKING: {wait_time_ms:.2f}ms > {cls._threshold_ms}ms "
                          f"at {layer_info} {operation_info} ({phase})")
        
        # Clear pending events after processing
        cls._pending_events = []
    @staticmethod
    def forward(ctx, compute_stream: torch.cuda.Stream, comm_stream: torch.cuda.Stream, layer_info: str, operation_info: str, *tensors, ):
        if comm_stream is not None:
            assert compute_stream is not None
            # sync the previous step
            compute_stream.wait_stream(comm_stream)
            
            # Measure comm_stream.wait_stream(compute_stream) for blocking detection
            if TickSync._enabled:
                # Create CUDA events to measure GPU timing
                start_event = torch.cuda.Event(enable_timing=True)
                end_event = torch.cuda.Event(enable_timing=True)
                
                # Record start event on compute stream
                start_event.record(compute_stream)
                comm_stream.wait_stream(compute_stream)
                # Record end event on comm stream
                end_event.record(comm_stream)
                
                # Store events for later synchronization and measurement
                TickSync._pending_events.append((start_event, end_event, 'forward', layer_info, operation_info))
            else:
                comm_stream.wait_stream(compute_stream)

        ctx.compute_stream = compute_stream
        ctx.comm_stream = comm_stream
        ctx.layer_info = layer_info
        ctx.operation_info = operation_info
        return tensors

    @staticmethod
    def backward(ctx, *grads):
        compute_stream = ctx.compute_stream
        comm_stream = ctx.comm_stream
        layer_info = ctx.layer_info
        operation_info = ctx.operation_info

        if comm_stream is not None:
            # sync the previous step
            compute_stream.wait_stream(comm_stream)
            
            # Measure comm_stream.wait_stream(compute_stream) for blocking detection
            if TickSync._enabled:
                # Create CUDA events to measure GPU timing
                start_event = torch.cuda.Event(enable_timing=True)
                end_event = torch.cuda.Event(enable_timing=True)
                
                # Record start event on compute stream
                start_event.record(compute_stream)
                comm_stream.wait_stream(compute_stream)
                # Record end event on comm stream
                end_event.record(comm_stream)
                
                # Store events for later synchronization and measurement
                TickSync._pending_events.append((start_event, end_event, 'backward', layer_info, operation_info))
            else:
                comm_stream.wait_stream(compute_stream)

        return (None, None, None, None, *grads,)


def tick_sync_with_info(compute_stream: torch.cuda.Stream, comm_stream: torch.cuda.Stream,
                       *tensors, layer_info: str = "unknown", operation_info: str = "unknown"):
    """
    Wrapper around TickSync that includes layer and operation information for blocking detection.
    """
    return TickSync.apply(compute_stream, comm_stream, layer_info, operation_info, *tensors)
