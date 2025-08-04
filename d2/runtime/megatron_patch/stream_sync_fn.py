"""
Wrapper torch function of the n2n communication to orchestrate the ping-pang parallel.
"""
import torch


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
    @staticmethod
    def forward(ctx, compute_stream: torch.cuda.Stream, comm_stream: torch.cuda.Stream,
                *tensors):
        if comm_stream is not None:
            assert compute_stream is not None
            # sync the previous step
            compute_stream.wait_stream(comm_stream)
            comm_stream.wait_stream(compute_stream)

        ctx.compute_stream = compute_stream
        ctx.comm_stream = comm_stream
        return tensors

    @staticmethod
    def backward(ctx, *grads):
        compute_stream = ctx.compute_stream
        comm_stream = ctx.comm_stream

        if comm_stream is not None:
            # sync the previous step
            compute_stream.wait_stream(comm_stream)
            comm_stream.wait_stream(compute_stream)

        return (None, None, *grads,)
