# profiler_ctx.py
import os
import torch
import torch.profiler as tp
import traceback

class ProfilerCtx:
    """
    Minimal torch.profiler wrapper.
    - Writes TensorBoard traces to outdir/
    - Exports a Chrome trace file at exit
    - Call .step() after each iteration you want recorded
    """
    def __init__(self, outdir: str, chrome_name: str = "trace.json"):
        self.outdir = outdir
        os.makedirs(self.outdir, exist_ok=True)
        self.chrome_path = os.path.join(self.outdir, chrome_name)
        self.prof = tp.profile(
            activities=[tp.ProfilerActivity.CPU, tp.ProfilerActivity.CUDA],
            # on_trace_ready=tp.tensorboard_trace_handler(self.outdir),
            record_shapes=True,
            with_stack=True,
            with_flops=True,
        )

    def __enter__(self):
        self.prof.__enter__()
        print(f"⚪ Entering prof.")
        return self

    def __exit__(self, exc_type, exc, tb):
        # Export a Chrome trace even if there was an exception
        print(f"⚪ Exiting prof.")
        self.prof.__exit__(exc_type, exc, tb)
        print(f"⚪ Exporting chrome trace for {self.chrome_path}")
        self.prof.export_chrome_trace(self.chrome_path)
        print(f"⚪ Finish exporting chrome trace for {self.chrome_path}")
        # except Exception as e:
        #     traceback.print_exc()
        #     pass
        return False  # don't swallow exceptions

    def step(self):
        """Mark one iteration; flushes to disk via on_trace_ready."""
        self.prof.step()