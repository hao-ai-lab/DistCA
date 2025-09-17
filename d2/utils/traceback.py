

# --------------------------------
# Better traceback formatting
# --------------------------------
import sys
import traceback
import os

RED = "\033[31m"
BLUE = "\033[34m"
RESET = "\033[0m"

should_enable_clickable_excepthook = os.environ.get("EXPERIMENT_PYTHON_BETTER_TRACEBACK", "1") == "1"
should_trace_calls = os.environ.get("EXPERIMENT_PYTHON_DEBUG_TRACE_CALLS", "0") == "1"

def clickable_excepthook(exc_type, exc_value, tb, file=None):
    for filename, lineno, func, text in traceback.extract_tb(tb):
        path = os.path.abspath(filename)
        print(f"{path}:{lineno}: in {func}", file=file)
        if text:
            print(f"    {text}", file=file)
    
    prefix = ""
    try:
        import torch
        rank = torch.distributed.get_rank()
        prefix = f"🔴 Rank {rank} "
    except:
        pass
    # error in red
    print(f"{RED}{prefix}{exc_type.__name__}: {exc_value}{RESET}", file=file)

def enable_clickable_excepthook():
    print("🟡 Enabling clickable excepthook.")
    sys.excepthook = clickable_excepthook


def trace_calls(frame, event, arg):
    if event == "call":
        code = frame.f_code
        print(f"--> Enter {code.co_name} ({code.co_filename}:{frame.f_lineno})")
    elif event == "return":
        code = frame.f_code
        print(f"<-- Exit {code.co_name} ({code.co_filename}:{frame.f_lineno})")
    return trace_calls



class TraceFunctions:
    def __init__(self, filter_path=None):
        self.filter_path = filter_path
        self._oldtrace = None
        self.indent = 0

    def _trace(self, frame, event, arg):
        if event in ("call", "return"):
            code = frame.f_code
            filename = code.co_filename
            if self.filter_path and self.filter_path not in filename:
                return
            if event == "call":
                print(f"{BLUE}{' ' * self.indent}--> Enter {code.co_name} ({filename}:{frame.f_lineno}){RESET}")
                self.indent += 1
            elif event == "return":
                print(f"{BLUE}{' ' * self.indent}<-- Exit  {code.co_name} ({filename}:{frame.f_lineno}){RESET}")
                self.indent -= 1
        return self._trace

    def __enter__(self):
        self._oldtrace = sys.gettrace()
        if not should_trace_calls:
            return self
        sys.settrace(self._trace)
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        sys.settrace(self._oldtrace)

def enable_trace_calls():
    print("🟡 Enabling python debug trace calls.")
    sys.settrace(trace_calls)


# if should_enable_clickable_excepthook:
#     enable_clickable_excepthook()
# if should_trace_calls:
#     enable_trace_calls()
