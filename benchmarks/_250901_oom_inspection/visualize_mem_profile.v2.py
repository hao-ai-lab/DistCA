import argparse
import pickle
import csv
from typing import Any, Iterable, List, Tuple

# ---- Helpers to be robust across PyTorch versions ----

def _iter_events(obj: Any):
    """
    Return a flat iterator of per-event records from various profiler pickles.
    Supports: profile result (has .events()), EventList, or list of events.
    """
    if obj is None:
        return []
    if hasattr(obj, "events") and callable(getattr(obj, "events")):
        return obj.events()
    # torch.autograd.profiler.EventList or plain list/tuple
    if isinstance(obj, (list, tuple)):
        return iter(obj)
    # Some versions store key_averages; not useful for per-event memory deltas.
    if hasattr(obj, "key_averages"):
        # Fall back to raw events if available
        if hasattr(obj, "events") and callable(getattr(obj, "events")):
            return obj.events()
    # Last resort: try to treat as iterable
    try:
        return iter(obj)
    except TypeError:
        return []

def _get_time_bounds(evt) -> Tuple[float, float]:
    """
    Get (start_us, end_us) for an event, trying several attribute spellings.
    Returns microseconds (float).
    """
    # Common kineto attrs
    for s_attr, e_attr, scale in [
        ("start_time", "end_time", 1.0),                 # some stores in us already
        ("start_time_us", "end_time_us", 1.0),
        ("start_time_ns", "end_time_ns", 1e-3),          # convert ns -> us
        ("cpu_time_total", None, None),                  # aggregated (skip)
    ]:
        if hasattr(evt, s_attr):
            s = getattr(evt, s_attr)
            if e_attr is None:
                return (float(s), float(s))  # degenerate, shouldn't happen
            if hasattr(evt, e_attr):
                e = getattr(evt, e_attr)
                return (float(s) * scale, float(e) * scale)
    # Fallback: 0-length
    return (0.0, 0.0)

def _get_name(evt) -> str:
    return getattr(evt, "name", getattr(evt, "key", "unknown"))

def _get_parent_name(evt) -> str:
    # Some formats expose .parent or .linked_correlation_id; keep simple
    p = getattr(evt, "parent", None)
    return getattr(p, "name", "") if p is not None else ""

def _get_stack(evt) -> str:
    # Try a few fields that may contain stack info
    for attr in ["stack", "stack_frames", "callstack", "python_stack"]:
        if hasattr(evt, attr) and getattr(evt, attr):
            frames = getattr(evt, attr)
            if isinstance(frames, (list, tuple)):
                # join nicely
                return " | ".join(str(f) for f in frames)
            return str(frames)
    # Sometimes extra fields carry file/line
    file = getattr(evt, "filename", "")
    line = getattr(evt, "line", "")
    if file or line:
        return f"{file}:{line}"
    return ""

def _mem_delta_fields(evt) -> List[Tuple[str, int]]:
    """
    Return list of (device, delta_bytes) for any memory deltas found on the event.
    We check multiple attribute names across versions.
    Positive delta => allocation; negative => free.
    """
    out = []
    # CUDA
    for attr in ["cuda_memory_usage", "self_cuda_memory_usage", "cuda_memory_allocated", "cuda_mem_usage"]:
        if hasattr(evt, attr):
            try:
                v = int(getattr(evt, attr))
                if v != 0:
                    out.append(("cuda", v))
                    break
            except Exception:
                pass
    # CPU
    for attr in ["cpu_memory_usage", "self_cpu_memory_usage", "cpu_memory_allocated", "cpu_mem_usage"]:
        if hasattr(evt, attr):
            try:
                v = int(getattr(evt, attr))
                if v != 0:
                    out.append(("cpu", v))
                    break
            except Exception:
                pass
    return out

def _event_contains_name(evt, region_name: str) -> bool:
    """
    True if this event is the region root (name matches).
    """
    return _get_name(evt) == region_name

def _find_region_windows(events: Iterable[Any], region_name: str) -> List[Tuple[float, float]]:
    """
    Find all [start_us, end_us] windows for events named region_name.
    """
    windows = []
    for evt in events:
        if _event_contains_name(evt, region_name):
            s, e = _get_time_bounds(evt)
            if e >= s:
                windows.append((s, e))
    return windows

def _is_within_any_window(evt, windows: List[Tuple[float, float]]) -> bool:
    if not windows:
        return False
    s, e = _get_time_bounds(evt)
    # Keep an event if it overlaps any window (inclusive overlap)
    for ws, we in windows:
        if not (e < ws or s > we):
            return True
    return False

# ---- Main ----

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Path to profiler pickle")
    ap.add_argument("--region", required=True, help="Region name to filter (e.g., forward_mlp)")
    ap.add_argument("--output", required=True, help="Output CSV path")
    args = ap.parse_args()

    with open(args.input, "rb") as f:
        prof_obj = pickle.load(f)

    # We will iterate events twice (find windows, then filter); so cache to list.
    all_events = list(_iter_events(prof_obj))

    # Find all time windows for the named region
    region_windows = _find_region_windows(all_events, args.region)

    if not region_windows:
        print(f"WARNING: No events found with name == '{args.region}'. "
              f"CSV will be empty unless your region name is different (check record_function label).")

    # Prepare CSV
    fieldnames = [
        "ts_us",              # event start (us)
        "te_us",              # event end (us)
        "duration_us",
        "device",             # cpu or cuda for the mem delta
        "mem_delta_bytes",    # +alloc / -free
        "event_name",
        "parent_name",
        "stack"
    ]
    with open(args.output, "w", newline="") as csvf:
        writer = csv.DictWriter(csvf, fieldnames=fieldnames)
        writer.writeheader()

        for evt in all_events:
            # Only keep events overlapping the region time window(s)
            if not _is_within_any_window(evt, region_windows):
                continue

            deltas = _mem_delta_fields(evt)
            if not deltas:
                continue  # skip events that did not change memory

            s, e = _get_time_bounds(evt)
            name = _get_name(evt)
            parent = _get_parent_name(evt)
            stack = _get_stack(evt)

            for dev, delta in deltas:
                writer.writerow({
                    "ts_us": f"{s:.3f}",
                    "te_us": f"{e:.3f}",
                    "duration_us": f"{max(0.0, e - s):.3f}",
                    "device": dev,
                    "mem_delta_bytes": delta,
                    "event_name": name,
                    "parent_name": parent,
                    "stack": stack,
                })

    print(f"✅ Wrote memory-delta CSV to: {args.output}")
    if region_windows:
        print(f"ℹ️ Found {len(region_windows)} region window(s) for '{args.region}'.")
    else:
        print("ℹ️ Tip: make sure you wrapped the target code with:")
        print('    with torch.profiler.record_function("forward_mlp"):')
        print("    ...")
        print("and use that same label for --region.")

if __name__ == "__main__":
    main()