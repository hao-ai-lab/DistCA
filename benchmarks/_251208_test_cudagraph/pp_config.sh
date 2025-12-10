

# Global variable overrides (use # $$ syntax)
# $$OUTPUT_DIR_PREFIX=/mnt/weka/home/hao.zhang/jd/d2/benchmarks/_251208_test_cudagraph/logs.v1
# $$FOLDER_SEPARATOR=1
# $$EXPERIMENT_DISTS=("wlbllm 0.0")


# <<< EXPERIMENT__SKIP=0,EXPERIMENT__COMPLETED=0,MAX_SAMPLE_ID=3,EXPERIMENT_REPEAT_TIMES=1,EXPERIMENT_WARMUP_TIMES=0,SHOULD_ADD_DEBUG_CASES=0,NUM_LAYERS=16,EXPERIMENT_NVSHMEM_BUFFER_SIZE_GB=1.5,MIN_TOLERANCE_FACTOR=0.15,ENABLE_NSYS=0

    # n  bs  mb   t         mode   cp  pp tp    comment        env_var
      2   1   2  8192        d2     1   2  8  'some_comment'  'EXPERIMENT_SCHEDULE_PRIORITY=100'

# >>>
# ------------ Stop here ------------