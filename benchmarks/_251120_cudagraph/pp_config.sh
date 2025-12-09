

# Global variable overrides (use # $$ syntax)
# $$OUTPUT_DIR_PREFIX=./logs.v1
# $$FOLDER_SEPARATOR=1
# $$EXPERIMENT_DISTS=("wlbllm 0.0")


# <<< EXPERIMENT__SKIP=0,EXPERIMENT__COMPLETED=0,MAX_SAMPLE_ID=3,EXPERIMENT_REPEAT_TIMES=1,EXPERIMENT_WARMUP_TIMES=0,SHOULD_ADD_DEBUG_CASES=0,NUM_LAYERS=16,EXPERIMENT_NVSHMEM_BUFFER_SIZE_GB=1.5,MIN_TOLERANCE_FACTOR=0.15

    # 384k tbs=4 node=16
    # n  bs  mb   t         mode   cp  pp tp    comment    env_var
    1   1   1     8192       d2     1  1  8 'some_comment'  'EXPERIMENT_SCHEDULE_PRIORITY=100'

# >>>
# ------------ Stop here ------------


# bash /mnt/weka/home/yonghao.zhuang/jd/d2/benchmarks/_251120_cudagraph/test_4d.sh --config /mnt/weka/home/yonghao.zhuang/jd/d2/benchmarks/_251120_cudagraph/pp_config.sh

# bash ../benchmarks/_251120_cudagraph/test_4d.sh --config ../benchmarks/_251120_cudagraph/pp_config.sh