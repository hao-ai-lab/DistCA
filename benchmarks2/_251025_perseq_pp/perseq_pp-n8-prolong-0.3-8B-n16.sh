
# WLBLLM PerSeq
# - Dataset: Prolong 
# - Model: 8B 
# - Node: 16

# Global variable overrides (use # $$ syntax)
# $$OUTPUT_DIR_PREFIX=/mnt/weka/home/yonghao.zhuang/jd/d2/benchmarks2/_251025_perseq_pp/logs.v1-n16-prolong-0.3-8B
# $$FOLDER_SEPARATOR=1
# $$EXPERIMENT_DISTS=("prolong 0.3")

# bash /mnt/weka/home/yonghao.zhuang/jd/d2/benchmarks2/_251025_perseq_pp/salloc_srun.pp_8b.v2.sh --config /mnt/weka/home/yonghao.zhuang/jd/d2/benchmarks2/_251025_perseq_pp/perseq_pp-n16-prolong-0.3-8B.sh

# MAX_SAMPLE_ID=30



# <<< EXPERIMENT__SKIP=0,EXPERIMENT__COMPLETED=0,EXPERIMENT_REPEAT_TIMES=1,EXPERIMENT_WARMUP_TIMES=0,SHOULD_ADD_DEBUG_CASES=0,NUM_LAYERS=32,CHANGE_LONG_DOC_RATIO=0.0,EXPERIMENT_NVSHMEM_BUFFER_SIZE_GB=1.5,MIN_TOLERANCE_FACTOR=0.15,MAX_SAMPLE_ID=5

# 128k tbs=64 node=16
    #n  bs  mb   t         mode          cp  pp tp    comment    env_var
    16   4  8 131072    wlbllm_perseq    4  2  8 'some_comment'  'EXPERIMENT_SCHEDULE_PRIORITY=100'
    16   4  8 131072    wlbllm           4  2  8 'some_comment'  'EXPERIMENT_SCHEDULE_PRIORITY=90'

# 256k tbs=32 node=16
    #n  bs  mb   t         mode          cp  pp tp    comment    env_var
    16   4  4 262144    wlbllm_perseq    4  2  8 'some_comment'  'EXPERIMENT_SCHEDULE_PRIORITY=100'
    16   4  4 262144    wlbllm           4  2  8 'some_comment'  'EXPERIMENT_SCHEDULE_PRIORITY=90'

# 512k tbs=8 node=16
    #n  bs  mb   t         mode          cp  pp tp    comment    env_var
    16   1  4 524288    wlbllm_perseq    4  2  8 'some_comment'  'EXPERIMENT_SCHEDULE_PRIORITY=100'
    16   1  4 524288    wlbllm           4  2  8 'some_comment'  'EXPERIMENT_SCHEDULE_PRIORITY=90'

# >>>


# # ------------ Stop here ------------