

# Global variable overrides (use # $$ syntax)
# $$OUTPUT_DIR_PREFIX=/mnt/weka/home/yonghao.zhuang/jd/d2/benchmarks/_250923_test_pp_34b/logs.v71-small-scale-pp-34b-16node-384k
# $$FOLDER_SEPARATOR=1
# $$EXPERIMENT_DISTS=("prolong 0.3")

# bash /mnt/weka/home/yonghao.zhuang/jd/d2/benchmarks/_250923_test_pp_34b/salloc_srun.pp_34b.v2.sh --config /mnt/weka/home/yonghao.zhuang/jd/d2/benchmarks/_250923_test_pp_34b/config_sweep_memory.config.v30_0930_0000PST_16node-prolong-0.3-384k.sh

# ================================================================================
#  ______                       _   ____              
# |  ____|                     | | |  _ \             
# | |__ ___  _ __ _ __ ___   __| | | |_) |_   _ _ __  
# |  __/ _ \| '__| '_ ` _ \ / _` | |  _ <| | | | '_ \ 
# | | | (_) | |  | | | | | | (_| | | |_) | |_| | | | |
# |_|  \___/|_|  |_| |_| |_|\__,_| |____/ \__,_|_| |_|
#
# ================================================================================

# ------------------------------------------------------------------------------------------------
#  384k pretrain small OOM sweep on wlbllm and d2 (2025/09/30 00:00:00 AM PST - 
# ------------------------------------------------------------------------------------------------

# $$OUTPUT_DIR_PREFIX=/mnt/weka/home/yonghao.zhuang/jd/d2/benchmarks/_250923_test_pp_34b/logs.v71-small-scale-pp-34b-16node-384k-debug-nsys

# <<< EXPERIMENT__SKIP=0,EXPERIMENT__COMPLETED=0,MAX_SAMPLE_ID=5,EXPERIMENT_REPEAT_TIMES=1,EXPERIMENT_WARMUP_TIMES=0,SHOULD_ADD_DEBUG_CASES=0,NUM_LAYERS=48,EXPERIMENT_NVSHMEM_BUFFER_SIZE_GB=1.5,MIN_TOLERANCE_FACTOR=0.15

    # 384k tbs=4 node=16
    # n  bs  mb   t         mode   cp  pp tp    comment    env_var
    16   1   4 393216    d2        8  2  8 'some_comment'  'EXPERIMENT_SCHEDULE_PRIORITY=100,ENABLE_NSYS=1'
    16   2   1 393216    d2       16  1  8 'some_comment'  'EXPERIMENT_SCHEDULE_PRIORITY=80,ENABLE_NSYS=1'

# >>>
# ------------ Stop here ------------

# <<< EXPERIMENT__SKIP=0,EXPERIMENT__COMPLETED=0,MAX_SAMPLE_ID=5,EXPERIMENT_REPEAT_TIMES=1,EXPERIMENT_WARMUP_TIMES=0,SHOULD_ADD_DEBUG_CASES=0,NUM_LAYERS=48,EXPERIMENT_NVSHMEM_BUFFER_SIZE_GB=1.5,MIN_TOLERANCE_FACTOR=0.15

    # 384k tbs=4 node=16
    # n  bs  mb   t         mode   cp  pp tp    comment    env_var
    16   1   4 393216    d2        8  2  8 'some_comment'  'EXPERIMENT_SCHEDULE_PRIORITY=100'
    16   4   1 393216    d2       16  1  8 'some_comment'  'EXPERIMENT_SCHEDULE_PRIORITY=100'


    16   1   4 393216    wlbllm    8  2  8 'some_comment'  'EXPERIMENT_SCHEDULE_PRIORITY=90'
    16   4   1 393216    wlbllm    8  1  8 'some_comment'  'EXPERIMENT_SCHEDULE_PRIORITY=90'
    16   4   1 393216    wlbllm   16  1  8 'some_comment'  'EXPERIMENT_SCHEDULE_PRIORITY=60'
    16   4   1 393216    wlbllm    4  1  8 'some_comment'  'EXPERIMENT_SCHEDULE_PRIORITY=60'
    16   4   1 393216    wlbllm    2  1  8 'some_comment'  'EXPERIMENT_SCHEDULE_PRIORITY=60'

    # 384k tbs=2 node=16
    # n  bs  mb   t         mode   cp  pp tp    comment    env_var
    16   2   1 393216    d2       16  1  8 'some_comment'  'EXPERIMENT_SCHEDULE_PRIORITY=80'
    16   2   1 393216    wlbllm    8  1  8 'some_comment'  'EXPERIMENT_SCHEDULE_PRIORITY=70'
    16   2   1 393216    wlbllm    4  1  8 'some_comment'  'EXPERIMENT_SCHEDULE_PRIORITY=60'
    16   2   1 393216    wlbllm   16  1  8 'some_comment'  'EXPERIMENT_SCHEDULE_PRIORITY=60'


# >>>

# ------------ Stop here ------------

    # 384k tbs=4 node=16
    # n  bs  mb   t         mode   cp  pp tp    comment    env_var
    16   1   4 393216    d2        8  2  8 'some_comment'  'EXPERIMENT_SCHEDULE_PRIORITY=60,EXPERIMENT__STATUS=FAIL,EXPERIMENT__FAIL_REASON=OOM,RESULT_DIR=27_145251.d2-n16-t393216-b1-mb4-cp8tp8pp2-codellama_CodeLlama-34b-hf-L48-wlbllm_0.0'
    16   4   1 393216    d2       16  1  8 'some_comment'  'EXPERIMENT_SCHEDULE_PRIORITY=60,EXPERIMENT__STATUS=FAIL,EXPERIMENT__FAIL_REASON=OOM,RESULT_DIR=27_145525.d2-n16-t393216-b4-mb1-cp16tp8pp1-codellama_CodeLlama-34b-hf-L48-wlbllm_0.0'
    16   1   4 393216    wlbllm    8  2  8 'some_comment'  'EXPERIMENT_SCHEDULE_PRIORITY=60,EXPERIMENT__STATUS=PARTIAL,RESULT_DIR=27_203205.wlbllm-n16-t393216-b1-mb4-cp8tp8pp2-codellama_CodeLlama-34b-hf-L48-wlbllm_0.0'
    16   4   1 393216    wlbllm   16  1  8 'some_comment'  ''
    16   4   1 393216    wlbllm    8  1  8 'some_comment'  'EXPERIMENT_SCHEDULE_PRIORITY=60,EXPERIMENT__STATUS=FAIL,EXPERIMENT__FAIL_REASON=OOM,RESULT_DIR=27_203656.wlbllm-n16-t393216-b4-mb1-cp8tp8pp1-codellama_CodeLlama-34b-hf-L48-wlbllm_0.0'
    16   4   1 393216    wlbllm    4  1  8 'some_comment'  ''
    16   4   1 393216    wlbllm    2  1  8 'some_comment'  ''

    # 384k tbs=2 node=16
    # n  bs  mb   t         mode   cp  pp tp    comment    env_var
    16   2   1 393216    d2       16  1  8 'some_comment'  'EXPERIMENT_SCHEDULE_PRIORITY=60,EXPERIMENT__STATUS=PASS,RESULT_DIR=27_145659.d2-n16-t393216-b2-mb1-cp16tp8pp1-codellama_CodeLlama-34b-hf-L48-wlbllm_0.0'
    16   2   1 393216    wlbllm    4  1  8 'some_comment'  ''
    16   2   1 393216    wlbllm    8  1  8 'some_comment'  'EXPERIMENT_SCHEDULE_PRIORITY=60,EXPERIMENT__STATUS=PASS,RESULT_DIR=27_150216.wlbllm-n16-t393216-b2-mb1-cp8tp8pp1-codellama_CodeLlama-34b-hf-L48-wlbllm_0.0'
    16   2   1 393216    wlbllm   16  1  8 'some_comment'  ''



# >>>



