# ▛▀▖      ▜             ▞▀▖  ▞▀▖  ▞▀▖▌ ▌▛▀▖  ▗▌ ▞▀▖        ▌   
# ▙▄▘▙▀▖▞▀▖▐ ▞▀▖▛▀▖▞▀▌▄▄▖▌▞▌   ▄▘   ▄▘▚▄▌▙▄▘   ▌ ▙▄ ▛▀▖▞▀▖▞▀▌▞▀▖
# ▌  ▌  ▌ ▌▐ ▌ ▌▌ ▌▚▄▌   ▛ ▌▗▖▖ ▌  ▖ ▌  ▌▌ ▌   ▌ ▌ ▌▌ ▌▌ ▌▌ ▌▛▀ 
# ▘  ▘  ▝▀  ▘▝▀ ▘ ▘▗▄▘   ▝▀ ▝▘▝▀   ▝▀   ▘▀▀   ▝▀ ▝▀ ▘ ▘▝▀ ▝▀▘▝▀▘

# ▗▌ ▞▀▖▞▀▖▌   ▞▀▖▛▀▘▞▀▖▌   ▛▀▘▗▌ ▞▀▖▌                          
#  ▌  ▗▘▚▄▘▌▗▘  ▗▘▙▄ ▙▄ ▌▗▘ ▙▄  ▌  ▗▘▌▗▘                        
#  ▌ ▗▘ ▌ ▌▛▚  ▗▘ ▖ ▌▌ ▌▛▚  ▖ ▌ ▌ ▗▘ ▛▚                         
# ▝▀ ▀▀▘▝▀ ▘ ▘ ▀▀▘▝▀ ▝▀ ▘ ▘ ▝▀ ▝▀ ▀▀▘▘ ▘                        

# ▞▀▖▞▀▖▞▀▖▞▀▖ ▞▀▖▞▀▖  ▞▀▖▞▀▖ ▛▀▖▞▀▖▀▛▘                         
# ▌▞▌▚▄▌ ▗▘▚▄▘ ▌▞▌▌▞▌▐▌▌▞▌▌▞▌ ▙▄▘▚▄  ▌                          
# ▛ ▌▖ ▌▗▘ ▌ ▌ ▛ ▌▛ ▌▗▖▛ ▌▛ ▌ ▌  ▖ ▌ ▌                          
# ▝▀ ▝▀ ▀▀▘▝▀  ▝▀ ▝▀ ▝▘▝▀ ▝▀  ▘  ▝▀  ▘                            
                                 
                                 


# Global variable overrides (use # $$ syntax)
# $$OUTPUT_DIR_PREFIX=/mnt/weka/home/yonghao.zhuang/jd/d2/benchmarks/_250923_test_pp_34b/logs.v10-large-scale-pp-34b-16node-128k-256k-384k--prolong-0.3
# $$FOLDER_SEPARATOR=1
# $$EXPERIMENT_DISTS=("prolong 0.3")
# $$EXPERIMENT_BALANCE_PING_PONG=1

# bash /mnt/weka/home/yonghao.zhuang/jd/d2/benchmarks/_250923_test_pp_34b/salloc_srun.pp_34b.v2.sh --config /mnt/weka/home/yonghao.zhuang/jd/d2/benchmarks/_250923_test_pp_34b/config_sweep_memory.config.v4_0928_0100PST_16node-prolong-0.3.sh



# ================================================================================
#  _____       _                   _____             
# |  __ \     | |                 |  __ \            
# | |  | | ___| |__  _   _  __ _ | |__) |   _ _ __  
# | |  | |/ _ \ '_ \| | | |/ _` ||  _  / | | | '_ \ 
# | |__| |  __/ |_) | |_| | (_| || | \ \ |_| | | | |
# |_____/ \___|_.__/ \__,_|\__, ||_|  \_\__,_|_| |_|
#                           __/ |                     
#                          |___/                      
# ================================================================================




# <<< EXPERIMENT__SKIP=0,EXPERIMENT__COMPLETED=0,MAX_SAMPLE_ID=5,EXPERIMENT_REPEAT_TIMES=1,EXPERIMENT_WARMUP_TIMES=0,SHOULD_ADD_DEBUG_CASES=0,NUM_LAYERS=48,CHANGE_LONG_DOC_RATIO=0.0,ENABLE_NSYS=1,OUTPUT_DIR_SUFFIX_ADDON=-debug-nsys,EXPERIMENT__DEBUG=1

    # # 384k tbs=4 node=16
    # # n  bs  mb   t         mode   cp  pp tp    comment    env_var
    # 16   1   4 393216    d2        8  2  8 'some_comment'  'EXPERIMENT_SCHEDULE_PRIORITY=60,EXPERIMENT__STATUS=FAIL,EXPERIMENT__FAIL_REASON=OOM,RESULT_DIR=27_145251.d2-n16-t393216-b1-mb4-cp8tp8pp2-codellama_CodeLlama-34b-hf-L48-wlbllm_0.0'
    # 16   4   1 393216    d2       16  1  8 'some_comment'  'EXPERIMENT_SCHEDULE_PRIORITY=60,EXPERIMENT__STATUS=FAIL,EXPERIMENT__FAIL_REASON=OOM,RESULT_DIR=27_145525.d2-n16-t393216-b4-mb1-cp16tp8pp1-codellama_CodeLlama-34b-hf-L48-wlbllm_0.0'
    # 16   1   4 393216    wlbllm    8  2  8 'some_comment'  'EXPERIMENT_SCHEDULE_PRIORITY=60,EXPERIMENT__STATUS=PARTIAL,RESULT_DIR=27_203205.wlbllm-n16-t393216-b1-mb4-cp8tp8pp2-codellama_CodeLlama-34b-hf-L48-wlbllm_0.0'
    # 16   4   1 393216    wlbllm   16  1  8 'some_comment'  ''
    # 16   4   1 393216    wlbllm    8  1  8 'some_comment'  'EXPERIMENT_SCHEDULE_PRIORITY=60,EXPERIMENT__STATUS=FAIL,EXPERIMENT__FAIL_REASON=OOM,RESULT_DIR=27_203656.wlbllm-n16-t393216-b4-mb1-cp8tp8pp1-codellama_CodeLlama-34b-hf-L48-wlbllm_0.0'
    # 16   4   1 393216    wlbllm    4  1  8 'some_comment'  ''
    # 16   4   1 393216    wlbllm    2  1  8 'some_comment'  ''

    # 384k tbs=2 node=16
    # n  bs  mb   t         mode   cp  pp tp    comment    env_var
    # 16   2   1 393216    d2       16  1  8 'some_comment'  ''
    # 16   2   1 393216    d2       16  1  8 'some_comment'  'EXPERIMENT_NVSHMEM_BUFFER_SIZE_GB=1.5'
    # 16   2   1 393216    wlbllm    4  1  8 'some_comment'  ''
    # 16   2   1 393216    wlbllm    8  1  8 'some_comment'  ''
    # 16   2   1 393216    wlbllm   16  1  8 'some_comment'  ''

# >>>







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
#      384k prolong 0.3 small OOM sweep on wlbllm and d2 (2025/09/28 01:00:00 AM PST - 
# ------------------------------------------------------------------------------------------------
# <<< EXPERIMENT__SKIP=1,EXPERIMENT__COMPLETED=0,MAX_SAMPLE_ID=30,EXPERIMENT_REPEAT_TIMES=1,EXPERIMENT_WARMUP_TIMES=0,SHOULD_ADD_DEBUG_CASES=0,NUM_LAYERS=48

    # 128k tbs=16 node=16
    # n  bs  mb   t         mode   cp  pp tp    comment    env_var
    16   2   8 131072    d2        8  2  8 'some_comment'  'EXPERIMENT_SCHEDULE_PRIORITY=100'
    16   1  16 131072    wlbllm    8  2  8 'some_comment'  'EXPERIMENT_SCHEDULE_PRIORITY=100'
    
    16   1  16 131072    d2        8  2  8 'some_comment'  ''
    16   2   8 131072    wlbllm    4  2  8 'some_comment'  ''

    # 16   1  16 131072    d2        4  4  8 'some_comment'  ''
    # 16   2   8 131072    wlbllm    4  4  8 'some_comment'  ''
    # 16   2   8 131072    wlbllm    8  2  8 'some_comment'  ''
    # 16   4   4 131072    wlbllm    2  2  8 'some_comment'  ''
    # 16   4   4 131072    wlbllm    4  2  8 'some_comment'  ''


    # 128k tbs=32 node=16
    #n  bs  mb   t         mode   cp  pp tp    comment    env_var
    16   2  16 131072    d2        8  2  8 'some_comment'  'EXPERIMENT_SCHEDULE_PRIORITY=80'
    16   1  32 131072    d2        8  2  8 'some_comment'  ''
    16   1  32 131072    d2        4  4  8 'some_comment'  ''
    16   2  16 131072    wlbllm    4  2  8 'some_comment'  'EXPERIMENT_SCHEDULE_PRIORITY=80'
    16   2  16 131072    wlbllm    4  4  8 'some_comment'  ''
    16   1  32 131072    wlbllm    8  2  8 'some_comment'  ''
    16   2  16 131072    wlbllm    8  2  8 'some_comment'  ''
    16   4   8 131072    wlbllm    2  2  8 'some_comment'  ''
    16   4   8 131072    wlbllm    4  2  8 'some_comment'  ''


    # 256k tbs=4 node=16
    # n  bs  mb   t         mode   cp  pp tp    comment    env_var
    16   1   4 262144    d2        8  2  8 'some_comment'  'EXPERIMENT_SCHEDULE_PRIORITY=70'
    16   1   4 262144    wlbllm    8  2  8 'some_comment'  'EXPERIMENT_SCHEDULE_PRIORITY=70'

    16   4   1 262144    d2       16  1  8 'some_comment'  ''
    16   4   1 262144    wlbllm    2  1  8 'some_comment'  ''
    16   4   1 262144    wlbllm    4  1  8 'some_comment'  ''
    16   4   1 262144    wlbllm    8  1  8 'some_comment'  ''
    16   4   1 262144    wlbllm   16  1  8 'some_comment'  ''
    
    # 256k tbs=8 node=16
    # n  bs  mb   t         mode   cp  pp tp    comment    env_var
    16   1   8 262144    d2        8  2  8 'some_comment'  'EXPERIMENT_SCHEDULE_PRIORITY=60'
    16   1   8 262144    wlbllm    8  2  8 'some_comment'  'EXPERIMENT_SCHEDULE_PRIORITY=60'

    16   8   1 262144    d2       16  1  8 'some_comment'  ''
    16   2   4 262144    wlbllm    4  2  8 'some_comment'  ''
    16   2   4 262144    wlbllm    8  2  8 'some_comment'  ''
    16   8   1 262144    wlbllm    1  1  8 'some_comment'  ''
    16   8   1 262144    wlbllm    2  1  8 'some_comment'  ''
    16   8   1 262144    wlbllm    4  1  8 'some_comment'  ''
    16   8   1 262144    wlbllm    8  1  8 'some_comment'  ''
    16   8   1 262144    wlbllm   16  1  8 'some_comment'  ''

    
    # 384k tbs=2 node=16
    # n  bs  mb   t         mode   cp  pp tp    comment    env_var
    16   2   1 393216    d2       16  1  8 'some_comment'  'EXPERIMENT_SCHEDULE_PRIORITY=50'
    16   2   1 393216    wlbllm    8  1  8 'some_comment'  'EXPERIMENT_SCHEDULE_PRIORITY=50'
    
    16   2   1 393216    wlbllm    4  1  8 'some_comment'  ''
    16   2   1 393216    wlbllm   16  1  8 'some_comment'  ''

    
# >>>

# <<< EXPERIMENT__SKIP=0,EXPERIMENT__COMPLETED=0,MAX_SAMPLE_ID=30,EXPERIMENT_REPEAT_TIMES=1,EXPERIMENT_WARMUP_TIMES=0,SHOULD_ADD_DEBUG_CASES=0,NUM_LAYERS=48

# 384k tbs=2 node=16
    # n  bs  mb   t         mode   cp  pp tp    comment    env_var
    16   2   1 393216    d2       16  1  8 'some_comment'  'EXPERIMENT_SCHEDULE_PRIORITY=50'
    16   2   1 393216    d2       8   2  8 'some_comment'  'EXPERIMENT_SCHEDULE_PRIORITY=50'
    16   2   1 393216    wlbllm    8  1  8 'some_comment'  'EXPERIMENT_SCHEDULE_PRIORITY=50'
    
    16   2   1 393216    wlbllm    4  1  8 'some_comment'  ''
    16   2   1 393216    wlbllm   16  1  8 'some_comment'  ''

# >>>


# Global variable overrides (use # $$ syntax)
# $$FOLDER_SEPARATOR=1
# $$EXPERIMENT_DISTS=("prolong 0.3")
# $$EXPERIMENT_BALANCE_PING_PONG=1
# $$OUTPUT_DIR_PREFIX=/mnt/weka/home/yonghao.zhuang/jd/d2/benchmarks/_250923_test_pp_34b/logs.v20-small-scale-pp-34b-16node-384k-prolong-0.3-sample5


# <<< EXPERIMENT__SKIP=0,EXPERIMENT__COMPLETED=0,MAX_SAMPLE_ID=5,EXPERIMENT_REPEAT_TIMES=1,EXPERIMENT_WARMUP_TIMES=0,SHOULD_ADD_DEBUG_CASES=0,NUM_LAYERS=48,CHANGE_LONG_DOC_RATIO=0.3,EXPERIMENT_NVSHMEM_BUFFER_SIZE_GB=2.0,MIN_TOLERANCE_FACTOR=0.125
# n  bs  mb   t         mode   cp  pp tp    comment    env_var
# 16   1   4 393216    d2        8  2  8 'some_comment'  'EXPERIMENT_SCHEDULE_PRIORITY=50'
16   2   2 393216    d2        8  2  8 'some_comment'  ''
16   4   1 393216    d2       16  1  8 'some_comment'  'EXPERIMENT_SCHEDULE_PRIORITY=40,EXPERIMENT__STATUS=FAIL,EXPERIMENT__FAIL_REASON=OOM'
# 16   1   4 393216    wlbllm    8  2  8 'some_comment'  ''
# 16   4   1 393216    wlbllm    2  1  8 'some_comment'  ''
# 16   4   1 393216    wlbllm    4  1  8 'some_comment'  'EXPERIMENT_SCHEDULE_PRIORITY=40'
# 16   4   1 393216    wlbllm    8  1  8 'some_comment'  'EXPERIMENT_SCHEDULE_PRIORITY=50'
# 16   4   1 393216    wlbllm   16  1  8 'some_comment'  ''
# >>>



# Global variable overrides (use # $$ syntax)
# $$OUTPUT_DIR_PREFIX=/mnt/weka/home/yonghao.zhuang/jd/d2/benchmarks/_250923_test_pp_34b/logs.v32-small-scale-pp-34b-16node-384k-prolong-0.8-sample5
# $$FOLDER_SEPARATOR=1
# $$EXPERIMENT_DISTS=("prolong 0.8")
# $$EXPERIMENT_BALANCE_PING_PONG=1

# <<< EXPERIMENT__SKIP=0,EXPERIMENT__COMPLETED=0,MAX_SAMPLE_ID=5,EXPERIMENT_REPEAT_TIMES=1,EXPERIMENT_WARMUP_TIMES=0,SHOULD_ADD_DEBUG_CASES=0,NUM_LAYERS=48,CHANGE_LONG_DOC_RATIO=0.8,EXPERIMENT_NVSHMEM_BUFFER_SIZE_GB=1.5,MIN_TOLERANCE_FACTOR=0.1
    
    # 384k tbs=4 node=16
    # n  bs  mb   t         mode   cp  pp tp    comment    env_var
    16   1   4 393216    d2        8  2  8 'some_comment'  'EXPERIMENT_SCHEDULE_PRIORITY=50'
    16   1   4 393216    wlbllm    8  2  8 'some_comment'  'EXPERIMENT_SCHEDULE_PRIORITY=50'
    16   2   2 393216    d2        8  2  8 'some_comment'  ''
    16   4   1 393216    d2       16  1  8 'some_comment'  'EXPERIMENT_SCHEDULE_PRIORITY=40'
    16   4   1 393216    wlbllm    2  1  8 'some_comment'  ''
    16   4   1 393216    wlbllm    4  1  8 'some_comment'  'EXPERIMENT_SCHEDULE_PRIORITY=40'
    16   4   1 393216    wlbllm    8  1  8 'some_comment'  'EXPERIMENT_SCHEDULE_PRIORITY=40'
    16   4   1 393216    wlbllm   16  1  8 'some_comment'  ''

    # 384k tbs=2 node=16
    # n  bs  mb   t         mode   cp  pp tp    comment    env_var
    16   2   1 393216    d2       16  1  8 'some_comment'  'EXPERIMENT_SCHEDULE_PRIORITY=30'
    16   2   1 393216    d2       8   2  8 'some_comment'  'EXPERIMENT_SCHEDULE_PRIORITY=30'
    16   2   1 393216    wlbllm    8  1  8 'some_comment'  'EXPERIMENT_SCHEDULE_PRIORITY=30'
    16   2   1 393216    wlbllm    4  1  8 'some_comment'  ''
    16   2   1 393216    wlbllm   16  1  8 'some_comment'  ''
    

# >>>


# ------------- $$$ Start here $$$ -----------------
# Global variable overrides (use # $$ syntax)
# $$OUTPUT_DIR_PREFIX=/mnt/weka/home/yonghao.zhuang/jd/d2/benchmarks/_250923_test_pp_34b/logs.v32-small-scale-pp-34b-16node-384k-prolong-0.8-sample5
# $$FOLDER_SEPARATOR=1
# $$EXPERIMENT_DISTS=("prolong 0.8")
# $$EXPERIMENT_BALANCE_PING_PONG=1

# <<< EXPERIMENT__SKIP=0,EXPERIMENT__COMPLETED=0,MAX_SAMPLE_ID=5,EXPERIMENT_REPEAT_TIMES=1,EXPERIMENT_WARMUP_TIMES=0,SHOULD_ADD_DEBUG_CASES=0,NUM_LAYERS=48,CHANGE_LONG_DOC_RATIO=0.8,EXPERIMENT_NVSHMEM_BUFFER_SIZE_GB=1.5,MIN_TOLERANCE_FACTOR=0.1
    
    # 384k tbs=4 node=16
    # n  bs  mb   t         mode   cp  pp tp    comment    env_var
    16   1   4 393216    d2        8  2  8 'some_comment'  'EXPERIMENT_SCHEDULE_PRIORITY=50'
    16   1   4 393216    wlbllm    8  2  8 'some_comment'  'EXPERIMENT_SCHEDULE_PRIORITY=50'
    16   2   2 393216    d2        8  2  8 'some_comment'  ''
    16   4   1 393216    d2       16  1  8 'some_comment'  'EXPERIMENT_SCHEDULE_PRIORITY=40'
    16   4   1 393216    wlbllm    2  1  8 'some_comment'  ''
    16   4   1 393216    wlbllm    4  1  8 'some_comment'  'EXPERIMENT_SCHEDULE_PRIORITY=40'
    16   4   1 393216    wlbllm    8  1  8 'some_comment'  'EXPERIMENT_SCHEDULE_PRIORITY=40'
    16   4   1 393216    wlbllm   16  1  8 'some_comment'  ''

    # 384k tbs=2 node=16
    # n  bs  mb   t         mode   cp  pp tp    comment    env_var
    16   2   1 393216    d2       16  1  8 'some_comment'  'EXPERIMENT_SCHEDULE_PRIORITY=30'
    16   2   1 393216    d2       8   2  8 'some_comment'  'EXPERIMENT_SCHEDULE_PRIORITY=30'
    16   2   1 393216    wlbllm    8  1  8 'some_comment'  'EXPERIMENT_SCHEDULE_PRIORITY=30'
    16   2   1 393216    wlbllm    4  1  8 'some_comment'  ''
    16   2   1 393216    wlbllm   16  1  8 'some_comment'  ''
    

# >>>
# ------------- Stop here -----------------

# # Global variable overrides (use # $$ syntax)
# # $$OUTPUT_DIR_PREFIX=/mnt/weka/home/yonghao.zhuang/jd/d2/benchmarks/_250923_test_pp_34b/logs.v8-small-scale-pp-34b-16node-128k-256k-384k-prolong-sample5
# # $$FOLDER_SEPARATOR=1
# # $$EXPERIMENT_DISTS=("prolong 0.3")
# # $$EXPERIMENT_BALANCE_PING_PONG=1

# # <<< EXPERIMENT__SKIP=0,EXPERIMENT__COMPLETED=0,MAX_SAMPLE_ID=5,EXPERIMENT_REPEAT_TIMES=1,EXPERIMENT_WARMUP_TIMES=0,SHOULD_ADD_DEBUG_CASES=0,NUM_LAYERS=48,CHANGE_LONG_DOC_RATIO=0.3,EXPERIMENT_NVSHMEM_BUFFER_SIZE_GB=1.5,MIN_TOLERANCE_FACTOR=0.125
# # n  bs  mb   t         mode   cp  pp tp    comment    env_var
# 16   1   4 393216    d2        8  2  8 'some_comment'  'EXPERIMENT_SCHEDULE_PRIORITY=50'
# 16   2   2 393216    d2        8  2  8 'some_comment'  ''
# 16   4   1 393216    d2       16  1  8 'some_comment'  'EXPERIMENT_SCHEDULE_PRIORITY=40'
# 16   1   4 393216    wlbllm    8  2  8 'some_comment'  ''
# 16   4   1 393216    wlbllm    2  1  8 'some_comment'  ''
# 16   4   1 393216    wlbllm    4  1  8 'some_comment'  'EXPERIMENT_SCHEDULE_PRIORITY=40'
# 16   4   1 393216    wlbllm    8  1  8 'some_comment'  'EXPERIMENT_SCHEDULE_PRIORITY=50'
# 16   4   1 393216    wlbllm   16  1  8 'some_comment'  ''
# # >>>
