# ▛▀▖      ▐        ▗     ▞▀▖▌ ▌▛▀▖ ▗▌ ▞▀▖        ▌   
# ▙▄▘▙▀▖▞▀▖▜▀ ▙▀▖▝▀▖▄ ▛▀▖  ▄▘▚▄▌▙▄▘  ▌ ▙▄ ▛▀▖▞▀▖▞▀▌▞▀▖
# ▌  ▌  ▛▀ ▐ ▖▌  ▞▀▌▐ ▌ ▌ ▖ ▌  ▌▌ ▌  ▌ ▌ ▌▌ ▌▌ ▌▌ ▌▛▀ 
# ▘  ▘  ▝▀▘ ▀ ▘  ▝▀▘▀▘▘ ▘ ▝▀   ▘▀▀  ▝▀ ▝▀ ▘ ▘▝▀ ▝▀▘▝▀▘

# ▗▌ ▞▀▖▞▀▖▌   ▞▀▖▛▀▘▞▀▖▌   ▛▀▘▗▌ ▞▀▖▌                
#  ▌  ▗▘▚▄▘▌▗▘  ▗▘▙▄ ▙▄ ▌▗▘ ▙▄  ▌  ▗▘▌▗▘              
#  ▌ ▗▘ ▌ ▌▛▚  ▗▘ ▖ ▌▌ ▌▛▚  ▖ ▌ ▌ ▗▘ ▛▚               
# ▝▀ ▀▀▘▝▀ ▘ ▘ ▀▀▘▝▀ ▝▀ ▘ ▘ ▝▀ ▝▀ ▀▀▘▘ ▘              

# ▞▀▖▞▀▖▞▀▖▞▀▖ ▞▀▖▗▌   ▞▀▖▞▀▖ ▛▀▖▞▀▖▀▛▘               
# ▌▞▌▚▄▌ ▗▘▚▄▘ ▌▞▌ ▌ ▐▌▌▞▌▌▞▌ ▙▄▘▚▄  ▌                
# ▛ ▌▖ ▌▗▘ ▌ ▌ ▛ ▌ ▌ ▗▖▛ ▌▛ ▌ ▌  ▖ ▌ ▌                
# ▝▀ ▝▀ ▀▀▘▝▀  ▝▀ ▝▀ ▝▘▝▀ ▝▀  ▘  ▝▀  ▘                




# Global variable overrides (use # $$ syntax)
# $$OUTPUT_DIR_PREFIX=/mnt/weka/home/yonghao.zhuang/jd/d2/benchmarks/_250923_test_pp_34b/logs.v9-large-scale-pp-34b-16node-128k-256k-384k-pretrain
# $$FOLDER_SEPARATOR=1
# $$EXPERIMENT_DISTS=("wlbllm 0.0")
# $$EXPERIMENT_BALANCE_PING_PONG=1

# bash /mnt/weka/home/yonghao.zhuang/jd/d2/benchmarks/_250923_test_pp_34b/salloc_srun.pp_34b.v2.sh --config /mnt/weka/home/yonghao.zhuang/jd/d2/benchmarks/_250923_test_pp_34b/config_sweep_memory.config.v4_0928_0000PST_16node-pretrain-0.0.sh



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
#      Pretrain 34B (2025/09/28 00:00:00 AM PST - 
# ------------------------------------------------------------------------------------------------
# <<< EXPERIMENT__SKIP=0,EXPERIMENT__COMPLETED=0,MAX_SAMPLE_ID=30,EXPERIMENT_REPEAT_TIMES=1,EXPERIMENT_WARMUP_TIMES=0,SHOULD_ADD_DEBUG_CASES=0,NUM_LAYERS=48,CHANGE_LONG_DOC_RATIO=0.0

    # 128k tbs=16 node=16
    # n  bs  mb   t         mode   cp  pp tp    comment    env_var
    16   2   8 131072    d2        8  2  8 'some_comment'  'EXPERIMENT_SCHEDULE_PRIORITY=100,EXPERIMENT__STATUS=PARTIAL,RESULT_DIR=28_012011.d2-n16-t131072-b2-mb8-cp8tp8pp2-codellama_CodeLlama-34b-hf-L48-wlbllm_0.0'
    16   1  16 131072    wlbllm    8  2  8 'some_comment'  'EXPERIMENT_SCHEDULE_PRIORITY=100,EXPERIMENT__STATUS=PASS,RESULT_DIR=28_020244.wlbllm-n16-t131072-b1-mb16-cp8tp8pp2-codellama_CodeLlama-34b-hf-L48-wlbllm_0.0'
    
    16   1  16 131072    d2        8  2  8 'some_comment'  ''
    16   2   8 131072    wlbllm    4  2  8 'some_comment'  ''

    # 16   1  16 131072    d2        4  4  8 'some_comment'  ''
    # 16   2   8 131072    wlbllm    4  4  8 'some_comment'  ''
    # 16   2   8 131072    wlbllm    8  2  8 'some_comment'  ''
    # 16   4   4 131072    wlbllm    2  2  8 'some_comment'  ''
    # 16   4   4 131072    wlbllm    4  2  8 'some_comment'  ''


    # 128k tbs=32 node=16
    #n  bs  mb   t         mode   cp  pp tp    comment    env_var
    16   2  16 131072    d2        8  2  8 'some_comment'  'EXPERIMENT_SCHEDULE_PRIORITY=80,EXPERIMENT__STATUS=FAIL,RESULT_DIR=28_025248.d2-n16-t131072-b2-mb16-cp8tp8pp2-codellama_CodeLlama-34b-hf-L48-wlbllm_0.0'
    16   1  32 131072    d2        8  2  8 'some_comment'  ''
    16   1  32 131072    d2        4  4  8 'some_comment'  ''
    16   2  16 131072    wlbllm    4  2  8 'some_comment'  'EXPERIMENT_SCHEDULE_PRIORITY=80,EXPERIMENT__STATUS=PARTIAL,RESULT_DIR=28_022651.wlbllm-n16-t131072-b2-mb16-cp4tp8pp2-codellama_CodeLlama-34b-hf-L48-wlbllm_0.0'
    16   2  16 131072    wlbllm    4  4  8 'some_comment'  'EXPERIMENT__STATUS=PARTIAL,RESULT_DIR=28_053717.wlbllm-n16-t131072-b2-mb16-cp4tp8pp4-codellama_CodeLlama-34b-hf-L48-wlbllm_0.0'
    16   1  32 131072    wlbllm    8  2  8 'some_comment'  'EXPERIMENT__STATUS=PARTIAL,RESULT_DIR=28_060550.wlbllm-n16-t131072-b1-mb32-cp8tp8pp2-codellama_CodeLlama-34b-hf-L48-wlbllm_0.0'
    16   2  16 131072    wlbllm    8  2  8 'some_comment'  'EXPERIMENT__STATUS=PARTIAL,RESULT_DIR=28_063247.wlbllm-n16-t131072-b2-mb16-cp8tp8pp2-codellama_CodeLlama-34b-hf-L48-wlbllm_0.0'
    16   4   8 131072    wlbllm    2  2  8 'some_comment'  'EXPERIMENT__STATUS=PARTIAL,RESULT_DIR=28_065943.wlbllm-n16-t131072-b4-mb8-cp2tp8pp2-codellama_CodeLlama-34b-hf-L48-wlbllm_0.0'
    16   4   8 131072    wlbllm    4  2  8 'some_comment'  'EXPERIMENT__STATUS=PARTIAL,RESULT_DIR=28_072605.wlbllm-n16-t131072-b4-mb8-cp4tp8pp2-codellama_CodeLlama-34b-hf-L48-wlbllm_0.0'


    # 256k tbs=4 node=16
    # n  bs  mb   t         mode   cp  pp tp    comment    env_var
    16   1   4 262144    d2        8  2  8 'some_comment'  'EXPERIMENT_SCHEDULE_PRIORITY=70,EXPERIMENT__STATUS=PARTIAL,RESULT_DIR=28_025822.d2-n16-t262144-b1-mb4-cp8tp8pp2-codellama_CodeLlama-34b-hf-L48-wlbllm_0.0'
    16   1   4 262144    wlbllm    8  2  8 'some_comment'  'EXPERIMENT_SCHEDULE_PRIORITY=70,EXPERIMENT__STATUS=PASS,RESULT_DIR=28_030539.wlbllm-n16-t262144-b1-mb4-cp8tp8pp2-codellama_CodeLlama-34b-hf-L48-wlbllm_0.0'

    16   4   1 262144    d2       16  1  8 'some_comment'  'EXPERIMENT__STATUS=PARTIAL,RESULT_DIR=28_075255.d2-n16-t262144-b4-mb1-cp16tp8pp1-codellama_CodeLlama-34b-hf-L48-wlbllm_0.0'
    16   4   1 262144    wlbllm    2  1  8 'some_comment'  'EXPERIMENT__STATUS=PASS,RESULT_DIR=28_080015.wlbllm-n16-t262144-b4-mb1-cp2tp8pp1-codellama_CodeLlama-34b-hf-L48-wlbllm_0.0'
    16   4   1 262144    wlbllm    4  1  8 'some_comment'  'EXPERIMENT__STATUS=FAIL,EXPERIMENT__FAIL_REASON=OOM,RESULT_DIR=28_082437.wlbllm-n16-t262144-b4-mb1-cp4tp8pp1-codellama_CodeLlama-34b-hf-L48-wlbllm_0.0'
    16   4   1 262144    wlbllm    8  1  8 'some_comment'  'EXPERIMENT__STATUS=FAIL,EXPERIMENT__FAIL_REASON=OOM,RESULT_DIR=28_082548.wlbllm-n16-t262144-b4-mb1-cp8tp8pp1-codellama_CodeLlama-34b-hf-L48-wlbllm_0.0'
    16   4   1 262144    wlbllm   16  1  8 'some_comment'  'EXPERIMENT__STATUS=FAIL,EXPERIMENT__FAIL_REASON=OOM,RESULT_DIR=28_082659.wlbllm-n16-t262144-b4-mb1-cp16tp8pp1-codellama_CodeLlama-34b-hf-L48-wlbllm_0.0'
    
    # 256k tbs=8 node=16
    # n  bs  mb   t         mode   cp  pp tp    comment    env_var
    16   1   8 262144    d2        8  2  8 'some_comment'  'EXPERIMENT_SCHEDULE_PRIORITY=60,EXPERIMENT__STATUS=PASS,RESULT_DIR=28_032550.d2-n16-t262144-b1-mb8-cp8tp8pp2-codellama_CodeLlama-34b-hf-L48-wlbllm_0.0'
    16   1   8 262144    wlbllm    8  2  8 'some_comment'  'EXPERIMENT_SCHEDULE_PRIORITY=60,EXPERIMENT__STATUS=PASS,RESULT_DIR=28_042108.wlbllm-n16-t262144-b1-mb8-cp8tp8pp2-codellama_CodeLlama-34b-hf-L48-wlbllm_0.0'

    16   8   1 262144    d2       16  1  8 'some_comment'  'EXPERIMENT__STATUS=FAIL,EXPERIMENT__FAIL_REASON=OOM,RESULT_DIR=28_082809.d2-n16-t262144-b8-mb1-cp16tp8pp1-codellama_CodeLlama-34b-hf-L48-wlbllm_0.0'
    16   2   4 262144    wlbllm    4  2  8 'some_comment'  'EXPERIMENT__STATUS=FAIL,RESULT_DIR=28_082935.wlbllm-n16-t262144-b2-mb4-cp4tp8pp2-codellama_CodeLlama-34b-hf-L48-wlbllm_0.0'
    16   2   4 262144    wlbllm    8  2  8 'some_comment'  ''
    16   8   1 262144    wlbllm    1  1  8 'some_comment'  ''
    16   8   1 262144    wlbllm    2  1  8 'some_comment'  ''
    16   8   1 262144    wlbllm    4  1  8 'some_comment'  ''
    16   8   1 262144    wlbllm    8  1  8 'some_comment'  ''
    16   8   1 262144    wlbllm   16  1  8 'some_comment'  ''

    
    # 384k tbs=2 node=16
    # n  bs  mb   t         mode   cp  pp tp    comment    env_var
    16   2   1 393216    d2       16  1  8 'some_comment'  'EXPERIMENT_SCHEDULE_PRIORITY=50,EXPERIMENT__STATUS=PASS,RESULT_DIR=28_045626.d2-n16-t393216-b2-mb1-cp16tp8pp1-codellama_CodeLlama-34b-hf-L48-wlbllm_0.0'
    16   2   1 393216    wlbllm    8  1  8 'some_comment'  'EXPERIMENT_SCHEDULE_PRIORITY=50,EXPERIMENT__STATUS=PASS,RESULT_DIR=28_052045.wlbllm-n16-t393216-b2-mb1-cp8tp8pp1-codellama_CodeLlama-34b-hf-L48-wlbllm_0.0'
    
    16   2   1 393216    wlbllm    4  1  8 'some_comment'  ''
    16   2   1 393216    wlbllm   16  1  8 'some_comment'  ''
    
# >>>









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



# <<< EXPERIMENT__SKIP=1,EXPERIMENT__COMPLETED=0,MAX_SAMPLE_ID=30,EXPERIMENT_REPEAT_TIMES=1,EXPERIMENT_WARMUP_TIMES=0,SHOULD_ADD_DEBUG_CASES=0,NUM_LAYERS=48,CHANGE_LONG_DOC_RATIO=0.0,EXPERIMENT_NVSHMEM_BUFFER_SIZE_GB=1.0,MIN_TOLERANCE_FACTOR=0.125


    # 384k tbs=2 node=16
    # n  bs  mb   t         mode   cp  pp tp    comment    env_var
    16   1   2 393216    d2       8   2  8 'some_comment'  'EXPERIMENT_SCHEDULE_PRIORITY=50'
    16   2   1 393216    d2       16  1  8 'some_comment'  'EXPERIMENT_SCHEDULE_PRIORITY=50'
    # 16   2   1 393216    wlbllm    8  1  8 'some_comment'  'EXPERIMENT_SCHEDULE_PRIORITY=50,EXPERIMENT__STATUS=PASS,RESULT_DIR=28_052045.wlbllm-n16-t393216-b2-mb1-cp8tp8pp1-codellama_CodeLlama-34b-hf-L48-wlbllm_0.0'
    
    # 16   2   1 393216    wlbllm    4  1  8 'some_comment'  ''
    # 16   2   1 393216    wlbllm   16  1  8 'some_comment'  ''

# >>>

# ------------- $$$ Start here $$$ -----------------
# Global variable overrides (use # $$ syntax)
# $$OUTPUT_DIR_PREFIX=/mnt/weka/home/yonghao.zhuang/jd/d2/benchmarks/_250923_test_pp_34b/logs.v30-small-scale-pp-34b-16node-384k-pretrain-sample5
# $$FOLDER_SEPARATOR=1
# $$EXPERIMENT_DISTS=("wlbllm 0.0")
# $$EXPERIMENT_BALANCE_PING_PONG=1

# <<< EXPERIMENT__SKIP=0,EXPERIMENT__COMPLETED=0,MAX_SAMPLE_ID=5,EXPERIMENT_REPEAT_TIMES=1,EXPERIMENT_WARMUP_TIMES=0,SHOULD_ADD_DEBUG_CASES=0,NUM_LAYERS=48,CHANGE_LONG_DOC_RATIO=0.0,EXPERIMENT_NVSHMEM_BUFFER_SIZE_GB=1.5,MIN_TOLERANCE_FACTOR=0.1
    # n  bs  mb   t         mode   cp  pp tp    comment    env_var
    16   1   4 393216    d2        8  2  8 'some_comment'  'EXPERIMENT_SCHEDULE_PRIORITY=50'
    16   1   4 393216    wlbllm    8  2  8 'some_comment'  'EXPERIMENT_SCHEDULE_PRIORITY=50'
    16   2   2 393216    d2        8  2  8 'some_comment'  ''
    16   4   1 393216    d2       16  1  8 'some_comment'  'EXPERIMENT_SCHEDULE_PRIORITY=40'
    16   4   1 393216    wlbllm    2  1  8 'some_comment'  ''
    16   4   1 393216    wlbllm    4  1  8 'some_comment'  'EXPERIMENT_SCHEDULE_PRIORITY=40'
    16   4   1 393216    wlbllm    8  1  8 'some_comment'  'EXPERIMENT_SCHEDULE_PRIORITY=40'
    16   4   1 393216    wlbllm   16  1  8 'some_comment'  ''
# >>>
# ------------- Stop here -----------------

