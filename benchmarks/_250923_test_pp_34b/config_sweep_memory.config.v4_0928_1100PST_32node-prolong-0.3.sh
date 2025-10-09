# ┏━┓┏━┓┏━┓╻  ┏━┓┏┓╻┏━╸   ┏━┓╻ ╻┏┓    ┏━┓┏━┓┏┓╻┏━┓╺┳┓┏━╸
# ┣━┛┣┳┛┃ ┃┃  ┃ ┃┃┗┫┃╺┓   ╺━┫┗━┫┣┻┓   ╺━┫┏━┛┃┗┫┃ ┃ ┃┃┣╸ 
# ╹  ╹┗╸┗━┛┗━╸┗━┛╹ ╹┗━┛   ┗━┛  ╹┗━┛   ┗━┛┗━╸╹ ╹┗━┛╺┻┛┗━╸
# ╺┓ ┏━┓┏━┓╻┏    ┏━┓┏━╸┏━┓╻┏    ┏━┓┏━┓╻ ╻╻┏             
#  ┃ ┏━┛┣━┫┣┻┓   ┏━┛┗━┓┣━┓┣┻┓   ╺━┫┣━┫┗━┫┣┻┓            
# ╺┻╸┗━╸┗━┛╹ ╹   ┗━╸┗━┛┗━┛╹ ╹   ┗━┛┗━┛  ╹╹ ╹            
# ┏━┓┏━┓┏━┓┏━┓   ╺┓ ╺┓  ┏━┓┏━┓   ┏━┓┏━┓╺┳╸              
# ┃┃┃┗━┫┏━┛┣━┫    ┃  ┃ ╹┃┃┃┃┃┃   ┣━┛┗━┓ ┃               
# ┗━┛┗━┛┗━╸┗━┛   ╺┻╸╺┻╸╹┗━┛┗━┛   ╹  ┗━┛ ╹                     


# Global variable overrides (use # $$ syntax)
# $$OUTPUT_DIR_PREFIX=/mnt/weka/home/yonghao.zhuang/jd/d2/benchmarks/_250923_test_pp_34b/logs.v12-large-scale-pp-34b-32node-128k-256k-384k-prolong-0.3
# $$FOLDER_SEPARATOR=1
# $$EXPERIMENT_DISTS=("prolong 0.3")
# $$EXPERIMENT_BALANCE_PING_PONG=1

# bash /mnt/weka/home/yonghao.zhuang/jd/d2/benchmarks/_250923_test_pp_34b/salloc_srun.pp_34b.v2.sh --config /mnt/weka/home/yonghao.zhuang/jd/d2/benchmarks/_250923_test_pp_34b/config_sweep_memory.config.v4_0928_1100PST_32node-prolong-0.3.sh



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
#   Node 32 pretrain small OOM sweep on wlbllm and d2 (2025/09/28 11:00:00 AM PST - 
# ------------------------------------------------------------------------------------------------



# <<< EXPERIMENT__SKIP=0,EXPERIMENT__COMPLETED=0,EXPERIMENT_REPEAT_TIMES=1,EXPERIMENT_WARMUP_TIMES=0,SHOULD_ADD_DEBUG_CASES=0,NUM_LAYERS=48,CHANGE_LONG_DOC_RATIO=0.0,EXPERIMENT_NVSHMEM_BUFFER_SIZE_GB=1.5,MIN_TOLERANCE_FACTOR=0.15,MAX_SAMPLE_ID=5

# 128k tbs=32 node=32
    #n  bs  mb   t         mode   cp  pp tp    comment    env_var
    32   4   8 131072    d2       16  2  8 'some_comment'  'EXPERIMENT_SCHEDULE_PRIORITY=100'
    32   2  16 131072    wlbllm    8  2  8 'some_comment'  'EXPERIMENT_SCHEDULE_PRIORITY=100'

    32   1  32 131072    d2        4  8  8 'some_comment'  'EXPERIMENT_SCHEDULE_PRIORITY=60'
    32   1  32 131072    d2        8  4  8 'some_comment'  'EXPERIMENT_SCHEDULE_PRIORITY=30'
    32   1  32 131072    d2       16  2  8 'some_comment'  'EXPERIMENT_SCHEDULE_PRIORITY=30'
    32   2  16 131072    d2        4  8  8 'some_comment'  ''
    32   2  16 131072    d2        8  4  8 'some_comment'  ''
    32   2  16 131072    d2       16  2  8 'some_comment'  'EXPERIMENT_SCHEDULE_PRIORITY=50'
    32   4   8 131072    d2        8  4  8 'some_comment'  ''
    32   8   4 131072    d2       16  2  8 'some_comment'  ''
    32  32   1 131072    d2       32  1  8 'some_comment'  '' # oom?

    32   1  32 131072    wlbllm   16  2  8 'some_comment'  ''
    32   2  16 131072    wlbllm    8  4  8 'some_comment'  ''
    32   2  16 131072    wlbllm   16  2  8 'some_comment'  ''
    32   4   8 131072    wlbllm    4  2  8 'some_comment'  ''
    32   4   8 131072    wlbllm    4  4  8 'some_comment'  ''
    32   4   8 131072    wlbllm    8  2  8 'some_comment'  ''
    32   4   8 131072    wlbllm    8  4  8 'some_comment'  ''
    32   4   8 131072    wlbllm   16  2  8 'some_comment'  ''
    32   8   4 131072    wlbllm    2  2  8 'some_comment'  '' # oom
    32   8   4 131072    wlbllm    4  2  8 'some_comment'  ''
    32   8   4 131072    wlbllm    8  2  8 'some_comment'  ''
    32   8   4 131072    wlbllm   16  2  8 'some_comment'  ''
    32  32   1 131072    wlbllm    1  1  8 'some_comment'  '' # oom
    32  32   1 131072    wlbllm    2  1  8 'some_comment'  '' # oom
    32  32   1 131072    wlbllm    4  1  8 'some_comment'  ''
    32  32   1 131072    wlbllm    8  1  8 'some_comment'  ''
    32  32   1 131072    wlbllm   16  1  8 'some_comment'  ''
    32  32   1 131072    wlbllm   32  1  8 'some_comment'  ''
    
# 256k tbs=16 node=32
    # n  bs  mb   t         mode   cp  pp tp    comment    env_var
    32   2   8 262144    d2       16  2  8 'some_comment'  'EXPERIMENT_SCHEDULE_PRIORITY=90,EXPERIMENT_NVSHMEM_BUFFER_SIZE_GB=1.5,MIN_TOLERANCE_FACTOR=0.15'
    32   2   8 262144    wlbllm    8  2  8 'some_comment'  'EXPERIMENT_SCHEDULE_PRIORITY=90'

    32   1  16 262144    d2        4  8  8 'some_comment'  'EXPERIMENT_NVSHMEM_BUFFER_SIZE_GB=1.5,MIN_TOLERANCE_FACTOR=0.15' # memory issue
    32   1  16 262144    d2        8  4  8 'some_comment'  'EXPERIMENT_NVSHMEM_BUFFER_SIZE_GB=1.5,MIN_TOLERANCE_FACTOR=0.15'
    32   1  16 262144    d2       16  2  8 'some_comment'  'EXPERIMENT_NVSHMEM_BUFFER_SIZE_GB=1.5,MIN_TOLERANCE_FACTOR=0.15'
    32   2   8 262144    d2        8  4  8 'some_comment'  'EXPERIMENT_NVSHMEM_BUFFER_SIZE_GB=1.5,MIN_TOLERANCE_FACTOR=0.15'
    32  16   1 262144    d2       32  1  8 'some_comment'  'EXPERIMENT_NVSHMEM_BUFFER_SIZE_GB=1.5,MIN_TOLERANCE_FACTOR=0.15'
    32   4   4 262144    d2       16  2  8 'some_comment'  'EXPERIMENT_NVSHMEM_BUFFER_SIZE_GB=1.5,MIN_TOLERANCE_FACTOR=0.15'
    32  16   1 262144    wlbllm   16  1  8 'some_comment'  ''
    32  16   1 262144    wlbllm    8  1  8 'some_comment'  ''
    32  16   1 262144    wlbllm    4  1  8 'some_comment'  ''
    32  16   1 262144    wlbllm    2  1  8 'some_comment'  ''
    32  16   1 262144    wlbllm    1  1  8 'some_comment'  ''
    32   4   4 262144    wlbllm    4  2  8 'some_comment'  ''
    32   4   4 262144    wlbllm    8  2  8 'some_comment'  ''
    32   2   8 262144    wlbllm   16  2  8 'some_comment'  ''
    32   2   8 262144    wlbllm    8  4  8 'some_comment'  ''
    32   1  16 262144    wlbllm   16  2  8 'some_comment'  ''
    32   4   4 262144    wlbllm   16  2  8 'some_comment'  ''
    32  16   1 262144    wlbllm   32  1  8 'some_comment'  ''

# 384k tbs=4 node=32
    # n  bs  mb   t         mode   cp  pp tp    comment    env_var
    32   1   4 393216    d2       16  2  8 'some_comment'  'EXPERIMENT_SCHEDULE_PRIORITY=80'
    32   4   1 393216    d2       32  1  8 'some_comment'  'EXPERIMENT_SCHEDULE_PRIORITY=70'
    32   1   4 393216    wlbllm   16  2  8 'some_comment'  'EXPERIMENT_SCHEDULE_PRIORITY=80'
    32   4   1 393216    wlbllm    4  1  8 'some_comment'  ''
    32   4   1 393216    wlbllm    8  1  8 'some_comment'  ''
    32   4   1 393216    wlbllm   16  1  8 'some_comment'  ''
    32   4   1 393216    wlbllm   32  1  8 'some_comment'  ''

# >>>