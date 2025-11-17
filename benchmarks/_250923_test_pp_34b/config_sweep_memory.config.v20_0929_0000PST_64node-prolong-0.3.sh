# ┏━┓┏━┓┏━┓╻  ┏━┓┏┓╻┏━╸   ┏━┓╻ ╻┏┓    ┏━┓╻ ╻┏┓╻┏━┓╺┳┓┏━╸┏━┓
# ┣━┛┣┳┛┃ ┃┃  ┃ ┃┃┗┫┃╺┓   ╺━┫┗━┫┣┻┓   ┣━┓┗━┫┃┗┫┃ ┃ ┃┃┣╸ ┗━┓
# ╹  ╹┗╸┗━┛┗━╸┗━┛╹ ╹┗━┛   ┗━┛  ╹┗━┛   ┗━┛  ╹╹ ╹┗━┛╺┻┛┗━╸┗━┛
# ╺┓ ┏━┓┏━┓╻┏    ┏━┓┏━╸┏━┓╻┏    ┏━┓┏━┓╻ ╻╻┏              
#  ┃ ┏━┛┣━┫┣┻┓   ┏━┛┗━┓┣━┓┣┻┓   ╺━┫┣━┫┗━┫┣┻┓             
# ╺┻╸┗━╸┗━┛╹ ╹   ┗━╸┗━┛┗━┛╹ ╹   ┗━┛┗━┛  ╹╹ ╹             
# ┏━┓┏━┓┏━┓┏━┓   ╺┓ ┏━┓┏━┓┏━┓   ┏━┓┏━┓╺┳╸                
# ┃┃┃┗━┫┏━┛┣━┫    ┃   ┃┃┃┃┃┃┃   ┣━┛┗━┓ ┃                 
# ┗━┛┗━┛┗━╸┗━┛   ╺┻╸  ╹┗━┛┗━┛   ╹  ┗━┛ ╹                 

# Global variable overrides (use # $$ syntax)
# $$FOLDER_SEPARATOR=1
# $$EXPERIMENT_BALANCE_PING_PONG=1
# $$EXPERIMENT_REPEAT_TIMES=1
# $$EXPERIMENT_WARMUP_TIMES=0
# $$SHOULD_ADD_DEBUG_CASES=0
# $$NUM_LAYERS=48
# $$EXPERIMENT_NVSHMEM_BUFFER_SIZE_GB=1.5
# $$MIN_TOLERANCE_FACTOR=0.15
# $$MAX_SAMPLE_ID=30
# $$EXPERIMENT_DISTS=("prolong 0.3")
# $$CHANGE_LONG_DOC_RATIO=0.3
# $$OUTPUT_DIR_PREFIX=/mnt/weka/home/yonghao.zhuang/jd/d2/benchmarks/_250923_test_pp_34b/logs.v60-large-scale-pp-34b-64node-128k-256k-384k-prolong-0.3

# bash /mnt/weka/home/yonghao.zhuang/jd/d2/benchmarks/_250923_test_pp_34b/salloc_srun.pp_34b.v2.sh --config /mnt/weka/home/yonghao.zhuang/jd/d2/benchmarks/_250923_test_pp_34b/config_sweep_memory.config.v20_0929_0000PST_64node-prolong-0.3.sh


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
#   Large Scale Running 64node prolong (2025/09/29 00:00:00 AM PST - 
# ------------------------------------------------------------------------------------------------

# ------------- $$$ Start here $$$ -----------------
# $$OUTPUT_DIR_PREFIX=/mnt/weka/home/yonghao.zhuang/jd/d2/benchmarks/_250923_test_pp_34b/logs.v62-small-scale-pp-34b-64node-128k-prolong-0.3

# Check the configs
# <<< EXPERIMENT__SKIP=0,EXPERIMENT__COMPLETED=0,EXPERIMENT_REPEAT_TIMES=1,EXPERIMENT_WARMUP_TIMES=0,SHOULD_ADD_DEBUG_CASES=0,NUM_LAYERS=48,SAMPLE_NAME=prolong,CHANGE_LONG_DOC_RATIO=0.3,EXPERIMENT_NVSHMEM_BUFFER_SIZE_GB=1.5,MIN_TOLERANCE_FACTOR=0.15,MAX_SAMPLE_ID=5

    # n  bs  mb   t         mode   cp  pp tp    comment    env_var
    64   4  16 131072    d2       16  4  8 'some_comment'  'EXPERIMENT_SCHEDULE_PRIORITY=100'
    64   4  16 131072    wlbllm   16  4  8 'some_comment'  'EXPERIMENT_SCHEDULE_PRIORITY=100'
    
    64   2  32 131072    d2       16  4  8 'some_comment'  ''
    64   1  64 131072    d2       16  4  8 'some_comment'  ''
    64   8   8 131072    d2       16  4  8 'some_comment'  ''
    64   2  32 131072    d2        8  8  8 'some_comment'  ''
    64   4  16 131072    d2        8  8  8 'some_comment'  ''
    64   1  64 131072    d2        8  8  8 'some_comment'  ''
    64   4  16 131072    d2       32  2  8 'some_comment'  ''
    64  16   4 131072    d2       32  2  8 'some_comment'  ''
    64   8   8 131072    d2       32  2  8 'some_comment'  ''
    64   2  32 131072    d2       32  2  8 'some_comment'  ''
    64   1  64 131072    d2       32  2  8 'some_comment'  ''
    64  64   1 131072    d2       64  1  8 'some_comment'  ''
    64   4  16 131072    wlbllm    8  8  8 'some_comment'  ''
    64   8   8 131072    wlbllm   16  4  8 'some_comment'  ''
    64   8   8 131072    wlbllm    4  4  8 'some_comment'  ''
    64   8   8 131072    wlbllm    8  4  8 'some_comment'  ''
    64   4  16 131072    wlbllm    8  4  8 'some_comment'  ''
    64   2  32 131072    wlbllm   16  4  8 'some_comment'  ''
    64  16   4 131072    wlbllm   16  2  8 'some_comment'  ''
    64  16   4 131072    wlbllm   32  2  8 'some_comment'  ''
    64  16   4 131072    wlbllm    8  2  8 'some_comment'  ''
    64  16   4 131072    wlbllm    4  2  8 'some_comment'  ''
    64  16   4 131072    wlbllm    2  2  8 'some_comment'  ''
    64   8   8 131072    wlbllm   32  2  8 'some_comment'  ''
    64   8   8 131072    wlbllm   16  2  8 'some_comment'  ''
    64   8   8 131072    wlbllm    8  2  8 'some_comment'  ''
    64   8   8 131072    wlbllm    4  2  8 'some_comment'  ''
    64   4  16 131072    wlbllm   32  2  8 'some_comment'  ''
    64   4  16 131072    wlbllm   16  2  8 'some_comment'  ''
    64   4  16 131072    wlbllm    8  2  8 'some_comment'  ''
    64   2  32 131072    wlbllm   32  2  8 'some_comment'  ''
    64   2  32 131072    wlbllm   16  2  8 'some_comment'  ''
    64   1  64 131072    wlbllm   32  2  8 'some_comment'  ''
    64  64   1 131072    wlbllm    1  1  8 'some_comment'  ''
    64  64   1 131072    wlbllm   32  1  8 'some_comment'  ''
    64  64   1 131072    wlbllm    8  1  8 'some_comment'  ''
    64  64   1 131072    wlbllm    2  1  8 'some_comment'  ''
    64  64   1 131072    wlbllm    4  1  8 'some_comment'  ''
    64  64   1 131072    wlbllm   16  1  8 'some_comment'  ''
    64  64   1 131072    wlbllm   64  1  8 'some_comment'  ''

# >>>
# ------------- $$$ Start here $$$ -----------------


    # 256k tbs=32 node=64
    # n  bs  mb   t         mode   cp  pp tp    comment    env_var
    64   2  16 262144    d2       16  4  8 'some_comment'  'EXPERIMENT_SCHEDULE_PRIORITY=100'
    64   4   8 262144    d2       32  2  8 'some_comment'  'EXPERIMENT_SCHEDULE_PRIORITY=80'
    64   4   8 262144    d2       16  4  8 'some_comment'  ''
    64   1  32 262144    d2       16  4  8 'some_comment'  ''
    64   8   4 262144    wlbllm   16  2  8 'some_comment'  'EXPERIMENT_SCHEDULE_PRIORITY=90'
    64   8   4 262144    d2       32  2  8 'some_comment'  ''
    64   2  16 262144    d2       32  2  8 'some_comment'  ''
    64   1  32 262144    d2       32  2  8 'some_comment'  ''
    64   2  16 262144    d2        8  8  8 'some_comment'  ''
    64   1  32 262144    d2        8  8  8 'some_comment'  ''
    64  32   1 262144    d2       64  1  8 'some_comment'  ''
    64   4   8 262144    wlbllm    8  2  8 'some_comment'  'EXPERIMENT_SCHEDULE_PRIORITY=70'
    64   2  16 262144    wlbllm   16  4  8 'some_comment'  'EXPERIMENT_SCHEDULE_PRIORITY=70'
    64   8   4 262144    wlbllm   32  2  8 'some_comment'  ''
    64   8   4 262144    wlbllm    4  2  8 'some_comment'  ''
    64   4   8 262144    wlbllm   32  2  8 'some_comment'  ''
    64   4   8 262144    wlbllm   16  2  8 'some_comment'  ''
    64   4   8 262144    wlbllm    8  4  8 'some_comment'  ''
    64   2  16 262144    wlbllm   32  2  8 'some_comment'  ''
    64   2  16 262144    wlbllm   16  2  8 'some_comment'  ''
    64   1  32 262144    wlbllm   32  2  8 'some_comment'  ''
    64   8   4 262144    wlbllm    8  2  8 'some_comment'  ''
    64  32   1 262144    wlbllm    8  1  8 'some_comment'  ''
    64  32   1 262144    wlbllm    4  1  8 'some_comment'  ''
    64  32   1 262144    wlbllm    2  1  8 'some_comment'  ''
    64  32   1 262144    wlbllm    1  1  8 'some_comment'  ''
    64  32   1 262144    wlbllm   32  1  8 'some_comment'  ''
    64  32   1 262144    wlbllm   16  1  8 'some_comment'  ''
    64   4   8 262144    wlbllm   16  4  8 'some_comment'  ''
    64  32   1 262144    wlbllm   64  1  8 'some_comment'  ''

# >>>

# ------------- $$$ Stop here $$$ -----------------

# Check the configs
# <<< EXPERIMENT__SKIP=0,EXPERIMENT__COMPLETED=0,EXPERIMENT_REPEAT_TIMES=1,EXPERIMENT_WARMUP_TIMES=0,SHOULD_ADD_DEBUG_CASES=0,NUM_LAYERS=48,SAMPLE_NAME=prolong,CHANGE_LONG_DOC_RATIO=0.3,EXPERIMENT_NVSHMEM_BUFFER_SIZE_GB=1.5,MIN_TOLERANCE_FACTOR=0.15,MAX_SAMPLE_ID=30

# 384k tbs=8 node=64
    #n  bs  mb   t         mode   cp  pp tp    comment    env_var
    64   1   8 393216    wlbllm   32  2  8 'some_comment'  'EXPERIMENT_SCHEDULE_PRIORITY=100'
    64   2   4 393216    d2       32  2  8 'some_comment'  'EXPERIMENT_SCHEDULE_PRIORITY=100'

    64   1   8 393216    d2       32  2  8 'some_comment'  'EXPERIMENT_SCHEDULE_PRIORITY=70'
    64   1   8 393216    d2       16  4  8 'some_comment'  'EXPERIMENT_SCHEDULE_PRIORITY=70'
    64   8   1 393216    d2       64  1  8 'some_comment'  'EXPERIMENT_SCHEDULE_PRIORITY=70'
    64   2   4 393216    wlbllm   16  2  8 'some_comment'  'EXPERIMENT_SCHEDULE_PRIORITY=60'
    64   8   1 393216    wlbllm   16  1  8 'some_comment'  'EXPERIMENT_SCHEDULE_PRIORITY=60'
    
    64   2   4 393216    wlbllm   32  2  8 'some_comment'  ''
    64   8   1 393216    wlbllm    4  1  8 'some_comment'  ''
    64   8   1 393216    wlbllm    8  1  8 'some_comment'  ''
    64   8   1 393216    wlbllm   32  1  8 'some_comment'  ''
    64   8   1 393216    wlbllm   64  1  8 'some_comment'  ''


# 256k tbs=16 node=64
# (2.0, 8, 32, 2, 32861.73)	(2.0, 8, 32, 2, 37196.99)	1.131924	262144	34b	64	
    #n  bs  mb   t         mode   cp  pp tp    comment    env_var
    64   2   8 262144    wlbllm   32  2  8 'some_comment'  'EXPERIMENT_SCHEDULE_PRIORITY=80'
    64   2   8 262144    d2       32  2  8 'some_comment'  'EXPERIMENT_SCHEDULE_PRIORITY=80'
    64   1  16 262144    d2        8  8  8 'some_comment'  'EXPERIMENT_SCHEDULE_PRIORITY=70'
    64   1  16 262144    d2       16  4  8 'some_comment'  'EXPERIMENT_SCHEDULE_PRIORITY=70'
    64   2   8 262144    d2       32  2  8 'some_comment'  'EXPERIMENT_SCHEDULE_PRIORITY=70'
    64   1  16 262144    d2       32  2  8 'some_comment'  'EXPERIMENT_SCHEDULE_PRIORITY=70'
    64   2   8 262144    d2       16  4  8 'some_comment'  'EXPERIMENT_SCHEDULE_PRIORITY=70'
    64  16   1 262144    d2       64  1  8 'some_comment'  'EXPERIMENT_SCHEDULE_PRIORITY=70'
    64   4   4 262144    d2       32  2  8 'some_comment'  'EXPERIMENT_SCHEDULE_PRIORITY=70'
    64   2   8 262144    wlbllm   16  4  8 'some_comment'  'EXPERIMENT_SCHEDULE_PRIORITY=60'
    64   4   4 262144    wlbllm   32  2  8 'some_comment'  'EXPERIMENT_SCHEDULE_PRIORITY=60'
    64   4   4 262144    wlbllm    8  2  8 'some_comment'  'EXPERIMENT_SCHEDULE_PRIORITY=60'
    64   4   4 262144    wlbllm   16  2  8 'some_comment'  'EXPERIMENT_SCHEDULE_PRIORITY=60'
    64   2   8 262144    wlbllm   16  2  8 'some_comment'  'EXPERIMENT_SCHEDULE_PRIORITY=60'
    64   1  16 262144    wlbllm   32  2  8 'some_comment'  'EXPERIMENT_SCHEDULE_PRIORITY=60'
    64  16   1 262144    wlbllm   64  1  8 'some_comment'  ''
    64  16   1 262144    wlbllm   16  1  8 'some_comment'  'EXPERIMENT_SCHEDULE_PRIORITY=50'
    64  16   1 262144    wlbllm   32  1  8 'some_comment'  ''
    64  16   1 262144    wlbllm    8  1  8 'some_comment'  ''
    64  16   1 262144    wlbllm    4  1  8 'some_comment'  ''
    64  16   1 262144    wlbllm    2  1  8 'some_comment'  ''


# >>>

# ------------- $$$ Stop here $$$ -----------------



# # MAX_SAMPLE_ID=30

# # ------------- $$$ Start here $$$ -----------------


# # <<< EXPERIMENT__SKIP=0,EXPERIMENT__COMPLETED=0,EXPERIMENT_REPEAT_TIMES=1,EXPERIMENT_WARMUP_TIMES=0,SHOULD_ADD_DEBUG_CASES=0,NUM_LAYERS=48,CHANGE_LONG_DOC_RATIO=0.0,EXPERIMENT_NVSHMEM_BUFFER_SIZE_GB=1.5,MIN_TOLERANCE_FACTOR=0.15,MAX_SAMPLE_ID=5

#     # 128k tbs=16 node=64
#     # n  bs  mb   t         mode   cp  pp tp    comment    env_var
#     64   2   8 131072    d2       32  2  8 'some_comment'  'EXPERIMENT_SCHEDULE_PRIORITY=90'
#     64   2   8 131072    wlbllm   16  2  8 'some_comment'  'EXPERIMENT_SCHEDULE_PRIORITY=90'

#     64   1  16 131072    d2        8  8  8 'some_comment'  ''
#     64   1  16 131072    d2       16  4  8 'some_comment'  ''
#     64   1  16 131072    d2       32  2  8 'some_comment'  ''
#     64   2   8 131072    d2       16  4  8 'some_comment'  ''
#     64  16   1 131072    d2       64  1  8 'some_comment'  ''
#     64   4   4 131072    d2       32  2  8 'some_comment'  ''
#     64  16   1 131072    wlbllm   32  1  8 'some_comment'  ''
#     64  16   1 131072    wlbllm   16  1  8 'some_comment'  ''
#     64  16   1 131072    wlbllm    8  1  8 'some_comment'  ''
#     64  16   1 131072    wlbllm    4  1  8 'some_comment'  ''
#     64  16   1 131072    wlbllm    2  1  8 'some_comment'  ''
#     64   4   4 131072    wlbllm    8  2  8 'some_comment'  ''
#     64   4   4 131072    wlbllm   16  2  8 'some_comment'  ''
#     64   2   8 131072    wlbllm   32  2  8 'some_comment'  ''
#     64   2   8 131072    wlbllm   16  4  8 'some_comment'  ''
#     64   1  16 131072    wlbllm   32  2  8 'some_comment'  ''
#     64   4   4 131072    wlbllm   32  2  8 'some_comment'  ''
#     64  16   1 131072    wlbllm   64  1  8 'some_comment'  ''

#     # 256k tbs=32 node=64
#     # n  bs  mb   t         mode   cp  pp tp    comment    env_var
#     64   1  32 262144    d2        8  8  8 'some_comment'  ''
#     64   4   8 262144    d2       32  2  8 'some_comment'  ''
#     64  32   1 262144    d2       64  1  8 'some_comment'  ''
#     64   4   8 262144    d2       16  4  8 'some_comment'  ''
#     64   8   4 262144    d2       32  2  8 'some_comment'  ''
#     64   2  16 262144    d2       16  4  8 'some_comment'  'EXPERIMENT_SCHEDULE_PRIORITY=80'
#     64   2  16 262144    d2       32  2  8 'some_comment'  ''
#     64   2  16 262144    d2        8  8  8 'some_comment'  'EXPERIMENT_SCHEDULE_PRIORITY=80'
#     64   1  32 262144    d2       32  2  8 'some_comment'  ''
#     64   1  32 262144    d2       16  4  8 'some_comment'  ''
#     64   8   4 262144    wlbllm   16  2  8 'some_comment'  'EXPERIMENT_SCHEDULE_PRIORITY=80'
#     64   4   8 262144    wlbllm    8  2  8 'some_comment'  'EXPERIMENT_SCHEDULE_PRIORITY=80'
#     64  32   1 262144    wlbllm    8  1  8 'some_comment'  ''
#     64  32   1 262144    wlbllm    4  1  8 'some_comment'  ''
#     64  32   1 262144    wlbllm    2  1  8 'some_comment'  ''
#     64  32   1 262144    wlbllm    1  1  8 'some_comment'  ''
#     64   8   4 262144    wlbllm   32  2  8 'some_comment'  ''
#     64  32   1 262144    wlbllm   32  1  8 'some_comment'  ''
#     64  32   1 262144    wlbllm   16  1  8 'some_comment'  ''
#     64   4   8 262144    wlbllm   16  4  8 'some_comment'  ''
#     64   8   4 262144    wlbllm    4  2  8 'some_comment'  ''
#     64   4   8 262144    wlbllm   32  2  8 'some_comment'  ''
#     64   4   8 262144    wlbllm   16  2  8 'some_comment'  ''
#     64   4   8 262144    wlbllm    8  4  8 'some_comment'  ''
#     64   2  16 262144    wlbllm   32  2  8 'some_comment'  ''
#     64   2  16 262144    wlbllm   16  4  8 'some_comment'  ''
#     64   2  16 262144    wlbllm   16  2  8 'some_comment'  ''
#     64   1  32 262144    wlbllm   32  2  8 'some_comment'  ''
#     64   8   4 262144    wlbllm    8  2  8 'some_comment'  ''
#     64  32   1 262144    wlbllm   64  1  8 'some_comment'  ''

#     # 384k tbs=16 node=64
#     # n  bs  mb   t         mode   cp  pp tp    comment    env_var
#     64   1  16 393216    d2        8  8  8 'some_comment'  ''
#     64   1  16 393216    d2       16  4  8 'some_comment'  'EXPERIMENT_SCHEDULE_PRIORITY=70'
#     64   1  16 393216    d2       32  2  8 'some_comment'  ''
#     64   2   8 393216    d2       16  4  8 'some_comment'  'EXPERIMENT_SCHEDULE_PRIORITY=70'
#     64   2   8 393216    d2       32  2  8 'some_comment'  ''
#     64  16   1 393216    d2       64  1  8 'some_comment'  ''
#     64   4   4 393216    d2       32  2  8 'some_comment'  ''
#     64  16   1 393216    wlbllm   32  1  8 'some_comment'  ''
#     64  16   1 393216    wlbllm   16  1  8 'some_comment'  ''
#     64  16   1 393216    wlbllm    8  1  8 'some_comment'  ''
#     64  16   1 393216    wlbllm    4  1  8 'some_comment'  ''
#     64  16   1 393216    wlbllm    2  1  8 'some_comment'  ''
#     64   4   4 393216    wlbllm    8  2  8 'some_comment'  ''
#     64   4   4 393216    wlbllm   16  2  8 'some_comment'  ''
#     64   2   8 393216    wlbllm   32  2  8 'some_comment'  ''
#     64   2   8 393216    wlbllm   16  4  8 'some_comment'  'EXPERIMENT_SCHEDULE_PRIORITY=70'
#     64   2   8 393216    wlbllm   16  2  8 'some_comment'  'EXPERIMENT_SCHEDULE_PRIORITY=70'
#     64   1  16 393216    wlbllm   32  2  8 'some_comment'  ''
#     64   4   4 393216    wlbllm   32  2  8 'some_comment'  ''
#     64  16   1 393216    wlbllm   64  1  8 'some_comment'  ''


#     # 384k tbs=8 node=64
#     # n  bs  mb   t         mode   cp  pp tp    comment    env_var
#     64   1   8 393216    d2       16  4  8 'some_comment'  ''
#     64   1   8 393216    d2       32  2  8 'some_comment'  ''
#     64   2   4 393216    d2       32  2  8 'some_comment'  'EXPERIMENT_SCHEDULE_PRIORITY=60'
#     64   8   1 393216    d2       64  1  8 'some_comment'  ''
#     64   1   8 393216    wlbllm   32  2  8 'some_comment'  ''
#     64   2   4 393216    wlbllm   16  2  8 'some_comment'  'EXPERIMENT_SCHEDULE_PRIORITY=60'
#     64   2   4 393216    wlbllm   32  2  8 'some_comment'  ''
#     64   8   1 393216    wlbllm    4  1  8 'some_comment'  ''
#     64   8   1 393216    wlbllm    8  1  8 'some_comment'  ''
#     64   8   1 393216    wlbllm   16  1  8 'some_comment'  ''
#     64   8   1 393216    wlbllm   32  1  8 'some_comment'  ''
#     64   8   1 393216    wlbllm   64  1  8 'some_comment'  ''
    


# # >>>



# # <<< EXPERIMENT__SKIP=0,EXPERIMENT__COMPLETED=0,EXPERIMENT_REPEAT_TIMES=1,EXPERIMENT_WARMUP_TIMES=0,SHOULD_ADD_DEBUG_CASES=0,NUM_LAYERS=48,CHANGE_LONG_DOC_RATIO=0.0,EXPERIMENT_NVSHMEM_BUFFER_SIZE_GB=1.5,MIN_TOLERANCE_FACTOR=0.15,MAX_SAMPLE_ID=5

#     # 128k tbs=64 node=64
#     # n  bs  mb   t         mode   cp  pp tp    comment    env_var
#     64   4  16 131072    d2       32  2  8 'some_comment'  ''
#     64   1  64 131072    d2        8  8  8 'some_comment'  ''
#     64  64   1 131072    d2       64  1  8 'some_comment'  ''
#     64   4  16 131072    d2       16  4  8 'some_comment'  ''
#     64  16   4 131072    d2       32  2  8 'some_comment'  ''
#     64   8   8 131072    d2       32  2  8 'some_comment'  'EXPERIMENT_SCHEDULE_PRIORITY=50'
#     64   2  32 131072    d2       32  2  8 'some_comment'  ''
#     64   4  16 131072    d2        8  8  8 'some_comment'  ''
#     64   2  32 131072    d2       16  4  8 'some_comment'  'EXPERIMENT_SCHEDULE_PRIORITY=50'
#     64   1  64 131072    d2       16  4  8 'some_comment'  ''
#     64   1  64 131072    d2       32  2  8 'some_comment'  ''
#     64   2  32 131072    d2        8  8  8 'some_comment'  'EXPERIMENT_SCHEDULE_PRIORITY=50'
#     64   8   8 131072    d2       16  4  8 'some_comment'  ''
#     64  16   4 131072    wlbllm   32  2  8 'some_comment'  ''
#     64  64   1 131072    wlbllm    1  1  8 'some_comment'  ''
#     64  64   1 131072    wlbllm   32  1  8 'some_comment'  ''
#     64  64   1 131072    wlbllm    8  1  8 'some_comment'  ''
#     64  64   1 131072    wlbllm    2  1  8 'some_comment'  ''
#     64  16   4 131072    wlbllm    8  2  8 'some_comment'  ''
#     64  16   4 131072    wlbllm    4  2  8 'some_comment'  ''
#     64  64   1 131072    wlbllm    4  1  8 'some_comment'  ''
#     64  16   4 131072    wlbllm    2  2  8 'some_comment'  ''
#     64   8   8 131072    wlbllm   32  2  8 'some_comment'  ''
#     64  64   1 131072    wlbllm   16  1  8 'some_comment'  ''
#     64  16   4 131072    wlbllm   16  2  8 'some_comment'  'EXPERIMENT_SCHEDULE_PRIORITY=50'
#     64   8   8 131072    wlbllm    4  4  8 'some_comment'  ''
#     64   8   8 131072    wlbllm   16  2  8 'some_comment'  'EXPERIMENT_SCHEDULE_PRIORITY=50'
#     64   8   8 131072    wlbllm    8  4  8 'some_comment'  ''
#     64   8   8 131072    wlbllm    8  2  8 'some_comment'  ''
#     64   8   8 131072    wlbllm    4  2  8 'some_comment'  ''
#     64   4  16 131072    wlbllm   32  2  8 'some_comment'  ''
#     64   4  16 131072    wlbllm   16  4  8 'some_comment'  ''
#     64   4  16 131072    wlbllm   16  2  8 'some_comment'  ''
#     64   4  16 131072    wlbllm    8  8  8 'some_comment'  ''
#     64   4  16 131072    wlbllm    8  4  8 'some_comment'  ''
#     64   4  16 131072    wlbllm    8  2  8 'some_comment'  ''
#     64   2  32 131072    wlbllm   32  2  8 'some_comment'  ''
#     64   2  32 131072    wlbllm   16  4  8 'some_comment'  ''
#     64   2  32 131072    wlbllm   16  2  8 'some_comment'  ''
#     64   1  64 131072    wlbllm   32  2  8 'some_comment'  ''
#     64   8   8 131072    wlbllm   16  4  8 'some_comment'  ''
#     64  64   1 131072    wlbllm   64  1  8 'some_comment'  ''

# # >>>

# # ------------- Stop here -----------------