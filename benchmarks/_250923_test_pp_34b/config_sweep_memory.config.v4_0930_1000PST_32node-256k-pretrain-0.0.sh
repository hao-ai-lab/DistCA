# ┏━┓┏━┓┏━╸╺┳╸┏━┓┏━┓╻┏┓╻   ┏━┓╻ ╻┏┓    ┏━┓┏━┓┏┓╻┏━┓╺┳┓┏━╸
# ┣━┛┣┳┛┣╸  ┃ ┣┳┛┣━┫┃┃┗┫   ╺━┫┗━┫┣┻┓   ╺━┫┏━┛┃┗┫┃ ┃ ┃┃┣╸ 
# ╹  ╹┗╸┗━╸ ╹ ╹┗╸╹ ╹╹╹ ╹   ┗━┛  ╹┗━┛   ┗━┛┗━╸╹ ╹┗━┛╺┻┛┗━╸
# ╺┓ ┏━┓┏━┓╻┏    ┏━┓┏━╸┏━┓╻┏    ┏━┓┏━┓╻ ╻╻┏              
#  ┃ ┏━┛┣━┫┣┻┓   ┏━┛┗━┓┣━┓┣┻┓   ╺━┫┣━┫┗━┫┣┻┓             
# ╺┻╸┗━╸┗━┛╹ ╹   ┗━╸┗━┛┗━┛╹ ╹   ┗━┛┗━┛  ╹╹ ╹             
# ┏━┓┏━┓┏━┓┏━┓   ╺┓ ┏━┓ ┏━┓┏━┓   ┏━┓┏━┓╺┳╸               
# ┃┃┃┗━┫┏━┛┣━┫    ┃ ┃┃┃╹┃┃┃┃┃┃   ┣━┛┗━┓ ┃                
# ┗━┛┗━┛┗━╸┗━┛   ╺┻╸┗━┛╹┗━┛┗━┛   ╹  ┗━┛ ╹                


# Global variable overrides (use # $$ syntax)
# $$OUTPUT_DIR_PREFIX=/mnt/weka/home/yonghao.zhuang/jd/d2/benchmarks/_250923_test_pp_34b/logs.v80-large-scale-pp-34b-32node-256k-pretrain
# $$FOLDER_SEPARATOR=1
# $$EXPERIMENT_DISTS=("wlbllm 0.0")

# bash /mnt/weka/home/yonghao.zhuang/jd/d2/benchmarks/_250923_test_pp_34b/salloc_srun.pp_34b.v2.sh --config /mnt/weka/home/yonghao.zhuang/jd/d2/benchmarks/_250923_test_pp_34b/config_sweep_memory.config.v4_0930_1000PST_32node-256k-pretrain-0.0.sh




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
#   Sweeping memory on 32node pretrain (2025/09/28 11:00:00 AM PST - 
# ------------------------------------------------------------------------------------------------


# <<< EXPERIMENT__SKIP=0,EXPERIMENT__COMPLETED=0,EXPERIMENT_REPEAT_TIMES=1,EXPERIMENT_WARMUP_TIMES=0,SHOULD_ADD_DEBUG_CASES=0,NUM_LAYERS=48,CHANGE_LONG_DOC_RATIO=0.0,EXPERIMENT_NVSHMEM_BUFFER_SIZE_GB=1.5,MIN_TOLERANCE_FACTOR=0.15,MAX_SAMPLE_ID=5



# 128k tbs=32 node=32
    #n  bs  mb   t         mode   cp  pp tp    comment    env_var
    32  32   1 131072    d2       32  1  8 'some_comment'  ''
    32   1  32 131072    d2       16  2  8 'some_comment'  ''
    32   2  16 131072    d2       16  2  8 'some_comment'  ''
    32   4   8 131072    d2       16  2  8 'some_comment'  'EXPERIMENT_SCHEDULE_PRIORITY=90'
    32   8   4 131072    d2       16  2  8 'some_comment'  ''
    32   2  16 131072    d2        8  4  8 'some_comment'  'EXPERIMENT_SCHEDULE_PRIORITY=90'
    32   1  32 131072    d2        8  4  8 'some_comment'  ''
    32   4   8 131072    d2        8  4  8 'some_comment'  ''
    32   1  32 131072    d2        4  8  8 'some_comment'  ''
    32   2  16 131072    d2        4  8  8 'some_comment'  'EXPERIMENT_SCHEDULE_PRIORITY=90'
    32  32   1 131072    wlbllm    1  1  8 'some_comment'  ''
    32  32   1 131072    wlbllm    2  1  8 'some_comment'  ''
    32  32   1 131072    wlbllm    4  1  8 'some_comment'  ''
    32  32   1 131072    wlbllm    8  1  8 'some_comment'  ''
    32  32   1 131072    wlbllm   16  1  8 'some_comment'  ''
    32  32   1 131072    wlbllm   32  1  8 'some_comment'  ''
    32   1  32 131072    wlbllm   16  2  8 'some_comment'  ''
    32   2  16 131072    wlbllm    8  2  8 'some_comment'  'EXPERIMENT_SCHEDULE_PRIORITY=90'
    32   2  16 131072    wlbllm   16  2  8 'some_comment'  ''
    32   4   8 131072    wlbllm    4  2  8 'some_comment'  'EXPERIMENT_SCHEDULE_PRIORITY=90'
    32   4   8 131072    wlbllm    8  2  8 'some_comment'  ''
    32   4   8 131072    wlbllm   16  2  8 'some_comment'  ''
    32   8   4 131072    wlbllm    2  2  8 'some_comment'  ''
    32   8   4 131072    wlbllm    4  2  8 'some_comment'  ''
    32   8   4 131072    wlbllm    8  2  8 'some_comment'  ''
    32   8   4 131072    wlbllm   16  2  8 'some_comment'  ''
    32   2  16 131072    wlbllm    8  4  8 'some_comment'  'EXPERIMENT_SCHEDULE_PRIORITY=90'
    32   4   8 131072    wlbllm    4  4  8 'some_comment'  ''
    32   4   8 131072    wlbllm    8  4  8 'some_comment'  ''

# 256k tbs=16 node=32
    # n  bs  mb   t         mode   cp  pp tp    comment    env_var
    32   2   8 262144    d2       16  2  8 'some_comment'  'EXPERIMENT_SCHEDULE_PRIORITY=90'
    32   2   8 262144    wlbllm    8  2  8 'some_comment'  'EXPERIMENT_SCHEDULE_PRIORITY=90'

    32   1  16 262144    d2        4  8  8 'some_comment'  ''
    32   1  16 262144    d2        8  4  8 'some_comment'  'EXPERIMENT_SCHEDULE_PRIORITY=90'
    32   1  16 262144    d2       16  2  8 'some_comment'  ''
    32   2   8 262144    d2        8  4  8 'some_comment'  ''
    32  16   1 262144    d2       32  1  8 'some_comment'  ''
    32   4   4 262144    d2       16  2  8 'some_comment'  ''
    32  16   1 262144    wlbllm   16  1  8 'some_comment'  ''
    32  16   1 262144    wlbllm    8  1  8 'some_comment'  ''
    32  16   1 262144    wlbllm    4  1  8 'some_comment'  ''
    32  16   1 262144    wlbllm    2  1  8 'some_comment'  ''
    32  16   1 262144    wlbllm    1  1  8 'some_comment'  ''
    32   4   4 262144    wlbllm    4  2  8 'some_comment'  ''
    32   4   4 262144    wlbllm    8  2  8 'some_comment'  ''
    32   2   8 262144    wlbllm   16  2  8 'some_comment'  'EXPERIMENT_SCHEDULE_PRIORITY=90'
    32   2   8 262144    wlbllm    8  4  8 'some_comment'  ''
    32   1  16 262144    wlbllm   16  2  8 'some_comment'  ''
    32   4   4 262144    wlbllm   16  2  8 'some_comment'  ''
    32  16   1 262144    wlbllm   32  1  8 'some_comment'  ''


# >>>