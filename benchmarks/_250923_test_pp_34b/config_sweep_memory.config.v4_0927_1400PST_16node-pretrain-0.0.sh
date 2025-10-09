



# Global variable overrides (use # $$ syntax)
# $$OUTPUT_DIR_PREFIX=/mnt/weka/home/yonghao.zhuang/jd/d2/benchmarks/_250923_test_pp_34b/logs.v7-small-scale-pp-34b-16node-128k-256k-384k
# $$FOLDER_SEPARATOR=1
# $$EXPERIMENT_DISTS=("wlbllm 0.0")

# bash /mnt/weka/home/yonghao.zhuang/jd/d2/benchmarks/_250923_test_pp_34b/salloc_srun.pp_34b.v2.sh --config /mnt/weka/home/yonghao.zhuang/jd/d2/benchmarks/_250923_test_pp_34b/config_sweep_memory.config.v4_0927_1400PST_16node-pretrain-0.0.sh



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
#      384k pretrain small OOM sweep on wlbllm and d2 (2025/09/27 11:00:00 AM PST - 22:00:00 PST)
# ------------------------------------------------------------------------------------------------
# <<< EXPERIMENT__SKIP=0,EXPERIMENT__COMPLETED=0,MAX_SAMPLE_ID=5,EXPERIMENT_REPEAT_TIMES=1,EXPERIMENT_WARMUP_TIMES=0,SHOULD_ADD_DEBUG_CASES=0,NUM_LAYERS=48,CHANGE_LONG_DOC_RATIO=0.0

    # 128k tbs=16 node=16
    # n  bs  mb   t         mode   cp  pp tp    comment    env_var
    16   1  16 131072    d2        2  8  8 'some_comment'  'EXPERIMENT__STATUS=FAIL,EXPERIMENT__FAIL_REASON=OOM,RESULT_DIR=27_150608.d2-n16-t131072-b1-mb16-cp2tp8pp8-codellama_CodeLlama-34b-hf-L48-wlbllm_0.0'
    16   1  16 131072    d2        4  4  8 'some_comment'  'EXPERIMENT__STATUS=PASS,RESULT_DIR=27_150936.d2-n16-t131072-b1-mb16-cp4tp8pp4-codellama_CodeLlama-34b-hf-L48-wlbllm_0.0'
    16   1  16 131072    d2        8  2  8 'some_comment'  'EXPERIMENT__STATUS=PASS,RESULT_DIR=27_152256.d2-n16-t131072-b1-mb16-cp8tp8pp2-codellama_CodeLlama-34b-hf-L48-wlbllm_0.0'
    16   1  16 131072    wlbllm    8  2  8 'some_comment'  'EXPERIMENT__STATUS=PASS,RESULT_DIR=27_153534.wlbllm-n16-t131072-b1-mb16-cp8tp8pp2-codellama_CodeLlama-34b-hf-L48-wlbllm_0.0'
    16   2   8 131072    wlbllm    4  2  8 'some_comment'  'EXPERIMENT_SCHEDULE_PRIORITY=100,EXPERIMENT__STATUS=PASS,RESULT_DIR=27_132626.wlbllm-n16-t131072-b2-mb8-cp4tp8pp2-codellama_CodeLlama-34b-hf-L48-wlbllm_0.0'
    16   2   8 131072    d2        4  4  8 'some_comment'  'EXPERIMENT__STATUS=FAIL,EXPERIMENT__FAIL_REASON=OOM,RESULT_DIR=27_154118.d2-n16-t131072-b2-mb8-cp4tp8pp4-codellama_CodeLlama-34b-hf-L48-wlbllm_0.0'
    16   2   8 131072    wlbllm    4  4  8 'some_comment'  'EXPERIMENT__STATUS=PASS,RESULT_DIR=27_154348.wlbllm-n16-t131072-b2-mb8-cp4tp8pp4-codellama_CodeLlama-34b-hf-L48-wlbllm_0.0'
    16   2   8 131072    d2        8  2  8 'some_comment'  'EXPERIMENT_SCHEDULE_PRIORITY=100,EXPERIMENT__STATUS=PASS,RESULT_DIR=27_133203.d2-n16-t131072-b2-mb8-cp8tp8pp2-codellama_CodeLlama-34b-hf-L48-wlbllm_0.0'
    16   2   8 131072    wlbllm    8  2  8 'some_comment'  'EXPERIMENT__STATUS=PASS,RESULT_DIR=27_155018.wlbllm-n16-t131072-b2-mb8-cp8tp8pp2-codellama_CodeLlama-34b-hf-L48-wlbllm_0.0'
    16   4   4 131072    wlbllm    2  2  8 'some_comment'  'EXPERIMENT__STATUS=PASS,RESULT_DIR=27_155558.wlbllm-n16-t131072-b4-mb4-cp2tp8pp2-codellama_CodeLlama-34b-hf-L48-wlbllm_0.0'
    16   4   4 131072    wlbllm    4  2  8 'some_comment'  'EXPERIMENT__STATUS=PASS,RESULT_DIR=27_160145.wlbllm-n16-t131072-b4-mb4-cp4tp8pp2-codellama_CodeLlama-34b-hf-L48-wlbllm_0.0'
    16   4   4 131072    d2        8  2  8 'some_comment'  'EXPERIMENT__STATUS=FAIL,EXPERIMENT__FAIL_REASON=OOM,RESULT_DIR=27_160732.d2-n16-t131072-b4-mb4-cp8tp8pp2-codellama_CodeLlama-34b-hf-L48-wlbllm_0.0'
    16   4   4 131072    wlbllm    8  2  8 'some_comment'  'EXPERIMENT__STATUS=FAIL,EXPERIMENT__FAIL_REASON=OOM,RESULT_DIR=27_160936.wlbllm-n16-t131072-b4-mb4-cp8tp8pp2-codellama_CodeLlama-34b-hf-L48-wlbllm_0.0'
    16  16   1 131072    wlbllm    1  1  8 'some_comment'  'EXPERIMENT__STATUS=FAIL,EXPERIMENT__FAIL_REASON=OOM,RESULT_DIR=27_161047.wlbllm-n16-t131072-b16-mb1-cp1tp8pp1-codellama_CodeLlama-34b-hf-L48-wlbllm_0.0'
    16  16   1 131072    wlbllm    2  1  8 'some_comment'  'EXPERIMENT__STATUS=FAIL,EXPERIMENT__FAIL_REASON=OOM,RESULT_DIR=27_161152.wlbllm-n16-t131072-b16-mb1-cp2tp8pp1-codellama_CodeLlama-34b-hf-L48-wlbllm_0.0'
    16  16   1 131072    wlbllm    4  1  8 'some_comment'  'EXPERIMENT__STATUS=FAIL,EXPERIMENT__FAIL_REASON=OOM,RESULT_DIR=27_161300.wlbllm-n16-t131072-b16-mb1-cp4tp8pp1-codellama_CodeLlama-34b-hf-L48-wlbllm_0.0'
    16  16   1 131072    wlbllm    8  1  8 'some_comment'  'EXPERIMENT__STATUS=FAIL,EXPERIMENT__FAIL_REASON=OOM,RESULT_DIR=27_161405.wlbllm-n16-t131072-b16-mb1-cp8tp8pp1-codellama_CodeLlama-34b-hf-L48-wlbllm_0.0'
    16  16   1 131072    d2       16  1  8 'some_comment'  'EXPERIMENT__STATUS=FAIL,EXPERIMENT__FAIL_REASON=OOM,RESULT_DIR=27_161510.d2-n16-t131072-b16-mb1-cp16tp8pp1-codellama_CodeLlama-34b-hf-L48-wlbllm_0.0'
    16  16   1 131072    wlbllm   16  1  8 'some_comment'  'EXPERIMENT__STATUS=FAIL,EXPERIMENT__FAIL_REASON=OOM,RESULT_DIR=27_161635.wlbllm-n16-t131072-b16-mb1-cp16tp8pp1-codellama_CodeLlama-34b-hf-L48-wlbllm_0.0'


    # 128k tbs=32 node=16
    #n  bs  mb   t         mode   cp  pp tp    comment    env_var
    16   1  32 131072    d2        2  8  8 'some_comment'  'EXPERIMENT__STATUS=FAIL,EXPERIMENT__FAIL_REASON=OOM,RESULT_DIR=27_161741.d2-n16-t131072-b1-mb32-cp2tp8pp8-codellama_CodeLlama-34b-hf-L48-wlbllm_0.0'
    16   1  32 131072    d2        4  4  8 'some_comment'  'EXPERIMENT__STATUS=PASS,RESULT_DIR=27_162221.d2-n16-t131072-b1-mb32-cp4tp8pp4-codellama_CodeLlama-34b-hf-L48-wlbllm_0.0'
    16   1  32 131072    d2        8  2  8 'some_comment'  'EXPERIMENT__STATUS=PASS,RESULT_DIR=27_164556.d2-n16-t131072-b1-mb32-cp8tp8pp2-codellama_CodeLlama-34b-hf-L48-wlbllm_0.0'
    16   1  32 131072    wlbllm    8  2  8 'some_comment'  'EXPERIMENT__STATUS=PASS,RESULT_DIR=27_170924.wlbllm-n16-t131072-b1-mb32-cp8tp8pp2-codellama_CodeLlama-34b-hf-L48-wlbllm_0.0'
    16   2  16 131072    d2        2  8  8 'some_comment'  'EXPERIMENT__STATUS=FAIL,EXPERIMENT__FAIL_REASON=OOM,RESULT_DIR=27_172131.d2-n16-t131072-b2-mb16-cp2tp8pp8-codellama_CodeLlama-34b-hf-L48-wlbllm_0.0'
    16   2  16 131072    wlbllm    4  2  8 'some_comment'  'EXPERIMENT__STATUS=PASS,RESULT_DIR=27_203816.wlbllm-n16-t131072-b2-mb16-cp4tp8pp2-codellama_CodeLlama-34b-hf-L48-wlbllm_0.0'
    16   2  16 131072    d2        4  4  8 'some_comment'  'EXPERIMENT_SCHEDULE_PRIORITY=90,EXPERIMENT__STATUS=FAIL,EXPERIMENT__FAIL_REASON=OOM,RESULT_DIR=27_134127.d2-n16-t131072-b2-mb16-cp4tp8pp4-codellama_CodeLlama-34b-hf-L48-wlbllm_0.0'
    16   2  16 131072    wlbllm    4  4  8 'some_comment'  'EXPERIMENT_SCHEDULE_PRIORITY=80,EXPERIMENT__STATUS=PASS,RESULT_DIR=27_134436.wlbllm-n16-t131072-b2-mb16-cp4tp8pp4-codellama_CodeLlama-34b-hf-L48-wlbllm_0.0'
    16   2  16 131072    d2        8  2  8 'some_comment'  'EXPERIMENT_SCHEDULE_PRIORITY=80,EXPERIMENT__STATUS=PASS,RESULT_DIR=27_135459.d2-n16-t131072-b2-mb16-cp8tp8pp2-codellama_CodeLlama-34b-hf-L48-wlbllm_0.0'
    16   2  16 131072    wlbllm    8  2  8 'some_comment'  'EXPERIMENT__STATUS=PASS,RESULT_DIR=27_205019.wlbllm-n16-t131072-b2-mb16-cp8tp8pp2-codellama_CodeLlama-34b-hf-L48-wlbllm_0.0'
    16   4   8 131072    wlbllm    2  2  8 'some_comment'  'EXPERIMENT__STATUS=PASS,RESULT_DIR=27_210249.wlbllm-n16-t131072-b4-mb8-cp2tp8pp2-codellama_CodeLlama-34b-hf-L48-wlbllm_0.0'
    16   4   8 131072    wlbllm    2  4  8 'some_comment'  'EXPERIMENT__STATUS=FAIL,EXPERIMENT__FAIL_REASON=OOM,RESULT_DIR=27_211330.wlbllm-n16-t131072-b4-mb8-cp2tp8pp4-codellama_CodeLlama-34b-hf-L48-wlbllm_0.0'
    16   4   8 131072    wlbllm    4  2  8 'some_comment'  'EXPERIMENT__STATUS=PASS,RESULT_DIR=27_211444.wlbllm-n16-t131072-b4-mb8-cp4tp8pp2-codellama_CodeLlama-34b-hf-L48-wlbllm_0.0'
    16   4   8 131072    d2        4  4  8 'some_comment'  'EXPERIMENT__STATUS=FAIL,EXPERIMENT__FAIL_REASON=OOM,RESULT_DIR=27_212549.d2-n16-t131072-b4-mb8-cp4tp8pp4-codellama_CodeLlama-34b-hf-L48-wlbllm_0.0'
    16   4   8 131072    wlbllm    4  4  8 'some_comment'  'EXPERIMENT__STATUS=FAIL,EXPERIMENT__FAIL_REASON=OOM,RESULT_DIR=27_212843.wlbllm-n16-t131072-b4-mb8-cp4tp8pp4-codellama_CodeLlama-34b-hf-L48-wlbllm_0.0'
    16   4   8 131072    d2        8  2  8 'some_comment'  'EXPERIMENT__STATUS=FAIL,EXPERIMENT__FAIL_REASON=OOM,RESULT_DIR=27_213000.d2-n16-t131072-b4-mb8-cp8tp8pp2-codellama_CodeLlama-34b-hf-L48-wlbllm_0.0'
    16   4   8 131072    wlbllm    8  2  8 'some_comment'  'EXPERIMENT__STATUS=FAIL,EXPERIMENT__FAIL_REASON=OOM,RESULT_DIR=27_213325.wlbllm-n16-t131072-b4-mb8-cp8tp8pp2-codellama_CodeLlama-34b-hf-L48-wlbllm_0.0'
    16   8   4 131072    wlbllm    1  2  8 'some_comment'  'EXPERIMENT__STATUS=FAIL,EXPERIMENT__FAIL_REASON=OOM,RESULT_DIR=27_213455.wlbllm-n16-t131072-b8-mb4-cp1tp8pp2-codellama_CodeLlama-34b-hf-L48-wlbllm_0.0'
    16   8   4 131072    wlbllm    2  2  8 'some_comment'  'EXPERIMENT__STATUS=FAIL,EXPERIMENT__FAIL_REASON=OOM,RESULT_DIR=27_213615.wlbllm-n16-t131072-b8-mb4-cp2tp8pp2-codellama_CodeLlama-34b-hf-L48-wlbllm_0.0'
    16   8   4 131072    wlbllm    4  2  8 'some_comment'  'EXPERIMENT__STATUS=FAIL,EXPERIMENT__FAIL_REASON=OOM,RESULT_DIR=27_213733.wlbllm-n16-t131072-b8-mb4-cp4tp8pp2-codellama_CodeLlama-34b-hf-L48-wlbllm_0.0'
    16   8   4 131072    d2        8  2  8 'some_comment'  'EXPERIMENT__STATUS=FAIL,EXPERIMENT__FAIL_REASON=OOM,RESULT_DIR=27_213852.d2-n16-t131072-b8-mb4-cp8tp8pp2-codellama_CodeLlama-34b-hf-L48-wlbllm_0.0'
    16   8   4 131072    wlbllm    8  2  8 'some_comment'  'EXPERIMENT__STATUS=FAIL,EXPERIMENT__FAIL_REASON=OOM,RESULT_DIR=27_214102.wlbllm-n16-t131072-b8-mb4-cp8tp8pp2-codellama_CodeLlama-34b-hf-L48-wlbllm_0.0'
    16  32   1 131072    wlbllm    1  1  8 'some_comment'  'EXPERIMENT__STATUS=FAIL,EXPERIMENT__FAIL_REASON=OOM,RESULT_DIR=27_214215.wlbllm-n16-t131072-b32-mb1-cp1tp8pp1-codellama_CodeLlama-34b-hf-L48-wlbllm_0.0'
    16  32   1 131072    wlbllm    2  1  8 'some_comment'  'EXPERIMENT__STATUS=FAIL,EXPERIMENT__FAIL_REASON=OOM,RESULT_DIR=27_214324.wlbllm-n16-t131072-b32-mb1-cp2tp8pp1-codellama_CodeLlama-34b-hf-L48-wlbllm_0.0'
    16  32   1 131072    wlbllm    4  1  8 'some_comment'  'EXPERIMENT__STATUS=FAIL,EXPERIMENT__FAIL_REASON=OOM,RESULT_DIR=27_214433.wlbllm-n16-t131072-b32-mb1-cp4tp8pp1-codellama_CodeLlama-34b-hf-L48-wlbllm_0.0'
    16  32   1 131072    wlbllm    8  1  8 'some_comment'  'EXPERIMENT__STATUS=FAIL,EXPERIMENT__FAIL_REASON=OOM,RESULT_DIR=27_214542.wlbllm-n16-t131072-b32-mb1-cp8tp8pp1-codellama_CodeLlama-34b-hf-L48-wlbllm_0.0'
    16  32   1 131072    d2       16  1  8 'some_comment'  'EXPERIMENT__STATUS=FAIL,RESULT_DIR=27_214647.d2-n16-t131072-b32-mb1-cp16tp8pp1-codellama_CodeLlama-34b-hf-L48-wlbllm_0.0'
    16  32   1 131072    wlbllm   16  1  8 'some_comment'  'EXPERIMENT__STATUS=FAIL,EXPERIMENT__FAIL_REASON=OOM,RESULT_DIR=27_214749.wlbllm-n16-t131072-b32-mb1-cp16tp8pp1-codellama_CodeLlama-34b-hf-L48-wlbllm_0.0'

    # 256k tbs=8 node=16
    # n  bs  mb   t         mode   cp  pp tp    comment    env_var
    16   1   8 262144    d2        4  4  8 'some_comment'  'EXPERIMENT__STATUS=FAIL,EXPERIMENT__FAIL_REASON=OOM,RESULT_DIR=27_214854.d2-n16-t262144-b1-mb8-cp4tp8pp4-codellama_CodeLlama-34b-hf-L48-wlbllm_0.0'
    16   1   8 262144    d2        8  2  8 'some_comment'  'EXPERIMENT_SCHEDULE_PRIORITY=70,EXPERIMENT__STATUS=PASS,RESULT_DIR=27_141125.d2-n16-t262144-b1-mb8-cp8tp8pp2-codellama_CodeLlama-34b-hf-L48-wlbllm_0.0'
    16   2   4 262144    d2        8  2  8 'some_comment'  'EXPERIMENT__STATUS=FAIL,RESULT_DIR=27_215301.d2-n16-t262144-b2-mb4-cp8tp8pp2-codellama_CodeLlama-34b-hf-L48-wlbllm_0.0'
    16   8   1 262144    d2       16  1  8 'some_comment'  ''
    16   1   8 262144    wlbllm    8  2  8 'some_comment'  'EXPERIMENT_SCHEDULE_PRIORITY=70,EXPERIMENT__STATUS=PASS,RESULT_DIR=27_142235.wlbllm-n16-t262144-b1-mb8-cp8tp8pp2-codellama_CodeLlama-34b-hf-L48-wlbllm_0.0'
    16   2   4 262144    wlbllm    4  2  8 'some_comment'  ''
    16   2   4 262144    wlbllm    8  2  8 'some_comment'  ''
    16   8   1 262144    wlbllm    1  1  8 'some_comment'  ''
    16   8   1 262144    wlbllm    2  1  8 'some_comment'  ''
    16   8   1 262144    wlbllm    4  1  8 'some_comment'  ''
    16   8   1 262144    wlbllm    8  1  8 'some_comment'  ''
    16   8   1 262144    wlbllm   16  1  8 'some_comment'  ''

    # 256k tbs=4 node=16
    # n  bs  mb   t         mode   cp  pp tp    comment    env_var
    16   1   4 262144    d2        8  2  8 'some_comment'  'EXPERIMENT_SCHEDULE_PRIORITY=60,EXPERIMENT__STATUS=PASS,RESULT_DIR=27_143001.d2-n16-t262144-b1-mb4-cp8tp8pp2-codellama_CodeLlama-34b-hf-L48-wlbllm_0.0'
    16   4   1 262144    d2       16  1  8 'some_comment'  'EXPERIMENT_SCHEDULE_PRIORITY=60,EXPERIMENT__STATUS=PASS,RESULT_DIR=27_143629.d2-n16-t262144-b4-mb1-cp16tp8pp1-codellama_CodeLlama-34b-hf-L48-wlbllm_0.0'
    16   1   4 262144    wlbllm    8  2  8 'some_comment'  'EXPERIMENT_SCHEDULE_PRIORITY=60,EXPERIMENT__STATUS=PASS,RESULT_DIR=27_144241.wlbllm-n16-t262144-b1-mb4-cp8tp8pp2-codellama_CodeLlama-34b-hf-L48-wlbllm_0.0'
    16   4   1 262144    wlbllm    2  1  8 'some_comment'  'EXPERIMENT_SCHEDULE_PRIORITY=60,EXPERIMENT__STATUS=PASS,RESULT_DIR=27_144724.wlbllm-n16-t262144-b4-mb1-cp2tp8pp1-codellama_CodeLlama-34b-hf-L48-wlbllm_0.0'
    16   4   1 262144    wlbllm    4  1  8 'some_comment'  ''
    16   4   1 262144    wlbllm    8  1  8 'some_comment'  ''
    16   4   1 262144    wlbllm   16  1  8 'some_comment'  ''

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



