



# ----------------------------------------------------------------------------
#      Tune Tolerance Factor (2025/09/26 12:50:00 PM PST)
#  MAX_SAMPLE_ID = 1
# ----------------------------------------------------------------------------
# <<< EXPERIMENT__SKIP=0,EXPERIMENT__COMPLETED=0,MAX_SAMPLE_ID=5,EXPERIMENT_REPEAT_TIMES=2
    64   4   4 393216    d2       64  1  8 'some_comment'  'COMMENT="Test without pingpong rebalance",MIN_TOLERANCE_FACTOR=0.15,OUTPUT_DIR_SUFFIX_ADDON=-tol0.15'
    
# >>>
# <<< EXPERIMENT__SKIP=1,EXPERIMENT__COMPLETED=1,MAX_SAMPLE_ID=1
    # 64   2   8 393216    d2       64  1  8 'some_comment'  'EXPERIMENT_SCHEDULE_PRIORITY=50,MIN_TOLERANCE_FACTOR=0.15,OUTPUT_DIR_SUFFIX_ADDON=-tol0.15,COMMENT="Slower than having PP"'
    # 64   2   8 393216    d2       32  2  8 'some_comment'  'EXPERIMENT_SCHEDULE_PRIORITY=50,MIN_TOLERANCE_FACTOR=0.15,OUTPUT_DIR_SUFFIX_ADDON=-tol0.15'
    # 64   2   8 393216    d2       32  2  8 'some_comment'  'EXPERIMENT_SCHEDULE_PRIORITY=50,MIN_TOLERANCE_FACTOR=0.125,OUTPUT_DIR_SUFFIX_ADDON=-tol0.125'
    # 64   2   8 393216  wlbllm     16  1  8 'some_comment'  'EXPERIMENT_SCHEDULE_PRIORITY=50'
    # 64   2   8 393216    d2       32  2  8 'some_comment'  'EXPERIMENT_SCHEDULE_PRIORITY=50,MIN_TOLERANCE_FACTOR=0.175,OUTPUT_DIR_SUFFIX_ADDON=-tol0.175'
    # 64   2   8 393216    d2       32  2  8 'some_comment'  'EXPERIMENT_SCHEDULE_PRIORITY=50,MIN_TOLERANCE_FACTOR=0.10,OUTPUT_DIR_SUFFIX_ADDON=-tol0.10' 
    # 64   2   8 393216    d2       32  2  8 'some_comment'  'EXPERIMENT_SCHEDULE_PRIORITY=50,MIN_TOLERANCE_FACTOR=0.20,OUTPUT_DIR_SUFFIX_ADDON=-tol0.20'

# >>>

# ------------- Stop here -----------------


# ----------------------------------------------------------------------------
#      Formal Experiment (2025/09/24 10:50:00 PM PST)
#  MAX_SAMPLE_ID = 3
# ----------------------------------------------------------------------------


# <<< EXPERIMENT__SKIP=0,EXPERIMENT__COMPLETED=0,MAX_SAMPLE_ID=1,EXPERIMENT__DEBUG=1,ENABLE_NSYS=1
# ---------------- 128k 64 node tbs=64 ----------------
    # n  bs  mb   t         mode   cp  pp tp    comment    env_var
    # Optimal Configurations so far
    # 64   4  16 131072    d2       16  4  8 'some_comment'  'EXPERIMENT_SCHEDULE_PRIORITY=50,EXPERIMENT__STATUS=PASS,RESULT_DIR=26_024644.d2-n64-t131072-b4-mb16-cp16tp8pp4-codellama_CodeLlama-34b-hf'
    64   4  16 131072    d2       16  4  8 'some_comment'  'EXPERIMENT_SCHEDULE_PRIORITY=50,EXPERIMENT__STATUS=PASS,RESULT_DIR=26_111259.d2-n64-t131072-b4-mb16-cp16tp8pp4-codellama_CodeLlama-34b-hf'
    64   2  32 131072    wlbllm   16  1  8 'some_comment'  'EXPERIMENT_SCHEDULE_PRIORITY=50,EXPERIMENT__STATUS=PASS,RESULT_DIR=26_073355.wlbllm-n64-t131072-b2-mb32-cp16tp8pp1-codellama_CodeLlama-34b-hf'
# >>>


# <<< EXPERIMENT__SKIP=0,EXPERIMENT__COMPLETED=0,MAX_SAMPLE_ID=1
# ---------------- 384k 64 node tbs=16 ----------------
   # n  bs  mb   t         mode   cp  pp tp    comment    env_var
    # Tuning tolerance factor of D2
    64   2   8 393216    d2       32  2  8 'some_comment'  'EXPERIMENT_SCHEDULE_PRIORITY=50,MIN_TOLERANCE_FACTOR=0.10,OUTPUT_DIR_SUFFIX_ADDON=-tol0.10'
    64   2   8 393216    d2       32  2  8 'some_comment'  'EXPERIMENT_SCHEDULE_PRIORITY=50,MIN_TOLERANCE_FACTOR=0.15,OUTPUT_DIR_SUFFIX_ADDON=-tol0.15'
    64   2   8 393216    d2       32  2  8 'some_comment'  'EXPERIMENT_SCHEDULE_PRIORITY=50,MIN_TOLERANCE_FACTOR=0.20,OUTPUT_DIR_SUFFIX_ADDON=-tol0.20'


    # Optimal Configurations so far
    64   4   4 393216    d2       64  1  8 'some_comment'  'OPTIMAL=1'
    64   2   8 393216    d2       32  2  8 'some_comment'  'EXPERIMENT_SCHEDULE_PRIORITY=50,EXPERIMENT__STATUS=PASS,RESULT_DIR=26_030441.d2-n64-t393216-b2-mb8-cp32tp8pp2-codellama_CodeLlama-34b-hf'
    64   2   8 393216  wlbllm     16  1  8 'some_comment'  'EXPERIMENT_SCHEDULE_PRIORITY=50'


    # Complete set except for optimal
    64   4   4 393216    d2       32  2  8 'some_comment'  ''
    64   8   2 393216    d2       64  1  8 'some_comment'  ''
    64   1  16 393216    d2       16  4  8 'some_comment'  'EXPERIMENT_SCHEDULE_PRIORITY=40,EXPERIMENT__STATUS=PASS,RESULT_DIR=26_032137.d2-n64-t393216-b1-mb16-cp16tp8pp4-codellama_CodeLlama-34b-hf'
    64  16   1 393216    d2       64  1  8 'somecomment'  'EXPERIMENT_SCHEDULE_PRIORITY=50,EXPERIMENT__STATUS=FAIL,EXPERIMENT__FAIL_REASON=OOM,RESULT_DIR=26_024102.d2-n64-t393216-b16-mb1-cp64tp8pp1-codellama_CodeLlama-34b-hf'
    64  16   1 393216    wlbllm   32  1  8 'some_comment'  'EXPERIMENT_SCHEDULE_PRIORITY=50,EXPERIMENT__STATUS=FAIL,EXPERIMENT__FAIL_REASON=OOM,RESULT_DIR=26_031233.wlbllm-n64-t393216-b16-mb1-cp32tp8pp1-codellama_CodeLlama-34b-hf'
    64   1  16 393216    d2       32  2  8 'some_comment'  'EXPERIMENT__STATUS=PASS,RESULT_DIR=26_100051.d2-n64-t393216-b1-mb16-cp32tp8pp2-codellama_CodeLlama-34b-hf,EXPERIMENT_SCHEDULE_PRIORITY=50'
    64   2   8 393216    wlbllm   32  2  8 'some_comment'  'EXPERIMENT_SCHEDULE_PRIORITY=50,EXPERIMENT__STATUS=FAIL,EXPERIMENT__FAIL_REASON=OOM,RESULT_DIR=26_031454.wlbllm-n64-t393216-b2-mb8-cp32tp8pp2-codellama_CodeLlama-34b-hf'



    64   2   8 393216    d2       64  1  8 'some_comment'  'EXPERIMENT__STATUS=FAIL,RESULT_DIR=26_105128.d2-n64-t393216-b2-mb8-cp64tp8pp1-codellama_CodeLlama-34b-hf'
    64   1  16 393216    d2        8  8  8 'some_comment'  'EXPERIMENT__STATUS=FAIL,EXPERIMENT__FAIL_REASON=OOM,RESULT_DIR=26_094608.d2-n64-t393216-b1-mb16-cp8tp8pp8-codellama_CodeLlama-34b-hf'
    64   2   8 393216    d2       16  4  8 'some_comment'  'EXPERIMENT__STATUS=PASS,RESULT_DIR=26_103748.d2-n64-t393216-b2-mb8-cp16tp8pp4-codellama_CodeLlama-34b-hf'
    64   4   4 393216    d2       32  2  8 'some_comment'  ''
    64   4   4 393216    d2       64  1  8 'some_comment'  ''
    64   8   2 393216    d2       64  1  8 'some_comment'  ''
    64   1  16 393216    d2       64  1  8 'some_comment'  'EXPERIMENT__STATUS=FAIL,EXPERIMENT__FAIL_REASON=OOM,RESULT_DIR=26_101415.d2-n64-t393216-b1-mb16-cp64tp8pp1-codellama_CodeLlama-34b-hf'
    64   4   4 393216    wlbllm    8  2  8 'some_comment'  'EXPERIMENT_SCHEDULE_PRIORITY=40,EXPERIMENT__STATUS=PASS,RESULT_DIR=26_081602.wlbllm-n64-t393216-b4-mb4-cp8tp8pp2-codellama_CodeLlama-34b-hf'
    64   2   8 393216    wlbllm   16  2  8 'some_comment'  'EXPERIMENT_SCHEDULE_PRIORITY=40,EXPERIMENT__STATUS=PASS,RESULT_DIR=26_082035.wlbllm-n64-t393216-b2-mb8-cp16tp8pp2-codellama_CodeLlama-34b-hf'
    64   1  16 393216    wlbllm   32  2  8 'some_comment'  'EXPERIMENT_SCHEDULE_PRIORITY=40,EXPERIMENT__STATUS=PASS,RESULT_DIR=26_082527.wlbllm-n64-t393216-b1-mb16-cp32tp8pp2-codellama_CodeLlama-34b-hf'
    64   4   4 393216    wlbllm   16  2  8 'some_comment'  'EXPERIMENT_SCHEDULE_PRIORITY=40,EXPERIMENT__STATUS=FAIL,EXPERIMENT__FAIL_REASON=OOM,RESULT_DIR=26_083042.wlbllm-n64-t393216-b4-mb4-cp16tp8pp2-codellama_CodeLlama-34b-hf'
    64   2   8 393216    wlbllm   16  4  8 'some_comment'  'EXPERIMENT_SCHEDULE_PRIORITY=40,EXPERIMENT__STATUS=FAIL,EXPERIMENT__FAIL_REASON=OOM,RESULT_DIR=26_083302.wlbllm-n64-t393216-b2-mb8-cp16tp8pp4-codellama_CodeLlama-34b-hf'
    64 0.5  32 393216    wlbllm   64  1  8 'some_comment'  'EXPERIMENT_SCHEDULE_PRIORITY=40,EXPERIMENT__STATUS=PASS,RESULT_DIR=26_083616.wlbllm-n64-t393216-b0.5-mb32-cp64tp8pp1-codellama_CodeLlama-34b-hf'
    64   1  16 393216    wlbllm   32  1  8 'some_comment'  'EXPERIMENT__STATUS=PASS,RESULT_DIR=26_095533.wlbllm-n64-t393216-b1-mb16-cp32tp8pp1-codellama_CodeLlama-34b-hf'
    64   1  16 393216    wlbllm   64  1  8 'some_comment'  'EXPERIMENT__STATUS=PASS,RESULT_DIR=26_102710.wlbllm-n64-t393216-b1-mb16-cp64tp8pp1-codellama_CodeLlama-34b-hf'
    64   2   8 393216    wlbllm   16  1  8 'some_comment'  'EXPERIMENT__STATUS=PASS,RESULT_DIR=26_103300.wlbllm-n64-t393216-b2-mb8-cp16tp8pp1-codellama_CodeLlama-34b-hf'
    64   2   8 393216    wlbllm   32  1  8 'some_comment'  'EXPERIMENT__STATUS=PASS,RESULT_DIR=26_104621.wlbllm-n64-t393216-b2-mb8-cp32tp8pp1-codellama_CodeLlama-34b-hf'
    64   2   8 393216    wlbllm   64  1  8 'some_comment'  ''
    64   4   4 393216    wlbllm    8  1  8 'some_comment'  ''
    64   4   4 393216    wlbllm   16  1  8 'some_comment'  ''
    64   4   4 393216    wlbllm   32  1  8 'some_comment'  ''
    64   4   4 393216    wlbllm   32  2  8 'some_comment'  ''
    64   4   4 393216    wlbllm   64  1  8 'some_comment'  ''
    64   8   2 393216    wlbllm    4  1  8 'some_comment'  ''
    64   8   2 393216    wlbllm    8  1  8 'some_comment'  ''
    64   8   2 393216    wlbllm   16  1  8 'some_comment'  ''
    64   8   2 393216    wlbllm   32  1  8 'some_comment'  ''
    64   8   2 393216    wlbllm   64  1  8 'some_comment'  ''
    64  16   1 393216    wlbllm    2  1  8 'some_comment'  'EXPERIMENT_SCHEDULE_PRIORITY=0'
    64  16   1 393216    wlbllm    4  1  8 'some_comment'  'EXPERIMENT_SCHEDULE_PRIORITY=0'
    64  16   1 393216    wlbllm    8  1  8 'some_comment'  ''
    64  16   1 393216    wlbllm   16  1  8 'some_comment'  ''
    64  16   1 393216    wlbllm   64  1  8 'some_comment'  ''


# ---------------- 384k 64 node tbs=8 ----------------

# ---------------- 256k 64 node tbs=32 ----------------
    # n  bs  mb   t         mode   cp  pp tp    comment    env_var
    64  32   1 262144    d2       64  1  8 'some_comment'  'EXPERIMENT_SCHEDULE_PRIORITY=50,EXPERIMENT__STATUS=FAIL,EXPERIMENT__FAIL_REASON=OOM,RESULT_DIR=26_031803.d2-n64-t262144-b32-mb1-cp64tp8pp1-codellama_CodeLlama-34b-hf'
    64   2  16 262144    d2       16  4  8 'some_comment'  'EXPERIMENT_SCHEDULE_PRIORITY=40,EXPERIMENT__STATUS=PASS,RESULT_DIR=26_033530.d2-n64-t262144-b2-mb16-cp16tp8pp4-codellama_CodeLlama-34b-hf'

    64   1  32 262144    wlbllm   32  2  8 'some_comment'  'EXPERIMENT_SCHEDULE_PRIORITY=40,EXPERIMENT__STATUS=PASS,RESULT_DIR=26_084637.wlbllm-n64-t262144-b1-mb32-cp32tp8pp2-codellama_CodeLlama-34b-hf'
    64   2  16 262144    wlbllm   16  2  8 'some_comment'  'EXPERIMENT_SCHEDULE_PRIORITY=40,EXPERIMENT__STATUS=PASS,RESULT_DIR=26_085230.wlbllm-n64-t262144-b2-mb16-cp16tp8pp2-codellama_CodeLlama-34b-hf'
    64   2  16 262144    wlbllm   16  4  8 'some_comment'  'EXPERIMENT_SCHEDULE_PRIORITY=40,EXPERIMENT__STATUS=PASS,RESULT_DIR=26_085701.wlbllm-n64-t262144-b2-mb16-cp16tp8pp4-codellama_CodeLlama-34b-hf'
    64   1  32 262144    d2       32  2  8 'some_comment'  'EXPERIMENT_SCHEDULE_PRIORITY=30,EXPERIMENT__STATUS=PASS,RESULT_DIR=26_091115.d2-n64-t262144-b1-mb32-cp32tp8pp2-codellama_CodeLlama-34b-hf'
    64   4   8 262144    wlbllm    8  2  8 'some_comment'  'EXPERIMENT_SCHEDULE_PRIORITY=40,EXPERIMENT__STATUS=PASS,RESULT_DIR=26_090149.wlbllm-n64-t262144-b4-mb8-cp8tp8pp2-codellama_CodeLlama-34b-hf'
    64   8   4 262144    wlbllm    4  2  8 'some_comment'  'EXPERIMENT_SCHEDULE_PRIORITY=40,EXPERIMENT__STATUS=PASS,RESULT_DIR=26_090652.wlbllm-n64-t262144-b8-mb4-cp4tp8pp2-codellama_CodeLlama-34b-hf'

    64 0.5  64 262144    wlbllm   64  1  8 'some_comment'  ''
    64   1  32 262144    d2        8  8  8 'some_comment'  ''
    64   1  32 262144    d2       16  4  8 'some_comment'  ''
    64   1  32 262144    wlbllm   32  1  8 'some_comment'  ''
    64   1  32 262144    d2       64  1  8 'some_comment'  ''
    64   1  32 262144    wlbllm   64  1  8 'some_comment'  ''
    64   2  16 262144    d2        8  8  8 'some_comment'  ''
    64   2  16 262144    wlbllm   16  1  8 'some_comment'  ''
    64   2  16 262144    wlbllm   32  1  8 'some_comment'  ''
    64   2  16 262144    d2       32  2  8 'some_comment'  ''
    64   2  16 262144    wlbllm   32  2  8 'some_comment'  ''
    64   2  16 262144    d2       64  1  8 'some_comment'  ''
    64   2  16 262144    wlbllm   64  1  8 'some_comment'  ''
    64   4   8 262144    wlbllm    8  1  8 'some_comment'  ''
    64   4   8 262144    wlbllm    8  4  8 'some_comment'  ''
    64   4   8 262144    wlbllm   16  1  8 'some_comment'  ''
    64   4   8 262144    wlbllm   16  2  8 'some_comment'  ''
    64   4   8 262144    d2       16  4  8 'some_comment'  ''
    64   4   8 262144    wlbllm   16  4  8 'some_comment'  ''
    64   4   8 262144    wlbllm   32  1  8 'some_comment'  ''
    64   4   8 262144    d2       32  2  8 'some_comment'  ''
    64   4   8 262144    wlbllm   32  2  8 'some_comment'  ''
    64   4   8 262144    d2       64  1  8 'some_comment'  ''
    64   4   8 262144    wlbllm   64  1  8 'some_comment'  ''
    64   8   4 262144    wlbllm    4  1  8 'some_comment'  ''
    64   8   4 262144    wlbllm    8  1  8 'some_comment'  ''
    64   8   4 262144    wlbllm    8  2  8 'some_comment'  ''
    64   8   4 262144    wlbllm   16  1  8 'some_comment'  ''
    64   8   4 262144    wlbllm   16  2  8 'some_comment'  ''
    64   8   4 262144    wlbllm   32  1  8 'some_comment'  ''
    64   8   4 262144    d2       32  2  8 'some_comment'  ',EXPERIMENT__STATUS=FAIL,EXPERIMENT__FAIL_REASON=OOM,RESULT_DIR=26_120430.d2-n64-t262144-b8-mb4-cp32tp8pp2-codellama_CodeLlama-34b-hf'
    64   8   4 262144    wlbllm   32  2  8 'some_comment'  ''
    64   8   4 262144    d2       64  1  8 'some_comment'  ',EXPERIMENT__STATUS=FAIL,RESULT_DIR=26_120814.d2-n64-t262144-b8-mb4-cp64tp8pp1-codellama_CodeLlama-34b-hf'
    64   8   4 262144    wlbllm   64  1  8 'some_comment'  'EXPERIMENT__STATUS=FAIL,EXPERIMENT__FAIL_REASON=OOM,RESULT_DIR=26_093222.wlbllm-n64-t262144-b8-mb4-cp64tp8pp1-codellama_CodeLlama-34b-hf'
    64  16   2 262144    wlbllm    2  1  8 'some_comment'  'EXPERIMENT__STATUS=FAIL,EXPERIMENT__FAIL_REASON=OOM,RESULT_DIR=26_093503.wlbllm-n64-t262144-b16-mb2-cp2tp8pp1-codellama_CodeLlama-34b-hf'
    64  16   2 262144    wlbllm    4  1  8 'some_comment'  'EXPERIMENT__STATUS=FAIL,EXPERIMENT__FAIL_REASON=OOM,RESULT_DIR=26_093737.wlbllm-n64-t262144-b16-mb2-cp4tp8pp1-codellama_CodeLlama-34b-hf'
    64  16   2 262144    wlbllm    8  1  8 'some_comment'  'EXPERIMENT__STATUS=FAIL,EXPERIMENT__FAIL_REASON=OOM,RESULT_DIR=26_094114.wlbllm-n64-t262144-b16-mb2-cp8tp8pp1-codellama_CodeLlama-34b-hf'
    64  16   2 262144    wlbllm   16  1  8 'some_comment'  'EXPERIMENT__STATUS=FAIL,EXPERIMENT__FAIL_REASON=OOM,RESULT_DIR=26_034536.wlbllm-n64-t262144-b16-mb2-cp16tp8pp1-codellama_CodeLlama-34b-hf'
    64  16   2 262144    wlbllm   32  1  8 'some_comment'  'EXPERIMENT__STATUS=FAIL,EXPERIMENT__FAIL_REASON=OOM,RESULT_DIR=26_034739.wlbllm-n64-t262144-b16-mb2-cp32tp8pp1-codellama_CodeLlama-34b-hf'
    64  16   2 262144    d2       64  1  8 'some_comment'  'EXPERIMENT__STATUS=PASS,RESULT_DIR=26_035107.d2-n64-t262144-b16-mb2-cp64tp8pp1-codellama_CodeLlama-34b-hf'
    64  16   2 262144    wlbllm   64  1  8 'some_comment'  'EXPERIMENT__STATUS=FAIL,EXPERIMENT__FAIL_REASON=OOM,RESULT_DIR=26_040019.wlbllm-n64-t262144-b16-mb2-cp64tp8pp1-codellama_CodeLlama-34b-hf'
    64  32   1 262144    wlbllm    1  1  8 'some_comment'  'EXPERIMENT__STATUS=FAIL,EXPERIMENT__FAIL_REASON=OOM,RESULT_DIR=26_040251.wlbllm-n64-t262144-b32-mb1-cp1tp8pp1-codellama_CodeLlama-34b-hf'
    64  32   1 262144    wlbllm    2  1  8 'some_comment'  'EXPERIMENT__STATUS=FAIL,EXPERIMENT__FAIL_REASON=OOM,RESULT_DIR=26_040608.wlbllm-n64-t262144-b32-mb1-cp2tp8pp1-codellama_CodeLlama-34b-hf'
    64  32   1 262144    wlbllm    4  1  8 'some_comment'  'EXPERIMENT__STATUS=FAIL,EXPERIMENT__FAIL_REASON=OOM,RESULT_DIR=26_041059.wlbllm-n64-t262144-b32-mb1-cp4tp8pp1-codellama_CodeLlama-34b-hf'
    64  32   1 262144    wlbllm    8  1  8 'some_comment'  'EXPERIMENT__STATUS=FAIL,EXPERIMENT__FAIL_REASON=OOM,RESULT_DIR=26_041259.wlbllm-n64-t262144-b32-mb1-cp8tp8pp1-codellama_CodeLlama-34b-hf'
    64  32   1 262144    wlbllm   16  1  8 'some_comment'  'EXPERIMENT__STATUS=FAIL,EXPERIMENT__FAIL_REASON=OOM,RESULT_DIR=26_041605.wlbllm-n64-t262144-b32-mb1-cp16tp8pp1-codellama_CodeLlama-34b-hf'
    64  32   1 262144    wlbllm   32  1  8 'some_comment'  'EXPERIMENT__STATUS=FAIL,EXPERIMENT__FAIL_REASON=OOM,RESULT_DIR=26_042051.wlbllm-n64-t262144-b32-mb1-cp32tp8pp1-codellama_CodeLlama-34b-hf'
    64  32   1 262144    wlbllm   64  1  8 'some_comment'  'EXPERIMENT__STATUS=FAIL,EXPERIMENT__FAIL_REASON=OOM,RESULT_DIR=26_042249.wlbllm-n64-t262144-b32-mb1-cp64tp8pp1-codellama_CodeLlama-34b-hf'

# ---------------- 128k 64 node tbs=64 ----------------
    # n  bs  mb   t         mode   cp  pp tp    comment    env_var
    # Optimal Configurations so far
    64   4  16 131072    d2       16  4  8 'some_comment'  'EXPERIMENT_SCHEDULE_PRIORITY=50,EXPERIMENT__STATUS=PASS,RESULT_DIR=26_024644.d2-n64-t131072-b4-mb16-cp16tp8pp4-codellama_CodeLlama-34b-hf'
    64  64   1 131072    wlbllm   16  1  8 'some_comment'  'EXPERIMENT_SCHEDULE_PRIORITY=50,EXPERIMENT__STATUS=FAIL,EXPERIMENT__FAIL_REASON=OOM,RESULT_DIR=26_025817.wlbllm-n64-t131072-b64-mb1-cp16tp8pp1-codellama_CodeLlama-34b-hf'

    
    64  64   1 131072    d2       64  1  8 'some_comment'  'EXPERIMENT_SCHEDULE_PRIORITY=50,EXPERIMENT__STATUS=FAIL,EXPERIMENT__FAIL_REASON=OOM,RESULT_DIR=26_024341.d2-n64-t131072-b64-mb1-cp64tp8pp1-codellama_CodeLlama-34b-hf'
    64  64   1 131072    wlbllm   32  1  8 'some_comment'  'EXPERIMENT_SCHEDULE_PRIORITY=50,EXPERIMENT__STATUS=FAIL,EXPERIMENT__FAIL_REASON=OOM,RESULT_DIR=26_030017.wlbllm-n64-t131072-b64-mb1-cp32tp8pp1-codellama_CodeLlama-34b-hf'

    64   1  64 131072    d2        8  8  8 'some_comment'  'EXPERIMENT__STATUS=PASS,RESULT_DIR=26_042613.d2-n64-t131072-b1-mb64-cp8tp8pp8-codellama_CodeLlama-34b-hf'
    64   1  64 131072    d2       16  4  8 'some_comment'  'EXPERIMENT__STATUS=PASS,RESULT_DIR=26_045305.d2-n64-t131072-b1-mb64-cp16tp8pp4-codellama_CodeLlama-34b-hf'
    64   1  64 131072    wlbllm   32  1  8 'some_comment'  'EXPERIMENT__STATUS=PASS,RESULT_DIR=26_052027.wlbllm-n64-t131072-b1-mb64-cp32tp8pp1-codellama_CodeLlama-34b-hf'
    64   1  64 131072    d2       32  2  8 'some_comment'  'EXPERIMENT__STATUS=PASS,RESULT_DIR=26_052923.d2-n64-t131072-b1-mb64-cp32tp8pp2-codellama_CodeLlama-34b-hf'
    64   1  64 131072    wlbllm   32  2  8 'some_comment'  'EXPERIMENT__STATUS=PASS,RESULT_DIR=26_060632.wlbllm-n64-t131072-b1-mb64-cp32tp8pp2-codellama_CodeLlama-34b-hf'
    64   1  64 131072    d2       64  1  8 'some_comment'  'EXPERIMENT__STATUS=PASS,RESULT_DIR=26_061251.d2-n64-t131072-b1-mb64-cp64tp8pp1-codellama_CodeLlama-34b-hf'
    64   1  64 131072    wlbllm   64  1  8 'some_comment'  'EXPERIMENT__STATUS=PASS,RESULT_DIR=26_070935.wlbllm-n64-t131072-b1-mb64-cp64tp8pp1-codellama_CodeLlama-34b-hf'
    64   2  32 131072    d2        8  8  8 'some_comment'  'EXPERIMENT__STATUS=PASS,RESULT_DIR=26_071948.d2-n64-t131072-b2-mb32-cp8tp8pp8-codellama_CodeLlama-34b-hf'
    64   2  32 131072    wlbllm   16  1  8 'some_comment'  'EXPERIMENT__STATUS=PASS,RESULT_DIR=26_073355.wlbllm-n64-t131072-b2-mb32-cp16tp8pp1-codellama_CodeLlama-34b-hf'
    64   2  32 131072    wlbllm   16  2  8 'some_comment'  'EXPERIMENT__STATUS=FAIL,RESULT_DIR=26_073939.wlbllm-n64-t131072-b2-mb32-cp16tp8pp2-codellama_CodeLlama-34b-hf'
    64   2  32 131072    d2       16  4  8 'some_comment'  ''
    64   2  32 131072    wlbllm   16  4  8 'some_comment'  ''
    64   2  32 131072    wlbllm   32  1  8 'some_comment'  ''
    64   2  32 131072    d2       32  2  8 'some_comment'  ''
    64   2  32 131072    wlbllm   32  2  8 'some_comment'  ''
    64   2  32 131072    d2       64  1  8 'some_comment'  ''
    64   2  32 131072    wlbllm   64  1  8 'some_comment'  ''
    64   4  16 131072    wlbllm    8  1  8 'some_comment'  ''
    64   4  16 131072    wlbllm    8  2  8 'some_comment'  ''
    64   4  16 131072    wlbllm    8  4  8 'some_comment'  ''
    64   4  16 131072    d2        8  8  8 'some_comment'  ''
    64   4  16 131072    wlbllm    8  8  8 'some_comment'  ''
    64   4  16 131072    wlbllm   16  1  8 'some_comment'  ''
    64   4  16 131072    wlbllm   16  2  8 'some_comment'  ''
    64   4  16 131072    wlbllm   16  4  8 'some_comment'  ''
    64   4  16 131072    wlbllm   32  1  8 'some_comment'  ''
    64   4  16 131072    d2       32  2  8 'some_comment'  ''
    64   4  16 131072    wlbllm   32  2  8 'some_comment'  ''
    64   4  16 131072    d2       64  1  8 'some_comment'  ''
    64   4  16 131072    wlbllm   64  1  8 'some_comment'  ''
    64   8   8 131072    wlbllm    4  1  8 'some_comment'  ''
    64   8   8 131072    wlbllm    4  2  8 'some_comment'  ''
    64   8   8 131072    wlbllm    4  4  8 'some_comment'  ''
    64   8   8 131072    wlbllm    8  1  8 'some_comment'  ''
    64   8   8 131072    wlbllm    8  2  8 'some_comment'  ''
    64   8   8 131072    wlbllm    8  4  8 'some_comment'  ''
    64   8   8 131072    wlbllm   16  1  8 'some_comment'  ''
    64   8   8 131072    wlbllm   16  2  8 'some_comment'  ''
    64   8   8 131072    d2       16  4  8 'some_comment'  ''
    64   8   8 131072    wlbllm   16  4  8 'some_comment'  ''
    64   8   8 131072    wlbllm   32  1  8 'some_comment'  ''
    64   8   8 131072    d2       32  2  8 'some_comment'  ''
    64   8   8 131072    wlbllm   32  2  8 'some_comment'  ''
    64   8   8 131072    d2       64  1  8 'some_comment'  ''
    64   8   8 131072    wlbllm   64  1  8 'some_comment'  ''
    64  16   4 131072    wlbllm    2  1  8 'some_comment'  ''
    64  16   4 131072    wlbllm    2  2  8 'some_comment'  ''
    64  16   4 131072    wlbllm    4  1  8 'some_comment'  ''
    64  16   4 131072    wlbllm    4  2  8 'some_comment'  ''
    64  16   4 131072    wlbllm    8  1  8 'some_comment'  ''
    64  16   4 131072    wlbllm    8  2  8 'some_comment'  ''
    64  16   4 131072    wlbllm   16  1  8 'some_comment'  ''
    64  16   4 131072    wlbllm   16  2  8 'some_comment'  ''
    64  16   4 131072    wlbllm   32  1  8 'some_comment'  ''
    64  16   4 131072    d2       32  2  8 'some_comment'  ''
    64  16   4 131072    wlbllm   32  2  8 'some_comment'  ''
    64  16   4 131072    d2       64  1  8 'some_comment'  ''
    64  16   4 131072    wlbllm   64  1  8 'some_comment'  ''
    64  32   2 131072    wlbllm    1  1  8 'some_comment'  'EXPERIMENT_SCHEDULE_PRIORITY=0'
    64  32   2 131072    wlbllm    2  1  8 'some_comment'  'EXPERIMENT_SCHEDULE_PRIORITY=0'
    64  32   2 131072    wlbllm    4  1  8 'some_comment'  ''
    64  32   2 131072    wlbllm    8  1  8 'some_comment'  ''
    64  32   2 131072    wlbllm   16  1  8 'some_comment'  ''
    64  32   2 131072    wlbllm   32  1  8 'some_comment'  ''
    64  32   2 131072    d2       64  1  8 'some_comment'  ''
    64  32   2 131072    wlbllm   64  1  8 'some_comment'  ''
    64  64   1 131072    wlbllm    1  1  8 'some_comment'  'EXPERIMENT_SCHEDULE_PRIORITY=0'
    64  64   1 131072    wlbllm    2  1  8 'some_comment'  'EXPERIMENT_SCHEDULE_PRIORITY=0'
    64  64   1 131072    wlbllm    4  1  8 'some_comment'  ''
    64  64   1 131072    wlbllm    8  1  8 'some_comment'  ''
    64  64   1 131072    wlbllm   64  1  8 'some_comment'  ''
    
# # >>>




# ----------------------------------------------------------------------------
#      Formal Experiment (2025/09/24 10:50:00 PM PST)
#  MAX_SAMPLE_ID = 3
# ----------------------------------------------------------------------------





# ---------------- 128k 64 node tbs=32 ----------------
# total batch size = 16, 32, 64
# <<< EXPERIMENT__SKIP=1,EXPERIMENT__COMPLETED=0
# #   n   bs   mb     t       mode   cp   pp  tp     comment        env_var
    # 64   2    16   131072     d2    32  2   8  'd2-pretrain'  'EXPERIMENT_ADD_SELECTIVE_CKPT=1,ENABLE_NSYS=0'
    # 64   2    16   131072  wlbllm   32  2   8  'd2-pretrain'  'EXPERIMENT_ADD_SELECTIVE_CKPT=1,ENABLE_NSYS=0'
# >>>



# # <<< EXPERIMENT__SKIP=1,EXPERIMENT__COMPLETED=0,COMMENT="We are slower, why?"
# # #   n   bs   mb     t       mode   cp   pp  tp     comment        env_var
#     64   2    16   131072     d2    32  2   8  'd2-pretrain'  'ENABLE_NSYS=0'
#     64   2    16   131072  wlbllm   32  2   8  'd2-pretrain'  'ENABLE_NSYS=0'
# # >>>

# # <<< SECTION=debug-nsys,EXPERIMENT__SKIP=0,MAX_SAMPLE_ID=1,ENABLE_NSYS=1
# # #   n   bs   mb     t       mode   cp   pp  tp     comment        env_var
#     64   2    16   131072     d2    32  2   8  'd2-pretrain'  ''
#     64   2    16   131072     d2    32  2   8  'd2-pretrain'  ''
#     64   2    16   131072  wlbllm   32  2   8  'd2-pretrain'  ''
# # >>>



# # Section 2: Debug configurations for development
# # <<< debug=1,schedule_priority=50
#     64   2    16   131072     d2    16  4   8  'd2-pretrain'  'EXPERIMENT_ADD_SELECTIVE_CKPT=1,ENABLE_NSYS=0'
#     64   2    16   131072  wlbllm   16  4   8  'd2-pretrain'  'EXPERIMENT_ADD_SELECTIVE_CKPT=1,ENABLE_NSYS=0'
# # >>>

# # Section 3: Production configurations (no special env vars)
# # <<<
#     64   2    16   131072  wlbllm   16  2   8  'd2-pretrain'  'EXPERIMENT_ADD_SELECTIVE_CKPT=1,ENABLE_NSYS=0'
#     64   2    16   131072  wlbllm    8  2   8  'd2-pretrain'  'EXPERIMENT_ADD_SELECTIVE_CKPT=1,ENABLE_NSYS=0'
# # >>>

# # Direct skip test (outside of sections)
#     64   2    16   131072     d2     8  2   8  'd2-pretrain'  'EXPERIMENT_ADD_SELECTIVE_CKPT=1,EXPERIMENT__SKIP=1'
