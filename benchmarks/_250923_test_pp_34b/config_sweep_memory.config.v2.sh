







# ----------------------------------------------------------------------------
#      Formal Experiment (2025/09/24 10:50:00 PM PST)
#  MAX_SAMPLE_ID = 3
# ----------------------------------------------------------------------------



# ---------------- 128k 64 node tbs=64 ----------------
# total batch size = 16, 32, 64
# #   n   bs   mb     t       mode   cp   pp  tp     comment        env_var
#     64   2    16   131072     d2    32  2   8  'd2-pretrain'  'EXPERIMENT_ADD_SELECTIVE_CKPT=1,ENABLE_NSYS=0'
#     64   2    16   131072  wlbllm   32  2   8  'd2-pretrain'  'EXPERIMENT_ADD_SELECTIVE_CKPT=1,ENABLE_NSYS=0'
    
#     64   2    16   131072     d2    16  4   8  'd2-pretrain'  'EXPERIMENT_ADD_SELECTIVE_CKPT=1,ENABLE_NSYS=0'
#     64   2    16   131072  wlbllm   16  4   8  'd2-pretrain'  'EXPERIMENT_ADD_SELECTIVE_CKPT=1,ENABLE_NSYS=0'
    
#     64   2    16   131072  wlbllm   16  2   8  'd2-pretrain'  'EXPERIMENT_ADD_SELECTIVE_CKPT=1,ENABLE_NSYS=0'
#     64   2    16   131072  wlbllm    8  2   8  'd2-pretrain'  'EXPERIMENT_ADD_SELECTIVE_CKPT=1,ENABLE_NSYS=0'



# ---------------- 128k 32 node tbs=32 ratio=8 ----------------
#   n   bs   mb     t       mode   cp   pp  tp     comment        env_var
    32   4    8    131072    d2    16   2   8  'd2-pretrain'  'EXPERIMENT_ADD_SELECTIVE_CKPT=1,ENABLE_NSYS=0,EXPERIMENT_SCHEDULE_PRIORITY=5'
    32   2    16   131072    d2    16   2   8  'd2-pretrain'  'EXPERIMENT_ADD_SELECTIVE_CKPT=1,ENABLE_NSYS=0'
    32   2    16   131072    d2     8   4   8  'd2-pretrain'  'EXPERIMENT_ADD_SELECTIVE_CKPT=1,ENABLE_NSYS=0'
    32   4    8    131072  wlbllm   8   2   8  'd2-pretrain'  'EXPERIMENT_ADD_SELECTIVE_CKPT=1,ENABLE_NSYS=0,EXPERIMENT_SCHEDULE_PRIORITY=5'
    32   2    16   131072  wlbllm   8   2   8  'd2-pretrain'  'EXPERIMENT_ADD_SELECTIVE_CKPT=1,ENABLE_NSYS=0'
    32   2    16   131072  wlbllm   4   4   8  'd2-pretrain'  'EXPERIMENT_ADD_SELECTIVE_CKPT=1,ENABLE_NSYS=0'


# ---------------- 128k 16 node tbs=16 ratio = 8 ----------------
#   n   bs   mb     t       mode   cp   pp  tp     comment        env_var
    16   16   1   131072     d2    16  1   8  'd2-pretrain'  'EXPERIMENT_ADD_SELECTIVE_CKPT=1,ENABLE_NSYS=0,EXPERIMENT_SCHEDULE_PRIORITY=5'
    16   2    8   131072     d2    8   2   8  'd2-pretrain'  'EXPERIMENT_ADD_SELECTIVE_CKPT=1,ENABLE_NSYS=0'
    16   1    16  131072     d2    8   2   8  'd2-pretrain'  'EXPERIMENT_ADD_SELECTIVE_CKPT=1,ENABLE_NSYS=0,EXPERIMENT_SCHEDULE_PRIORITY=5'
    16   4    4   131072     d2    8   2   8  'd2-pretrain'  'EXPERIMENT_ADD_SELECTIVE_CKPT=1,ENABLE_NSYS=0'
    16   1    16  131072  wlbllm   4   4   8  'd2-pretrain'  'EXPERIMENT_ADD_SELECTIVE_CKPT=1,ENABLE_NSYS=0,EXPERIMENT_SCHEDULE_PRIORITY=5,COMMENT="used in wlbllm paper."'
    16   2    8   131072  wlbllm   4   2   8  'd2-pretrain'  'EXPERIMENT_ADD_SELECTIVE_CKPT=1,ENABLE_NSYS=0'
    16   2    8   131072  wlbllm   8   2   8  'd2-pretrain'  'EXPERIMENT_ADD_SELECTIVE_CKPT=1,ENABLE_NSYS=0'


# ---------------- 256k 32 node tbs=16 ----------------
#   n   bs   mb     t       mode   cp   pp  tp     comment        env_var
    32   16   1    262144    d2    32   1   8  'd2-pretrain'  'EXPERIMENT_ADD_SELECTIVE_CKPT=1,ENABLE_NSYS=0,EXPERIMENT_SCHEDULE_PRIORITY=5'
    32   2    8    262144    d2    16   2   8  'd2-pretrain'  'EXPERIMENT_ADD_SELECTIVE_CKPT=1,ENABLE_NSYS=0'
    32   16   1    262144  wlbllm  32   1   8  'd2-pretrain'  'EXPERIMENT_ADD_SELECTIVE_CKPT=1,ENABLE_NSYS=0,EXPERIMENT_SCHEDULE_PRIORITY=5'
    32   2    8    262144  wlbllm  16   2   8  'd2-pretrain'  'EXPERIMENT_ADD_SELECTIVE_CKPT=1,ENABLE_NSYS=0'
    32   2    8    262144  wlbllm  16   2   8  'd2-pretrain'  'EXPERIMENT_ADD_SELECTIVE_CKPT=1,ENABLE_NSYS=0,EXPERIMENT_SCHEDULE_PRIORITY=5'
    32   2    8    262144  wlbllm   8   2   8  'd2-pretrain'  'EXPERIMENT_ADD_SELECTIVE_CKPT=1,ENABLE_NSYS=0'
    32   2    8    262144  wlbllm   8   4   8  'd2-pretrain'  'EXPERIMENT_ADD_SELECTIVE_CKPT=1,ENABLE_NSYS=0'
    32   2    8    262144  wlbllm   4   4   8  'd2-pretrain'  'EXPERIMENT_ADD_SELECTIVE_CKPT=1,ENABLE_NSYS=0'


# ---------------- 256k 16 node tbs=16 ----------------
#   n   bs   mb     t       mode   cp   pp  tp     comment        env_var
    16   2    8    262144    d2    8    2   8  'd2-pretrain'  'EXPERIMENT_ADD_SELECTIVE_CKPT=1,ENABLE_NSYS=0,EXPERIMENT_SCHEDULE_PRIORITY=5'
    16   1    16   262144    d2    8    2   8  'd2-pretrain'  'EXPERIMENT_ADD_SELECTIVE_CKPT=1,ENABLE_NSYS=0'
    16   16   1    262144    d2    16   1   8  'd2-pretrain'  'EXPERIMENT_ADD_SELECTIVE_CKPT=1,ENABLE_NSYS=0'
    16   16   1    262144  wlbllm  16   1   8  'd2-pretrain'  'EXPERIMENT_ADD_SELECTIVE_CKPT=1,ENABLE_NSYS=0,EXPERIMENT_SCHEDULE_PRIORITY=5'
    16   2    8    262144  wlbllm  8    2   8  'd2-pretrain'  'EXPERIMENT_ADD_SELECTIVE_CKPT=1,ENABLE_NSYS=0,EXPERIMENT_SCHEDULE_PRIORITY=5'
    16   1   16    262144  wlbllm  4    4   8  'd2-pretrain'  'EXPERIMENT_ADD_SELECTIVE_CKPT=1,ENABLE_NSYS=0'


# ---------------- 256k 16 node tbs=8 ----------------
#   n   bs   mb     t       mode   cp   pp  tp     comment        env_var
    16   1    8    262144    d2    8    2   8  'd2-pretrain'  'EXPERIMENT_ADD_SELECTIVE_CKPT=1,ENABLE_NSYS=0,EXPERIMENT_SCHEDULE_PRIORITY=5'
    16   8    1    262144    d2    16   1   8  'd2-pretrain'  'EXPERIMENT_ADD_SELECTIVE_CKPT=1,ENABLE_NSYS=0'
    16   8    1    262144  wlbllm  16   1   8  'd2-pretrain'  'EXPERIMENT_ADD_SELECTIVE_CKPT=1,ENABLE_NSYS=0,EXPERIMENT_SCHEDULE_PRIORITY=5'
    16   1    8    262144  wlbllm  8    2   8  'd2-pretrain'  'EXPERIMENT_ADD_SELECTIVE_CKPT=1,ENABLE_NSYS=0,EXPERIMENT_SCHEDULE_PRIORITY=5'
    16   1   16    262144  wlbllm  4    4   8  'd2-pretrain'  'EXPERIMENT_ADD_SELECTIVE_CKPT=1,ENABLE_NSYS=0,EXPERIMENT_SCHEDULE_PRIORITY=5'
    

# # ---------------- 512k 32 node tbs=8 ----------------
# #   n   bs   mb     t       mode   cp   pp  tp     comment        env_var
#     32   8   1   524288     d2     32   1   8  'd2-pretrain'  'EXPERIMENT_ADD_SELECTIVE_CKPT=1,ENABLE_NSYS=0,EXPERIMENT_SCHEDULE_PRIORITY=5'
#     32   2   4   524288     d2     16   2   8  'd2-pretrain'  'EXPERIMENT_ADD_SELECTIVE_CKPT=1,ENABLE_NSYS=0,EXPERIMENT_SCHEDULE_PRIORITY=5'
#     32   8   1   524288   wlbllm   16   1   8  'd2-pretrain'  'EXPERIMENT_ADD_SELECTIVE_CKPT=1,ENABLE_NSYS=0,EXPERIMENT_SCHEDULE_PRIORITY=5'
#     32   1   8   524288   wlbllm   16   2   8  'd2-pretrain'  'EXPERIMENT_ADD_SELECTIVE_CKPT=1,ENABLE_NSYS=0,EXPERIMENT_SCHEDULE_PRIORITY=5'
#     32   1   8   524288   wlbllm    8   4   8  'd2-pretrain'  'EXPERIMENT_ADD_SELECTIVE_CKPT=1,ENABLE_NSYS=0'


# # ---------------- 512k 16 node tbs=4 ----------------
# #   n   bs   mb     t       mode   cp   pp  tp     comment        env_var
#     16   4    1   524288     d2    16   1   8  'd2-pretrain'  'EXPERIMENT_ADD_SELECTIVE_CKPT=1,ENABLE_NSYS=0,EXPERIMENT_SCHEDULE_PRIORITY=5'
#     16   1    4   524288     d2     8   2   8  'd2-pretrain'  'EXPERIMENT_ADD_SELECTIVE_CKPT=1,ENABLE_NSYS=0,EXPERIMENT_SCHEDULE_PRIORITY=5'
#     16   4    1   524288   wlbllm   8   1   8  'd2-pretrain'  'EXPERIMENT_ADD_SELECTIVE_CKPT=1,ENABLE_NSYS=0,EXPERIMENT_SCHEDULE_PRIORITY=5'
#     16   1    4   524288   wlbllm   8   2   8  'd2-pretrain'  'EXPERIMENT_ADD_SELECTIVE_CKPT=1,ENABLE_NSYS=0,EXPERIMENT_SCHEDULE_PRIORITY=5'
#     16   2    2   524288   wlbllm   4   2   8  'd2-pretrain'  'EXPERIMENT_ADD_SELECTIVE_CKPT=1,ENABLE_NSYS=0'

# ---------------- 512k 32 node tbs=8 ----------------
#   n   bs   mb     t       mode   cp   pp  tp     comment        env_var
    32   1   8   524288     d2    32   1   8  'd2-pretrain'  'EXPERIMENT_ADD_SELECTIVE_CKPT=1,ENABLE_NSYS=0,EXPERIMENT_SCHEDULE_PRIORITY=5,MIN_TOLERANCE_FACTOR=0.2'
    32   1   8   524288     d2    16   2   8  'd2-pretrain'  'EXPERIMENT_ADD_SELECTIVE_CKPT=1,ENABLE_NSYS=0,EXPERIMENT_SCHEDULE_PRIORITY=5,MIN_TOLERANCE_FACTOR=0.2'
    32   1   8   524288   wlbllm  16   1   8  'd2-pretrain'  'EXPERIMENT_ADD_SELECTIVE_CKPT=1,ENABLE_NSYS=0,EXPERIMENT_SCHEDULE_PRIORITY=5'
    32   1   8   524288   wlbllm   8   2   8  'd2-pretrain'  'EXPERIMENT_ADD_SELECTIVE_CKPT=1,ENABLE_NSYS=0,EXPERIMENT_SCHEDULE_PRIORITY=5'

#     32   1   4   524288     d2    16   2   8  'd2-pretrain'  'EXPERIMENT_ADD_SELECTIVE_CKPT=1,ENABLE_NSYS=0,EXPERIMENT_SCHEDULE_PRIORITY=5'
#     32   2   2   524288     d2    16   2   8  'd2-pretrain'  'EXPERIMENT_ADD_SELECTIVE_CKPT=1,ENABLE_NSYS=0'
#     32   4   1   524288   wlbllm   8   1   8  'd2-pretrain'  'EXPERIMENT_ADD_SELECTIVE_CKPT=1,ENABLE_NSYS=0,EXPERIMENT_SCHEDULE_PRIORITY=5'
#     32   4   1   524288   wlbllm   4   1   8  'd2-pretrain'  'EXPERIMENT_ADD_SELECTIVE_CKPT=1,ENABLE_NSYS=0'
#     32   1   4   524288   wlbllm   8   2   8  'd2-pretrain'  'EXPERIMENT_ADD_SELECTIVE_CKPT=1,ENABLE_NSYS=0'

# ---------------- 512k 16 node tbs=4 ----------------
#   n   bs   mb     t       mode   cp   pp  tp     comment        env_var
    16   1    4   524288     d2    16   1   8  'd2-pretrain'  'EXPERIMENT_ADD_SELECTIVE_CKPT=1,ENABLE_NSYS=0,EXPERIMENT_SCHEDULE_PRIORITY=5,MIN_TOLERANCE_FACTOR=0.2'
    # 16   1    4   524288     d2    8   2   8  'd2-pretrain'  'EXPERIMENT_ADD_SELECTIVE_CKPT=1,ENABLE_NSYS=0,EXPERIMENT_SCHEDULE_PRIORITY=5,MIN_TOLERANCE_FACTOR=0.2'
    16   1    4   524288   wlbllm  16   1   8  'd2-pretrain'  'EXPERIMENT_ADD_SELECTIVE_CKPT=1,ENABLE_NSYS=0,EXPERIMENT_SCHEDULE_PRIORITY=5'
    # 16   1    4   524288   wlbllm  8    2   8  'd2-pretrain'  'EXPERIMENT_ADD_SELECTIVE_CKPT=1,ENABLE_NSYS=0,EXPERIMENT_SCHEDULE_PRIORITY=5'
#     16   1    2   524288     d2     8   2   8  'd2-pretrain'  'EXPERIMENT_ADD_SELECTIVE_CKPT=1,ENABLE_NSYS=0,EXPERIMENT_SCHEDULE_PRIORITY=5'
#     16   2    1   524288   wlbllm   8   1   8  'd2-pretrain'  'EXPERIMENT_ADD_SELECTIVE_CKPT=1,ENABLE_NSYS=0,EXPERIMENT_SCHEDULE_PRIORITY=5'
#     16   1    2   524288   wlbllm   8   2   8  'd2-pretrain'  'EXPERIMENT_ADD_SELECTIVE_CKPT=1,ENABLE_NSYS=0'


# ---------------- 512k 16 node tbs=2 ----------------
#   n   bs   mb     t       mode   cp   pp  tp     comment        env_var
    16   1    2   524288     d2    16   1   8  'd2-pretrain'  'EXPERIMENT_ADD_SELECTIVE_CKPT=1,ENABLE_NSYS=0,EXPERIMENT_SCHEDULE_PRIORITY=5,MIN_TOLERANCE_FACTOR=0.2'
    16   1    2   524288     d2    8   2   8  'd2-pretrain'  'EXPERIMENT_ADD_SELECTIVE_CKPT=1,ENABLE_NSYS=0,EXPERIMENT_SCHEDULE_PRIORITY=5,MIN_TOLERANCE_FACTOR=0.2'
    # 16   1    4   524288     d2    8   2   8  'd2-pretrain'  'EXPERIMENT_ADD_SELECTIVE_CKPT=1,ENABLE_NSYS=0,EXPERIMENT_SCHEDULE_PRIORITY=5,MIN_TOLERANCE_FACTOR=0.2'
    16   1    2   524288   wlbllm  16   1   8  'd2-pretrain'  'EXPERIMENT_ADD_SELECTIVE_CKPT=1,ENABLE_NSYS=0,EXPERIMENT_SCHEDULE_PRIORITY=5'
    16   1    2   524288   wlbllm   8   1   8  'd2-pretrain'  'EXPERIMENT_ADD_SELECTIVE_CKPT=1,ENABLE_NSYS=0,EXPERIMENT_SCHEDULE_PRIORITY=5'
    16   1    2   524288   wlbllm   4   2   8  'd2-pretrain'  'EXPERIMENT_ADD_SELECTIVE_CKPT=1,ENABLE_NSYS=0,EXPERIMENT_SCHEDULE_PRIORITY=5'
    # 16   1    4   524288   wlbllm  8    2   8  'd2-pretrain'  'EXPERIMENT_ADD_SELECTIVE_CKPT=1,ENABLE_NSYS=0,EXPERIMENT_SCHEDULE_PRIORITY=5'
#     16   1    2   524288     d2     8   2   8  'd2-pretrain'  'EXPERIMENT_ADD_SELECTIVE_CKPT=1,ENABLE_NSYS=0,EXPERIMENT_SCHEDULE_PRIORITY=5'
#     16   2    1   524288   wlbllm   8   1   8  'd2-pretrain'  'EXPERIMENT_ADD_SELECTIVE_CKPT=1,ENABLE_NSYS=0,EXPERIMENT_SCHEDULE_PRIORITY=5'
#     16   1    2   524288   wlbllm   8   2   8  'd2-pretrain'  'EXPERIMENT_ADD_SELECTIVE_CKPT=1,ENABLE_NSYS=0'


# ---------------- 512k 16 node tbs=1 ----------------
#   n   bs   mb     t       mode   cp   pp  tp     comment        env_var
    16   1    1   524288     d2    16   1   8  'd2-pretrain'  'EXPERIMENT_ADD_SELECTIVE_CKPT=1,ENABLE_NSYS=0,EXPERIMENT_SCHEDULE_PRIORITY=5,MIN_TOLERANCE_FACTOR=0.2'
    16   1    1   524288   wlbllm  16   1   8  'd2-pretrain'  'EXPERIMENT_ADD_SELECTIVE_CKPT=1,ENABLE_NSYS=0,EXPERIMENT_SCHEDULE_PRIORITY=5'


# ---------------- 384k 32 node tbs=4 ----------------
#   n   bs   mb     t       mode   cp   pp  tp     comment        env_var
    32   4   1   393216     d2    32   1   8  'd2-pretrain'  'EXPERIMENT_ADD_SELECTIVE_CKPT=1,ENABLE_NSYS=0,EXPERIMENT_SCHEDULE_PRIORITY=5,MIN_TOLERANCE_FACTOR=0.2'
    32   1   4   393216     d2    16   2   8  'd2-pretrain'  'EXPERIMENT_ADD_SELECTIVE_CKPT=1,ENABLE_NSYS=0,MIN_TOLERANCE_FACTOR=0.25'
    32   4   1   393216   wlbllm  16   1   8  'd2-pretrain'  'EXPERIMENT_ADD_SELECTIVE_CKPT=1,ENABLE_NSYS=0,EXPERIMENT_SCHEDULE_PRIORITY=5'
    32   4   1   393216   wlbllm   8   1   8  'd2-pretrain'  'EXPERIMENT_ADD_SELECTIVE_CKPT=1,ENABLE_NSYS=0'
    32   1   4   393216   wlbllm   8   2   8  'd2-pretrain'  'EXPERIMENT_ADD_SELECTIVE_CKPT=1,ENABLE_NSYS=0,EXPERIMENT_SCHEDULE_PRIORITY=5'


# ---------------- 384k 32 node tbs=4 debug (0925 9PM) ----------------
#   n   bs   mb     t       mode   cp   pp  tp     comment        env_var
    # 32   1   4   393216     d2    16   2   8  'd2-pretrain'  'EXPERIMENT_ADD_SELECTIVE_CKPT=1,ENABLE_NSYS=1,MAX_SAMPLE_ID=1,MIN_TOLERANCE_FACTOR=0.25,EXPERIMENT__DEBUG=1'
    # 32   1   4   393216     d2    16   2   8  'd2-pretrain'  'EXPERIMENT_ADD_SELECTIVE_CKPT=1,ENABLE_NSYS=1,MAX_SAMPLE_ID=1,MIN_TOLERANCE_FACTOR=0.35,EXPERIMENT__DEBUG=1'
    # 32   4   1   393216   wlbllm   8   1   8  'd2-pretrain'  'EXPERIMENT_ADD_SELECTIVE_CKPT=1,ENABLE_NSYS=1,MAX_SAMPLE_ID=1,EXPERIMENT__DEBUG=1'

# ---------------- 384k 32 node tbs=8 debug (0925 9PM) ----------------
    # 32   8   1   393216     d2    32   1   8  'd2-pretrain'  'EXPERIMENT_ADD_SELECTIVE_CKPT=1,ENABLE_NSYS=1,MAX_SAMPLE_ID=1,MIN_TOLERANCE_FACTOR=0.25,EXPERIMENT__DEBUG=1,EXPERIMENT__STATUS=FAIL,EXPERIMENT__FAIL_REASON=OOM'
    32   1   8   393216     d2    16   2   8  'd2-pretrain'  'EXPERIMENT_ADD_SELECTIVE_CKPT=1,ENABLE_NSYS=1,MAX_SAMPLE_ID=1,MIN_TOLERANCE_FACTOR=0.25,EXPERIMENT__DEBUG=1'
    32   8   1   393216   wlbllm   8   1   8  'd2-pretrain'  'EXPERIMENT_ADD_SELECTIVE_CKPT=1,ENABLE_NSYS=1,MAX_SAMPLE_ID=1,EXPERIMENT__DEBUG=1'
    

# ---------------- 384k 16 node tbs=2 ----------------
#   n   bs   mb     t       mode   cp   pp  tp     comment        env_var
    16   2   1   393216     d2    16   1   8  'd2-pretrain'  'EXPERIMENT_ADD_SELECTIVE_CKPT=1,ENABLE_NSYS=0,EXPERIMENT_SCHEDULE_PRIORITY=5,MIN_TOLERANCE_FACTOR=0.2'
    16   1   2   393216     d2     8   2   8  'd2-pretrain'  'EXPERIMENT_ADD_SELECTIVE_CKPT=1,ENABLE_NSYS=0,MIN_TOLERANCE_FACTOR=0.2'
    16   1   2   393216   wlbllm   8   2   8  'd2-pretrain'  'EXPERIMENT_ADD_SELECTIVE_CKPT=1,ENABLE_NSYS=0,EXPERIMENT_SCHEDULE_PRIORITY=5'
    16   2   1   393216   wlbllm   8   1   8  'd2-pretrain'  'EXPERIMENT_ADD_SELECTIVE_CKPT=1,ENABLE_NSYS=0,EXPERIMENT_SCHEDULE_PRIORITY=5'