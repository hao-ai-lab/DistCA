
# 🟢
# NNODES=2 GPUS=$((NNODES * 8))
# NNODES=4 GPUS=$((NNODES * 8)) CP_SIZE=2 PP_SIZE=2 TP_SIZE=8 NUM_MICROBATCH=2 NUM_TOKENS=8192
# NNODES=8 GPUS=$((NNODES * 8)) CP_SIZE=4 PP_SIZE=2 TP_SIZE=8 NUM_MICROBATCH=2 NUM_TOKENS=16384
# NNODES=16 GPUS=$((NNODES * 8)) CP_SIZE=8 PP_SIZE=2 TP_SIZE=8 NUM_MICROBATCH=2 NUM_TOKENS=32768
# NNODES=16 GPUS=$((NNODES * 8)) CP_SIZE=4 PP_SIZE=4 TP_SIZE=8 NUM_MICROBATCH=4 NUM_TOKENS=32768
# NNODES=32 GPUS=$((NNODES * 8)) CP_SIZE=8 PP_SIZE=4 TP_SIZE=8 NUM_MICROBATCH=8 NUM_TOKENS=65536
# NNODES=32 GPUS=$((NNODES * 8)) CP_SIZE=8 PP_SIZE=4 TP_SIZE=8 NUM_MICROBATCH=8 NUM_TOKENS=65536


# ⚪
NNODES=32 GPUS=$((NNODES * 8)) CP_SIZE=4 PP_SIZE=8 TP_SIZE=8 NUM_MICROBATCH=8 NUM_TOKENS=65536
# 🔴


TS=$(TZ=America/Los_Angeles date +%Y%m%d_%H%M%S)
LOG_DIR=/mnt/weka/home/yonghao.zhuang/jd/d2/benchmarks/_250909_pp_debug/logs/${TS}/n${NNODES}-cp${CP_SIZE}-pp${PP_SIZE}-tp${TP_SIZE}-mb${NUM_MICROBATCH}-t${NUM_TOKENS}
mkdir -p $LOG_DIR
srun -N $NNODES -G $GPUS -w 'fs-mbz-gpu-038' --jobid=710588 torchrun --nnodes $NNODES --nproc_per_node 8 --rdzv-endpoint fs-mbz-gpu-038:29500 --rdzv-backend c10d --rdzv-id pp --no-python bash ./bind_and_exec.sh python test_megatron_e2e_pipeline_with_cp.py --num-nodes $NNODES --num-gpus-per-node 8  --num-microbatch $NUM_MICROBATCH --use-planner --num-batches 1 --num-tokens $NUM_TOKENS --pp-size $PP_SIZE --tp-size $TP_SIZE --cp-degree $CP_SIZE
