# ------------------
# Baseline
# ------------------

HOSTNAME=$(hostname)

echo " --- Baseline --- "

TORCHARGS="--nnodes 2 --nproc_per_node 8 --master_addr ${HOSTNAME}"
# NUM_TOKENS=73728
# NUM_TOKENS=66560
# NUM_TOKENS=65536
# NUM_TOKENS=49152
# NUM_TOKENS=40960
# NUM_TOKENS=36864
NUM_TOKENS=32768
# NUM_TOKENS=8192
NUM_LAYERS=4
# NUM_LAYERS=32
MODE="baseline"
REPLAN_ITER=0
MAX_SAMPLE_ID=3
# ENABLE_NSYS=0
ENABLE_NSYS=1
if [ ${ENABLE_NSYS} -eq 1 ]; then
    EXTRA_ARGS=""
else
    EXTRA_ARGS="--force-exit"
fi

for NODE_RANK in 0 1; do
    FILENAME="test_e2e_anchor.n2g8.l${NUM_LAYERS}.${MODE}.t${NUM_TOKENS}.node${NODE_RANK}"
    if [ ${ENABLE_NSYS} -eq 1 ]; then
        NSYS_CMD="nsys profile --force-overwrite=true -o nsys-profile/${FILENAME}.nsys-rep -t cuda,nvtx"
    else
        NSYS_CMD=""
    fi
    
    echo NVSHMEM_IB_ENABLE_IBGDA=true NVTE_ALLOW_NONDETERMINISTIC_ALGO=0 ${NSYS_CMD} torchrun ${TORCHARGS} --node_rank=${NODE_RANK} --master_port=29400 test_e2e_anchor.py --num-nodes=2 --num-gpus-per-node=8 --tp-size=8 --num-tokens ${NUM_TOKENS} --num-layers ${NUM_LAYERS} --mode ${MODE} --replan-iter ${REPLAN_ITER} --max-sample-id ${MAX_SAMPLE_ID} --output-file nsys-profile/${FILENAME}.json ${EXTRA_ARGS}
done

echo " --- D2 without Planner --- "

MODE="d2"
REPLAN_ITER=0

for NODE_RANK in 0 1; do
    FILENAME="test_e2e_anchor.n2g8.l${NUM_LAYERS}.${MODE}p${REPLAN_ITER}.t${NUM_TOKENS}.node${NODE_RANK}"
    if [ ${ENABLE_NSYS} -eq 1 ]; then
        NSYS_CMD="nsys profile --force-overwrite=true -o nsys-profile/${FILENAME}.nsys-rep -t cuda,nvtx"
    else
        NSYS_CMD=""
    fi

    echo NVSHMEM_IB_ENABLE_IBGDA=true NVTE_ALLOW_NONDETERMINISTIC_ALGO=0 ${NSYS_CMD} torchrun ${TORCHARGS} --node_rank=${NODE_RANK}  --master_port=29500 test_e2e_anchor.py  --num-nodes=2 --num-gpus-per-node=8 --tp-size=8 --num-tokens ${NUM_TOKENS} --num-layers ${NUM_LAYERS} --mode ${MODE} --replan-iter ${REPLAN_ITER} --max-sample-id ${MAX_SAMPLE_ID} --output-file nsys-profile/${FILENAME}.json ${EXTRA_ARGS}
done


echo " --- D2 with Planner --- "

MODE="d2"
REPLAN_ITER=1

for NODE_RANK in 0 1; do
    FILENAME="test_e2e_anchor.n2g8.l${NUM_LAYERS}.${MODE}p${REPLAN_ITER}.t${NUM_TOKENS}.node${NODE_RANK}"
    if [ ${ENABLE_NSYS} -eq 1 ]; then
        NSYS_CMD="nsys profile --force-overwrite=true -o nsys-profile/${FILENAME}.nsys-rep -t cuda,nvtx"
    else
        NSYS_CMD=""
    fi

    echo NVSHMEM_IB_ENABLE_IBGDA=true NVTE_ALLOW_NONDETERMINISTIC_ALGO=0 ${NSYS_CMD} torchrun ${TORCHARGS} --node_rank=${NODE_RANK}  --master_port=32000 test_e2e_anchor.py --num-nodes=2 --num-gpus-per-node=8 --tp-size=8 --num-tokens ${NUM_TOKENS} --num-layers ${NUM_LAYERS} --mode ${MODE} --replan-iter ${REPLAN_ITER} --max-sample-id ${MAX_SAMPLE_ID} --output-file nsys-profile/${FILENAME}.json ${EXTRA_ARGS}
done