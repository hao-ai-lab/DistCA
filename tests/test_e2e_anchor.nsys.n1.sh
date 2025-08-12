# ------------------
# Baseline
# ------------------

HOSTNAME=$(hostname)
# OUTPUT_DIR="nsys-profile"
OUTPUT_DIR="profile-find-attn"

# Function to generate filename
generate_filename() {
    local num_layers=$1
    local mode=$2
    local replan_iter=$3
    local num_tokens=$4
    local node_rank=$5
    
    if [ "$mode" = "baseline" ]; then
        echo "test_e2e_anchor.n1g1.l${num_layers}.${mode}.t${num_tokens}.node${node_rank}"
    else
        echo "test_e2e_anchor.n1g1.l${num_layers}.${mode}p${replan_iter}.t${num_tokens}.node${node_rank}"
    fi
}

echo " --- Baseline --- "

TORCHARGS="--nnodes 1 --nproc_per_node 1 --master_addr ${HOSTNAME} --rdzv_backend=c10d --rdzv_conf read_timeout=120,join_timeout=1200,last_call_timeout=60"
# NUM_TOKENS=73728
# NUM_TOKENS=66560
# NUM_TOKENS=65536
# NUM_TOKENS=49152
# NUM_TOKENS=40960
# NUM_TOKENS=36864
# NUM_TOKENS=32768
NUM_TOKENS=16384
# NUM_TOKENS=8192
# NUM_TOKENS=2048
NUM_LAYERS=2
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

MASTER_PORT=29400
for NODE_RANK in 0; do
    FILENAME=$(generate_filename ${NUM_LAYERS} ${MODE} ${REPLAN_ITER} ${NUM_TOKENS} ${NODE_RANK})
    if [ ${ENABLE_NSYS} -eq 1 ]; then
        NSYS_CMD="nsys profile --force-overwrite=true -o ${OUTPUT_DIR}/${FILENAME}.nsys-rep -t cuda,nvtx"
    else
        NSYS_CMD=""
    fi
    
    echo NVSHMEM_IB_ENABLE_IBGDA=true NVTE_ALLOW_NONDETERMINISTIC_ALGO=1 ${NSYS_CMD} torchrun ${TORCHARGS} --node_rank=${NODE_RANK} --master_port=${MASTER_PORT} test_e2e_anchor.py --num-nodes=1 --num-gpus-per-node=1 --tp-size=1 --num-tokens ${NUM_TOKENS} --num-layers ${NUM_LAYERS} --mode ${MODE} --replan-iter ${REPLAN_ITER} --max-sample-id ${MAX_SAMPLE_ID} --output-file ${OUTPUT_DIR}/${FILENAME}.json ${EXTRA_ARGS}
done

echo " --- D2 without Planner --- "

MODE="d2"
REPLAN_ITER=0

MASTER_PORT=29500
for NODE_RANK in 0; do
    FILENAME=$(generate_filename ${NUM_LAYERS} ${MODE} ${REPLAN_ITER} ${NUM_TOKENS} ${NODE_RANK})
    if [ ${ENABLE_NSYS} -eq 1 ]; then
        NSYS_CMD="nsys profile --force-overwrite=true -o ${OUTPUT_DIR}/${FILENAME}.nsys-rep -t cuda,nvtx"
    else
        NSYS_CMD=""
    fi

    echo NVSHMEM_IB_ENABLE_IBGDA=true NVTE_ALLOW_NONDETERMINISTIC_ALGO=1 ${NSYS_CMD} torchrun ${TORCHARGS} --node_rank=${NODE_RANK}  --master_port=${MASTER_PORT} test_e2e_anchor.py  --num-nodes=1 --num-gpus-per-node=1 --tp-size=1 --num-tokens ${NUM_TOKENS} --num-layers ${NUM_LAYERS} --mode ${MODE} --replan-iter ${REPLAN_ITER} --max-sample-id ${MAX_SAMPLE_ID} --output-file ${OUTPUT_DIR}/${FILENAME}.json ${EXTRA_ARGS}
done


echo " --- D2 with Planner --- "

MODE="d2"
REPLAN_ITER=1

for NODE_RANK in 0; do
    FILENAME=$(generate_filename ${NUM_LAYERS} ${MODE} ${REPLAN_ITER} ${NUM_TOKENS} ${NODE_RANK})
    if [ ${ENABLE_NSYS} -eq 1 ]; then
        NSYS_CMD="nsys profile --force-overwrite=true -o ${OUTPUT_DIR}/${FILENAME}.nsys-rep -t cuda,nvtx"
    else
        NSYS_CMD=""
    fi

    echo NVSHMEM_IB_ENABLE_IBGDA=true NVTE_ALLOW_NONDETERMINISTIC_ALGO=1 ${NSYS_CMD} torchrun ${TORCHARGS} --node_rank=${NODE_RANK}  --master_port=32000 test_e2e_anchor.py --num-nodes=1 --num-gpus-per-node=1 --tp-size=1 --num-tokens ${NUM_TOKENS} --num-layers ${NUM_LAYERS} --mode ${MODE} --replan-iter ${REPLAN_ITER} --max-sample-id ${MAX_SAMPLE_ID} --output-file ${OUTPUT_DIR}/${FILENAME}.json ${EXTRA_ARGS}
done