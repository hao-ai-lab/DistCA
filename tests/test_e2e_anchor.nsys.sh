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
        echo "test_e2e_anchor.n2g8.l${num_layers}.${mode}.t${num_tokens}.node${node_rank}"
    else
        echo "test_e2e_anchor.n2g8.l${num_layers}.${mode}p${replan_iter}.t${num_tokens}.node${node_rank}"
    fi
}

NVTE_ALLOW_NONDETERMINISTIC_ALGO=1

echo " --- Baseline --- "

TORCHARGS="--nnodes 2 --nproc_per_node 8 --rdzv_backend=c10d --rdzv_endpoint=${HOSTNAME}:29400 --rdzv_conf read_timeout=120,join_timeout=1200,last_call_timeout=60"
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
    FILENAME=$(generate_filename ${NUM_LAYERS} ${MODE} ${REPLAN_ITER} ${NUM_TOKENS} ${NODE_RANK})
    if [ ${ENABLE_NSYS} -eq 1 ]; then
        NSYS_CMD="nsys profile --force-overwrite=true -o ${OUTPUT_DIR}/${FILENAME}.nsys-rep -t cuda,nvtx"
    else
        NSYS_CMD=""
    fi
    
    echo NVSHMEM_IB_ENABLE_IBGDA=true NVTE_ALLOW_NONDETERMINISTIC_ALGO=${NVTE_ALLOW_NONDETERMINISTIC_ALGO} ${NSYS_CMD} torchrun ${TORCHARGS} --node_rank=${NODE_RANK} test_e2e_anchor.py --num-nodes=2 --num-gpus-per-node=8 --tp-size=8 --num-tokens ${NUM_TOKENS} --num-layers ${NUM_LAYERS} --mode ${MODE} --replan-iter ${REPLAN_ITER} --max-sample-id ${MAX_SAMPLE_ID} --output-file ${OUTPUT_DIR}/${FILENAME}.json ${EXTRA_ARGS}
done

echo " --- D2 without Planner --- "

MODE="d2"
REPLAN_ITER=0

TORCHARGS="--nnodes 2 --nproc_per_node 8 --rdzv_backend=c10d --rdzv_endpoint=${HOSTNAME}:29500 --rdzv_conf read_timeout=120,join_timeout=1200,last_call_timeout=60"
for NODE_RANK in 0 1; do
    FILENAME=$(generate_filename ${NUM_LAYERS} ${MODE} ${REPLAN_ITER} ${NUM_TOKENS} ${NODE_RANK})
    if [ ${ENABLE_NSYS} -eq 1 ]; then
        NSYS_CMD="nsys profile --force-overwrite=true -o ${OUTPUT_DIR}/${FILENAME}.nsys-rep -t cuda,nvtx"
    else
        NSYS_CMD=""
    fi

    echo NVSHMEM_IB_ENABLE_IBGDA=true NVTE_ALLOW_NONDETERMINISTIC_ALGO=${NVTE_ALLOW_NONDETERMINISTIC_ALGO} ${NSYS_CMD} torchrun ${TORCHARGS} --node_rank=${NODE_RANK} test_e2e_anchor.py  --num-nodes=2 --num-gpus-per-node=8 --tp-size=8 --num-tokens ${NUM_TOKENS} --num-layers ${NUM_LAYERS} --mode ${MODE} --replan-iter ${REPLAN_ITER} --max-sample-id ${MAX_SAMPLE_ID} --output-file ${OUTPUT_DIR}/${FILENAME}.json ${EXTRA_ARGS}
done


echo " --- D2 with Planner --- "

MODE="d2"
REPLAN_ITER=1

TORCHARGS="--nnodes 2 --nproc_per_node 8 --rdzv_backend=c10d --rdzv_endpoint=${HOSTNAME}:32000 --rdzv_conf read_timeout=120,join_timeout=1200,last_call_timeout=60"
for NODE_RANK in 0 1; do
    FILENAME=$(generate_filename ${NUM_LAYERS} ${MODE} ${REPLAN_ITER} ${NUM_TOKENS} ${NODE_RANK})
    if [ ${ENABLE_NSYS} -eq 1 ]; then
        NSYS_CMD="nsys profile --force-overwrite=true -o ${OUTPUT_DIR}/${FILENAME}.nsys-rep -t cuda,nvtx"
    else
        NSYS_CMD=""
    fi

    echo NVSHMEM_IB_ENABLE_IBGDA=true NVTE_ALLOW_NONDETERMINISTIC_ALGO=${NVTE_ALLOW_NONDETERMINISTIC_ALGO} ${NSYS_CMD} torchrun ${TORCHARGS} --node_rank=${NODE_RANK} test_e2e_anchor.py --num-nodes=2 --num-gpus-per-node=8 --tp-size=8 --num-tokens ${NUM_TOKENS} --num-layers ${NUM_LAYERS} --mode ${MODE} --replan-iter ${REPLAN_ITER} --max-sample-id ${MAX_SAMPLE_ID} --output-file ${OUTPUT_DIR}/${FILENAME}.json ${EXTRA_ARGS}
done