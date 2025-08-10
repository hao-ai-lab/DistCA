#!/bin/bash

NUM_TOKENS=32768
NUM_LAYERS=4

# Read hostfile and get current hostname
HOSTFILE="hostfile"
CURRENT_HOSTNAME=$(hostname)

# Read hosts from hostfile into an array
mapfile -t HOSTS < "$HOSTFILE"

# Find the node rank (index of current hostname in hostfile)
NODE_RANK=-1
for i in "${!HOSTS[@]}"; do
    if [[ "${HOSTS[$i]}" == "$CURRENT_HOSTNAME" ]]; then
        NODE_RANK=$i
        break
    fi
done

# Check if hostname was found
if [[ $NODE_RANK -eq -1 ]]; then
    echo "Error: Current hostname '$CURRENT_HOSTNAME' not found in hostfile"
    exit 1
fi

# Get world size (number of hosts)
WORLD_SIZE=${#HOSTS[@]}

# Get master address (first host in hostfile)
MASTER_ADDR=${HOSTS[0]}

echo "Current hostname: $CURRENT_HOSTNAME"
echo "Node rank: $NODE_RANK"
echo "World size: $WORLD_SIZE"
echo "Master address: $MASTER_ADDR"

# Run torchrun with dynamic values
NVSHMEM_IB_ENABLE_IBGDA=true 
NVTE_ALLOW_NONDETERMINISTIC_ALGO=0

this_dir=$(dirname $(realpath $0))
root_dir=$(realpath $this_dir/../..)
test_dir=$(realpath $root_dir/tests)

PYTHON_PATH=$test_dir
FILE_PATH=$(realpath $test_dir/test_e2e_combined.py)
if [ ! -f $FILE_PATH ]; then
    echo "Error: test_e2e_combined.py not found in $FILE_PATH. Maybe you specified the wrong file path?"
    exit 1
fi


TORCHRUN_DISTCONFIG=(
    --nnodes=$WORLD_SIZE
    --nproc_per_node=8 
    --node_rank=$NODE_RANK 
    --master_addr=$MASTER_ADDR 
    --master_port=29500
)

now=$(date +%Y%m%d_%H%M%S)
EXEC_CONFIG=(
    --num-nodes $WORLD_SIZE 
    --num-gpus-per-node 8 
    --tp-size 8 
    --num-tokens $NUM_TOKENS 
    --num-layers $NUM_LAYERS
    --max-sample-id 32
    --output-dir $this_dir/data/$now.v3/
)


echo "Running d2 mode"

set -x
# nsys profile -o $this_dir/data/profile.d2.%p.nsys-rep -t cuda,nvtx \
torchrun ${TORCHRUN_DISTCONFIG[@]} $FILE_PATH ${EXEC_CONFIG[@]} --mode d2 
set +x
sleep 5

echo "Running baseline mode"

set -x
# nsys profile -o $this_dir/data/profile.baseline.%p.nsys-rep -t cuda,nvtx \
torchrun ${TORCHRUN_DISTCONFIG[@]} $FILE_PATH ${EXEC_CONFIG[@]} --mode baseline
set +x