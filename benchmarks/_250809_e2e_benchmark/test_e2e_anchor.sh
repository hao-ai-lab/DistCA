
#!/bin/bash

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
FILE_PATH=$(realpath $test_dir/test_e2e_anchor.py)
if [ ! -f $FILE_PATH ]; then
    echo "Error: test_e2e_anchor.py not found in $FILE_PATH. Maybe you specified the wrong file path?"
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
    --num-layers 4
    --max-sample-id 4
)


set -x

# Define token sizes and replan iterations
TOKEN_SIZES=(8192 16384 32768 65536)
TOKEN_LABELS=(8k 16k 32k 64k)

for i in "${!TOKEN_SIZES[@]}"; do
    TOKEN_SIZE=${TOKEN_SIZES[$i]}
    TOKEN_LABEL=${TOKEN_LABELS[$i]}
    
    echo "# ${TOKEN_LABEL}"
    
    # Run d2 mode with different replan iterations
    torchrun ${TORCHRUN_DISTCONFIG[@]} $FILE_PATH ${EXEC_CONFIG[@]} \
        --output-file $this_dir/data/$now.anchor.v1/d2.t${TOKEN_LABEL}.p0.json --replan-iter 0 --mode d2 --num-tokens $TOKEN_SIZE
    sleep 5

    torchrun ${TORCHRUN_DISTCONFIG[@]} $FILE_PATH ${EXEC_CONFIG[@]} \
        --output-file $this_dir/data/$now.anchor.v1/d2.t${TOKEN_LABEL}.p1.json --replan-iter 1 --mode d2 --num-tokens $TOKEN_SIZE
    sleep 5
    
    # Run baseline mode
    torchrun ${TORCHRUN_DISTCONFIG[@]} $FILE_PATH ${EXEC_CONFIG[@]} \
        --output-file $this_dir/data/$now.anchor.v1/baseline.t${TOKEN_LABEL}.json --replan-iter 0 --mode baseline --num-tokens $TOKEN_SIZE
    sleep 5
done



set +x

