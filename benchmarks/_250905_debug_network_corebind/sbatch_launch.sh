export NUM_TOKENS=16384
export NUM_LAYERS=4
export MAX_SAMPLE_ID=5
export BATCH_SIZE=2
export EXPERIMENT_WARMUP_TIMES=1 
export NNODES=2
export DRY_RUN=0
export ELONGATE_FACTOR=1

export ENABLE_NSYS=0
export SHOULD_ADD_DEBUG_CASES=1
export OUTPUT_DIR_PREFIX="/mnt/weka/home/yonghao.zhuang/jd/d2/benchmarks/_250905_debug_network_corebind/logs"

priority="--partition=lowprio --qos=lowprio"
# priority="--partition=main --qos=iq"


export EXPERIMENT_DEBUG_SET_METADATA_TRANSFER_SIZE_TO_0=0
sbatch $priority --nodes=$NNODES test_e2e_combined.slurm.v5.sh
# sbatch $priority --nodes=$NNODES test_e2e_combined.slurm.v2.sh
# sbatch $priority --nodes=$NNODES test_e2e_combined.slurm.v3.sh

# export EXPERIMENT_DEBUG_SET_METADATA_TRANSFER_SIZE_TO_0=1
# sbatch $priority --nodes=$NNODES  test_e2e_combined.slurm.v3.sh




# export NUM_TOKENS=131072
# export NUM_LAYERS=32
# export MAX_SAMPLE_ID=10
# export BATCH_SIZE=2
# export ENABLE_NSYS=0
# export EXPERIMENT_WARMUP_TIMES=1 
# export NNODES=8
# export DRY_RUN=0

# export SHOULD_ADD_DEBUG_CASES=1
# export OUTPUT_DIR_PREFIX="/mnt/weka/home/yonghao.zhuang/jd/d2/benchmarks/_250905_debug_network_corebind/logs"


# export EXPERIMENT_DEBUG_SET_METADATA_TRANSFER_SIZE_TO_0=0
# sbatch $priority --nodes=$NNODES test_e2e_combined.slurm.v3.sh

# export EXPERIMENT_DEBUG_SET_METADATA_TRANSFER_SIZE_TO_0=1
# sbatch $priority --nodes=$NNODES  test_e2e_combined.slurm.v3.sh

# # sbatch --nodes=$NNODES test_e2e_combined.slurm.v2.sh