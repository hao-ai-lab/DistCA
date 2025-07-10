#!/bin/bash

#SBATCH --output=wandb_%j.log
#SBATCH --error=wandb_%j.err
#SBATCH --time=06:00:00
#SBATCH -q regular
#SBATCH -C cpu
#SBATCH -N 1
#SBATCH --mem=128G

export OMP_NUM_THREADS=128
export OMP_PLACES=threads
export OMP_PROC_BIND=spread

module load conda
conda activate d2

MAX_LENGTH=${MAX_LENGTH:-256}
UPSAMPLE=${UPSAMPLE:-8}

# Run the Python script
python zrun_wandb_optimizer_wlbupsample.py \
    --max_length "$MAX_LENGTH" \
    --upsample_long_factor "$UPSAMPLE" \
    --num_total_devices $NUM_TOTAL_DEVICES \
    --max_num_workers_attnserver $MAX_NUM_WORKERS_ATTNSERVER
