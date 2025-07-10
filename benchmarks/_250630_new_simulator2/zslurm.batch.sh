#!/bin/bash
set -x
for max_length in 1024 512 256 128 64; do
    
    factors=(64 32 16 8 4 2 1)

    for upsample_factor in "${factors[@]}"; do
        num_total_devices=128
        max_num_workers_attnserver=16
        job_name="wandb_ctx${max_length}_f${upsample_factor}_gpu${num_total_devices}_wkrs${max_num_workers_attnserver}"
        
        echo "Submitting $job_name"
        sbatch --job-name="$job_name" \
               --export=MAX_LENGTH=$max_length,UPSAMPLE=$upsample_factor,NUM_TOTAL_DEVICES=$num_total_devices,MAX_NUM_WORKERS_ATTNSERVER=$max_num_workers_attnserver \
               zslurm.sh
    done
done
set +x