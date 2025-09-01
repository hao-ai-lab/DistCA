

for repeat in 1 1 1; do
    for num_layers in 4 32; do
        for num_tokens in 65536 131072 262144 524288; do
            num_layers=4
            # Calculate elongate_factor based on num_tokens
            # Formula: num_tokens / 64 / 1024
            elongate_factor=$((num_tokens / 65536))
            
            # Ensure we have at least elongate_factor=1 for the smallest case
            if [ "$elongate_factor" -eq 0 ]; then
                elongate_factor=1
            fi

            for nodes in 32; do
                for batch_size in 1 2 4 8 16 32; do
                    # Look up buffer size from table based on num_tokens, nodes, batch_size
                    if [ "$num_tokens" -eq 131072 ]; then
                        buffer_size=8    
                    
                    elif [ "$num_tokens" -eq 262144 ]; then
                        buffer_size=20
                        
                    elif [ "$num_tokens" -eq 524288 ]; then
                        buffer_size=20
                    fi

                    # Run D2
                    # CUR_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
                    OUTPUT_DIR_PREFIX="$HOME/jd/d2/benchmarks/_250830_large_scale_v3/logs.v3" MODE=d2 ELONGATE_FACTOR=$elongate_factor BATCH_SIZE=$batch_size NUM_TOKENS=$num_tokens MAX_SAMPLE_ID=50 TP_SIZE=8 CP_SIZE=1 NUM_LAYERS=$num_layers EXPERIMENT_REPEAT_TIMES=3 EXPERIMENT_WARMUP_TIMES=3 EXPERIMENT_WARMUP_TIMEOUT_SEC=90 EXPERIMENT_TIMEOUT_WARMUP_START=120 EXPERIMENT_NVSHMEM_BUFFER_SIZE_GB=$buffer_size sbatch --nodes $nodes --job-name=d2-v3 --partition=lowprio --qos=lowprio test_e2e_combined.slurm.sh
                    sleep 2
                    # OUTPUT_DIR_PREFIX="$CUR_DIR/logs.v3" MODE=d2 ELONGATE_FACTOR=$elongate_factor BATCH_SIZE=$batch_size NUM_TOKENS=$num_tokens MAX_SAMPLE_ID=50 TP_SIZE=8 CP_SIZE=1 NUM_LAYERS=$num_layers EXPERIMENT_REPEAT_TIMES=3 EXPERIMENT_WARMUP_TIMES=3 EXPERIMENT_WARMUP_TIMEOUT_SEC=90 EXPERIMENT_TIMEOUT_WARMUP_START=120 EXPERIMENT_NVSHMEM_BUFFER_SIZE_GB=$buffer_size bash test_e2e_combined.slurm.sh
                    # exit 1
                    
                done
            done
        done
    done
done 