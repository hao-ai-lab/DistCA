

CURDIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export OUTPUT_DIR_PREFIX="$CURDIR/logs.v4" 


# if OUTPUT_DIR_PREFIX exists, execute python analyze_v4.py
if [ -d "$OUTPUT_DIR_PREFIX" ]; then
    # cd $CURDIR && pwd && python analyze_v4.py && cd -
    success_eids=$(cat "$CURDIR/success_eids.txt")
    echo "Skipping success_eids=$success_eids"
else
    echo "OUTPUT_DIR_PREFIX=$OUTPUT_DIR_PREFIX does not exist. Not running analyze_v4.py"
fi


for repeat in 1; do
    for num_tokens in 262144 131072 65536 524288; do
        for num_layers in 4 32; do
            # Calculate elongate_factor based on num_tokens
            # Formula: num_tokens / 64 / 1024
            elongate_factor=$((num_tokens / 65536))
            
            # Ensure we have at least elongate_factor=1 for the smallest case
            if [ "$elongate_factor" -eq 0 ]; then
                elongate_factor=1
            fi

            for nodes in 32 16 8; do
                for batch_size in 1 2 4 8 16 32; do
                    # Look up buffer size from table based on num_tokens, nodes, batch_size
                    # Run D2
                    for buffer_size in 1 2 4; do

                        d2_eid="${nodes}_${num_tokens}_${batch_size}_d2_1_${num_layers}"

                        if [[ "$success_eids" =~ "$d2_eid" ]]; then
                            echo "Skip: $d2_eid"
                            continue
                        fi
                        echo "Should run: $d2_eid"
                        # continue
                        
                        OUTPUT_DIR_PREFIX=$OUTPUT_DIR_PREFIX MODE=d2 ELONGATE_FACTOR=$elongate_factor BATCH_SIZE=$batch_size NUM_TOKENS=$num_tokens MAX_SAMPLE_ID=50 TP_SIZE=8 CP_SIZE=1 NUM_LAYERS=$num_layers EXPERIMENT_REPEAT_TIMES=3 EXPERIMENT_WARMUP_TIMES=3 EXPERIMENT_WARMUP_TIMEOUT_SEC=90 EXPERIMENT_TIMEOUT_WARMUP_START=120 EXPERIMENT_NVSHMEM_BUFFER_SIZE_GB=$buffer_size sbatch --nodes $nodes --job-name=d2-v4 --partition=lowprio --qos=lowprio test_e2e_combined.slurm.sh
                        # SLURM_JOB_ID=676824 SLURM_NODELIST="fs-mbz-gpu-[004,036,064,138,041,184,124,144,137,143,153,217,209,272,268,294,341,279,311,369,402,444,488,441,481,460,649,646,753,805,743,880]" DRY_RUN=1 SLURM_GPUS_ON_NODE=8 SLURM_NNODES=$nodes OUTPUT_DIR_PREFIX=$OUTPUT_DIR_PREFIX MODE=d2 ELONGATE_FACTOR=$elongate_factor BATCH_SIZE=$batch_size NUM_TOKENS=$num_tokens MAX_SAMPLE_ID=50 TP_SIZE=8 CP_SIZE=1 NUM_LAYERS=$num_layers EXPERIMENT_REPEAT_TIMES=3 EXPERIMENT_WARMUP_TIMES=3 EXPERIMENT_WARMUP_TIMEOUT_SEC=90 EXPERIMENT_TIMEOUT_WARMUP_START=120 EXPERIMENT_NVSHMEM_BUFFER_SIZE_GB=$buffer_size bash test_e2e_combined.salloc-exp.sh
                        exit 1
                    done

                    # # Run WLBLLM
                    # for cp_size in 32 16 8 4 2 1; do
                    #     if [ $cp_size -gt $nodes ]; then
                    #         continue
                    #     fi

                    #     wlbllm_eid="${nodes}_${num_tokens}_${batch_size}_wlbllm_${cp_size}_${num_layers}"
                    #     if [[ "$success_eids" =~ "$wlbllm_eid" ]]; then
                    #         echo "Skip: $wlbllm_eid"
                    #         continue
                    #     fi
                    #     echo "Should run: $wlbllm_eid"
                    #     # continue

                    #     OUTPUT_DIR_PREFIX=$OUTPUT_DIR_PREFIX MODE=wlbllm ELONGATE_FACTOR=$elongate_factor BATCH_SIZE=$batch_size NUM_TOKENS=$num_tokens MAX_SAMPLE_ID=50 TP_SIZE=8 CP_SIZE=$cp_size NUM_LAYERS=$num_layers EXPERIMENT_REPEAT_TIMES=3 EXPERIMENT_WARMUP_TIMES=3 EXPERIMENT_WARMUP_TIMEOUT_SEC=90 EXPERIMENT_TIMEOUT_WARMUP_START=120 EXPERIMENT_NVSHMEM_BUFFER_SIZE_GB=$buffer_size sbatch --nodes $nodes --job-name=d2-v4 --partition=lowprio --qos=lowprio test_e2e_combined.slurm.sh
                    # done
                done
            done
        done
    done
done