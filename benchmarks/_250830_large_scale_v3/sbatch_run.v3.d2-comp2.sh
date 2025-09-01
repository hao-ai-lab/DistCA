CURDIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
success_eids=$(cat "$CURDIR/success_eids.txt")

for repeat in 1; do
    for num_tokens in 65536 131072 262144 524288; do
        for num_layers in 4 32; do

            # Calculate elongate_factor based on num_tokens
            # Formula: num_tokens / 64 / 1024
            elongate_factor=$((num_tokens / 65536))
            
            # Ensure we have at least elongate_factor=1 for the smallest case
            if [ "$elongate_factor" -eq 0 ]; then
                elongate_factor=1
            fi

            for nodes in 8 16 32; do
                for batch_size in 1 2 4 8 16 32; do
                    # Look up buffer size from table based on num_tokens, nodes, batch_size
                    buffer_size=16
                    if [ "$nodes" -eq 8 ]; then
                        if [ "$num_tokens" -eq 131072 ]; then
                            buffer_size=4
                        elif [ "$num_tokens" -eq 262144 ]; then
                            buffer_size=8
                        elif [ "$num_tokens" -eq 524288 ]; then
                            buffer_size=16
                        fi
                    elif [ "$nodes" -eq 16 ]; then
                        if [ "$num_tokens" -eq 131072 ]; then
                            buffer_size=4
                        elif [ "$num_tokens" -eq 262144 ]; then
                            buffer_size=8
                        elif [ "$num_tokens" -eq 524288 ]; then
                            buffer_size=16
                        fi
                    elif [ "$nodes" -eq 32 ]; then
                        if [ "$num_tokens" -eq 131072 ]; then
                            buffer_size=16
                        elif [ "$num_tokens" -eq 262144 ]; then
                            buffer_size=16
                        elif [ "$num_tokens" -eq 524288 ]; then
                            buffer_size=16
                        fi
                    fi

                    # success_eids
                    # df_display['eid'] = df_display['nnodes'].astype(str) + "_" + df_display['NUM_TOKENS'].astype(str) + "_" + df_display['BATCH_SIZE'].astype(str) + "_" + df_display['MODE'].astype(str) + "_" + df_display['CP_SIZE'].astype(str) + "_" + df_display['num_layers'].astype(str)

                    # d2_eid="${nodes}_${num_tokens}_${batch_size}_d2_1_${num_layers}"
                    # wlbllm_eid="${nodes}_${num_tokens}_${batch_size}_wlbllm_${cp_size}_${num_layers}"

                    

                    # Run D2
                    
                    for repeat in 1 1 1; do
                        d2_eid="${nodes}_${num_tokens}_${batch_size}_d2_1_${num_layers}"
                        if [[ "$success_eids" =~ "$d2_eid" ]]; then
                            echo "Skip: $d2_eid"
                            continue
                        fi
                        echo "Should run: $d2_eid"
                        # continue
                        OUTPUT_DIR_PREFIX="/mnt/weka/home/yonghao.zhuang/jd/d2/benchmarks/_250830_large_scale_v3/logs.v3" MODE=d2 ELONGATE_FACTOR=$elongate_factor BATCH_SIZE=$batch_size NUM_TOKENS=$num_tokens MAX_SAMPLE_ID=50 TP_SIZE=8 CP_SIZE=1 NUM_LAYERS=$num_layers EXPERIMENT_REPEAT_TIMES=3 EXPERIMENT_WARMUP_TIMES=3 EXPERIMENT_WARMUP_TIMEOUT_SEC=90 EXPERIMENT_TIMEOUT_WARMUP_START=120 EXPERIMENT_NVSHMEM_BUFFER_SIZE_GB=$buffer_size sbatch --nodes $nodes --job-name=d2-v3 --partition=lowprio --qos=lowprio test_e2e_combined.slurm.sh
                    done

                    # Run WLBLLM
                    for cp_size in 32 16 8 4 2 1; do
                        if [ $cp_size -gt $nodes ]; then
                            continue
                        fi
                        wlbllm_eid="${nodes}_${num_tokens}_${batch_size}_wlbllm_${cp_size}_${num_layers}"
                        if [[ "$success_eids" =~ "$wlbllm_eid" ]]; then
                            echo "Skip: $wlbllm_eid"
                            continue
                        fi
                        echo "Should run: $wlbllm_eid"
                        # continue
                        OUTPUT_DIR_PREFIX="/mnt/weka/home/yonghao.zhuang/jd/d2/benchmarks/_250830_large_scale_v3/logs.v3" MODE=wlbllm ELONGATE_FACTOR=$elongate_factor BATCH_SIZE=$batch_size NUM_TOKENS=$num_tokens MAX_SAMPLE_ID=50 TP_SIZE=8 CP_SIZE=$cp_size NUM_LAYERS=$num_layers EXPERIMENT_REPEAT_TIMES=3 EXPERIMENT_WARMUP_TIMES=3 EXPERIMENT_WARMUP_TIMEOUT_SEC=90 EXPERIMENT_TIMEOUT_WARMUP_START=120 EXPERIMENT_NVSHMEM_BUFFER_SIZE_GB=$buffer_size sbatch --nodes $nodes --job-name=d2-v3 --partition=lowprio --qos=lowprio test_e2e_combined.slurm.sh

                    done
                    
                    sleep 1

                
                done
            done
        done
    done
done