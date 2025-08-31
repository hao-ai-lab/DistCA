for repeat in 1; do
    for num_tokens in 65536 131072 262144 524288; do
        num_layers=4
        # Calculate elongate_factor based on num_tokens
        # Formula: num_tokens / 64 / 1024
        elongate_factor=$((num_tokens / 65536))
        
        # Ensure we have at least elongate_factor=1 for the smallest case
        if [ "$elongate_factor" -eq 0 ]; then
            elongate_factor=1
        fi

        for nodes in 8 16 32; do
            for batch_size in 1 2 4; do
                # Look up buffer size from table based on num_tokens, nodes, batch_size
                if [ "$nodes" -eq 8 ]; then
                    if [ "$num_tokens" -eq 131072 ]; then
                        if [ "$batch_size" -eq 1 ]; then
                            buffer_size=4
                        else
                            buffer_size=3
                        fi
                    elif [ "$num_tokens" -eq 262144 ]; then
                        if [ "$batch_size" -eq 1 ]; then
                            buffer_size=7
                        else
                            buffer_size=6
                        fi
                    elif [ "$num_tokens" -eq 524288 ]; then
                        if [ "$batch_size" -eq 1 ]; then
                            buffer_size=15
                        else
                            buffer_size=12
                        fi
                    fi
                elif [ "$nodes" -eq 16 ]; then
                    if [ "$num_tokens" -eq 131072 ]; then
                        if [ "$batch_size" -eq 1 ]; then
                            buffer_size=6
                        elif [ "$batch_size" -eq 2 ]; then
                            buffer_size=4
                        else
                            buffer_size=3
                        fi
                    elif [ "$num_tokens" -eq 262144 ]; then
                        if [ "$batch_size" -eq 1 ]; then
                            buffer_size=14
                        elif [ "$batch_size" -eq 2 ]; then
                            buffer_size=7
                        else
                            buffer_size=6
                        fi
                    elif [ "$num_tokens" -eq 524288 ]; then
                        if [ "$batch_size" -eq 1 ]; then
                            buffer_size=20
                        elif [ "$batch_size" -eq 2 ]; then
                            buffer_size=12
                        else
                            buffer_size=10
                        fi
                    fi
                elif [ "$nodes" -eq 32 ]; then
                    if [ "$num_tokens" -eq 131072 ]; then
                        if [ "$batch_size" -eq 1 ]; then
                            buffer_size=13
                        elif [ "$batch_size" -eq 2 ]; then
                            buffer_size=5
                        else
                            buffer_size=3
                        fi
                    elif [ "$num_tokens" -eq 262144 ]; then
                        if [ "$batch_size" -eq 1 ]; then
                            buffer_size=20
                        elif [ "$batch_size" -eq 2 ]; then
                            buffer_size=12
                        else
                            buffer_size=5
                        fi
                    elif [ "$num_tokens" -eq 524288 ]; then
                        if [ "$batch_size" -eq 1 ]; then
                            buffer_size=20
                        elif [ "$batch_size" -eq 2 ]; then
                            buffer_size=16
                        else
                            buffer_size=16
                        fi
                    fi
                fi

                # Run D2
                for repeat in 1 1 1; do
                    OUTPUT_DIR_PREFIX="/mnt/weka/home/yonghao.zhuang/jd/d2/benchmarks/_250830_large_scale_v3/logs.v1" MODE=d2 ELONGATE_FACTOR=$elongate_factor BATCH_SIZE=$batch_size NUM_TOKENS=$num_tokens MAX_SAMPLE_ID=50 TP_SIZE=8 CP_SIZE=1 NUM_LAYERS=$num_layers EXPERIMENT_REPEAT_TIMES=3 EXPERIMENT_WARMUP_TIMES=3 EXPERIMENT_WARMUP_TIMEOUT_SEC=90 EXPERIMENT_TIMEOUT_WARMUP_START=120 EXPERIMENT_NVSHMEM_BUFFER_SIZE_GB=$buffer_size sbatch --nodes $nodes --job-name=d2-v3 --partition=lowprio --qos=lowprio test_e2e_combined.slurm.sh
                done

                # Run WLBLLM
                for cp_size in 32 16 8 4 2 1; do
                    if [ $cp_size -gt $nodes ]; then
                        continue
                    fi
                    OUTPUT_DIR_PREFIX="/mnt/weka/home/yonghao.zhuang/jd/d2/benchmarks/_250830_large_scale_v3/logs.v1" MODE=wlbllm ELONGATE_FACTOR=$elongate_factor BATCH_SIZE=$batch_size NUM_TOKENS=$num_tokens MAX_SAMPLE_ID=50 TP_SIZE=8 CP_SIZE=$cp_size NUM_LAYERS=$num_layers EXPERIMENT_REPEAT_TIMES=3 EXPERIMENT_WARMUP_TIMES=3 EXPERIMENT_WARMUP_TIMEOUT_SEC=90 EXPERIMENT_TIMEOUT_WARMUP_START=120 EXPERIMENT_NVSHMEM_BUFFER_SIZE_GB=$buffer_size sbatch --nodes $nodes --job-name=d2-v3 --partition=lowprio --qos=lowprio test_e2e_combined.slurm.sh

                done

                
            done
        done
    done
done
