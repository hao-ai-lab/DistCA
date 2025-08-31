# for repeat in 1; do
#     for num_tokens in 65536 131072 262144 524288; do
        
#         if [ "$num_tokens" -eq 262144 ]; then
#             num_layers=8
#             elongate_factor=4
#             buffer_size=6
#         elif [ "$num_tokens" -eq 524288 ]; then
#             num_layers=4
#             elongate_factor=8
#             buffer_size=6
#         elif [ "$num_tokens" -eq 1048576 ]; then
#             elongate_factor=2
#             buffer_size=6
#         else
#             num_layers=8
#             buffer_size=6
#         fi

#         for nodes in 8 16 32; do
#             for batch_size in 1 2 4; do
#                 for cp_size in 32 16 8 4 2 1; do
#                     if [ $cp_size -gt $nodes ]; then
#                         continue
#                     fi
#                     MODE=wlbllm ELONGATE_FACTOR=$elongate_factor BATCH_SIZE=$batch_size NUM_TOKENS=$num_tokens MAX_SAMPLE_ID=50 TP_SIZE=8 CP_SIZE=$cp_size NUM_LAYERS=$num_layers EXPERIMENT_REPEAT_TIMES=3 EXPERIMENT_WARMUP_TIMES=3 sbatch --nodes $nodes --partition=lowprio --qos=lowprio test_e2e_combined.slurm.sh
#                 done
#                 for repeat in 1 1 1; do
#                     EXPERIMENT_NVSHMEM_BUFFER_SIZE_GB=$buffer_size MODE=d2 ELONGATE_FACTOR=$elongate_factor BATCH_SIZE=$batch_size NUM_TOKENS=$num_tokens MAX_SAMPLE_ID=50 TP_SIZE=8 CP_SIZE=$cp_size NUM_LAYERS=$num_layers EXPERIMENT_REPEAT_TIMES=3 EXPERIMENT_WARMUP_TIMES=3 sbatch --nodes $nodes --partition=lowprio --qos=lowprio test_e2e_combined.slurm.sh
#                 done
#             done
#         done
#     done
# done


for repeat in 1; do
    for num_tokens in 65536 131072 262144 524288; do
        num_layers=4
        
        if [ "$num_tokens" -eq 131072 ]; then
            elongate_factor=2
        elif [ "$num_tokens" -eq 262144 ]; then
            elongate_factor=4
        elif [ "$num_tokens" -eq 524288 ]; then
            elongate_factor=8
        else
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
                            buffer_size=29
                        elif [ "$batch_size" -eq 2 ]; then
                            buffer_size=13
                        else
                            buffer_size=11
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
                            buffer_size=29
                        elif [ "$batch_size" -eq 2 ]; then
                            buffer_size=13
                        else
                            buffer_size=5
                        fi
                    elif [ "$num_tokens" -eq 524288 ]; then
                        if [ "$batch_size" -eq 1 ]; then
                            buffer_size=43
                        elif [ "$batch_size" -eq 2 ]; then
                            buffer_size=19
                        else
                            buffer_size=10
                        fi
                    fi
                fi

                for repeat in 1 1 1; do
                    EXPERIMENT_NVSHMEM_BUFFER_SIZE_GB=$buffer_size MODE=d2 ELONGATE_FACTOR=$elongate_factor BATCH_SIZE=$batch_size NUM_TOKENS=$num_tokens MAX_SAMPLE_ID=50 TP_SIZE=8 CP_SIZE=$cp_size NUM_LAYERS=$num_layers EXPERIMENT_REPEAT_TIMES=3 EXPERIMENT_WARMUP_TIMES=3 sbatch --nodes $nodes --partition=lowprio --qos=lowprio test_e2e_combined.slurm.sh
                done
            done
        done
    done
done
