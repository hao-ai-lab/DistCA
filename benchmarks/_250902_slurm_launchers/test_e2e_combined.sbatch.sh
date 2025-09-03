# Simple example of how to use the `test_e2e_combined.slurm.sh` script.

# # ✅ Passed
# sbatch test_e2e_combined.slurm.sh

# MODE=d2 BATCH_SIZE=2 PP_SIZE=1 TP_SIZE=8 NUM_TOKENS=65536 MAX_SAMPLE_ID=3 NUM_LAYERS=4 sbatch test_e2e_combined.slurm.sh
# MODE=d2 TP_SIZE=8 BATCH_SIZE=2 sbatch --nodes 2 test_e2e_combined.slurm.sh

# MODE=wlbllm CP_SIZE=4 TP_SIZE=8 sbatch --nodes 4 test_e2e_combined.slurm.sh
# MODE=wlbllm CP_SIZE=2 TP_SIZE=8 sbatch --nodes 4 test_e2e_combined.slurm.sh
# MODE=wlbllm CP_SIZE=4 TP_SIZE=8 NUM_TOKENS=131072 sbatch --nodes 16 test_e2e_combined.slurm.sh

# MODE=wlbllm CP_SIZE=8 TP_SIZE=8 NUM_TOKENS=131072 sbatch --nodes 32 test_e2e_combined.slurm.sh
# MODE=d2 BATCH_SIZE=2 PP_SIZE=1 TP_SIZE=8 NUM_TOKENS=65536 MAX_SAMPLE_ID=3 NUM_LAYERS=4 sbatch --nodes 32 test_e2e_combined.slurm.sh

# ⚠️ Finished running.
# - All WLBLLM has no troulbe finishing.
# - All D2 fails, except 
#      - 256k bs = 2 nlayer = 4 buffer_size = 4 ( 12, 8 will fail??? why??)
#      - 256k bs = 4 nlayer = 4 buffer_size = 12 (4, 8 will fail???)

for num_tokens in 262144; do
    for nodes in 32; do
        for batch_size in 1 2 4; do
            for cp_size in 32 16 8 4 2 1; do
                if [ $cp_size -gt $nodes ]; then
                    continue
                fi
                # MODE=wlbllm BATCH_SIZE=$batch_size NUM_TOKENS=$num_tokens MAX_SAMPLE_ID=5 TP_SIZE=8 CP_SIZE=$cp_size NUM_LAYERS=4 EXPERIMENT_REPEAT_TIMES=1 EXPERIMENT_WARMUP_TIMES=1 sbatch --nodes $nodes --partition=lowprio --qos=lowprio --time=00:10:00 test_e2e_combined.slurm.sh
                # sleep 5
            done
            EXPERIMENT_NVSHMEM_BUFFER_SIZE_GB="8" MODE=d2 BATCH_SIZE=$batch_size NUM_TOKENS=$num_tokens MAX_SAMPLE_ID=5 TP_SIZE=8 CP_SIZE=$cp_size NUM_LAYERS=4 EXPERIMENT_REPEAT_TIMES=1 EXPERIMENT_WARMUP_TIMES=1 sbatch --nodes $nodes --partition=lowprio --qos=lowprio --time=00:10:00 test_e2e_combined.slurm.sh

            sleep 5

            EXPERIMENT_NVSHMEM_BUFFER_SIZE_GB="4" MODE=d2 BATCH_SIZE=$batch_size NUM_TOKENS=$num_tokens MAX_SAMPLE_ID=5 TP_SIZE=8 CP_SIZE=$cp_size NUM_LAYERS=4 EXPERIMENT_REPEAT_TIMES=1 EXPERIMENT_WARMUP_TIMES=1 sbatch --nodes $nodes --partition=lowprio --qos=lowprio --time=00:10:00 test_e2e_combined.slurm.sh

            sleep 5

            EXPERIMENT_NVSHMEM_BUFFER_SIZE_GB="12" MODE=d2 BATCH_SIZE=$batch_size NUM_TOKENS=$num_tokens MAX_SAMPLE_ID=5 TP_SIZE=8 CP_SIZE=$cp_size NUM_LAYERS=4 EXPERIMENT_REPEAT_TIMES=1 EXPERIMENT_WARMUP_TIMES=1 sbatch --nodes $nodes --partition=lowprio --qos=lowprio --time=00:10:00 test_e2e_combined.slurm.sh
            sleep 5
        done
    done
done


# ✅ Done (D2 nlayer = 2 only)
# 256k
# 1	✅	8
# 2	✅	2
# 2	✅	4
# 2	✅	12
# 4	✅	6
# 4	✅	12
for num_tokens in 262144; do
    for nodes in 32; do
        for batch_size in 1 2 4; do
            for buffer_size in 1 2 4 6 8 10 12; do
                EXPERIMENT_NVSHMEM_BUFFER_SIZE_GB="$buffer_size" MODE=d2 BATCH_SIZE=$batch_size NUM_TOKENS=$num_tokens MAX_SAMPLE_ID=5 TP_SIZE=8 CP_SIZE=$cp_size NUM_LAYERS=2 EXPERIMENT_REPEAT_TIMES=1 EXPERIMENT_WARMUP_TIMES=1 sbatch --nodes $nodes --partition=lowprio --qos=lowprio --time=00:10:00 test_e2e_combined.slurm.sh
            done
        done
    done
done

# ⚪ Running




for num_tokens in 131072; do
    for nodes in 32; do
        for batch_size in 1; do
            for num_layers in 2 4 8 16 32; do
                EXPERIMENT_NVSHMEM_BUFFER_SIZE_GB="6" MODE=d2 BATCH_SIZE=$batch_size NUM_TOKENS=$num_tokens MAX_SAMPLE_ID=5 TP_SIZE=8 CP_SIZE=$cp_size NUM_LAYERS=$num_layers EXPERIMENT_REPEAT_TIMES=1 EXPERIMENT_WARMUP_TIMES=1 sbatch --nodes $nodes --partition=lowprio --qos=lowprio --time=00:10:00 test_e2e_combined.slurm.sh
            done
        done
    done
done



# for num_tokens in 262144; do
#     for nodes in 32; do
#         for batch_size in 1; do
#             for num_layers in 32; do
#                 EXPERIMENT_NVSHMEM_BUFFER_SIZE_GB="6" MODE=d2 BATCH_SIZE=$batch_size NUM_TOKENS=$num_tokens MAX_SAMPLE_ID=5 TP_SIZE=8 CP_SIZE=$cp_size NUM_LAYERS=$num_layers EXPERIMENT_REPEAT_TIMES=1 EXPERIMENT_WARMUP_TIMES=1 sbatch --nodes $nodes --partition=lowprio --qos=lowprio --time=00:10:00 test_e2e_combined.slurm.sh
#             done
#         done
#     done
# done


# for num_tokens in 262144; do
#     for nodes in 16; do
#         for batch_size in 1; do
#             for num_layers in 2; do
#                 EXPERIMENT_NVSHMEM_BUFFER_SIZE_GB="6" MODE=d2 BATCH_SIZE=$batch_size NUM_TOKENS=$num_tokens MAX_SAMPLE_ID=5 TP_SIZE=8 CP_SIZE=$cp_size NUM_LAYERS=$num_layers EXPERIMENT_REPEAT_TIMES=1 EXPERIMENT_WARMUP_TIMES=1 sbatch --nodes $nodes --partition=lowprio --qos=lowprio --time=00:10:00 test_e2e_combined.slurm.sh
#             done
#         done
#     done
# done



# for num_tokens in 262144; do
#     for nodes in 8 16 32; do
#         for batch_size in 1; do
#             for num_layers in 8 16 32; do
#                 EXPERIMENT_NVSHMEM_BUFFER_SIZE_GB="6" MODE=d2 BATCH_SIZE=$batch_size NUM_TOKENS=$num_tokens MAX_SAMPLE_ID=5 TP_SIZE=8 CP_SIZE=$cp_size NUM_LAYERS=$num_layers EXPERIMENT_REPEAT_TIMES=1 EXPERIMENT_WARMUP_TIMES=1 sbatch --nodes $nodes --partition=lowprio --qos=lowprio --time=00:10:00 test_e2e_combined.slurm.sh
#             done
#         done
#     done
# done

for num_tokens in 262144; do
    for nodes in 8 16 32; do
        for batch_size in 1; do
            for num_layers in 8 16 32; do
                EXPERIMENT_NVSHMEM_BUFFER_SIZE_GB="6" MODE=d2 BATCH_SIZE=$batch_size NUM_TOKENS=$num_tokens MAX_SAMPLE_ID=5 TP_SIZE=8 CP_SIZE=$cp_size NUM_LAYERS=$num_layers EXPERIMENT_REPEAT_TIMES=1 EXPERIMENT_WARMUP_TIMES=1 sbatch --nodes $nodes --partition=lowprio --qos=lowprio --time=00:10:00 test_e2e_combined.slurm.sh
            done
        done
    done
done

for num_tokens in 262144; do
    for nodes in 8 16 32; do
        for num_layers in 4; do
            EXPERIMENT_NVSHMEM_BUFFER_SIZE_GB="6" MODE=d2 BATCH_SIZE=$batch_size NUM_TOKENS=$num_tokens MAX_SAMPLE_ID=5 TP_SIZE=8 CP_SIZE=$cp_size NUM_LAYERS=$num_layers EXPERIMENT_REPEAT_TIMES=1 EXPERIMENT_WARMUP_TIMES=1 sbatch --nodes $nodes --partition=lowprio --qos=lowprio --time=00:10:00 test_e2e_combined.slurm.sh
        done
    done
done



for repeat in 1; do
    for num_tokens in 65536 131072 262144 524288; do
        
        if [ "$num_tokens" -eq 262144 ]; then
            num_layers=8
            elongate_factor=4
        elif [ "$num_tokens" -eq 524288 ]; then
            num_layers=4
            elongate_factor=8
        elif [ "$num_tokens" -eq 1048576 ]; then
            elongate_factor=2
        else
            num_layers=8
        fi

        for nodes in 8 16 32; do
            for batch_size in 1 2 4; do
                for cp_size in 32 16 8 4 2 1; do
                    MODE=wlbllm BATCH_SIZE=$batch_size NUM_TOKENS=$num_tokens MAX_SAMPLE_ID=50 TP_SIZE=8 CP_SIZE=$cp_size NUM_LAYERS=$num_layers EXPERIMENT_REPEAT_TIMES=3 EXPERIMENT_WARMUP_TIMES=3 sbatch --nodes $nodes --partition=lowprio --qos=lowprio test_e2e_combined.slurm.sh
                done
                EXPERIMENT_NVSHMEM_BUFFER_SIZE_GB="6" MODE=d2 BATCH_SIZE=$batch_size NUM_TOKENS=$num_tokens MAX_SAMPLE_ID=50 TP_SIZE=8 CP_SIZE=$cp_size NUM_LAYERS=$num_layers EXPERIMENT_REPEAT_TIMES=3 EXPERIMENT_WARMUP_TIMES=3 sbatch --nodes $nodes --partition=lowprio --qos=lowprio test_e2e_combined.slurm.sh
                EXPERIMENT_NVSHMEM_BUFFER_SIZE_GB="6" MODE=d2 BATCH_SIZE=$batch_size NUM_TOKENS=$num_tokens MAX_SAMPLE_ID=50 TP_SIZE=8 CP_SIZE=$cp_size NUM_LAYERS=$num_layers EXPERIMENT_REPEAT_TIMES=3 EXPERIMENT_WARMUP_TIMES=3 sbatch --nodes $nodes --partition=lowprio --qos=lowprio test_e2e_combined.slurm.sh
                EXPERIMENT_NVSHMEM_BUFFER_SIZE_GB="6" MODE=d2 BATCH_SIZE=$batch_size NUM_TOKENS=$num_tokens MAX_SAMPLE_ID=50 TP_SIZE=8 CP_SIZE=$cp_size NUM_LAYERS=$num_layers EXPERIMENT_REPEAT_TIMES=3 EXPERIMENT_WARMUP_TIMES=3 sbatch --nodes $nodes --partition=lowprio --qos=lowprio test_e2e_combined.slurm.sh
            done
        done
    done
done




# # ⚪ Running

# for num_tokens in 65536 131072 262144 524288 1048576; do
#     for nodes in 32 16 8 4 2 1; do
#         for batch_size in 1 2 4 8; do
#             for cp_size in 32 16 8 4 2 1; do
#                 if [ $cp_size -gt $nodes ]; then
#                     continue
#                 fi
#                 echo MODE=wlbllm BATCH_SIZE=$batch_size NUM_TOKENS=$num_tokens MAX_SAMPLE_ID=100 TP_SIZE=8 CP_SIZE=$cp_size NUM_LAYERS=4 sbatch --nodes $nodes test_e2e_combined.slurm.sh
#                 # sleep 2
#             done
#             echo MODE=d2 BATCH_SIZE=$batch_size NUM_TOKENS=$num_tokens MAX_SAMPLE_ID=100 TP_SIZE=8 CP_SIZE=$cp_size NUM_LAYERS=4 sbatch --nodes $nodes test_e2e_combined.slurm.sh
#             # sleep 2
#         done
#     done
# done


# # ELONGATE_FACTOR=4 EXPERIMENT_NVSHMEM_BUFFER_SIZE_GB="6" MODE=d2 BATCH_SIZE=4 NUM_TOKENS=65535 D2_SHOULD_REPLAN=1 SHOULD_ADD_DEBUG_CASES=1 MAX_SAMPLE_ID=3 TP_SIZE=8 CP_SIZE=1 NUM_LAYERS=4 sbatch --nodes 4 test_e2e_combined.slurm.sh