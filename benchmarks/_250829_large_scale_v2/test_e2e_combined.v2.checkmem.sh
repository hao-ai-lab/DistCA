

squeue --me

now="$(TZ='America/Los_Angeles' date +%Y%m%d%H%M%S)_PST"

for num_tokens in 65536 131072 262144 524288; do
    
    num_layers=4
    
    elongate_factor=1 # 65535
    if [ "$num_tokens" -eq 131072 ]; then
        elongate_factor=2
    elif [ "$num_tokens" -eq 262144 ]; then
        elongate_factor=4
    elif [ "$num_tokens" -eq 524288 ]; then
        elongate_factor=8
    fi

    for nodes in 8 16 32; do
        for batch_size in 1 2 4; do
            for buffer_size in 20; do
                    OUTPUT_DIR_PREFIX="/mnt/weka/home/yonghao.zhuang/jd/d2/benchmarks/_250829_large_scale_v2/logs.checkmem.${now}" EXPERIMENT_NVSHMEM_BUFFER_SIZE_GB=$buffer_size MODE=d2 ELONGATE_FACTOR=$elongate_factor BATCH_SIZE=$batch_size NUM_TOKENS=$num_tokens MAX_SAMPLE_ID=10 TP_SIZE=8 CP_SIZE=$cp_size NUM_LAYERS=$num_layers EXPERIMENT_REPEAT_TIMES=3 EXPERIMENT_WARMUP_TIMES=3 sbatch --nodes $nodes --job-name=d2-ck-mem --time=00:10:00 --partition=lowprio --qos=lowprio test_e2e_combined.slurm.sh
            done
        done
    done
done
