
salloc -N 8 -G 64 --ntasks-per-node=1 --cpus-per-task=96 --mem=1440G --time=12:00:00 --exclusive --partition=lowprio --qos=lowprio --job-name=d2-interact --exclude=fs-mbz-gpu-684,fs-mbz-gpu-697,fs-mbz-gpu-286,fs-mbz-gpu-877,fs-mbz-gpu-757,fs-mbz-gpu-806,fs-mbz-gpu-377,fs-mbz-gpu-906,fs-mbz-gpu-168,fs-mbz-gpu-708,fs-mbz-gpu-868,fs-mbz-gpu-223,fs-mbz-gpu-954,fs-mbz-gpu-707

salloc -N 16 -G 128 --ntasks-per-node=1 --cpus-per-task=96 --mem=1440G --time=12:00:00 --exclusive --partition=lowprio --qos=lowprio --job-name=d2-interact --exclude=fs-mbz-gpu-684,fs-mbz-gpu-697,fs-mbz-gpu-286,fs-mbz-gpu-877,fs-mbz-gpu-757,fs-mbz-gpu-806,fs-mbz-gpu-377,fs-mbz-gpu-906,fs-mbz-gpu-168,fs-mbz-gpu-708,fs-mbz-gpu-868,fs-mbz-gpu-223,fs-mbz-gpu-954,fs-mbz-gpu-707

salloc -N 32 -G 256 --ntasks-per-node=1 --cpus-per-task=96 --mem=1440G --time=12:00:00 --exclusive --partition=lowprio --qos=lowprio --job-name=d2-interact --exclude=fs-mbz-gpu-684,fs-mbz-gpu-697,fs-mbz-gpu-286,fs-mbz-gpu-877,fs-mbz-gpu-757,fs-mbz-gpu-806,fs-mbz-gpu-377,fs-mbz-gpu-906,fs-mbz-gpu-168,fs-mbz-gpu-708,fs-mbz-gpu-868,fs-mbz-gpu-223,fs-mbz-gpu-954,fs-mbz-gpu-707

squeue -j 668403 -o "%.18i %.9P %.8j %.8u %.2t %.10M %.6D %R"


conda activate jd-d2

# 8_65536_32_d2_1_4
EXPERIMENT_LOG_MEMORY_USAGE=1 \
JOBID=668871 \
EXPERIMENT_NVSHMEM_BUFFER_SIZE_GB=4 \
SLURM_GPUS_ON_NODE=8 \
SLURM_NNODES=8 NNODES=8  \
SLURM_JOB_NODELIST="fs-mbz-gpu-[051,061,070,091,157,192,206,214]" \
MODE="d2" BATCH_SIZE=32 NUM_TOKENS=65536 MAX_SAMPLE_ID=3 \
NUM_LAYERS=4 TP_SIZE=8 PP_SIZE=1 CP_SIZE=1 \
bash test_e2e_combined.salloc-exp.sh


# 8_65536_2_wlbllm_8_4
EXPERIMENT_LOG_MEMORY_USAGE=1 \
JOBID=668871 \
SLURM_GPUS_ON_NODE=8 \
SLURM_NNODES=8 NNODES=8  \
SLURM_JOB_NODELIST="fs-mbz-gpu-[051,061,070,091,157,192,206,214]" \
MODE="wlbllm" BATCH_SIZE=32 NUM_TOKENS=65536 MAX_SAMPLE_ID=3 \
NUM_LAYERS=4 TP_SIZE=8 PP_SIZE=1 CP_SIZE=8 \
bash test_e2e_combined.salloc-exp.sh

