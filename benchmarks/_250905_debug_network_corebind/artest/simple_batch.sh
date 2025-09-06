#!/bin/bash

#SBATCH --job-name=torchrun-dummy
#SBATCH --nodes=4
# # #SBATCH --core-spec=2 # not supported by the cluster orz
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=8
#SBATCH --mem=1440G
#SBATCH --exclusive
#SBATCH --output=logs/%x.%j.%t.log
#SBATCH --error=logs/%x.%j.%t.log
#SBATCH --time=00:05:00
#SBATCH --qos=hao

pwd
cd /mnt/weka/home/hao.zhang/jd/d2/benchmarks/_250905_torchrun
pwd
TS=$(TZ=America/Los_Angeles date +%Y%m%d_%H%M%S)
echo $TS
mkdir -p ./logs/${TS}

set -x
# srun --label torchrun --nnodes 1 --nproc_per_node 8 --rdzv_backend=c10d --rdzv_endpoint=localhost:29600 --rdzv_id=0000 --no-python bash -lc "
#     exec > logs/${TS}/output.rank${RANK}.log 2>&1 
#     echo $CUDA_VISIBLE_DEVICES
#     nvidia-smi
#     # show the gpu uuid
#     nvidia-smi -i $CUDA_VISIBLE_DEVICES --query-gpu=gpu_uuid --format=csv,noheader
#     env
#     taskset -c ${LOCAL_RANK} python torchsample.py
# "

start_time=$(date +%s)

srun --label --cpu-bind=cores --gpu-bind=closest \
nsys profile -t nvtx,cuda -o logs/${TS}/profile.%h.nsys-rep \
torchrun --nnodes 1 --nproc_per_node 8 --rdzv_backend=c10d --rdzv_endpoint=localhost:29600 --rdzv_id=0000 --no-python bash -l torch_launcher.sh

# bash -lc "
#     exec > logs/${TS}/rank\${RANK}.log 2>&1 
#     echo LOCAL_RANK=\${LOCAL_RANK} RANK=\${RANK}
#     nsys profile -t nvtx,cuda -o logs/${TS}/\${RANK}.nsys-rep python torchsample.py
# "
end_time=$(date +%s)
echo "Time taken: $((end_time - start_time)) seconds"
set +x