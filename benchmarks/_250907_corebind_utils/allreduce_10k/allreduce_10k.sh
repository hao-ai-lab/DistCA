


# srun -w fs-mbz-gpu-012,fs-mbz-gpu-018 --label --jobid=703724 -N 2 -G 16 --ntasks-per-node=8 --gpus-per-task=1 --cpu-bind=cores bash -lc '
# echo "$(hostname) R=$SLURM_PROCID L=$SLURM_LOCALID CVD=$CUDA_VISIBLE_DEVICES"
# python -c "import torch, os; i = torch.cuda.current_device(); p = torch.cuda.get_device_properties(i); print(i, p);"
# '




# srun -w fs-mbz-gpu-012 --label --jobid=703724 -N 32 -G 256 --ntasks-per-node=8 --gpus-per-task=1 --cpu-bind=cores bash -lc '
# echo "$(hostname) R=$SLURM_PROCID L=$SLURM_LOCALID CVD=$CUDA_VISIBLE_DEVICES"
# python -c "import torch, os; i = torch.cuda.current_device(); p = torch.cuda.get_device_properties(i); print(i, p);"
# '


# srun -w fs-mbz-gpu-012 --label --jobid=703724 -N 32 -G 256 --ntasks-per-node=8 --gpus-per-task=1 --cpu-bind=cores bash -lc '
# echo "$(hostname) R=$SLURM_PROCID L=$SLURM_LOCALID CVD=$CUDA_VISIBLE_DEVICES"
# python allreduce_10k.py
# '

# srun -w fs-mbz-gpu-012 --label --jobid=703724 -N 32 -G 256 --ntasks-per-node=8 --gpus-per-task=1 --cpu-bind=cores torchrun --nnodes=32 --nproc_per_node=8 --rdzv_backend=c10d --rdzv_endpoint=fs-mbz-gpu-012:29500 --rdzv_id=703724 allreduce_10k.py


# srun -w fs-mbz-gpu-012 --label --jobid=703724 -N 32 -G 256 --ntasks-per-node=1 --gpus-per-task=8 --cpu-bind=cores torchrun --nnodes=32 --nproc_per_node=8 --rdzv_backend=c10d --rdzv_endpoint=fs-mbz-gpu-012:29500 --rdzv_id=703724 allreduce_10k.py

# srun -w fs-mbz-gpu-012 --label --jobid=703724 -N 1 -G 8 --ntasks-per-node=1 --gpus-per-task=8 --cpu-bind=cores torchrun --nnodes=1 --nproc_per_node=8 --rdzv_backend=c10d --rdzv_endpoint=fs-mbz-gpu-012:29500 --rdzv_id=703724 allreduce_10k.py


# export OMP_NUM_THREADS=1

# srun -w fs-mbz-gpu-012 --label --jobid=703724 -N 1 -G 8 --ntasks-per-node=8 --gpus-per-task=1 --cpu-bind=cores torchrun --nnodes=32 --nproc_per_node=8 --rdzv_backend=c10d --rdzv_endpoint=fs-mbz-gpu-012:29500 --rdzv_id=703724 allreduce_10k.py

# # -----

# # 🟢
# srun --output=logs/allreduce_10k.%N.%j.log --error=logs/allreduce_10k.%N.%j.log -w fs-mbz-gpu-012 --label --jobid=703746 -N 1 -G 8 --ntasks-per-node=1 --gpus-per-task=8 --cpu-bind=cores torchrun --nnodes=1 --nproc_per_node=8 --rdzv_backend=c10d --rdzv_endpoint=fs-mbz-gpu-012:29500 --rdzv_id=703724 allreduce_10k.py

# # --output=logs/${TS}/allreduce_10k.%N.%j.log --error=logs/${TS}/allreduce_10k.%N.%j.log 

# TS=$(TZ=America/Los_Angeles date +%Y%m%d_%H%M%S)
# mkdir -p ./logs/${TS}
# srun -vv -w fs-mbz-gpu-012 --label --jobid=703746 -N 1 -G 8 --ntasks-per-node=1 --gpus-per-task=8 --cpu-bind=cores \
# nsys profile -t nvtx,cuda -o ./logs/${TS}/allreduce_10k.%h.nsys-rep \
# torchrun --nnodes=1 --nproc_per_node=8 --rdzv_backend=c10d --rdzv_endpoint=fs-mbz-gpu-012:29500 --rdzv_id=703724 allreduce_10k.py --output-dir ./logs/${TS}/


# TS=$(TZ=America/Los_Angeles date +%Y%m%d_%H%M%S)
# mkdir -p ./logs/${TS}
# srun -vv -w fs-mbz-gpu-012 --label --jobid=703746 -N 1 -G 8 --ntasks-per-node=1 --gpus-per-task=8 --cpu-bind=cores \
# nsys profile -t nvtx,cuda -o ./logs/${TS}/allreduce_10k.%h.nsys-rep \
# torchrun --nnodes=1 --nproc_per_node=8 --rdzv_backend=c10d --rdzv_endpoint=fs-mbz-gpu-012:29500 --rdzv_id=703724 allreduce_10k.py --output-dir ./logs/${TS}/




# TS=$(TZ=America/Los_Angeles date +%Y%m%d_%H%M%S)
# mkdir -p ./logs/${TS}
# srun -vv -w fs-mbz-gpu-039 --label --jobid=703746 -N 1 -G 8 --ntasks-per-node=1 --gpus-per-task=8 --cpu-bind=cores \
# nsys profile -t nvtx,cuda -o ./logs/${TS}/allreduce_10k.%h.nsys-rep \
# torchrun --nnodes=1 --nproc_per_node=8 --rdzv_backend=c10d --rdzv_endpoint=fs-mbz-gpu-012:29500 --rdzv_id=703724 allreduce_10k.py --output-dir ./logs/${TS}/


# # 1 node, 8 GPUs, 8 cores per node
TS=$(TZ=America/Los_Angeles date +%Y%m%d_%H%M%S)
head_node=fs-mbz-gpu-039
endpoint=${head_node}:29500
mkdir -p ./logs/${TS}
srun -vv -w ${head_node} --label --jobid=703746 \
-N 1 -G 8 --ntasks-per-node=1 --gpus-per-task=8 --cpu-bind=cores \
nsys profile -t nvtx,cuda -o ./logs/${TS}/allreduce_10k.%h.nsys-rep \
torchrun --nnodes=1 --nproc_per_node=8 \
--rdzv_backend=c10d --rdzv_endpoint=${endpoint} \
--rdzv_id=703724 \
allreduce_10k.py --output-dir ./logs/${TS}/

# N node, 8N GPUs, 8 cores per node
TS=$(TZ=America/Los_Angeles date +%Y%m%d_%H%M%S)
head_node=fs-mbz-gpu-039
endpoint=${head_node}:29500
mkdir -p ./logs/${TS}
NNODES=4
gpus=$((NNODES * 8))
srun -vv -w ${head_node} --label --jobid=703746 \
-N $NNODES -G $gpus --ntasks-per-node=1 --gpus-per-task=8 --cpu-bind=cores \
nsys profile -t nvtx,cuda -o ./logs/${TS}/allreduce_10k.%h.nsys-rep \
torchrun --nnodes=$NNODES --nproc_per_node=8 \
--rdzv_backend=c10d --rdzv_endpoint=${endpoint} --rdzv_id=703724 \
allreduce_10k.py --output-dir ./logs/${TS}/


# N node, 8N GPUs, 8 cores per node
TS=$(TZ=America/Los_Angeles date +%Y%m%d_%H%M%S)
head_node=fs-mbz-gpu-039
endpoint=${head_node}:29500
mkdir -p ./logs/${TS}
NNODES=32
gpus=$((NNODES * 8))
srun -vv -w ${head_node} --label --jobid=703746 \
-N $NNODES -G $gpus --ntasks-per-node=1 --gpus-per-task=8 --cpu-bind=cores \
nsys profile -t nvtx,cuda -o ./logs/${TS}/allreduce_10k.%h.nsys-rep \
torchrun --nnodes=$NNODES --nproc_per_node=8 \
--rdzv_backend=c10d --rdzv_endpoint=${endpoint} --rdzv_id=703724 \
allreduce_10k.py --output-dir ./logs/${TS}/


echo "Done"
echo '\a'