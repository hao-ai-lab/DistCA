# export OMP_NUM_THREADS=1
# export NCCL_DEBUG=WARN
# export NCCL_IB_DISABLE=1
# export NCCL_P2P_LEVEL=NVL

# Important for --capture-range=nvtx: allow any NVTX string (not just registered ones)
# export NSYS_NVTX_PROFILER_REGISTER_ONLY=0


srun_strs=""
if [ ${JOBID} -ne 0 ]; then
    srun_strs="--jobid=${JOBID}"
fi

export PS4='\033[1;36m+ \033[0m'
set -x
srun -N 1 -G 8 --ntasks-per-node=1 $srun_strs nsys profile -t cuda,nvtx -o ./nsys_torch_nvtx_mainonly.nsys-rep \
  torchrun --nproc-per-node=8 --rdzv_backend=c10d --rdzv_endpoint=localhost:29500 \
    cuda_multiseg_torch_nvtx.py
set +x
# --capture-range=nvtx \
#       --capture-range-end=stop \