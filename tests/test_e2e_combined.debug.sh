NNODES=1
NPROC_PER_NODE=8
RZV_BACKEND=c10d
RZV_ENDPOINT=fs-mbz-gpu-012:29900
RZV_ID=megatron_d2_unique_id
MODE=baseline

# deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B
# MODE=d2
export NVTE_NVTX_ENABLED=1
export NSYS_NVTX_PROFILER_REGISTER_ONLY=0 
export D2_FA2A_DISABLE_SEND_RECV=0 
export NVSHMEM_IB_ENABLE_IBGDA=true 
export NVTE_ALLOW_NONDETERMINISTIC_ALGO=1 
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH


# nsys profile -t cuda,nvtx -s none --wait=primary --show-output=true \
#  --force-overwrite=true --export=sqlite -o profile_output \
#  python your_training_script.py
nsys profile \
  --show-output=true \
  --force-overwrite=true \
  -o nsys-profile/debug.nsys-rep \
  --sample=none \
  -t cuda,nvtx \
torchrun \
  --nnodes=${NNODES} \
  --nproc_per_node=${NPROC_PER_NODE} \
  --rdzv_backend=${RZV_BACKEND} \
  --rdzv_endpoint=${RZV_ENDPOINT} \
  --rdzv_id=${RZV_ID} \
  --max_restarts=0 \
  test_e2e_combined.py \
    --mode ${MODE} \
    --replan-iter 0 \
    --num-nodes ${NNODES} \
    --num-gpus-per-node ${NPROC_PER_NODE} \
    --tp-size 4 \
    --num-layers 4 \
    --max-sample-id 3 \
    --up-sample-factor 2 \
    --num-tokens 65536

# nsys stats nsys-profile/debug.nsys-rep 
# nsys stats nsys-profile/debug.nsys-rep --report cuda_gpu_kern_sum --format table --force-export=true

# nsys export --type sqlite --output nsys-profile/debug.sqlite nsys-profile/debug.nsys-rep
# nsys stats nsys-profile/debug.nsys-rep --report nvtxppsum

# --mode d2 \
# --mode baseline \

# pass🟢：
# - 128k num layer 4 
# - 170000 num layer 2

# fail🔴：
# - 128k num layer 6 