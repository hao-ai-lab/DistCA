D2_FA2A_DISABLE_SEND_RECV=0 NVSHMEM_IB_ENABLE_IBGDA=true NVTE_ALLOW_NONDETERMINISTIC_ALGO=1 \
torchrun \
  --nnodes=4:4 \
  --nproc_per_node=8 \
  --rdzv_backend=c10d \
  --rdzv_endpoint=fs-mbz-gpu-012:29800 \
  --rdzv_id=megatron_d2_unique_id \
  --max_restarts=0 \
  test_e2e_combined.py \
    --mode d2 \
    --replan-iter 1 \
    --num-nodes 4 \
    --num-gpus-per-node 8 \
    --tp-size 8 \
    --num-layers 4 \
    --max-sample-id 10 \
    --num-tokens 32768

