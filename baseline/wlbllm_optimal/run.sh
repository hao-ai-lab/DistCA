#!/bin/bash

rm -rf gpt-checkpoint

export CUDA_DEVICE_MAX_CONNECTIONS="1"
GPUS_PER_NODE=1
MASTER_ADDR=localhost
MASTER_PORT=6001
NNODES=1
NODE_RANK=0
WORLD_SIZE=$(($GPUS_PER_NODE*$NNODES))
DISTRIBUTED_ARGS="--nproc_per_node $GPUS_PER_NODE --nnodes $NNODES --node_rank $NODE_RANK --master_addr $MASTER_ADDR --master_port $MASTER_PORT"
CHECKPOINT_PATH=gpt-checkpoint
VOCAB_FILE=vocab.json
MERGE_FILE=merges.txt
DATA_PATH=my-gpt2_text_document
GPT_ARGS="--num-layers 4
--hidden-size 768
--num-attention-heads 12
--seq-length 256
--max-position-embeddings 256
--micro-batch-size 1
--global-batch-size 1
--lr 0.0005
--train-iters 1
--lr-decay-iters 150000
--lr-decay-style cosine
--lr-warmup-iters 1
--weight-decay .1
--adam-beta2 .999
--fp16
--log-interval 1
--save-interval 4
--eval-interval 4
--eval-iters 4
"
MOE_ARGS="
--num-experts 8
--expert-model-parallel-size 8
--moe-grouped-gemm
--moe-permute-fusion
--moe-router-load-balancing-type aux_loss
--moe-router-topk 2
--moe-aux-loss-coeff 1e-2
--use-distributed-optimizer
--moe-token-dispatcher-type alltoall
"
CONTROL_ARGS="
--mock-data
"
TENSORBOARD_ARGS="--tensorboard-dir tensorboard-logs/"

set -x
python3 -m torch.distributed.launch $DISTRIBUTED_ARGS \
        pretrain_gpt.py \
        --tensor-model-parallel-size 1 \
        --pipeline-model-parallel-size 1 \
        $GPT_ARGS \
        --vocab-file $VOCAB_FILE \
        --merge-file $MERGE_FILE \
        --save $CHECKPOINT_PATH \
        --load $CHECKPOINT_PATH \
        --logging-level 0 \
        $CONTROL_ARGS \
        $TENSORBOARD_ARGS \
        $@
set +x