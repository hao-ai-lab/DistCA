#!/bin/bash

set -x

this_dir=$(dirname $(realpath $0))
# source $this_dir/../env/login.env

# Runs the "175B" parameter model

export CUDA_DEVICE_MAX_CONNECTIONS=1

GPUS_PER_NODE=2
# Change for multinode config
MASTER_ADDR=localhost
MASTER_PORT=6000
NUM_NODES=1
NODE_RANK=0
WORLD_SIZE=$(($GPUS_PER_NODE*$NUM_NODES))

CHECKPOINT_PATH=$1 #<Specify path>
VOCAB_FILE=$this_dir/vocab/gpt2-vocab.json
MERGE_FILE=$this_dir/vocab/gpt2-merges.txt

DISTRIBUTED_ARGS=(
    --nproc_per_node $GPUS_PER_NODE 
    --nnodes $NUM_NODES 
    --master_addr $MASTER_ADDR 
    --master_port $MASTER_PORT
)

GPT_MODEL_ARGS=(
    --num-layers 4
    --hidden-size 4096 
    --num-attention-heads 16 
    --seq-length 2048 
    --max-position-embeddings 2048 
    --attention-backend auto # Can use (flash/fused/unfused/local)
)

TRAINING_ARGS=(
    --micro-batch-size 1 
    --global-batch-size 16 
    --train-iters 4 
    --weight-decay 0.1 
    --adam-beta1 0.9 
    --adam-beta2 0.95 
    --init-method-std 0.006 
    --clip-grad 1.0 
    --fp16
    --lr 6.0e-5 
    --lr-decay-style cosine 
    --min-lr 6.0e-6
    --lr-warmup-fraction .001 
    --lr-decay-iters 430000 
)

MODEL_PARALLEL_ARGS=(
	--tensor-model-parallel-size $GPUS_PER_NODE
	--pipeline-model-parallel-size 1
    --context-parallel-size 1
)

DATA_ARGS=(
    --mock-data
    # --data-path $this_dir/gpt2_text_document
    --vocab-file $VOCAB_FILE 
    --merge-file $MERGE_FILE 
    --split 949,50,1
)

EVAL_AND_LOGGING_ARGS=(
    --log-interval 1
    --save-interval 10000 
    --eval-interval 1000 
    --save /tmp/checkpoint 
    # --load /tmp/checkpoint 
    --eval-iters 10
)

export PYTHONPATH=$this_dir/../Megatron-LM/:$PYTHONPATH
torchrun ${DISTRIBUTED_ARGS[@]} $this_dir/test_basic_megatron_e2e.py \
    ${GPT_MODEL_ARGS[@]} \
    ${TRAINING_ARGS[@]} \
    ${MODEL_PARALLEL_ARGS[@]} \
    ${DATA_ARGS[@]} \
    ${EVAL_AND_LOGGING_ARGS[@]}

set +x