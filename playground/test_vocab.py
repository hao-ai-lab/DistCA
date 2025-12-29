# -----------------------------------------
# VocabParallelEmbedding
# -----------------------------------------

import torch
from distca.utils.logging import time_it, log_module, log_tensor_stats, get_logger
from megatron.core.tensor_parallel.layers import VocabParallelEmbedding

logger = get_logger()

with time_it("initialize VocabParallelEmbedding"):
    vocab_parallel_embedding = VocabParallelEmbedding(
        num_embeddings=128256,
        embedding_dim=128,
        # init_method=torch.nn.init.normal_,
        init_method=torch.nn.init.kaiming_normal_,
        reduce_scatter_embeddings=True,
        config=config,
    )
    log_module(vocab_parallel_embedding, name="VocabParallelEmbedding", preview=0)

    # Do a quick forward pass on a random input tensor and log the output.
    with torch.no_grad():
        sample_input = torch.randint(
            low=0,
            high=128256,
            size=(1, 8),  # batch=1, seq=8 for a tiny sanity check
            device=torch.cuda.current_device(),
            dtype=torch.int64,
        )
        sample_output = vocab_parallel_embedding(sample_input)
    logger.info(
        f"Forward output shape: {tuple(sample_output.shape)}, "
        f"dtype: {sample_output.dtype}, device: {sample_output.device}"
    )
    # Quick sanity stats to catch zeros/nans/abnormal values.
    log_tensor_stats(sample_input, name="VocabParallelEmbedding.input", preview=10)
    log_tensor_stats(sample_output, name="VocabParallelEmbedding.output", preview=10)
    # Log a small slice so we can see actual numbers without flooding logs.
    # logger.info(f"Forward output sample[0, 0, :8]: {sample_output[0, 0, :8].tolist()}")
