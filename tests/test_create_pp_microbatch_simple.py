

from test_megatron_e2e_pipeline_with_cp import create_pp_microbatches
from global_batch_provider import setup_global_batch, get_next_batch
import global_batch_provider

global_batch_provider.GLOBAL_BATCH = None

setup_global_batch(
    total_seq_len=262144,
    up_sample_factor=4,
    elongate_factor=1,
    filter_threshold=64 * 1024,
    filter_ratio=0.,
    should_add_debug_cases=False,
    change_long_doc_ratio=0.0,
    sample_name='wlbllm',
    # balance_ping_pong_batch_size=True,
    balance_ping_pong_batch_size=False,
)

# Reimport to get any updates
import importlib
import test_megatron_e2e_pipeline_with_cp
importlib.reload(test_megatron_e2e_pipeline_with_cp)
from test_megatron_e2e_pipeline_with_cp import create_pp_microbatches


import d2.runtime.attn_kernels.ops
DispatcherWrapper = d2.runtime.attn_kernels.ops.DispatcherWrapper

class MockDispatcher:
    def __init__(self):
        self.buffer_size = 1024 * 1024 * 1024 # 1GB default buffer size

class MockDispatcherWrapper:
    instance = [MockDispatcher()]

DispatcherWrapper.instance = MockDispatcherWrapper.instance
DispatcherWrapper.instance[0].buffer_size

import os
# Set minimum tolerance factor for batch balancing via environment variable
os.environ["MIN_TOLERANCE_FACTOR"] = "0.15"



import time
start_time = time.time()
microbatches_0, tick_per_rank_doc_lens_0 = create_pp_microbatches(
num_microbatch=16,
pp_degree=2,
as_rank=0,
as_world_size=16,
total_seq_len=262144,
num_seqs=3,
max_cp_degree=16,
hidden_size_q_tp=1024,
hidden_size_k_tp=128,
element_size=2,
num_head_in_dtype=16,
tp_size=8,
dp_size=8,
num_token_per_rank=32768,
num_batches=1,
use_planner=True,
return_seq_lens=True,
)
end_time = time.time()
print(f"🟡 create_pp_microbatches duration: {end_time - start_time} seconds")

duration = end_time - start_time
print(f"🟡 create_pp_microbatches duration: {duration} seconds")
