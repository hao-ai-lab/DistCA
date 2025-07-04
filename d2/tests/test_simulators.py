from typing import Deque, Iterable, List, Sequence

import numpy as np
import time
from collections import deque
import matplotlib.pyplot as plt

from d2.simulator.optimizers.attnserver import AttnServerSolver
from d2.simulator.optimizers.wlbllm import WlbLlmSolver
from d2.simulator.optimizers.samples import (
    sample_multimodal_gaussian, 
    sample_random_docs,
    sample_wlbllm_docs,
    batch_documents,
)

K = 1024
M = 1024 ** 2


from d2.simulator.compare import wlbllm_vs_attnserver

def test_simple_batch():
    batch = [4 * K, 4 * K]
    wlbllm_vs_attnserver(
        batch,
        num_workers=4,
        num_total_devices=32,
    )

if __name__ == "__main__":  
    test_simple_batch()