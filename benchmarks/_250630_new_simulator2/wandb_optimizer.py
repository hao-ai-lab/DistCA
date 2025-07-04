from samples import (sample_wlbllm_docs, sample_wlbllm_docs_altered, batch_documents)
import timemodule as tm

batches = list(batch_documents(
    sample_wlbllm_docs(size=1000), 
    max_ctx_length=64 * tm.K * 8
))

from wandb_optimizer_attnserver_lessvar import main as attnserver_solver
# import wandb_optimizer_wlbllm as wlbllm


for idx, batch in enumerate(batches):
    for tp in (1, 2, 4, 8):
        for cp in (1, 2, 4, 8):
            attnserver_solver(
                batch=batch,
                num_total_devices=64,
                mlp_tp=tp,
                mlp_cp=cp,
                max_time_in_seconds=360,
                sample_name="wlbllm_docs",
                sample_id=idx,
            )
            break