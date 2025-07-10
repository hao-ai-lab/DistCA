import wandb
import os
import argparse

os.environ["WANDB_API_KEY"] = "02575b6c73e438f9885daa7cf691a45939d26a71"

wandb.login(
    key="02575b6c73e438f9885daa7cf691a45939d26a71"
)

from samples import (
    sample_wlbllm_docs, 
    sample_wlbllm_docs_altered,
    sample_multimodal_gaussian,
    sample_wlbllm_docs_upsample,
    batch_documents,
)
import timemodule as tm
from wandb_optimizer_attnserver_lessvar import main as attnserver_solver
from wandb_optimizer_wlbllm import main as wlbllm_solver


def run_wlbdocs_upsample_experiment(
    *,
    max_ctx_length: int,
    upsample_long_factor: int,
    num_total_devices: int,
    max_num_workers_attnserver: int,
):
    # Define sample configuration
    sample_name = "wlbdocs_upsample"

    batches = list(batch_documents(
        sample_wlbllm_docs_upsample(
            size=1000, 
            seed=42,
            filter_threshold=10000,
            filter_ratio=1.0,
            upsample_long_factor=upsample_long_factor,
        ), 
        max_ctx_length=max_ctx_length,
    ))

    # Create a run name with sample name and max context length
    ctx_str = f"{int(max_ctx_length/tm.K)}K"
    run_name = f"{sample_name}_ctx_{ctx_str}_gpu{num_total_devices}_atworker{max_num_workers_attnserver}_f{upsample_long_factor}"

    wandb.init(
        entity="junda-d2",
        project="d2", 
        name=run_name,
    )

    # Define metrics for W&B
    wandb.define_metric("step")
    wandb.define_metric("ratio/*", step_metric="step")
    wandb.define_metric("wlb/*", step_metric="step")
    wandb.define_metric("attnserver/*", step_metric="step")
    wandb.define_metric("input/batch", step_metric="step")
    wandb.define_metric("solver/*", step_metric="step")
    wandb.define_metric("best/*", step_metric="step")

    wandb.config.update(dict(
        sample_name=sample_name,
        sample_configs=dict(
            size=1000,
            filter_threshold=10000,
            filter_ratio=1.0,
            upsample_long_factor=upsample_long_factor,
        ),
        max_ctx_length=max_ctx_length,
        num_total_devices=num_total_devices,
        max_num_workers_attnserver=max_num_workers_attnserver,
    ))

    for idx, batch in enumerate(batches):
        wlbllm_results = {}
        for tp in (1, 2, 4, 8):
            for cp in (1, 2, 4, 8):
                wlbllm_results[(tp, cp)] = wlbllm_solver(
                    batch=batch,
                    num_total_devices=num_total_devices,
                    tp=tp,
                    cp=cp,
                    max_time_in_seconds=360,
                    sample_name=sample_name,
                    sample_id=idx,
                )
        
        attnserver_results = attnserver_solver(
            batch=batch,
            num_total_devices=num_total_devices,
            mlp_tp=8,
            mlp_cp=4,
            num_workers=max_num_workers_attnserver,
            max_time_in_seconds=360,
            sample_name=sample_name,
            sample_id=idx,
            sweep_all_mlp_plans=True,
        )

        log_data = {"step": idx}
        log_items = []

        best_attn_time = None
        best_wlb_time = None
        for tp in (1, 2, 4, 8):
            for cp in (1, 2, 4, 8):
                dp = num_total_devices // (tp * cp)
                wlb_result = wlbllm_results[(tp, cp)]
                attnserver_result = attnserver_results['sweep_results'][(tp, cp)]

                if best_attn_time is None or attnserver_result["batch_total_time"] < best_attn_time:
                    best_attn_time = attnserver_result["batch_total_time"]
                if best_wlb_time is None or wlb_result["max_worker_latency_us"] < best_wlb_time:
                    best_wlb_time = wlb_result["max_worker_latency_us"]

                ratio = attnserver_result["batch_total_time"] / wlb_result["max_worker_latency_us"]
                key = f"ratio/tp{tp}_cp{cp}_dp{dp}"
                log_data[key] = ratio
                key = f"wlb/tp{tp}_cp{cp}_dp{dp}"
                log_data[key] = wlb_result["max_worker_latency_us"]
                key = f"attnserver/tp{tp}_cp{cp}_dp{dp}"
                log_data[key] = attnserver_result["batch_total_time"]

                log_item = (dict(
                    idx=idx,
                    input_batch=batch,
                    run_name=run_name,
                    tp=tp, cp=cp, dp=dp,
                    ratio=ratio,
                    wlb_result=wlb_result,
                    attnserver_result=attnserver_result
                ))
                log_items.append(log_item)

        key = f"input/batch"
        log_data[key] = batch

        key = f"best/attn_time"
        log_data[key] = best_attn_time
        key = f"best/wlb_time"
        log_data[key] = best_wlb_time
        key = f"best/ratio"
        log_data[key] = best_attn_time / best_wlb_time

        key = f"solver/status"
        log_data[key] = attnserver_results['solver_status']
        key = f"solver/status_code"
        log_data[key] = attnserver_results['solver_status_code']
        key = f"solver/solved_time_s"
        log_data[key] = attnserver_results['solved_time_s']
        key = f"solver/num_vars"
        log_data[key] = attnserver_results['num_vars']
        key = f"solver/num_cons"
        log_data[key] = attnserver_results['num_cons']

        key = f"solver/time_s"
        log_data[key] = attnserver_results['solved_time_s']
        key = "solver/num_vars"
        log_data[key] = attnserver_results['num_vars']
        key = "solver/num_cons"
        log_data[key] = attnserver_results['num_cons']

        wandb.log(log_data)
        with open(f"logs/{run_name}.jsonl", "a") as f:
            import json
            for log_item in log_items:
                log_item_str = json.dumps(log_item)
                f.write(f"{log_item_str}\n")
            f.flush()
    
    # Finish the run
    wandb.finish()

def main():
    parser = argparse.ArgumentParser(description="Run WLB Docs Upsample Experiment")
    parser.add_argument("--max_length", type=int, required=True, help="Maximum context length")
    parser.add_argument("--upsample_long_factor", type=int, required=True, help="Upsample long factor")
    parser.add_argument("--num_total_devices", type=int, default=128, help="Total number of devices")
    parser.add_argument("--max_num_workers_attnserver", type=int, default=16, help="Max number of workers for attnserver")
    args = parser.parse_args()

    run_wlbdocs_upsample_experiment(
        max_ctx_length=args.max_length * tm.K,
        upsample_long_factor=args.upsample_long_factor,
        num_total_devices=args.num_total_devices,
        max_num_workers_attnserver=args.max_num_workers_attnserver,
    )

if __name__ == "__main__":
    main()


"""
python zrun_wandb_optimizer_wlbupsample.py \
    --max_length 256 \
    --upsample_long_factor 32 \
    --num_total_devices 128 \
    --max_num_workers_attnserver 16
"""