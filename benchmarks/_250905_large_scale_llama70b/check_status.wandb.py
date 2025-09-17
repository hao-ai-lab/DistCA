# %%

import time
import wandb
import importlib
from IPython.display import display
wandb.login()

run = wandb.init(
    # Set the wandb entity where your project will be logged (generally your team name).
    entity="junda-d2",
    # Set the wandb project where this run will be logged.
    project="d2-dpcp-0906-exp",
    # Set run name and enable resuming
    name="v3",
    resume="allow",
    # Track hyperparameters and run metadata.
    config={},
)

while True:
    import check_status
    importlib.reload(check_status)
    mem_dfs = check_status.mem_dfs
    mem_dfs['max_mem'] =mem_dfs['max_mem'].astype(str)
    mem_dfs['exp_total_time'] =mem_dfs['exp_total_time'].astype(str)

    mem_dfs_short = check_status.mem_dfs[[
        'gid', 'reason', 'is_finished', 
        'avg_duration', 
        'max_mem',
        'exp_total_time', 
        'mode', 'CP_SIZE', 'nodes', 
        # "ckpt", "resend", 
        "exp", 
        # 'it', 
    ]]
    mem_dfs_short.sort_values(by='gid', inplace=True)
    display(mem_dfs_short)
    
    table = wandb.Table(dataframe=mem_dfs_short)
    run.log({"df_display": table})
    time.sleep(10)

run.finish()

# %%