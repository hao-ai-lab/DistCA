# %%
import importlib
import analyze_v1
import time
importlib.reload(analyze_v1)
# %%

import random

import wandb

# Start a new wandb run to track this script.
run = wandb.init(
    # Set the wandb entity where your project will be logged (generally your team name).
    entity="junda-d2",
    # Set the wandb project where this run will be logged.
    project="d2-cpdp-exp",
    # Track hyperparameters and run metadata.
    config={},
)

# %%
while True:
    time.sleep(30)
    importlib.reload(analyze_v1)

    # analyze_v1.df_display
    # - make the ""

    # Move the columns ['is_past_one_test', 'it_reached'] to after the column 'speedup.
    # Get current column order
    cols = analyze_v1.df_display.columns.tolist()

    # Remove the columns we want to move
    cols.remove('is_past_one_test')
    cols.remove('it_reached')

    # Find index of 'speedup' column
    speedup_idx = cols.index('speedup')

    # Insert columns after 'speedup'
    cols.insert(speedup_idx + 1, 'it_reached')
    cols.insert(speedup_idx + 1, 'is_past_one_test')

    # Reorder columns
    analyze_v1.df_display = analyze_v1.df_display[cols]

    from IPython.display import display
    display(analyze_v1.df_display)

    # Convert dataframe to wandb Table and log it
    table = wandb.Table(dataframe=analyze_v1.df_display)
    run.log({"df_display": table})
    # run.log({"df_display_html": wandb.Html(
    #     f"<head><meta charset=\"utf-8\"></head><body>{analyze_v1.df_display.to_html()}</body>"
    # ))



# %%
