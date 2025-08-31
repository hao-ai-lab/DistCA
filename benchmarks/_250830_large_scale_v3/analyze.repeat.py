# %%
import importlib
import analyze_v1
import time
importlib.reload(analyze_v1)
# %%

while True:
    time.sleep(10)
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



# %%
