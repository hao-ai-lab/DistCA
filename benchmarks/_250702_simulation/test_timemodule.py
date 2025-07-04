import timemodule as tm
import rich

a = tm.get_mlp_time(
    1, 1, 
    32 * tm.K, 
    return_extrainfo=True)
rich.print(f"tp=1, cp=1, t=32K")
rich.print(a)