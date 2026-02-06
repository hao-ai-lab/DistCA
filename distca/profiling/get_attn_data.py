import os
from pathlib import Path

import pandas as pd


def get_attn_data() -> "dict[tuple[int, int], dict[int, float]]":
    try:
        this_dir = os.path.dirname(__file__)
    except:
        import distca
        import distca.profiling

        this_dir = Path(distca.profiling.__path__[0])

    this_dir = Path(this_dir)
    # filepath = this_dir / "data" / "compute-attn-H100.psv"
    filepath = this_dir / "data" / "compute-attn-H100.csv"
    df = pd.read_csv(filepath)

    result = {}
    for _, row in df.iterrows():
        tp, cp = row["tp"], row["cp"]
        if (tp, cp) not in result:
            result[(tp, cp)] = {}
        result[(tp, cp)][row["L"]] = row["duration_per_doc"]
    return result


if __name__ == "__main__":
    df = get_attn_data()
