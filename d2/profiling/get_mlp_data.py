import pandas as pd
import os
from pathlib import Path

def get_mlp_data() -> "dict[tuple[int, int], dict[int, float]]":
    try:
        this_dir = os.path.dirname(__file__)
    except:
        import d2
        import d2.profiling
        this_dir = Path(d2.profiling.__path__[0])

    this_dir = Path(this_dir)
    filepath = this_dir / "data" / "compute-mlp-only.bs4.tsv"
    df = pd.read_csv(filepath, sep="\t")

    columns = [
        "ctx_length",
        "tp",
        "cp",
        "megatron.core.transformer.attention.forward.qkv_1",
        "megatron.core.transformer.attention.forward.linear_proj_1",
        "megatron.core.transformer.transformer_layer._forward_attention.self_attn_bda_1",
        "megatron.core.transformer.transformer_layer._forward_mlp.mlp_1",
        "megatron.core.transformer.transformer_layer._forward_mlp.mlp_bda_1",
    ]
    assert set(df.columns) == set(columns)

    df.rename(columns={
        "ctx_length": "seq_len",
        "megatron.core.transformer.attention.forward.qkv_1": "qkv",
        "megatron.core.transformer.attention.forward.linear_proj_1": "linear_proj",
        "megatron.core.transformer.transformer_layer._forward_attention.self_attn_bda_1": "attn_bda",
        "megatron.core.transformer.transformer_layer._forward_mlp.mlp_1": "mlp",
        "megatron.core.transformer.transformer_layer._forward_mlp.mlp_bda_1": "mlp_bda",
    }, inplace=True)

    df['latency(ms)'] = df['qkv'] + df['linear_proj'] + df['attn_bda'] + df['mlp'] + df['mlp_bda']
    
    # divide by 4 because data is of batch size 4
    df['latency(ms)'] = df['latency(ms)'] / 4


    result = {}
    for _, row in df.iterrows():
        tp, cp = row["tp"], row["cp"]
        if (tp, cp) not in result:
            result[(tp, cp)] = {}
        result[(tp, cp)][row["seq_len"]] = row["latency(ms)"]
    return result