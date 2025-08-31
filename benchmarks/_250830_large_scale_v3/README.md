# Experiment Description

```bash
for model = Qwen3/8B -> 30B
    for distribution in [pretrain, long-mostly, rl]
        for nnode in {8, 16, 32}
            for seqlen in {128k, 256k, 512k}
                sample 100 times
                run each wlbllm(dp,cp) 
                run d2(dpcp=nnode)
                report avg , std
```