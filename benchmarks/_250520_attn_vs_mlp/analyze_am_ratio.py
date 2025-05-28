# %%
import time_module
from time_module import compute, network
from models import Llama8B
# %%

model = Llama8B(tp=1, cp=4)
# %%

# One Sequence Data
K = 1024

for k in [32, 48, 64, 96, 128]:
    a = model._attn(k * K)
    m = model._mlp(k * K)
    print(f"[One Sequence] {k}K: attn: {a:.2f} ms, mlp: {m:.2f} ms, a/m: {(a/m):.2f}")

"""
[One Sequence] 32K: attn: 13360.00 ms, mlp: 30767.50 ms, a/m: 0.43
[One Sequence] 48K: attn: 26630.00 ms, mlp: 25421.00 ms, a/m: 1.05
[One Sequence] 64K: attn: 44050.00 ms, mlp: 26169.00 ms, a/m: 1.68
[One Sequence] 96K: attn: 95120.00 ms, mlp: 27677.50 ms, a/m: 3.44
[One Sequence] 128K: attn: 165480.00 ms, mlp: 29186.00 ms, a/m: 5.67
"""

# %%
def am_ratio(model, batch: list[int], verbose=True):
    a = sum(model._attn(i) for i in batch)
    m = model._mlp(sum(batch))
    aom = a / m
    if verbose:
        print(f"attn: {a:.2f} ms, mlp: {m:.2f} ms, a/m: {(a/m):.2f}")
    return a, m, aom


# %%

# Multi Sequence Data
attn_lens = [16] + [1] * (64 - 16)
attn_lens = [i * K for i in attn_lens]
a, m, aom = am_ratio(model,attn_lens, verbose=True)
"""
attn: 55930.00 ms, mlp: 26169.00 ms, a/m: 2.14
"""

# %%
for tp in [1, 2, 4, 8]:
    for cp in [1, 2, 4, 8]:
        if tp * cp <= 16 and tp <= 8:
            model = Llama8B(tp=tp, cp=cp)
            a, m, aom = am_ratio(model,[32* K] , verbose=False)
            a = a / 1000
            m = m / 1000
            print(f"TP: {tp}, CP: {cp}, A: {a:.2f}ms, M: {m:.2f}ms, A/M: {aom:.2f}")

print("---")
for tp in [1, 2, 4, 8]:
    for cp in [1, 2, 4, 8]:
        if tp * cp <= 16 and tp <= 8:
            model = Llama8B(tp=tp, cp=cp)
            a, m, aom = am_ratio(model,[1* K] , verbose=False)
            a = a / 1000
            m = m / 1000
            print(f"TP: {tp}, CP: {cp}, A: {a:.2f}ms, M: {m:.2f}ms, A/M: {aom:.2f}")
# %%

