# %% 
from datasets import load_dataset
import numpy as np
import matplotlib.pyplot as plt
from transformers import AutoTokenizer
from multiprocessing import Pool

# %%
tokenizer = AutoTokenizer.from_pretrained("princeton-nlp/Llama-3-8B-ProLong-512k-Base")


# %%
# ds = load_dataset("Tongyi-Zhiwen/DocQA-RL-1.6K")
# %%

ds = load_dataset("bigcode/the-stack", streaming=True, split="train")
def encode_text(item):
    return len(tokenizer.encode(item['content']))
token_lengths_v1 = []
# %%
cnt = 0
for sample in iter(ds):
    if cnt > 10000:
        break
    token_lengths_v1.append(encode_text(sample))
    cnt += 1
    print(cnt)

token_lengths = token_lengths_v1
print("\nToken statistics:")
print(f"Number of samples: {len(token_lengths)}")
print(f"Mean tokens: {np.mean(token_lengths):.2f}")
print(f"Median tokens: {np.median(token_lengths):.2f}") 
print(f"Min tokens: {min(token_lengths)}")
print(f"Max tokens: {max(token_lengths)}")

plt.figure(figsize=(10, 6))
plt.hist(token_lengths, bins=50)
plt.title('Distribution of Token Lengths')
plt.xlabel('Length (tokens)')
plt.ylabel('Count')
# plt.ylim(0, 10)
plt.show()



# %%
ds = load_dataset("Tongyi-Zhiwen/DocQA-RL-1.6K")

# %%
ds['train']
# %%
def encode_text(item):
    return len(tokenizer.encode(
        item['prompt'][0]['content']
    ))

# %%
token_lengths_v2 = []
# %%
cnt = 0
for sample in (ds['train']):
    if cnt > 10000:
        break
    token_lengths_v2.append(encode_text(sample))
    cnt += 1
    print(cnt)

# %%
token_lengths = token_lengths_v2
print("\nToken statistics:")
print(f"Number of samples: {len(token_lengths)}")
print(f"Mean tokens: {np.mean(token_lengths):.2f}")
print(f"Median tokens: {np.median(token_lengths):.2f}") 
print(f"Min tokens: {min(token_lengths)}")
print(f"Max tokens: {max(token_lengths)}")

plt.figure(figsize=(10, 6))
plt.hist(token_lengths, bins=50)
plt.title('Distribution of Token Lengths')
plt.xlabel('Length (tokens)')
plt.ylabel('Count')
# plt.ylim(0, 10)
plt.show()


# %%

ds = load_dataset("Tongyi-Zhiwen/ruler-128k-subset")
token_lengths_v3 = []
# %%
def encode_text(item):
    return len(tokenizer.encode(
        item['prompt'][0]['content']
    ))

cnt = 0
for sample in (ds['train']):
    if cnt > 10000:
        break
    token_lengths_v2.append(encode_text(sample))
    cnt += 1
    print(cnt)
token_lengths = token_lengths_v3
print("\nToken statistics:")
print(f"Number of samples: {len(token_lengths)}")
print(f"Mean tokens: {np.mean(token_lengths):.2f}")
print(f"Median tokens: {np.median(token_lengths):.2f}") 
print(f"Min tokens: {min(token_lengths)}")
print(f"Max tokens: {max(token_lengths)}")

plt.figure(figsize=(10, 6))
plt.hist(token_lengths, bins=50)
plt.title('Distribution of Token Lengths')
plt.xlabel('Length (tokens)')
plt.ylabel('Count')
# plt.ylim(0, 10)
plt.show()
# %%
ds['niah']
# %%
