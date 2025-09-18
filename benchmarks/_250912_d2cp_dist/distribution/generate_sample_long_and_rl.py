# %%
from datasets import load_dataset
import numpy as np
import matplotlib.pyplot as plt
from multiprocessing import Pool


# Use fast tokenizer to get token counts
from transformers import AutoTokenizer

# tokenizer = AutoTokenizer.from_pretrained("codellama/CodeLlama-34b-hf", use_fast=True)
tokenizer = AutoTokenizer.from_pretrained("princeton-nlp/Llama-3-8B-ProLong-512k-Base")

# %%
# Load dataset
ds = load_dataset("princeton-nlp/TextbookChapters")

# %%
def encode_text(item):
    return len(tokenizer.encode(item['chapter']))

with Pool() as pool:
    token_lengths = pool.map(encode_text, ds['train'])

# %%
orig_token_lengths = token_lengths
# %%
import random
token_lengths = orig_token_lengths.copy()
random.seed(42)
random.shuffle(orig_token_lengths)

# %%
token_lengths = sorted(token_lengths, reverse=True)
# %% 
token_lengths = orig_token_lengths[:1000]
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
token_lengths

# %%

plt.figure(figsize=(10, 6))
plt.hist(token_lengths, bins=50)
plt.title('Distribution of Token Lengths')
plt.xlabel('Length (tokens)')
plt.ylabel('Count')
plt.ylim(0, 10)
plt.show()

# %%
from huggingface_hub import login

# Login to Hugging Face
login("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")

# %%

from datasets import load_dataset
ds = load_dataset("open-web-math/open-web-math")
with Pool() as pool:
    token_lengths = pool.map(encode_text, ds['train'])

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
plt.show()
# %%
# dataset streaming (will only download the data as needed)
ds = load_dataset("bigcode/the-stack", streaming=True, split="train")
# for sample in iter(ds): print(sample["content"])


# %%
ds = load_dataset("cerebras/SlimPajama-627B", split="train")

# %%
for sample in iter(ds):
    print(sample)
    break

# %%
token_lengths_v2 = []

def encode_text(item):
    return len(tokenizer.encode(item['text']))
# %%
cnt = 0
for sample in iter(ds):
    if cnt > 10000:
        break
    token_lengths_v2.append(encode_text(sample))
    cnt += 1

    print(cnt)
