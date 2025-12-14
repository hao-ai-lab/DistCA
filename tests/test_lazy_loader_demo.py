"""
Minimal, dependency-light demo for the lazy token iterator in `training_utils`.

This file is designed to:
  - NOT depend on Megatron or HF datasets.
  - Show that `lazy_token_iterator_from_texts` tokenizes lazily.
"""

import torch

from training_utils import lazy_token_iterator_from_texts


class DummyTokenizer:
    """Very simple tokenizer for demo purposes.

    - Splits on whitespace.
    - Maps each distinct token string to an integer ID.
    """

    def __init__(self):
        self.vocab = {}
        self.next_id = 1  # reserve 0 for padding if needed

    def encode(self, text: str, add_special_tokens: bool = True, truncation: bool = False):
        tokens = text.split()
        ids = []
        for tok in tokens:
            if tok not in self.vocab:
                self.vocab[tok] = self.next_id
                self.next_id += 1
            ids.append(self.vocab[tok])
        return ids


def main():
    # A tiny in-memory "dataset"
    raw_texts = [
        "hello world",
        "  ",  # will be skipped
        "this is a test",
        "hello again",
    ]

    tokenizer = DummyTokenizer()

    token_iter = lazy_token_iterator_from_texts(
        texts=raw_texts,
        tokenizer=tokenizer,
        max_samples=10,
    )

    print("Iterating over lazily tokenized samples:")
    for i, token_tensor in enumerate(token_iter):
        print(f"  sample {i}: shape={tuple(token_tensor.shape)}, tokens={token_tensor.tolist()}")


if __name__ == "__main__":
    main()


