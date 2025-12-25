


```bash
#if nltk isn't installed
pip install nltk
# /mnt/weka/home/hao.zhang/jd/Megatron-LM/tools
# python tools/preprocess_data.py \
python /mnt/weka/home/hao.zhang/jd/Megatron-LM/tools/preprocess_data.py \
--input codeparrot_data.json \
--output-prefix code \
--json-keys 'content' \
--tokenizer-type HuggingFaceTokenizer \
--tokenizer-model deepseek-ai/DeepSeek-R1-Distill-Llama-8B \
--workers 1 \
--append-eod
# --split-sentences \
```




Debug example trying to use the tokenizer
```python
import argparse
from megatron.training.tokenizer import build_tokenizer

# Create a minimal args namespace with required fields
args = argparse.Namespace(
    # Required for build_tokenizer
    tokenizer_type='HuggingFaceTokenizer',
    tokenizer_model='deepseek-ai/DeepSeek-R1-Distill-Llama-8B',
    
    # Required defaults (set in preprocess_data.py's get_args)
    rank=0,  # Set to 0 to see the "building tokenizer" message
    make_vocab_size_divisible_by=128,
    tensor_model_parallel_size=1,
    vocab_extra_ids=0,
    
    # Optional (not used by HuggingFaceTokenizer but may be checked)
    vocab_file=None,
    merge_file=None,
    padded_vocab_size=None,
)

# Build the tokenizer
tokenizer = build_tokenizer(args)

# Test it out!
text = "Hello, how are you today?"
token_ids = tokenizer.tokenize(text)
print(f"Text: {text}")
print(f"Token IDs: {token_ids}")
print(f"Vocab size: {tokenizer.vocab_size}")
print(f"EOD token ID: {tokenizer.eod}")

# Decode back
decoded = tokenizer.detokenize(token_ids)
print(f"Decoded: {decoded}")
```