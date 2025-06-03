"""
Build a prototype system for the attention and MLP pipeline.
"""

import modal

vllm_image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("vllm-flash-attn")
)
