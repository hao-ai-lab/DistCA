"""
Patch to skip float conversion when D2_SKIP_FLOAT_CONVERSION is set.
    Megatron-LM/megatron/core/transformer/module.py
"""

import os

def float16_to_fp32(val):
    from typing import Optional, Tuple

    import torch
    from torch.autograd import Variable
    from torch.nn.parameter import Parameter

    from megatron.core.transformer.module import _BF16_TYPES, _HALF_TYPES

    def float_conversion(val):
        if os.getenv('D2_SKIP_FLOAT_CONVERSION', '0') == "1":
            print("D2_SKIP_FLOAT_CONVERSION is set, skipping float conversion")
            return val
        val_typecheck = val
        if isinstance(val_typecheck, (Parameter, Variable)):
            val_typecheck = val.data
        if isinstance(val_typecheck, (_BF16_TYPES, _HALF_TYPES)):
            val = val.float()
        return val

    return conversion_helper(val, float_conversion)


@contextmanager
def patch_float16_to_fp32():
    import megatron.core.transformer.module
    original_float16_to_fp32 = megatron.core.transformer.module.float16_to_fp32
    megatron.core.transformer.module.float16_to_fp32 = float16_to_fp32
    yield
    megatron.core.transformer.module.float16_to_fp32 = original_float16_to_fp32