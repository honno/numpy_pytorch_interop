import torch

# These two dicts are autogenerated with autogen/gen_dtypes.py,
# using numpy version 1.23.5.

_can_cast_dict =  {'no': {torch.float16: {torch.float16: True, torch.float32: False, torch.float64: False, torch.complex64: False, torch.complex128: False, torch.uint8: False, torch.int8: False, torch.int16: False, torch.int32: False, torch.int64: False, torch.bool: False}, torch.float32: {torch.float16: False, torch.float32: True, torch.float64: False, torch.complex64: False, torch.complex128: False, torch.uint8: False, torch.int8: False, torch.int16: False, torch.int32: False, torch.int64: False, torch.bool: False}, torch.float64: {torch.float16: False, torch.float32: False, torch.float64: True, torch.complex64: False, torch.complex128: False, torch.uint8: False, torch.int8: False, torch.int16: False, torch.int32: False, torch.int64: False, torch.bool: False}, torch.complex64: {torch.float16: False, torch.float32: False, torch.float64: False, torch.complex64: True, torch.complex128: False, torch.uint8: False, torch.int8: False, torch.int16: False, torch.int32: False, torch.int64: False, torch.bool: False}, torch.complex128: {torch.float16: False, torch.float32: False, torch.float64: False, torch.complex64: False, torch.complex128: True, torch.uint8: False, torch.int8: False, torch.int16: False, torch.int32: False, torch.int64: False, torch.bool: False}, torch.uint8: {torch.float16: False, torch.float32: False, torch.float64: False, torch.complex64: False, torch.complex128: False, torch.uint8: True, torch.int8: False, torch.int16: False, torch.int32: False, torch.int64: False, torch.bool: False}, torch.int8: {torch.float16: False, torch.float32: False, torch.float64: False, torch.complex64: False, torch.complex128: False, torch.uint8: False, torch.int8: True, torch.int16: False, torch.int32: False, torch.int64: False, torch.bool: False}, torch.int16: {torch.float16: False, torch.float32: False, torch.float64: False, torch.complex64: False, torch.complex128: False, torch.uint8: False, torch.int8: False, torch.int16: True, torch.int32: False, torch.int64: False, torch.bool: False}, torch.int32: {torch.float16: False, torch.float32: False, torch.float64: False, torch.complex64: False, torch.complex128: False, torch.uint8: False, torch.int8: False, torch.int16: False, torch.int32: True, torch.int64: False, torch.bool: False}, torch.int64: {torch.float16: False, torch.float32: False, torch.float64: False, torch.complex64: False, torch.complex128: False, torch.uint8: False, torch.int8: False, torch.int16: False, torch.int32: False, torch.int64: True, torch.bool: False}, torch.bool: {torch.float16: False, torch.float32: False, torch.float64: False, torch.complex64: False, torch.complex128: False, torch.uint8: False, torch.int8: False, torch.int16: False, torch.int32: False, torch.int64: False, torch.bool: True}}, 'equiv': {torch.float16: {torch.float16: True, torch.float32: False, torch.float64: False, torch.complex64: False, torch.complex128: False, torch.uint8: False, torch.int8: False, torch.int16: False, torch.int32: False, torch.int64: False, torch.bool: False}, torch.float32: {torch.float16: False, torch.float32: True, torch.float64: False, torch.complex64: False, torch.complex128: False, torch.uint8: False, torch.int8: False, torch.int16: False, torch.int32: False, torch.int64: False, torch.bool: False}, torch.float64: {torch.float16: False, torch.float32: False, torch.float64: True, torch.complex64: False, torch.complex128: False, torch.uint8: False, torch.int8: False, torch.int16: False, torch.int32: False, torch.int64: False, torch.bool: False}, torch.complex64: {torch.float16: False, torch.float32: False, torch.float64: False, torch.complex64: True, torch.complex128: False, torch.uint8: False, torch.int8: False, torch.int16: False, torch.int32: False, torch.int64: False, torch.bool: False}, torch.complex128: {torch.float16: False, torch.float32: False, torch.float64: False, torch.complex64: False, torch.complex128: True, torch.uint8: False, torch.int8: False, torch.int16: False, torch.int32: False, torch.int64: False, torch.bool: False}, torch.uint8: {torch.float16: False, torch.float32: False, torch.float64: False, torch.complex64: False, torch.complex128: False, torch.uint8: True, torch.int8: False, torch.int16: False, torch.int32: False, torch.int64: False, torch.bool: False}, torch.int8: {torch.float16: False, torch.float32: False, torch.float64: False, torch.complex64: False, torch.complex128: False, torch.uint8: False, torch.int8: True, torch.int16: False, torch.int32: False, torch.int64: False, torch.bool: False}, torch.int16: {torch.float16: False, torch.float32: False, torch.float64: False, torch.complex64: False, torch.complex128: False, torch.uint8: False, torch.int8: False, torch.int16: True, torch.int32: False, torch.int64: False, torch.bool: False}, torch.int32: {torch.float16: False, torch.float32: False, torch.float64: False, torch.complex64: False, torch.complex128: False, torch.uint8: False, torch.int8: False, torch.int16: False, torch.int32: True, torch.int64: False, torch.bool: False}, torch.int64: {torch.float16: False, torch.float32: False, torch.float64: False, torch.complex64: False, torch.complex128: False, torch.uint8: False, torch.int8: False, torch.int16: False, torch.int32: False, torch.int64: True, torch.bool: False}, torch.bool: {torch.float16: False, torch.float32: False, torch.float64: False, torch.complex64: False, torch.complex128: False, torch.uint8: False, torch.int8: False, torch.int16: False, torch.int32: False, torch.int64: False, torch.bool: True}}, 'safe': {torch.float16: {torch.float16: True, torch.float32: True, torch.float64: True, torch.complex64: True, torch.complex128: True, torch.uint8: False, torch.int8: False, torch.int16: False, torch.int32: False, torch.int64: False, torch.bool: False}, torch.float32: {torch.float16: False, torch.float32: True, torch.float64: True, torch.complex64: True, torch.complex128: True, torch.uint8: False, torch.int8: False, torch.int16: False, torch.int32: False, torch.int64: False, torch.bool: False}, torch.float64: {torch.float16: False, torch.float32: False, torch.float64: True, torch.complex64: False, torch.complex128: True, torch.uint8: False, torch.int8: False, torch.int16: False, torch.int32: False, torch.int64: False, torch.bool: False}, torch.complex64: {torch.float16: False, torch.float32: False, torch.float64: False, torch.complex64: True, torch.complex128: True, torch.uint8: False, torch.int8: False, torch.int16: False, torch.int32: False, torch.int64: False, torch.bool: False}, torch.complex128: {torch.float16: False, torch.float32: False, torch.float64: False, torch.complex64: False, torch.complex128: True, torch.uint8: False, torch.int8: False, torch.int16: False, torch.int32: False, torch.int64: False, torch.bool: False}, torch.uint8: {torch.float16: True, torch.float32: True, torch.float64: True, torch.complex64: True, torch.complex128: True, torch.uint8: True, torch.int8: False, torch.int16: True, torch.int32: True, torch.int64: True, torch.bool: False}, torch.int8: {torch.float16: True, torch.float32: True, torch.float64: True, torch.complex64: True, torch.complex128: True, torch.uint8: False, torch.int8: True, torch.int16: True, torch.int32: True, torch.int64: True, torch.bool: False}, torch.int16: {torch.float16: False, torch.float32: True, torch.float64: True, torch.complex64: True, torch.complex128: True, torch.uint8: False, torch.int8: False, torch.int16: True, torch.int32: True, torch.int64: True, torch.bool: False}, torch.int32: {torch.float16: False, torch.float32: False, torch.float64: True, torch.complex64: False, torch.complex128: True, torch.uint8: False, torch.int8: False, torch.int16: False, torch.int32: True, torch.int64: True, torch.bool: False}, torch.int64: {torch.float16: False, torch.float32: False, torch.float64: True, torch.complex64: False, torch.complex128: True, torch.uint8: False, torch.int8: False, torch.int16: False, torch.int32: False, torch.int64: True, torch.bool: False}, torch.bool: {torch.float16: True, torch.float32: True, torch.float64: True, torch.complex64: True, torch.complex128: True, torch.uint8: True, torch.int8: True, torch.int16: True, torch.int32: True, torch.int64: True, torch.bool: True}}, 'same_kind': {torch.float16: {torch.float16: True, torch.float32: True, torch.float64: True, torch.complex64: True, torch.complex128: True, torch.uint8: False, torch.int8: False, torch.int16: False, torch.int32: False, torch.int64: False, torch.bool: False}, torch.float32: {torch.float16: True, torch.float32: True, torch.float64: True, torch.complex64: True, torch.complex128: True, torch.uint8: False, torch.int8: False, torch.int16: False, torch.int32: False, torch.int64: False, torch.bool: False}, torch.float64: {torch.float16: True, torch.float32: True, torch.float64: True, torch.complex64: True, torch.complex128: True, torch.uint8: False, torch.int8: False, torch.int16: False, torch.int32: False, torch.int64: False, torch.bool: False}, torch.complex64: {torch.float16: False, torch.float32: False, torch.float64: False, torch.complex64: True, torch.complex128: True, torch.uint8: False, torch.int8: False, torch.int16: False, torch.int32: False, torch.int64: False, torch.bool: False}, torch.complex128: {torch.float16: False, torch.float32: False, torch.float64: False, torch.complex64: True, torch.complex128: True, torch.uint8: False, torch.int8: False, torch.int16: False, torch.int32: False, torch.int64: False, torch.bool: False}, torch.uint8: {torch.float16: True, torch.float32: True, torch.float64: True, torch.complex64: True, torch.complex128: True, torch.uint8: True, torch.int8: True, torch.int16: True, torch.int32: True, torch.int64: True, torch.bool: False}, torch.int8: {torch.float16: True, torch.float32: True, torch.float64: True, torch.complex64: True, torch.complex128: True, torch.uint8: False, torch.int8: True, torch.int16: True, torch.int32: True, torch.int64: True, torch.bool: False}, torch.int16: {torch.float16: True, torch.float32: True, torch.float64: True, torch.complex64: True, torch.complex128: True, torch.uint8: False, torch.int8: True, torch.int16: True, torch.int32: True, torch.int64: True, torch.bool: False}, torch.int32: {torch.float16: True, torch.float32: True, torch.float64: True, torch.complex64: True, torch.complex128: True, torch.uint8: False, torch.int8: True, torch.int16: True, torch.int32: True, torch.int64: True, torch.bool: False}, torch.int64: {torch.float16: True, torch.float32: True, torch.float64: True, torch.complex64: True, torch.complex128: True, torch.uint8: False, torch.int8: True, torch.int16: True, torch.int32: True, torch.int64: True, torch.bool: False}, torch.bool: {torch.float16: True, torch.float32: True, torch.float64: True, torch.complex64: True, torch.complex128: True, torch.uint8: True, torch.int8: True, torch.int16: True, torch.int32: True, torch.int64: True, torch.bool: True}}, 'unsafe': {torch.float16: {torch.float16: True, torch.float32: True, torch.float64: True, torch.complex64: True, torch.complex128: True, torch.uint8: True, torch.int8: True, torch.int16: True, torch.int32: True, torch.int64: True, torch.bool: True}, torch.float32: {torch.float16: True, torch.float32: True, torch.float64: True, torch.complex64: True, torch.complex128: True, torch.uint8: True, torch.int8: True, torch.int16: True, torch.int32: True, torch.int64: True, torch.bool: True}, torch.float64: {torch.float16: True, torch.float32: True, torch.float64: True, torch.complex64: True, torch.complex128: True, torch.uint8: True, torch.int8: True, torch.int16: True, torch.int32: True, torch.int64: True, torch.bool: True}, torch.complex64: {torch.float16: True, torch.float32: True, torch.float64: True, torch.complex64: True, torch.complex128: True, torch.uint8: True, torch.int8: True, torch.int16: True, torch.int32: True, torch.int64: True, torch.bool: True}, torch.complex128: {torch.float16: True, torch.float32: True, torch.float64: True, torch.complex64: True, torch.complex128: True, torch.uint8: True, torch.int8: True, torch.int16: True, torch.int32: True, torch.int64: True, torch.bool: True}, torch.uint8: {torch.float16: True, torch.float32: True, torch.float64: True, torch.complex64: True, torch.complex128: True, torch.uint8: True, torch.int8: True, torch.int16: True, torch.int32: True, torch.int64: True, torch.bool: True}, torch.int8: {torch.float16: True, torch.float32: True, torch.float64: True, torch.complex64: True, torch.complex128: True, torch.uint8: True, torch.int8: True, torch.int16: True, torch.int32: True, torch.int64: True, torch.bool: True}, torch.int16: {torch.float16: True, torch.float32: True, torch.float64: True, torch.complex64: True, torch.complex128: True, torch.uint8: True, torch.int8: True, torch.int16: True, torch.int32: True, torch.int64: True, torch.bool: True}, torch.int32: {torch.float16: True, torch.float32: True, torch.float64: True, torch.complex64: True, torch.complex128: True, torch.uint8: True, torch.int8: True, torch.int16: True, torch.int32: True, torch.int64: True, torch.bool: True}, torch.int64: {torch.float16: True, torch.float32: True, torch.float64: True, torch.complex64: True, torch.complex128: True, torch.uint8: True, torch.int8: True, torch.int16: True, torch.int32: True, torch.int64: True, torch.bool: True}, torch.bool: {torch.float16: True, torch.float32: True, torch.float64: True, torch.complex64: True, torch.complex128: True, torch.uint8: True, torch.int8: True, torch.int16: True, torch.int32: True, torch.int64: True, torch.bool: True}}}


_result_type_dict =  {torch.float16: {torch.float16: torch.float16, torch.float32: torch.float32, torch.float64: torch.float64, torch.complex64: torch.complex64, torch.complex128: torch.complex128, torch.uint8: torch.float16, torch.int8: torch.float16, torch.int16: torch.float32, torch.int32: torch.float64, torch.int64: torch.float64, torch.bool: torch.float16}, torch.float32: {torch.float16: torch.float32, torch.float32: torch.float32, torch.float64: torch.float64, torch.complex64: torch.complex64, torch.complex128: torch.complex128, torch.uint8: torch.float32, torch.int8: torch.float32, torch.int16: torch.float32, torch.int32: torch.float64, torch.int64: torch.float64, torch.bool: torch.float32}, torch.float64: {torch.float16: torch.float64, torch.float32: torch.float64, torch.float64: torch.float64, torch.complex64: torch.complex128, torch.complex128: torch.complex128, torch.uint8: torch.float64, torch.int8: torch.float64, torch.int16: torch.float64, torch.int32: torch.float64, torch.int64: torch.float64, torch.bool: torch.float64}, torch.complex64: {torch.float16: torch.complex64, torch.float32: torch.complex64, torch.float64: torch.complex128, torch.complex64: torch.complex64, torch.complex128: torch.complex128, torch.uint8: torch.complex64, torch.int8: torch.complex64, torch.int16: torch.complex64, torch.int32: torch.complex128, torch.int64: torch.complex128, torch.bool: torch.complex64}, torch.complex128: {torch.float16: torch.complex128, torch.float32: torch.complex128, torch.float64: torch.complex128, torch.complex64: torch.complex128, torch.complex128: torch.complex128, torch.uint8: torch.complex128, torch.int8: torch.complex128, torch.int16: torch.complex128, torch.int32: torch.complex128, torch.int64: torch.complex128, torch.bool: torch.complex128}, torch.uint8: {torch.float16: torch.float16, torch.float32: torch.float32, torch.float64: torch.float64, torch.complex64: torch.complex64, torch.complex128: torch.complex128, torch.uint8: torch.uint8, torch.int8: torch.int16, torch.int16: torch.int16, torch.int32: torch.int32, torch.int64: torch.int64, torch.bool: torch.uint8}, torch.int8: {torch.float16: torch.float16, torch.float32: torch.float32, torch.float64: torch.float64, torch.complex64: torch.complex64, torch.complex128: torch.complex128, torch.uint8: torch.int16, torch.int8: torch.int8, torch.int16: torch.int16, torch.int32: torch.int32, torch.int64: torch.int64, torch.bool: torch.int8}, torch.int16: {torch.float16: torch.float32, torch.float32: torch.float32, torch.float64: torch.float64, torch.complex64: torch.complex64, torch.complex128: torch.complex128, torch.uint8: torch.int16, torch.int8: torch.int16, torch.int16: torch.int16, torch.int32: torch.int32, torch.int64: torch.int64, torch.bool: torch.int16}, torch.int32: {torch.float16: torch.float64, torch.float32: torch.float64, torch.float64: torch.float64, torch.complex64: torch.complex128, torch.complex128: torch.complex128, torch.uint8: torch.int32, torch.int8: torch.int32, torch.int16: torch.int32, torch.int32: torch.int32, torch.int64: torch.int64, torch.bool: torch.int32}, torch.int64: {torch.float16: torch.float64, torch.float32: torch.float64, torch.float64: torch.float64, torch.complex64: torch.complex128, torch.complex128: torch.complex128, torch.uint8: torch.int64, torch.int8: torch.int64, torch.int16: torch.int64, torch.int32: torch.int64, torch.int64: torch.int64, torch.bool: torch.int64}, torch.bool: {torch.float16: torch.float16, torch.float32: torch.float32, torch.float64: torch.float64, torch.complex64: torch.complex64, torch.complex128: torch.complex128, torch.uint8: torch.uint8, torch.int8: torch.int8, torch.int16: torch.int16, torch.int32: torch.int32, torch.int64: torch.int64, torch.bool: torch.bool}}

