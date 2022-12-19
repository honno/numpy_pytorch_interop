""" Define the supported dtypes and numpy <--> torch dtype mapping, define casting rules. 
"""

# TODO: 1. define torch_np dtypes, make this work without numpy.
#       2. mimic numpy's various aliases (np.half == np.float16, dtype='i8' etc)
#       3. convert from python types: np.ones(3, dtype=float) etc

import torch

# Define analogs of numpy dtypes supported by pytorch.

class dtype:
    def __init__(self, name):
        self._name = name

    @property
    def name(self):
        return self._name

    def __repr__(self):
        return f'dtype("{self._name}")'

    __str__ = __repr__


dt_names = ['float16', 'float32', 'float64',
         'complex64', 'complex128',
         'uint8',
         'int8',
         'int16',
         'int32',
         'int64',
         'bool']


float16 = dtype("float16")
float32 = dtype("float32")
float64 = dtype("float64")
complex64 = dtype("complex64")
complex128 = dtype("complex128")
uint8 = dtype("uint8")
int8 = dtype("int8")
int16 = dtype("int16")
int32 = dtype("int32")
int64 = dtype("int64")
bool = dtype("bool")


# Map the torch-suppored subset dtypes to local analogs
# "quantized" types not available in numpy, skip
_dtype_from_torch_dict = {
        # floating-point
        torch.float16: float16,
        torch.float32: float32,
        torch.float64 : float64,
        # np.complex32 does not exist
        torch.complex64: complex64,
        torch.complex128: complex128,
        # integer, unsigned (unit8 only, torch.uint32 etc do not exist)
        torch.uint8: uint8,
        # integer
        torch.int8: int8,
        torch.int16: int16,
        torch.int32: int32,
        torch.int64: int64,
        # boolean
        torch.bool : bool
}


# reverse mapping
_torch_dtype_from_dtype_dict = {_dtype_from_torch_dict[key]: key
                                for key in _dtype_from_torch_dict}

def dtype_from_torch(torch_dtype):
    try:
        return _dtype_from_torch_dict[torch_dtype]
    except KeyError:
        # mimic numpy: >>> np.dtype('unknown') -->  TypeError
        raise TypeError


def torch_dtype_from_dtype(dtype):
    if dtype is None:
        return None
    try:
        return _torch_dtype_from_dtype_dict[dtype]
    except KeyError:
        # mimic numpy: >>> np.dtype('unknown') -->  TypeError
        raise TypeError



# The casting below is defined *with dtypes only*, so no value-based casting!

# These two dicts are autogenerated with autogen/gen_dtypes.py,
# using numpy version 1.23.5.

_can_cast_dict = {
'no': {'float16': {'float16': True, 'float32': False, 'float64': False, 'complex64': False, 'complex128': False, 'uint8': False, 'int8': False, 'int16': False, 'int32': False, 'int64': False, 'bool': False}, 'float32': {'float16': False, 'float32': True, 'float64': False, 'complex64': False, 'complex128': False, 'uint8': False, 'int8': False, 'int16': False, 'int32': False, 'int64': False, 'bool': False}, 'float64': {'float16': False, 'float32': False, 'float64': True, 'complex64': False, 'complex128': False, 'uint8': False, 'int8': False, 'int16': False, 'int32': False, 'int64': False, 'bool': False}, 'complex64': {'float16': False, 'float32': False, 'float64': False, 'complex64': True, 'complex128': False, 'uint8': False, 'int8': False, 'int16': False, 'int32': False, 'int64': False, 'bool': False}, 'complex128': {'float16': False, 'float32': False, 'float64': False, 'complex64': False, 'complex128': True, 'uint8': False, 'int8': False, 'int16': False, 'int32': False, 'int64': False, 'bool': False}, 'uint8': {'float16': False, 'float32': False, 'float64': False, 'complex64': False, 'complex128': False, 'uint8': True, 'int8': False, 'int16': False, 'int32': False, 'int64': False, 'bool': False}, 'int8': {'float16': False, 'float32': False, 'float64': False, 'complex64': False, 'complex128': False, 'uint8': False, 'int8': True, 'int16': False, 'int32': False, 'int64': False, 'bool': False}, 'int16': {'float16': False, 'float32': False, 'float64': False, 'complex64': False, 'complex128': False, 'uint8': False, 'int8': False, 'int16': True, 'int32': False, 'int64': False, 'bool': False}, 'int32': {'float16': False, 'float32': False, 'float64': False, 'complex64': False, 'complex128': False, 'uint8': False, 'int8': False, 'int16': False, 'int32': True, 'int64': False, 'bool': False}, 'int64': {'float16': False, 'float32': False, 'float64': False, 'complex64': False, 'complex128': False, 'uint8': False, 'int8': False, 'int16': False, 'int32': False, 'int64': True, 'bool': False}, 'bool': {'float16': False, 'float32': False, 'float64': False, 'complex64': False, 'complex128': False, 'uint8': False, 'int8': False, 'int16': False, 'int32': False, 'int64': False, 'bool': True}},

'equiv': {'float16': {'float16': True, 'float32': False, 'float64': False, 'complex64': False, 'complex128': False, 'uint8': False, 'int8': False, 'int16': False, 'int32': False, 'int64': False, 'bool': False}, 'float32': {'float16': False, 'float32': True, 'float64': False, 'complex64': False, 'complex128': False, 'uint8': False, 'int8': False, 'int16': False, 'int32': False, 'int64': False, 'bool': False}, 'float64': {'float16': False, 'float32': False, 'float64': True, 'complex64': False, 'complex128': False, 'uint8': False, 'int8': False, 'int16': False, 'int32': False, 'int64': False, 'bool': False}, 'complex64': {'float16': False, 'float32': False, 'float64': False, 'complex64': True, 'complex128': False, 'uint8': False, 'int8': False, 'int16': False, 'int32': False, 'int64': False, 'bool': False}, 'complex128': {'float16': False, 'float32': False, 'float64': False, 'complex64': False, 'complex128': True, 'uint8': False, 'int8': False, 'int16': False, 'int32': False, 'int64': False, 'bool': False}, 'uint8': {'float16': False, 'float32': False, 'float64': False, 'complex64': False, 'complex128': False, 'uint8': True, 'int8': False, 'int16': False, 'int32': False, 'int64': False, 'bool': False}, 'int8': {'float16': False, 'float32': False, 'float64': False, 'complex64': False, 'complex128': False, 'uint8': False, 'int8': True, 'int16': False, 'int32': False, 'int64': False, 'bool': False}, 'int16': {'float16': False, 'float32': False, 'float64': False, 'complex64': False, 'complex128': False, 'uint8': False, 'int8': False, 'int16': True, 'int32': False, 'int64': False, 'bool': False}, 'int32': {'float16': False, 'float32': False, 'float64': False, 'complex64': False, 'complex128': False, 'uint8': False, 'int8': False, 'int16': False, 'int32': True, 'int64': False, 'bool': False}, 'int64': {'float16': False, 'float32': False, 'float64': False, 'complex64': False, 'complex128': False, 'uint8': False, 'int8': False, 'int16': False, 'int32': False, 'int64': True, 'bool': False}, 'bool': {'float16': False, 'float32': False, 'float64': False, 'complex64': False, 'complex128': False, 'uint8': False, 'int8': False, 'int16': False, 'int32': False, 'int64': False, 'bool': True}},

'safe': {'float16': {'float16': True, 'float32': True, 'float64': True, 'complex64': True, 'complex128': True, 'uint8': False, 'int8': False, 'int16': False, 'int32': False, 'int64': False, 'bool': False}, 'float32': {'float16': False, 'float32': True, 'float64': True, 'complex64': True, 'complex128': True, 'uint8': False, 'int8': False, 'int16': False, 'int32': False, 'int64': False, 'bool': False}, 'float64': {'float16': False, 'float32': False, 'float64': True, 'complex64': False, 'complex128': True, 'uint8': False, 'int8': False, 'int16': False, 'int32': False, 'int64': False, 'bool': False}, 'complex64': {'float16': False, 'float32': False, 'float64': False, 'complex64': True, 'complex128': True, 'uint8': False, 'int8': False, 'int16': False, 'int32': False, 'int64': False, 'bool': False}, 'complex128': {'float16': False, 'float32': False, 'float64': False, 'complex64': False, 'complex128': True, 'uint8': False, 'int8': False, 'int16': False, 'int32': False, 'int64': False, 'bool': False}, 'uint8': {'float16': True, 'float32': True, 'float64': True, 'complex64': True, 'complex128': True, 'uint8': True, 'int8': False, 'int16': True, 'int32': True, 'int64': True, 'bool': False}, 'int8': {'float16': True, 'float32': True, 'float64': True, 'complex64': True, 'complex128': True, 'uint8': False, 'int8': True, 'int16': True, 'int32': True, 'int64': True, 'bool': False}, 'int16': {'float16': False, 'float32': True, 'float64': True, 'complex64': True, 'complex128': True, 'uint8': False, 'int8': False, 'int16': True, 'int32': True, 'int64': True, 'bool': False}, 'int32': {'float16': False, 'float32': False, 'float64': True, 'complex64': False, 'complex128': True, 'uint8': False, 'int8': False, 'int16': False, 'int32': True, 'int64': True, 'bool': False}, 'int64': {'float16': False, 'float32': False, 'float64': True, 'complex64': False, 'complex128': True, 'uint8': False, 'int8': False, 'int16': False, 'int32': False, 'int64': True, 'bool': False}, 'bool': {'float16': True, 'float32': True, 'float64': True, 'complex64': True, 'complex128': True, 'uint8': True, 'int8': True, 'int16': True, 'int32': True, 'int64': True, 'bool': True}},

'same_kind': {'float16': {'float16': True, 'float32': True, 'float64': True, 'complex64': True, 'complex128': True, 'uint8': False, 'int8': False, 'int16': False, 'int32': False, 'int64': False, 'bool': False}, 'float32': {'float16': True, 'float32': True, 'float64': True, 'complex64': True, 'complex128': True, 'uint8': False, 'int8': False, 'int16': False, 'int32': False, 'int64': False, 'bool': False}, 'float64': {'float16': True, 'float32': True, 'float64': True, 'complex64': True, 'complex128': True, 'uint8': False, 'int8': False, 'int16': False, 'int32': False, 'int64': False, 'bool': False}, 'complex64': {'float16': False, 'float32': False, 'float64': False, 'complex64': True, 'complex128': True, 'uint8': False, 'int8': False, 'int16': False, 'int32': False, 'int64': False, 'bool': False}, 'complex128': {'float16': False, 'float32': False, 'float64': False, 'complex64': True, 'complex128': True, 'uint8': False, 'int8': False, 'int16': False, 'int32': False, 'int64': False, 'bool': False}, 'uint8': {'float16': True, 'float32': True, 'float64': True, 'complex64': True, 'complex128': True, 'uint8': True, 'int8': True, 'int16': True, 'int32': True, 'int64': True, 'bool': False}, 'int8': {'float16': True, 'float32': True, 'float64': True, 'complex64': True, 'complex128': True, 'uint8': False, 'int8': True, 'int16': True, 'int32': True, 'int64': True, 'bool': False}, 'int16': {'float16': True, 'float32': True, 'float64': True, 'complex64': True, 'complex128': True, 'uint8': False, 'int8': True, 'int16': True, 'int32': True, 'int64': True, 'bool': False}, 'int32': {'float16': True, 'float32': True, 'float64': True, 'complex64': True, 'complex128': True, 'uint8': False, 'int8': True, 'int16': True, 'int32': True, 'int64': True, 'bool': False}, 'int64': {'float16': True, 'float32': True, 'float64': True, 'complex64': True, 'complex128': True, 'uint8': False, 'int8': True, 'int16': True, 'int32': True, 'int64': True, 'bool': False}, 'bool': {'float16': True, 'float32': True, 'float64': True, 'complex64': True, 'complex128': True, 'uint8': True, 'int8': True, 'int16': True, 'int32': True, 'int64': True, 'bool': True}},

'unsafe': {'float16': {'float16': True, 'float32': True, 'float64': True, 'complex64': True, 'complex128': True, 'uint8': True, 'int8': True, 'int16': True, 'int32': True, 'int64': True, 'bool': True}, 'float32': {'float16': True, 'float32': True, 'float64': True, 'complex64': True, 'complex128': True, 'uint8': True, 'int8': True, 'int16': True, 'int32': True, 'int64': True, 'bool': True}, 'float64': {'float16': True, 'float32': True, 'float64': True, 'complex64': True, 'complex128': True, 'uint8': True, 'int8': True, 'int16': True, 'int32': True, 'int64': True, 'bool': True}, 'complex64': {'float16': True, 'float32': True, 'float64': True, 'complex64': True, 'complex128': True, 'uint8': True, 'int8': True, 'int16': True, 'int32': True, 'int64': True, 'bool': True}, 'complex128': {'float16': True, 'float32': True, 'float64': True, 'complex64': True, 'complex128': True, 'uint8': True, 'int8': True, 'int16': True, 'int32': True, 'int64': True, 'bool': True}, 'uint8': {'float16': True, 'float32': True, 'float64': True, 'complex64': True, 'complex128': True, 'uint8': True, 'int8': True, 'int16': True, 'int32': True, 'int64': True, 'bool': True}, 'int8': {'float16': True, 'float32': True, 'float64': True, 'complex64': True, 'complex128': True, 'uint8': True, 'int8': True, 'int16': True, 'int32': True, 'int64': True, 'bool': True}, 'int16': {'float16': True, 'float32': True, 'float64': True, 'complex64': True, 'complex128': True, 'uint8': True, 'int8': True, 'int16': True, 'int32': True, 'int64': True, 'bool': True}, 'int32': {'float16': True, 'float32': True, 'float64': True, 'complex64': True, 'complex128': True, 'uint8': True, 'int8': True, 'int16': True, 'int32': True, 'int64': True, 'bool': True}, 'int64': {'float16': True, 'float32': True, 'float64': True, 'complex64': True, 'complex128': True, 'uint8': True, 'int8': True, 'int16': True, 'int32': True, 'int64': True, 'bool': True}, 'bool': {'float16': True, 'float32': True, 'float64': True, 'complex64': True, 'complex128': True, 'uint8': True, 'int8': True, 'int16': True, 'int32': True, 'int64': True, 'bool': True}}
}


_result_type_dict = {
'float16': {'float16': 'float16', 'float32': 'float32', 'float64': 'float64', 'complex64': 'complex64', 'complex128': 'complex128', 'uint8': 'float16', 'int8': 'float16', 'int16': 'float32', 'int32': 'float64', 'int64': 'float64', 'bool': 'float16'}, 
'float32': {'float16': 'float32', 'float32': 'float32', 'float64': 'float64', 'complex64': 'complex64', 'complex128': 'complex128', 'uint8': 'float32', 'int8': 'float32', 'int16': 'float32', 'int32': 'float64', 'int64': 'float64', 'bool': 'float32'},
'float64': {'float16': 'float64', 'float32': 'float64', 'float64': 'float64', 'complex64': 'complex128', 'complex128': 'complex128', 'uint8': 'float64', 'int8': 'float64', 'int16': 'float64', 'int32': 'float64', 'int64': 'float64', 'bool': 'float64'},
'complex64': {'float16': 'complex64', 'float32': 'complex64', 'float64': 'complex128', 'complex64': 'complex64', 'complex128': 'complex128', 'uint8': 'complex64', 'int8': 'complex64', 'int16': 'complex64', 'int32': 'complex128', 'int64': 'complex128', 'bool': 'complex64'},
'complex128': {'float16': 'complex128', 'float32': 'complex128', 'float64': 'complex128', 'complex64': 'complex128', 'complex128': 'complex128', 'uint8': 'complex128', 'int8': 'complex128', 'int16': 'complex128', 'int32': 'complex128', 'int64': 'complex128', 'bool': 'complex128'},
'uint8': {'float16': 'float16', 'float32': 'float32', 'float64': 'float64', 'complex64': 'complex64', 'complex128': 'complex128', 'uint8': 'uint8', 'int8': 'int16', 'int16': 'int16', 'int32': 'int32', 'int64': 'int64', 'bool': 'uint8'},
'int8': {'float16': 'float16', 'float32': 'float32', 'float64': 'float64', 'complex64': 'complex64', 'complex128': 'complex128', 'uint8': 'int16', 'int8': 'int8', 'int16': 'int16', 'int32': 'int32', 'int64': 'int64', 'bool': 'int8'},
'int16': {'float16': 'float32', 'float32': 'float32', 'float64': 'float64', 'complex64': 'complex64', 'complex128': 'complex128', 'uint8': 'int16', 'int8': 'int16', 'int16': 'int16', 'int32': 'int32', 'int64': 'int64', 'bool': 'int16'},
'int32': {'float16': 'float64', 'float32': 'float64', 'float64': 'float64', 'complex64': 'complex128', 'complex128': 'complex128', 'uint8': 'int32', 'int8': 'int32', 'int16': 'int32', 'int32': 'int32', 'int64': 'int64', 'bool': 'int32'},
'int64': {'float16': 'float64', 'float32': 'float64', 'float64': 'float64', 'complex64': 'complex128', 'complex128': 'complex128', 'uint8': 'int64', 'int8': 'int64', 'int16': 'int64', 'int32': 'int64', 'int64': 'int64', 'bool': 'int64'},
'bool': {'float16': 'float16', 'float32': 'float32', 'float64': 'float64', 'complex64': 'complex64', 'complex128': 'complex128', 'uint8': 'uint8', 'int8': 'int8', 'int16': 'int16', 'int32': 'int32', 'int64': 'int64', 'bool': 'bool'}}

########################## end autogenerated part


__all__ = ['dtype_from_torch', 'dtype'] + dt_names

