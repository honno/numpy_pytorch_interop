""" Define the supported dtypes and numpy <--> torch dtype mapping, define casting rules. 
"""

# TODO: 1. define torch_np dtypes, make this work without numpy.
#       2. mimic numpy's various aliases (np.half == np.float16, dtype='i8' etc)
#       3. convert from python types: np.ones(3, dtype=float) etc

import numpy as np
import torch


class dtype:
    def __init__(self, name):
        self._name = name

dt_names = ['float16', 'float32', 'float64',
         'complex64', 'complex128',
         'uint8',
         'int8',
         'int16',
         'int32',
         'int64',
         'bool']

templ = """\
{name} = dtype("{name}")
"""


############### Output the dtypes #############

src_lines = [templ.format(name=name) for name in dt_names]
src = "".join(src_lines)
print(src)



############### Output the casting dict ############3

_casting_modes = ['no', 'equiv', 'safe', 'same_kind',  'unsafe']

# The structure is 
#_can_cast_dict["safe"]["dtyp1"]["dtyp2"]


def generate_can_cast(casting):
    """Dump the np casting table"""
    dct = {}
    for dtyp1 in dt_names:
        dct_dtyp1 = {}
        for dtyp2 in dt_names:
            can_cast = np.can_cast(np.dtype(dtyp1), np.dtype(dtyp2),
                                            casting=casting)
            dct_dtyp1[dtyp2] = can_cast
        dct[dtyp1] = dct_dtyp1
    return dct


def generate_result_type():
    """Dump the np.result_type table."""
    dct = {}
    for dtyp1 in dt_names:
        dct_dtyp1 = {}
        for dtyp2 in dt_names:
            result_type = np.result_type(np.dtype(dtyp1), np.dtype(dtyp2))
            dct_dtyp1[dtyp2] = result_type.name
        dct[dtyp1] = dct_dtyp1
    return dct


# pprint compact=True doesn't quite work :-)
#import pprint
#pprint.pprint(_can_cast_dict['no']['int32'], compact=True, width=100)


preamble = f"""
# These two dicts are autogenerated with autogen/gen_dtypes.py,
# using numpy version {np.__version__}.
"""
print(preamble)

_can_cast_dict = {}
for casting in _casting_modes:
    _can_cast_dict[casting] = generate_can_cast(casting)


print("_can_cast_dict = ", _can_cast_dict)
print("\n")
print("_result_type_dict = ", generate_result_type())

