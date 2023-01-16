import functools

import torch

from . import _dtypes, _helpers, _ufunc_impl, _util
from ._ndarray import asarray

#
# Functions in _ufunc_impl receive arrays, implement common tasks with ufunc args
# and delegate heavy lifting to pytorch equivalents.
#
# Functions in this file implement binary ufuncs: wrap two first arguments in
# asarray and delegate to functions from _ufunc_impl.
#
# One other user of _ufunc_impl functions in ndarray, where its __add__ method
# calls _ufunc_impl.add and so on. Note that ndarray dunders already know
# that its first arg is an array, so they only convert the second argument.
#
# XXX: While it sounds tempting to merge _binary_ufuncs.py and _ufunc_impl.py
# files, doing it would currently create import cycles.
#

# TODO: deduplicate with _unary_ufuncs/deco_unary_ufunc_from_impl,
# _ndarray/asarray_replacer, and _wrapper/concatenate et al
def deco_ufunc_from_impl(impl_func):
    @functools.wraps(impl_func)
    def wrapped(x1, x2, *args, **kwds):
        x1_array = asarray(x1)
        x2_array = asarray(x2)
        return impl_func(x1_array, x2_array, *args, **kwds)

    return wrapped


# the list is autogenerated, cf autogen/gen_ufunc_2.py
add = deco_ufunc_from_impl(_ufunc_impl.add)
arctan2 = deco_ufunc_from_impl(_ufunc_impl.arctan2)
bitwise_and = deco_ufunc_from_impl(_ufunc_impl.bitwise_and)
bitwise_or = deco_ufunc_from_impl(_ufunc_impl.bitwise_or)
bitwise_xor = deco_ufunc_from_impl(_ufunc_impl.bitwise_xor)
copysign = deco_ufunc_from_impl(_ufunc_impl.copysign)
divide = deco_ufunc_from_impl(_ufunc_impl.divide)
equal = deco_ufunc_from_impl(_ufunc_impl.equal)
float_power = deco_ufunc_from_impl(_ufunc_impl.float_power)
floor_divide = deco_ufunc_from_impl(_ufunc_impl.floor_divide)
fmax = deco_ufunc_from_impl(_ufunc_impl.fmax)
fmin = deco_ufunc_from_impl(_ufunc_impl.fmin)
fmod = deco_ufunc_from_impl(_ufunc_impl.fmod)
gcd = deco_ufunc_from_impl(_ufunc_impl.gcd)
greater = deco_ufunc_from_impl(_ufunc_impl.greater)
greater_equal = deco_ufunc_from_impl(_ufunc_impl.greater_equal)
heaviside = deco_ufunc_from_impl(_ufunc_impl.heaviside)
hypot = deco_ufunc_from_impl(_ufunc_impl.hypot)
lcm = deco_ufunc_from_impl(_ufunc_impl.lcm)
ldexp = deco_ufunc_from_impl(_ufunc_impl.ldexp)
left_shift = deco_ufunc_from_impl(_ufunc_impl.left_shift)
less = deco_ufunc_from_impl(_ufunc_impl.less)
less_equal = deco_ufunc_from_impl(_ufunc_impl.less_equal)
logaddexp = deco_ufunc_from_impl(_ufunc_impl.logaddexp)
logaddexp2 = deco_ufunc_from_impl(_ufunc_impl.logaddexp2)
logical_and = deco_ufunc_from_impl(_ufunc_impl.logical_and)
logical_or = deco_ufunc_from_impl(_ufunc_impl.logical_or)
logical_xor = deco_ufunc_from_impl(_ufunc_impl.logical_xor)
matmul = deco_ufunc_from_impl(_ufunc_impl.matmul)
maximum = deco_ufunc_from_impl(_ufunc_impl.maximum)
minimum = deco_ufunc_from_impl(_ufunc_impl.minimum)
remainder = deco_ufunc_from_impl(_ufunc_impl.remainder)
multiply = deco_ufunc_from_impl(_ufunc_impl.multiply)
nextafter = deco_ufunc_from_impl(_ufunc_impl.nextafter)
not_equal = deco_ufunc_from_impl(_ufunc_impl.not_equal)
power = deco_ufunc_from_impl(_ufunc_impl.power)
remainder = deco_ufunc_from_impl(_ufunc_impl.remainder)
right_shift = deco_ufunc_from_impl(_ufunc_impl.right_shift)
subtract = deco_ufunc_from_impl(_ufunc_impl.subtract)
divide = deco_ufunc_from_impl(_ufunc_impl.divide)
