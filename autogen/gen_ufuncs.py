from dump_namespace import grab_namespace, get_signature

import numpy as np

namespace = np

dct = grab_namespace(namespace)


# SKIP these (need special handling)
skip = {np.frexp, np.modf,    # non-standard unary ufunc signatures
        np.isnat,
        np.invert,      # bitwise NOT operator
        np.spacing,     # niche, does not have a direct equivalent
}

# np functions where torch names differ
torch_names = {np.radians : "deg2rad",
               np.degrees : "rad2deg",
               np.conjugate : "conj_physical",
               np.fabs : "absolute",       # FIXME: np.fabs raises form complex
               np.rint : "round"
}


# np functions which do not have a torch equivalent
default_stanza = "torch.{torch_name}(torch.as_tensor(x), out=out)"

stanzas = {np.cbrt : "torch.pow(torch.as_tensor(x), 1/3, out=out)",

           # XXX what on earth is np.positive
           np.positive: "+torch.as_tensor(x)",

           # these three do not have an out arg
           np.isinf: "torch.isinf(torch.as_tensor(x))",
           np.isnan: "torch.isnan(torch.as_tensor(x))",
           np.isfinite: "torch.isfinite(torch.as_tensor(x))",
}


# for these np functions, pytorch analog does not have the out= arg
needs_out = {np.isinf, np.isnan, np.isfinite, np.positive}
add_out_stanza = """
    if out is not None:
        out[...] = result
"""


header = """\
# this file is autogenerated via gen_ufuncs.py
# do not edit manually!

import torch

import _util
from _ndarray import asarray_replacer_1

"""

test_header = header + """\
import numpy as np
import torch

from _unary_ufuncs import *
from testing import assert_allclose
"""


template = """

@asarray_replacer_1
def {np_name}(x, /, out=None, *, where=True, casting='same_kind', order='K',
          dtype=None, subok=False, **kwds):
    _util.subok_not_ok(subok=subok)
    if order != 'K' or casting != 'same_kind' or not where:
        raise NotImplementedError
    if out is not None:
      # XXX dtypes, casting
        out = out.to(dtype)
    result = {torch_stanza}
    if dtype is not None:
        result = result.to(dtype)
    {out_stanza}
    return result

"""

test_template = """

def test_{np_name}():
    assert_allclose(np.{np_name}(0.5),
                    {np_name}(0.5), atol=1e-14, check_dtype=False)

"""


###### UNARY UFUNCS ###################################

_all_list = []
main_text = header
test_text = test_header

for ufunc in dct['ufunc']:
    if ufunc in skip:
        continue

    if ufunc.nin == 1:
        # print(get_signature(ufunc))

        torch_name = torch_names.get(ufunc)
        if torch_name is None:
            torch_name = ufunc.__name__

        torch_stanza = stanzas.get(ufunc)
        if torch_stanza is None:
            torch_stanza = default_stanza.format(torch_name=torch_name)

        out_stanza= add_out_stanza if ufunc in needs_out else ""

        main_text += template.format(np_name=ufunc.__name__,
                                     torch_stanza=torch_stanza,
                                     out_stanza=out_stanza)
        test_text += test_template.format(np_name=ufunc.__name__)

        _all_list.append(ufunc.__name__)

main_text += "\n\n__all__ = %s" % _all_list


with open("_unary_ufuncs.py", "w") as f:
    f.write(main_text)

with open("test_unary_ufuncs.py", "w") as f:
    f.write(test_text)


###### BINARY UFUNCS ###################################



test_header = header + """\
import numpy as np
import torch

from _binary_ufuncs import *
"""


template = """

def {np_name}(x1, x2, /, out=None, *, where=True, casting='same_kind', order='K',
            dtype=None, subok=False, **kwds):
    _util.subok_not_ok(subok=subok)
    if order != 'K' or casting != 'same_kind' or not where:
        raise NotImplementedError
    if out is not None:
        # XXX: dtypes, casting
        out = out.to(dtype)
    result = torch.{torch_name}(torch.as_tensor(x1), torch.as_tensor(x2), out=out)
    if dtype is not None:
        result = result.to(dtype)
    return result

"""

test_template = """

def test_{np_name}():
    np.testing.assert_allclose(np.{np_name}(0.5, 0.6),
                               {np_name}(0.5, 0.6), atol=1e-7)

"""



skip = {np.divmod,    # two outputs
}


torch_names = {np.power: "pow",
               np.equal: "eq",
}


_all_list = []
main_text = header
test_text = test_header

for ufunc in dct['ufunc']:

    if ufunc in skip:
        continue

    if ufunc.nin == 2:
   #     print(get_signature(ufunc))

        torch_name = torch_names.get(ufunc)
        if torch_name is None:
            torch_name = ufunc.__name__


        main_text += template.format(np_name=ufunc.__name__,
                                     torch_name=torch_name,)
#                                     out_stanza=out_stanza)
        test_text += test_template.format(np_name=ufunc.__name__)



with open("_binary_ufuncs.py", "w") as f:
    f.write(main_text)

with open("test_binary_ufuncs.py", "w") as f:
    f.write(test_text)

