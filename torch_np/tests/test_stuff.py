import warnings

import pytest

pytest.importorskip("hypothesis")

from hypothesis import given
from hypothesis import strategies as st
from hypothesis.errors import HypothesisWarning
from hypothesis.extra.array_api import make_strategies_namespace

import torch_np

__all__ = ["xps"]

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=HypothesisWarning)
    xps = make_strategies_namespace(torch_np, api_version="2021.12")


def integer_array_indices(shape, result_shape) -> st.SearchStrategy[tuple]:
    # See hypothesis.extra.numpy.integer_array_indices()
    # n.b. result_shape only accepts a shape, as opposed to only accepting a strategy
    def array_for(index_shape, size):
        return xps.arrays(
            dtype=xps.integer_dtypes(),
            shape=index_shape,
            elements=st.integers(-size, size - 1),
        )

    return st.tuples(*(array_for(result_shape, size) for size in shape))


@given(
    x=xps.arrays(dtype=xps.integer_dtypes(), shape=xps.array_shapes()),
    data=st.data(),
)
def test_integer_indexing(x, data):
    result_shape = data.draw(xps.array_shapes(), label="result_shape")
    idx = data.draw(integer_array_indices(x.shape, result_shape), label="idx")
    result = x[idx]
    assert result.shape == result_shape
