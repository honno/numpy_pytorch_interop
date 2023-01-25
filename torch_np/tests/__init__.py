import warnings

from hypothesis.errors import HypothesisWarning
from hypothesis.extra.array_api import make_strategies_namespace

import torch_np

__all__ = ["xps"]

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=HypothesisWarning)
    xps = make_strategies_namespace(torch_np, api_version="2021.12")
