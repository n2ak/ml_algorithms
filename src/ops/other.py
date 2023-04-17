from __future__ import annotations
from src.grad import MeanGradFn, SumGradFn  # TODO ,MaximumGradFn
import numpy as np
from .unary_ops import _unary_op
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from ..tensor import Tensor
from src.grad.utils import register_grad_fn


def linear(a: Tensor, w: Tensor, b: Tensor) -> Tensor:
    """
    returns a*w+b
    """
    assert a.shape[-1] == w.shape[0]
    assert b is None or b.shape[-1] == w.shape[-1]
    res = (a @ w).biased(b)
    return res


def biased(x: Tensor, bias: Tensor = None) -> Tensor:
    if bias is not None:
        # assert tuple(x.shape) == tuple(bias.shape), f"{x.shape} != {bias.shape}"
        x += bias
    return x


def sequential(x: Tensor, layers) -> Tensor:
    for layer in layers:
        x = layer(x)
    return x


@register_grad_fn(MeanGradFn)
def mean(x: Tensor, axis=None) -> Tensor:
    return _unary_op(np.mean, x, axis=axis)


@register_grad_fn(SumGradFn)
def sum(x: Tensor, axis=None) -> Tensor:
    return _unary_op(np.sum, x, axis=axis)


# @register_grad_fn(MaximumGradFn)
# def sum(x: Tensor) -> Tensor:
#     return _unary_op(np.maximum, x)
