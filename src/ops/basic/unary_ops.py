from __future__ import annotations
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from src._tensor import _Tensor
from src.grad.utils import register_grad_fn
from src.grad import MeanGradFn, SumGradFn, ExpGradFn, LogGradFn
import numpy as np
from src.utils import printed_ops


def _unary_op(func, x: _Tensor, **kwargs):
    from src import tensor
    return tensor.from_numpy(func(x.numpy(), **kwargs))


@printed_ops
@register_grad_fn(MeanGradFn)
def mean(x: _Tensor, axis=None) -> _Tensor:
    return _unary_op(np.mean, x, axis=axis)


@printed_ops
@register_grad_fn(SumGradFn)
def sum(x: _Tensor, axis=None, keepdim=False) -> _Tensor:
    return _unary_op(np.sum, x, axis=axis, keepdims=keepdim)


@printed_ops
@register_grad_fn(ExpGradFn)
def exp(x: _Tensor) -> _Tensor:
    return _unary_op(np.exp, x)


@printed_ops
@register_grad_fn(LogGradFn)
def log(x: _Tensor) -> _Tensor:
    res = _unary_op(np.log, x)
    # from src.tensor import tensor
    return res
