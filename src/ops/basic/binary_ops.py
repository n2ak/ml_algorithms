from __future__ import annotations
from typing import TYPE_CHECKING
from src.grad import AddGradFn, SubGradFn, MulGradFn, PowGradFn, DivGradFn, MatMulGradFn
import numpy as np
from src.grad.utils import register_grad_fn
from ..utils import printed_ops

if TYPE_CHECKING:
    from src._tensor import _Tensor

# @printed_ops


def _bin_op(func, x, other):
    from src._tensor import tensor
    res = tensor(func(np.array(x), np.array(other)))
    if np.isfinite(res).mean() != 1:
        print(
            "Infinite value, func:",
            func.__name__,
            x,
            other,
        )
        assert False
    return res


@printed_ops
@register_grad_fn(AddGradFn)
def add(x: _Tensor, other) -> _Tensor:
    return _bin_op(np.add, x, other)


@printed_ops
@register_grad_fn(SubGradFn)
def sub(x: _Tensor, other) -> _Tensor:
    return _bin_op(np.subtract, x, other)


@printed_ops
@register_grad_fn(MulGradFn)
def mul(x: _Tensor, other) -> _Tensor:
    return _bin_op(np.multiply, x, other)


@printed_ops
@register_grad_fn(PowGradFn)
def pow(x: _Tensor, other) -> _Tensor:
    return _bin_op(np.power, x, other)


@printed_ops
@register_grad_fn(DivGradFn)
def truediv(x: _Tensor, other) -> _Tensor:
    return _bin_op(np.divide, x, other)


@printed_ops
@register_grad_fn(DivGradFn, reverse=True)
def rtruediv(x: _Tensor, other) -> _Tensor:
    return _bin_op(np.divide, other, x)


@printed_ops
@register_grad_fn(MatMulGradFn)
def matmul(x: _Tensor, other) -> _Tensor:
    return _bin_op(np.matmul, x, other)


@printed_ops
def neg(x: _Tensor) -> _Tensor:
    return mul(x, -1)
