from __future__ import annotations
import traceback
from src.grad import *
from typing import TYPE_CHECKING
import numpy as np
from src.grad.utils import register_grad_fn
from src.utils import _printed
printed_binary_ops = _printed("binary_ops")


if TYPE_CHECKING:
    from src._tensor import _Tensor

# @printed_binary_ops


def _bin_op(func, x, other):
    from src._tensor import tensor
    res = tensor(func(np.array(x), np.array(other)))
    if np.isfinite(res).mean() != 1:
        print("Infinite value, func:", func.__name__)
        pass
    return res


@printed_binary_ops
@register_grad_fn(AddGradFn)
def add(x: _Tensor, other) -> _Tensor:
    return _bin_op(np.add, x, other)


@printed_binary_ops
@register_grad_fn(SubGradFn)
def sub(x: _Tensor, other) -> _Tensor:
    return _bin_op(np.subtract, x, other)


@printed_binary_ops
@register_grad_fn(MulGradFn)
def mul(x: _Tensor, other) -> _Tensor:
    return _bin_op(np.multiply, x, other)


@printed_binary_ops
@register_grad_fn(PowGradFn)
def pow(x: _Tensor, other) -> _Tensor:
    return _bin_op(np.power, x, other)


@printed_binary_ops
@register_grad_fn(DivGradFn)
def truediv(x: _Tensor, other) -> _Tensor:
    return _bin_op(np.divide, x, other)


@printed_binary_ops
@register_grad_fn(DivGradFn, reverse=True)
def rtruediv(x: _Tensor, other) -> _Tensor:
    return _bin_op(np.divide, other, x)


@printed_binary_ops
@register_grad_fn(MatMulGradFn)
def matmul(x: _Tensor, other) -> _Tensor:
    return _bin_op(np.matmul, x, other)


@printed_binary_ops
@register_grad_fn(MulGradFn)
def neg(x: _Tensor) -> _Tensor:
    # TODO np.negative is wrong
    return _bin_op(np.multiply, x, -1)
