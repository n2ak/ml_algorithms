from __future__ import annotations
from src.grad import *
from typing import TYPE_CHECKING
import numpy as np
from src.grad.utils import register_grad_fn


if TYPE_CHECKING:
    from src.tensor import Tensor


def _bin_op(func, x, other):
    from src.tensor import tensor
    return tensor(func(x.numpy(), np.array(other)))


@register_grad_fn(AddGradFn)
def add(x: Tensor, other) -> Tensor:
    return _bin_op(np.add, x, other)


@register_grad_fn(SubGradFn)
def sub(x: Tensor, other) -> Tensor:
    return _bin_op(np.subtract, x, other)


@register_grad_fn(MulGradFn)
def mul(x: Tensor, other) -> Tensor:
    return _bin_op(np.multiply, x, other)


@register_grad_fn(PowGradFn)
def pow(x: Tensor, other) -> Tensor:
    return _bin_op(np.power, x, other)


@register_grad_fn(DivGradFn)
def truediv(x: Tensor, other) -> Tensor:
    return _bin_op(np.divide, x, other)


@register_grad_fn(MatMulGradFn)
def matmul(x: Tensor, other) -> Tensor:
    return _bin_op(np.matmul, x, other)


@register_grad_fn(MulGradFn)
def neg(x: Tensor) -> Tensor:
    # TODO np.negative is wrong
    return _bin_op(np.multiply, x, -1)
