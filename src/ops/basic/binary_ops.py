from __future__ import annotations
from typing import TYPE_CHECKING
from src.grad import correct_shape
import numpy as np
from src.utils import printed_ops

from src.grad.utils import register_grad, _pass_gradient
if TYPE_CHECKING:
    from src._tensor import _Tensor


def _bin_op(func, x, other):
    from src import tensor
    res = tensor.from_numpy(func(np.array(x), np.array(other)))
    if np.isfinite(res).mean() != 1:
        print(
            "Infinite value, func:",
            func.__name__,
            x,
            other,
        )
        assert False
    return res


@register_grad(binary=True)
def add(x: _Tensor, other) -> _Tensor:
    def backward(gradient):
        from src import tensor
        from src._tensor import _Tensor
        new_grad = gradient
        for var in [x, other]:
            if isinstance(gradient, _Tensor) and isinstance(var, _Tensor) and var.shape != gradient.shape:
                new_grad = correct_shape(var, gradient)
            _pass_gradient(var, new_grad)
    return _bin_op(np.add, x, other), backward


@printed_ops
@register_grad(binary=True)
def sub(x: _Tensor, other) -> _Tensor:
    def backward(gradient):
        from src import tensor
        from src._tensor import _Tensor
        new_grad = gradient
        for var, M in zip([x, other], [1, -1]):
            if isinstance(gradient, _Tensor) and isinstance(var, _Tensor) and var.shape != gradient.shape:
                new_grad = correct_shape(var, gradient)
            _pass_gradient(var, M*new_grad)
    return _bin_op(np.subtract, x, other), backward


@printed_ops
@register_grad(binary=True)
def mul(x: _Tensor, other) -> _Tensor:
    def backward(gradient):
        _pass_gradient(x, other * gradient)
        _pass_gradient(other, x * gradient)

    return _bin_op(np.multiply, x, other), backward


@printed_ops
@register_grad(binary=True)
def pow(x: _Tensor, other) -> _Tensor:
    def backward(gradient):

        _pass_gradient(x, other * (x ** (other-1)) * gradient)
        _pass_gradient(other, (x ** other)*x.log() * gradient)
    return _bin_op(np.power, x, other), backward


@printed_ops
@register_grad(binary=True)
def truediv(x: _Tensor, other) -> _Tensor:
    def backward(gradient):
        gradient2 = -1*(x/(other**2))*gradient
        from src._tensor import _Tensor
        if isinstance(gradient2, _Tensor) and isinstance(other, _Tensor) and other.shape != gradient2.shape:
            gradient2 = correct_shape(other, gradient2)

        _pass_gradient(x, 1/other * gradient)
        _pass_gradient(other, gradient2)
    return _bin_op(np.divide, x, other), backward


@printed_ops
@register_grad(binary=True)
def rtruediv(x: _Tensor, other) -> _Tensor:
    x, other = other, x

    def backward(gradient):
        gradient2 = -1*(x/(other**2))*gradient
        from src._tensor import _Tensor
        if isinstance(gradient2, _Tensor) and isinstance(other, _Tensor) and other.shape != gradient2.shape:
            gradient2 = correct_shape(other, gradient2)

        _pass_gradient(x, 1/other * gradient)
        _pass_gradient(other, gradient2)
    return _bin_op(np.divide, x, other), backward


@printed_ops
@register_grad(binary=True)
def matmul(x: _Tensor, other) -> _Tensor:
    def backward(gradient):
        _pass_gradient(x, gradient @ other.T)
        _pass_gradient(other, x.T @ gradient)
    return _bin_op(np.matmul, x, other), backward


@printed_ops
def neg(x: _Tensor) -> _Tensor:
    return mul(x, -1)
