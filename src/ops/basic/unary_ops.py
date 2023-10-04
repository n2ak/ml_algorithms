from __future__ import annotations
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from src._tensor import _Tensor
import numpy as np
from src.utils import printed_ops
from src.grad.utils import register_grad, _pass_gradient


def _unary_op(func, x: _Tensor, **kwargs):
    from src import tensor
    return tensor.from_numpy(func(x.numpy(), **kwargs))


@printed_ops
@register_grad()
def mean(x: _Tensor, axis=None) -> _Tensor:
    def backward(gradient):
        if (axis is not None) and (gradient.shape != () and gradient.shape != x.shape):
            gradient_shape = list(x.shape)
            gradient_shape[axis] = 1
            gradient = gradient.reshape(shape=gradient_shape)
        from src import tensor
        _pass_gradient(x, tensor.ns_like(x, 1/x.size) * gradient)
    return _unary_op(np.mean, x, axis=axis), backward


@register_grad()
def sum(x: _Tensor, axis=None, keepdim=False) -> _Tensor:
    from src import tensor

    def backward(gradient):
        if (axis is not None) and (gradient.shape != () and gradient.shape != x.shape):
            gradient_shape = list(x.shape)
            gradient_shape[axis] = 1
            gradient = gradient.reshape(shape=gradient_shape)
        _pass_gradient(x, tensor.ones(x.shape) * gradient)
    return _unary_op(np.sum, x, axis=axis, keepdims=keepdim), backward


@printed_ops
@register_grad()
def exp(x: _Tensor) -> _Tensor:
    def backward(gradient):
        _pass_gradient(x, x.exp() * gradient)
    return _unary_op(np.exp, x), backward


@printed_ops
@register_grad()
def log(x: _Tensor) -> _Tensor:
    def backward(gradient):
        _pass_gradient(x, 1/x * gradient)
    return _unary_op(np.log, x), backward
