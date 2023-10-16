from __future__ import annotations
from functools import partial
from typing import TYPE_CHECKING
from src.utils import printed_act, as_activation_layer
from src.grad.utils import register_grad, _pass_gradient

if TYPE_CHECKING:
    from src._tensor import _Tensor


@as_activation_layer(name="ReLU")
@printed_act
@register_grad()
def relu(x: _Tensor) -> _Tensor:
    def backward(gradient):
        import numpy as np
        gradient.data[np.where(x.relu().data == 0)] = 0
        _pass_gradient(x, gradient)
    xx = x.copy()
    d = xx.numpy()
    d[d < 0] = 0
    return xx, backward


@as_activation_layer(name="Sigmoid")
@printed_act
def sigmoid(tensor: _Tensor) -> _Tensor:
    return 1 / ((-tensor).exp() + 1)


@as_activation_layer(name="Softmax")
@printed_act
def softmax(x: _Tensor, dim: int = -1) -> _Tensor:
    m = x - x.data.max(axis=dim, keepdims=True)
    e = m.exp()
    _, e, ss = m, e, e.sum(axis=dim, keepdim=True)
    return e/ss
    # avoids overflow , https://timvieira.github.io/blog/post/2014/02/11/exp-normalize-trick/
    x = (x - x.numpy().max()).exp()
    x = x/(x.sum(axis=dim, keepdim=True))
    return x


@as_activation_layer(name="LogSoftmax")
@printed_act
def log_softmax(x: _Tensor, dim=-1) -> _Tensor:
    # https://stackoverflow.com/questions/61567597/how-is-log-softmax-implemented-to-compute-its-value-and-gradient-with-better
    new_x = x-x.data.max(axis=dim, keepdims=True)
    res = new_x - new_x.exp().sum(axis=dim, keepdim=True).log()
    return res


@as_activation_layer(name="Tanh")
@printed_act
def tanh(x):
    a, b = x.exp(), (-x).exp()
    res = (a-b)/(a+b)
    return res
