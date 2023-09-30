from __future__ import annotations
from functools import partial
from typing import TYPE_CHECKING
from src._base import Function
from src.utils import _printed, as_layer
from src.grad import register_grad_fn, ReLUGradFn
printed_act = _printed(type="act")

if TYPE_CHECKING:
    from src._tensor import _Tensor


as_activation_layer = partial(as_layer, module_name=__name__, base=(Function,))


@printed_act
@as_activation_layer(name="ReLU")
@register_grad_fn(ReLUGradFn)
def relu(tensor: _Tensor) -> _Tensor:
    tensor = tensor.copy()
    d = tensor.numpy()
    d[d < 0] = 0
    return tensor


@printed_act
def sigmoid(tensor: _Tensor) -> _Tensor:
    return 1 / ((-tensor).exp() + 1)


@printed_act
@as_activation_layer(name="Softmax")
def softmax(x: _Tensor, dim: int = -1) -> _Tensor:
    m = x - x.data.max(axis=dim, keepdims=True)
    e = m.exp()
    _, e, ss = m, e, e.sum(axis=dim, keepdim=True)
    return e/ss
    # avoids overflow , https://timvieira.github.io/blog/post/2014/02/11/exp-normalize-trick/
    x = (x - x.numpy().max()).exp()
    x = x/(x.sum(axis=dim, keepdim=True))
    return x


@printed_act
@as_activation_layer(name="LogSoftmax")
def log_softmax(x: _Tensor, dim=-1) -> _Tensor:
    # https://stackoverflow.com/questions/61567597/how-is-log-softmax-implemented-to-compute-its-value-and-gradient-with-better
    new_x = x-x.data.max(axis=dim, keepdims=True)
    res = new_x - new_x.exp().sum(axis=dim, keepdim=True).log()
    return res
