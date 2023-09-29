from __future__ import annotations
from src.grad import MeanGradFn, SumGradFn  # TODO ,MaximumGradFn
import numpy as np
from .unary_ops import _unary_op
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from src._tensor import _Tensor
from src.grad.utils import register_grad_fn

from src.utils import _printed
printed_other_ops = _printed("other_ops")


@printed_other_ops
# @register_grad_fn(LinearGradFn)
def linear(a: _Tensor, w: _Tensor, b: _Tensor) -> _Tensor:
    """
    returns a*w+b
    """
    assert a.shape[-1] == w.shape[0], f"{a.shape}@{b.shape}"
    assert b is None or b.shape[-1] == w.shape[-1]
    res = (a @ w).biased(b)
    return res


@printed_other_ops
def biased(x: _Tensor, bias: _Tensor = None) -> _Tensor:
    if bias is not None:
        # assert tuple(x.shape) == tuple(bias.shape), f"{x.shape} != {bias.shape}"
        x += bias
    return x


@printed_other_ops
def sequential(x: _Tensor, layers) -> _Tensor:
    for layer in layers:
        x = layer(x)
    return x


@printed_other_ops
@register_grad_fn(MeanGradFn)
def mean(x: _Tensor, axis=None) -> _Tensor:
    return _unary_op(np.mean, x, axis=axis)


@printed_other_ops
@register_grad_fn(SumGradFn)
def sum(x: _Tensor, axis=None, keepdim=False) -> _Tensor:
    return _unary_op(np.sum, x, axis=axis, keepdims=keepdim)


@printed_other_ops
def _conv2d_output_shape(x: _Tensor, out_, ks, p=0, s=1, d=0):
    b, _, w, h = tuple(x.shape)
    s1, s2 = s if isinstance(s, tuple) else (s, s)
    p1, p2 = p if isinstance(p, tuple) else (p, p)
    d1, d2 = d if isinstance(d, tuple) else (d, d)
    ks1, ks2 = ks
    from math import ceil
    # w,h = (w-ks1+p1+s1)/s1,(h-ks2+p2+s2)/s2
    # w = ceil(w) if w - int(w) < .5 else ceil(w)+1
    # h = ceil(h) if h - int(h) < .5 else ceil(h)+1

    w = (w+2*p1-d1*(ks1-1)-1)//s1 + 1
    h = (h+2*p2-d2*(ks2-1)-1)//s2 + 1
    out_shape = b, out_, w, h
    return out_shape


@printed_other_ops
def conv2d(
    x: _Tensor,
    weights: _Tensor,
    bias: _Tensor = None,
):

    return x
