from __future__ import annotations
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from src._tensor import _Tensor
from src.grad import *
import numpy as np
from src.grad.utils import register_grad_fn
from src.utils import _printed
printed_unary_ops = _printed("unary_ops")


def _unary_op(func, x: _Tensor, **kwargs):
    from src._tensor import tensor
    return tensor(func(x.numpy(), **kwargs))


@printed_unary_ops
@register_grad_fn(ExpGradFn)
def exp(x: _Tensor) -> _Tensor:
    return _unary_op(np.exp, x)


@printed_unary_ops
@register_grad_fn(LogGradFn)
def log(x: _Tensor) -> _Tensor:
    res = _unary_op(np.log, x)
    # from src.tensor import tensor
    return res
# @register_grad_fn(PowGradFn)
# def pow(x: _Tensor) -> _Tensor:
#     return _unary_op(np.power, x)
