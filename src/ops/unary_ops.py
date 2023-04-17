from __future__ import annotations
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from ..tensor import Tensor
from src.grad import *
import numpy as np
from src.grad.utils import register_grad_fn


def _unary_op(func, x: Tensor, **kwargs):
    from src.tensor import tensor
    return tensor(func(x.numpy(), **kwargs))


@register_grad_fn(LogGradFn)
def log(x: Tensor) -> Tensor:
    return _unary_op(np.log, x)


@register_grad_fn(ExpGradFn)
def exp(x: Tensor) -> Tensor:
    return _unary_op(np.exp, x)


# @register_grad_fn(PowGradFn)
# def pow(x: Tensor) -> Tensor:
#     return _unary_op(np.power, x)
