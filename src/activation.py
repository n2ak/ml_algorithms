from __future__ import annotations
from typing import TYPE_CHECKING
from src.function import Function

from src.utils import printed
if TYPE_CHECKING:
    from .tensor import Tensor


def as_activation_layer(name, base=(Function,)):
    import functools
    import sys

    def decorator_factory(method):
        new_type = type(name, base, {
            "forward": lambda self, *
            args, **kwargs: method(*args, **kwargs)
        })
        setattr(sys.modules[__name__], name, new_type)

        @functools.wraps(method)
        def new_method():
            return method()  # return the method to be used normally
        return new_method
    return decorator_factory


@printed
@as_activation_layer(name="ReLU")
def relu(tensor: Tensor) -> Tensor:
    tensor = tensor.copy()
    tensor[tensor < 0] = 0
    return tensor


@printed
def sigmoid(tensor: Tensor) -> Tensor:
    return 1 / ((-tensor).exp() + 1)


@printed
def softmax(tensor: Tensor, dim: int = 0) -> Tensor:
    # avoids overflow , https://timvieira.github.io/blog/post/2014/02/11/exp-normalize-trick/
    max_ = tensor.max()
    tensor = (tensor - max_).exp()
    tensor = tensor/tensor.sum(axis=dim, keepdims=True)
    return tensor


@printed
def log_softmax(x: Tensor, dim=0) -> Tensor:
    x = x.softmax(dim=dim)
    x = x.log()
    return x
