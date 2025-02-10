from ._tensor import Tensor, is_tensor
from .grad_utils import differentiable_function
import numpy as np


@differentiable_function(1)
def cross_entropy(input: Tensor, target, dim=-1, reduction="mean") -> Tensor:
    assert target.dtype == np.int32
    assert is_tensor(input)
    assert is_tensor(target)
    assert input.ndim == 2
    assert target.ndim == 1
    assert input.shape[0] == target.shape[0]

    def backward(gradient):
        dx = input.softmax(dim=dim).data
        n, *_ = input.shape
        dx[list(range(n)), target.data.astype(int)] -= 1
        if reduction == "mean":
            dx /= n
        elif reduction == "none":
            gradient = gradient[:, None]
        elif reduction == "sum":
            pass
        else:
            raise NotImplementedError()
        return dx * gradient
    x = input.log_softmax(dim)
    x = negative_log_likelihood(x, target, reduction=reduction)
    return x, backward


@differentiable_function(1)
def negative_log_likelihood(input: Tensor, target: Tensor, reduction="mean"):
    """`x` is typically a result of `x = logits.log_softmax()`"""
    assert is_tensor(input)
    assert is_tensor(target)
    assert np.all(input.data <= 0), input.data <= 0
    assert reduction in ["mean", "sum", "none"]

    def backward(gradient):
        len_ = len(t)
        dx = np.zeros((input.shape))
        if reduction == "mean":
            v = 1 / len_
        elif reduction == "none":
            v = 1
            gradient = gradient[:, None]
        elif reduction == "sum":
            v = 1
        dx[list(range(len_)), t] = -v
        # a = dx * gradient
        # del gradient
        return dx * gradient

    t = target.data.copy().astype(int)
    y = np.zeros((len(t), input.shape[-1]))
    y[list(range(len(t))), t] = 1

    res = -(input*y).sum(dim=1)
    if reduction == "mean":
        res = res.mean()
    elif reduction == "sum":
        res = res.sum()
    elif reduction == "none":
        pass
    return res, backward
