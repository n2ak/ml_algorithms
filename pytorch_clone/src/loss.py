from ._tensor import Tensor, is_tensor
from .grad_utils import differentiable_function, pass_gradient
import numpy as np


@differentiable_function(1)
def cross_entropy(x: Tensor, t, dim=-1, reduction="mean") -> Tensor:
    assert t.dtype == np.int32
    assert is_tensor(x)
    assert is_tensor(t)
    assert x.ndim == 2
    assert t.ndim == 1
    assert x.shape[0] == t.shape[0]

    def backward(gradient):
        n, *_ = x.shape
        dx = x.softmax(dim=1).data
        dx[list(range(n)), t.data.astype(int)] -= 1
        dx /= n
        pass_gradient(x, dx * gradient)
    xx = x.log_softmax(dim)
    xx = negative_log_likelihood(xx, t, reduction=reduction)
    return xx, backward


@differentiable_function(1)
def negative_log_likelihood(x, tt, reduction="mean"):
    assert is_tensor(x)
    assert is_tensor(tt)
    assert np.all(x.data <= 0), x.data <= 0

    def backward(gradient):
        len_ = len(t)
        y = Tensor.zeros((x.shape))
        y.data[list(range(len_)), tt.numpy().astype(int)] = -(1/len_)
        pass_gradient(x, y * gradient)

    t = tt.data.astype(int)
    y = Tensor.zeros((len(t), x.shape[-1]))
    y.data[list(range(len(t))), t] = 1
    res = (x*y).sum(axis=1)
    if reduction == "mean":
        res = - res.mean()
    elif reduction == "sum":
        res = - res.sum()
    else:
        raise NotImplementedError()
    return res, backward
