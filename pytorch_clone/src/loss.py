from ._tensor import Tensor
from .grad_utils import differentiable_function, _pass_gradient


@differentiable_function(1)
def cross_entropy(x, t, dim=-1, reduction="mean"):
    def backward(gradient):
        n, *_ = x.shape
        dx = x.softmax(dim=1).data
        dx[list(range(n)), t.data.astype(int)] -= 1
        dx /= n
        _pass_gradient(x, dx * gradient)
    xx = x.log_softmax(dim)
    xx = negative_log_likelihood(x, t, reduction=reduction)
    return xx, backward


@differentiable_function(1)
def negative_log_likelihood(x, tt, reduction="mean"):
    def backward(gradient):
        len_ = len(t)
        y = Tensor.zeros((x.shape))
        y.data[list(range(len_)), tt.numpy().astype(int)] = -(1/len_)
        _pass_gradient(x, y * gradient)

    t = tt.data.astype(int)
    y = Tensor.zeros((len(t), x.shape[-1]))
    y.data[list(range(len(t))), t] = -1
    res = (x*y).sum(axis=1)
    if reduction == "mean":
        res = res.mean()
    elif reduction == "sum":
        res = res.sum()
    return res, backward
