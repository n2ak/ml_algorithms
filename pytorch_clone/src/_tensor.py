import numpy as np
from .ops import ops


class Tensor:
    @property
    def T(self): return Tensor(self.data.T)
    @property
    def size(self): return self.data.size
    __add__ = ops.add
    __sub__ = ops.sub
    __mul__ = ops.mul
    __pow__ = ops.pow
    __div__ = ops.truediv
    __truediv__ = ops.truediv
    __rtruediv__ = ops.rtruediv
    __matmul__ = ops.matmul
    __neg__ = ops.neg

    __rmul__ = __mul__
    __radd__ = __add__
    __rsub__ = __sub__
    __iadd__ = __add__
    __isub__ = __sub__
    __imul__ = __mul__

    mean = ops.mean
    sum = ops.sum
    log = ops.log
    exp = ops.exp

    # ------------activations--------------
    tanh = ops.tanh
    relu = ops.relu
    sigmoid = ops.sigmoid
    softmax = ops.softmax
    log_softmax = ops.log_softmax
    # ------------loss--------------
    linear = ops.linear
    conv2d = ops.conv2d
    dropout = ops.dropout
    flatten = ops.flatten
    reshape = ops.reshape
    squeeze = ops.squeeze
    unsqueeze = ops.unsqueeze

    def __init__(self, data, requires_grad=False) -> None:
        if not isinstance(data, np.ndarray):
            data = np.array(data)
        self.data = data
        self.make_require_grad(requires_grad)

    def make_require_grad(self, val: bool):
        if val:
            self.gradient = None
            self._backward = self.accumulate_grad
        self.requires_grad = val

    def accumulate_grad(self, grad):
        assert isinstance(grad, np.ndarray)
        assert grad.shape == self.shape
        if self.gradient is None:
            self.gradient = np.asarray(0).astype(grad.dtype)
        self.gradient = self.gradient + grad

    def requires_grad_(self, val=True):
        self.make_require_grad(val)
        return self

    @property
    def shape(self): return self.data.shape

    @staticmethod
    def normal(shape):
        return Tensor(np.random.normal(0, 1, shape))

    from functools import partialmethod

    @staticmethod
    def ns(shape, n):
        return Tensor(np.zeros(shape) + n)

    @staticmethod
    def ns_like(x, n):
        return Tensor.ns(x.shape, n)

    ones = partialmethod(ns_like, n=1)
    zeros = partialmethod(ns_like, n=0)

    def copy(self): return Tensor(self.data.copy())

    def backward(self, gradient=1, strict=True):
        if not isinstance(gradient, np.ndarray):
            gradient = np.array(gradient)
        assert self.requires_grad
        assert gradient.shape == self.shape, (gradient.shape, self.shape)
        self._backward(gradient)
