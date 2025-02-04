import numpy as np
from .ops import bin_ops, unary_ops, other_ops, reduction_ops


def is_tensor(x):
    return isinstance(x, Tensor)


class Tensor:
    @property
    def T(self): return Tensor(self.data.T)
    @property
    def size(self): return self.data.size
    @property
    def ndim(self): return self.data.ndim
    @property
    def dtype(self): return self.data.dtype
    __add__ = bin_ops.add
    __sub__ = bin_ops.sub
    __mul__ = bin_ops.mul
    __pow__ = bin_ops.pow
    __div__ = bin_ops.truediv
    __truediv__ = bin_ops.truediv
    __rtruediv__ = bin_ops.rtruediv
    __matmul__ = bin_ops.matmul
    __neg__ = unary_ops.neg

    __rmul__ = __mul__
    __radd__ = __add__
    __rsub__ = __sub__
    __iadd__ = __add__
    __isub__ = __sub__
    __imul__ = __mul__

    mean = reduction_ops.mean
    sum = reduction_ops.sum
    log = unary_ops.log
    exp = unary_ops.exp

    # ------------activations--------------
    tanh = unary_ops.tanh
    relu = unary_ops.relu
    sigmoid = unary_ops.sigmoid
    softmax = unary_ops.softmax
    log_softmax = unary_ops.log_softmax
    # ------------other--------------
    linear = other_ops.linear
    conv2d = other_ops.conv2d
    conv2d_slow = other_ops.conv2d_slow
    conv2d_fast = other_ops.conv2d_fast
    dropout = other_ops.dropout
    flatten = other_ops.flatten
    reshape = other_ops.reshape
    squeeze = other_ops.squeeze
    unsqueeze = other_ops.unsqueeze

    def __init__(self, data, requires_grad=False) -> None:
        if not isinstance(data, np.ndarray):
            data = np.array(data)
        assert not isinstance(data, Tensor)
        self.data = data
        self.make_require_grad(requires_grad)

    def make_require_grad(self, val: bool):
        self.gradient = None
        if val:
            self._backward = self.accumulate_grad
        self.requires_grad = val

    def accumulate_grad(self, grad):
        assert isinstance(grad, np.ndarray)
        assert grad.shape == self.shape, \
            f"Expected gradient shape to match tensor shape,but found {self.shape}=/={grad.shape}"
        if self.gradient is None:
            self.gradient = np.asarray(0).astype(grad.dtype)
        self.gradient = self.gradient + grad

    def requires_grad_(self, val=True):
        self.make_require_grad(val)
        return self

    @property
    def shape(self): return self.data.shape

    @staticmethod
    def normal(shape, loc=0, scale=1):
        return Tensor(np.random.normal(loc, scale, shape))

    @staticmethod
    def randn(*shape):
        return Tensor(np.random.randn(*shape))

    def numpy(self):
        return self.data.copy()

    from functools import partialmethod

    @staticmethod
    def ns(shape, n):
        return Tensor(np.zeros(shape) + n)

    @staticmethod
    def ns_like(x, n):
        return Tensor.ns(x.shape, n)

    ones = partialmethod(ns, n=1)
    zeros = partialmethod(ns, n=0)
    ones_like = partialmethod(ns_like, n=1)
    zeros_like = partialmethod(ns_like, n=0)

    def item(self):
        assert self.size == 1, f"Expected size to be 1 but found: {self.size}"
        return self.data.tolist()

    def copy(self): return Tensor(self.data.copy())

    def backward(self, gradient=1, strict=True):
        if not isinstance(gradient, np.ndarray):
            gradient = np.array(gradient)
        assert self.requires_grad
        assert gradient.shape == self.shape, (gradient.shape, self.shape)
        # try:
        from .grad_utils import grad_off
        with grad_off():

            self._backward(gradient)
        # except:
            # print("Error")

    def __repr__(self) -> str:
        return self.data.__repr__()

    def __getitem__(self, item):
        return Tensor(self.data[item])
