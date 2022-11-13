from __future__ import annotations
from typing import List, Callable
import numpy as np
from .grad import AddGradFn, DivGradFn, ExpGradFn, GradFn, IdentityGradFn, MatMulGradFn, MeanGradFn, MulGradFn, PowGradFn, SubGradFn, SumGradFn

from .utils import biased, cross_entropy, is_scalar, linear, log_softmax, negative_log_likelihood, relu, sequential, sigmoid, softmax


class Tensor(np.ndarray):
    _is_leaf = True
    _grad = None
    grad_fn: GradFn | None = None
    requires_grad = False

    def __init__(self, requires_grad=False) -> None:
        super().__init__()

        self._init(requires_grad)

    def requires_grad_(self):
        self.requires_grad = True
        return self

    @property
    def grad(self):
        return self._grad

    def set_grad(self, grad):
        self._grad = grad

    def _init(self, requires_grad):
        self._is_leaf = True
        self._grad = None
        self.grad_fn = None
        self.requires_grad = requires_grad

    def array(arr: Tensor | np.ndarray | list, dtype=np.float32, requires_grad=False) -> Tensor:
        arr = np.array(arr, dtype=dtype).view(Tensor)
        arr._init(requires_grad)
        return arr

    @staticmethod
    def zeros_like(tensor):
        return Tensor.array(np.zeros_like(tensor))

    @staticmethod
    def ns_like(tensor, n):
        return Tensor.zeros_like(tensor) + n

    @staticmethod
    def ones_like(tensor):
        return Tensor.ns_like(tensor, 1)

    def is_a_leaf(self): return self._is_leaf

    def can_calculatebackward(self, g):
        if self.is_a_leaf() and self.grad_fn is None:
            # TODO
            # self.grad_fn = IdentityGradFn([self])
            pass
        if self.grad_fn is None:
            raise Exception("No grad function")
        if self.requires_grad == False:
            raise Exception(
                "Tensor must require grad to perform this operation")
        if not self.is_scalar():
            raise Exception("Cannot calculate gradient of a non-scalar")
        return True

    def backward(self, gradient=1):
        assert self.can_calculatebackward(gradient)
        self.grad_fn.calculate()

    def is_scalar(self):
        # return np.isscalar(self)
        return self.shape == ()

    def from_op(self, grad_fn):
        self._is_leaf = False
        self.grad_fn = grad_fn
        self.requires_grad = True

    @classmethod
    def filter(cls, dep):
        for v in dep:
            if isinstance(v, Tensor) and v.requires_grad:
                return True
        return False

    def bin_op(self, other, func, grad_fn, *args, **kwargs):
        a = Tensor.array(func(self, other, *args, **kwargs))
        return a.check_required(grad_fn, [self, other])

    def op(self, func, grad_fn, *args, **kwargs):
        a = Tensor.array(func(self, *args, **kwargs))
        return a.check_required(grad_fn, [self])

    def __add__(self, other): return self.bin_op(other, np.add, AddGradFn)
    def __sub__(self, other): return self.bin_op(other, np.subtract, SubGradFn)
    def __mul__(self, other): return self.bin_op(other, np.multiply, MulGradFn)
    def __pow__(self, other): return self.bin_op(other, np.power, PowGradFn)

    def __truediv__(self, other): return self.bin_op(
        other, np.divide, DivGradFn)
    def __matmul__(self, other): return self.bin_op(
        other, np.matmul, MatMulGradFn)

    # TODO : might need its own GradFn
    def __neg__(self): return self.bin_op(-1, lambda x, _: -1*x, MulGradFn)

    def __rmul__(self, other): return self.__mul__(other)
    def __radd__(self, other): return self.__add__(other)
    def __rsub__(self, other): return self.__sub__(other)

    # TODO
    def __iadd__(self, other): return self.bin_op(other, np.add, AddGradFn)

    def __isub__(self, other): return self.bin_op(
        other, np.subtract, SubGradFn)
    def __imul__(self, other): return self.bin_op(
        other, np.multiply, MulGradFn)

    def numpy(self): return self.view(np.ndarray)

    def torch(self):
        import torch
        return torch.tensor(self.numpy(), requires_grad=self.requires_grad)

    def mean(self, *args, **kwargs):
        r = Tensor.array(self.numpy().mean(*args, **kwargs))
        return r.check_required(MeanGradFn, [self])

    def sum(self, *args, **kwargs):
        r = Tensor.array(self.numpy().sum(*args, **kwargs))
        return r.check_required(SumGradFn, [self])

    def check_required(self, grad_fn, dep):
        requires_grad = Tensor.filter(dep)
        if not requires_grad:
            return self
        self.from_op(grad_fn(dep))
        return self

    @classmethod
    def rand(cls, *args): return Tensor.array(np.random.rand(*args))
    @classmethod
    def zeros(cls, *args): return Tensor.array(np.zeros(*args))

    def __repr__(self) -> str:
        v = (f".val          ={self.numpy()}",
             f".requires_grad={self.requires_grad}",
             f".grad_fn      ={self.grad_fn}")
        return "<tensor.Tensor\n%s\n%s\n%s>" % v

    def __str__(self) -> str:
        return self.__repr__()

    linear = linear
    relu = relu
    sigmoid = sigmoid
    sequential = sequential
    softmax = softmax
    log_softmax = log_softmax
    negative_log_likelihood = negative_log_likelihood
    cross_entropy = cross_entropy
    biased = biased

    unique = lambda self, *args, **kwargs: np.unique(self, *args, **kwargs)

    def exp(self, *args, **kwargs) -> Tensor: ...
    def log(self, *args, **kwargs) -> Tensor: ...


def bind():
    def _bind(cls, func: Callable[[Tensor], Tensor], grad_fn: GradFn) -> None:
        def perform(tensor, *args, **kwargs):
            r = func(tensor, *args, **kwargs)
            dep = [tensor, *args]
            if not grad_fn:
                return r
            return r.check_required(grad_fn, dep)
        return setattr(cls, func.__name__, lambda self, *args, **kwargs: perform(self, *args, **kwargs))
    methods = [
        (np.exp, ExpGradFn),
        (np.log, None),

        # (np.unique,None),

        # (linear,None),
        # (relu,None),
        # (sigmoid,None),
        # (sequential,None),
        # (softmax,None),
        # (log_softmax,None),
        # (negative_log_likelihood,None),
        # (cross_entropy,None),
        # (biased,None),
    ]
    print("Bound to numpy")
    for method, grad_fn in methods:
        _bind(Tensor, method, grad_fn)


bind()
