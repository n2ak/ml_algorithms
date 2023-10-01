from __future__ import annotations
import numpy as np


class _Tensor:
    from src import nn
    import src.ops as ops

    _is_leaf = True
    _grad = None
    from src.grad import GradFn
    grad_fn: GradFn | None = None
    requires_grad = False

    def __init__(self, data: np.ndarray, requires_grad=False) -> None:
        super().__init__()
        self._init(requires_grad)
        self.data = data
        self.name = ""

    def requires_grad_(self, requires_grad=True):
        self.requires_grad = requires_grad
        return self

    def set_grad_fn(self, grad_fn):
        assert self.grad_fn is None, str(self.grad_fn)
        self.grad_fn = grad_fn

    @property
    def grad(self) -> _Tensor: return self._grad
    @property
    def shape(self) -> _Tensor: return self.data.shape
    @property
    def size(self) -> _Tensor: return self.data.size
    @property
    def dtype(self) -> _Tensor: return self.data.dtype

    @property
    def T(self) -> _Tensor:
        # TODO: make it use np.transpose()
        copy = self.copy()
        copy.data = copy.data.T
        return copy

    def copy(self):
        # TODO: might be wrong
        copy = tensor(self.data.copy(), requires_grad=self.requires_grad)
        return copy

    def reshape(self, *shape):
        # No copy
        t = self.copy()
        t.data = t.data.reshape(*shape)
        return t

    def _accumulate_grad(self, gradient):
        grad = self.grad
        if grad is None:
            grad = tensor(0)
        grad = grad + gradient
        self.set_grad(grad)

    def set_grad(self, grad):
        assert isinstance(grad, _Tensor)
        self_shape, grad_shape = tuple(self.shape), tuple(grad.shape)
        assert self_shape == grad_shape, f"Expected gradient of shape: {self_shape},recieved: {grad_shape}"
        self._grad = grad

    def zero_grad(self):
        # self._grad *=0
        if self._grad is not None:
            self.set_grad(tensor_zeros_like(self._grad))

    def _init(self, requires_grad):
        self._is_leaf = True
        self._grad = None
        self.grad_fn = None
        self.requires_grad = requires_grad

    def float(self):
        self.data = self.data.astype(float)
        return self

    def astype(self, type):
        self.data = self.data.astype(type)
        return self

    @classmethod
    def array(cls, arr: _Tensor | np.ndarray | list, dtype=np.float32, requires_grad=False) -> _Tensor:
        arr = np.array(arr, dtype=dtype)
        arr = cls(arr)
        arr._init(requires_grad)
        return arr

    @staticmethod
    def ns_like(x, n): return _Tensor.ns(x.shape, n)
    @staticmethod
    def zeros_like(x): return _Tensor.ns_like(x, 0)
    @staticmethod
    def ones_like(x): return _Tensor.ns_like(x, 1)
    @staticmethod
    def ns(shape, k): return tensor(np.zeros(shape)+k)
    @staticmethod
    def ones(shape): return _Tensor.ns(shape, 1)
    @staticmethod
    def zeros(shape): return _Tensor.ns(shape, 0)

    # def numpy(self): return np.add(self.data, 0)
    def numpy(self): return self.data

    def to_numpy(self, *args):
        return self.data.copy()

    __array__ = to_numpy

    def is_a_leaf(self): return self._is_leaf

    def can_calculatebackward(self, g):
        if self.is_a_leaf() and self.grad_fn is None:
            # TODO: IdentityGradFn
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

    def backward(self, gradient=1, print_ok=False):
        if not isinstance(gradient, _Tensor):
            gradient = tensor(gradient)
        assert self.can_calculatebackward(gradient)
        assert np.isfinite(self.data), f"Value is infinite: {self.data}"
        self.grad_fn.calculate(gradient, print_ok=print_ok)

    def is_scalar(self):
        # return np.isscalar(self)
        return self.shape == ()

    __add__ = ops.add
    __sub__ = ops.sub
    __mul__ = ops.mul
    __pow__ = ops.pow
    __div__ = ops.truediv
    __truediv__ = ops.truediv
    __rtruediv__ = ops.rtruediv
    __matmul__ = ops.matmul
    __neg__ = ops.neg
    mean = ops.mean
    sum = ops.sum
    log = ops.log
    exp = ops.exp
    __rmul__ = __mul__
    __radd__ = __add__
    __rsub__ = __sub__
    __iadd__ = __add__
    __isub__ = __sub__
    __imul__ = __mul__

    # ------------activations--------------
    relu = nn.activation.relu
    sigmoid = nn.activation.sigmoid
    softmax = nn.activation.softmax
    log_softmax = nn.activation.log_softmax
    # ------------loss--------------
    mse = nn.loss.mse
    cross_entropy = nn.loss.cross_entropy
    nll = nn.loss.negative_log_likelihood
    # ------------ops--------------
    biased = ops.biased
    linear = ops.linear
    conv2d = ops.conv2d
    sequential = ops.sequential
    flatten = ops.flatten

    def __len__(self):
        return self.data.__len__()
    # ------------$--------------
    unique = lambda self, *args, **kwargs: np.unique(self, *args, **kwargs)

    # NOTE: For tests
    def detach(self): return self

    def torch(self):
        import torch
        return torch.from_numpy(self.numpy()).requires_grad_(self.requires_grad)

    @classmethod
    def rand(cls, *args): return tensor(np.random.rand(*args))

    @classmethod
    def uniform(cls, shape, low=0, high=1): return tensor(
        np.random.uniform(low, high, shape))

    def __repr__(self) -> str:
        v = (
            # f".val          = {self.numpy()}",
            f".requires_grad= {self.requires_grad}",
            f".grad_fn      = {type(self.grad_fn).__name__}"
        )
        t = "\n".join(v)
        return f'<nn.Tensor\n{t}/>'

    def __str__(self) -> str:
        return str(self.numpy())

    # def __getattr__(self, f):
    #     copy = self.copy()
    #     func = getattr(np, f)
    #     copy.data = func()

    def argmax(self, *args):
        return tensor(self.data.argmax(*args), requires_grad=True)


tensor = _Tensor.array
from_numpy = tensor
tensor_zeros = _Tensor.zeros
tensor_zeros_like = _Tensor.zeros_like
tensor_ones = _Tensor.ones
tensor_ones_like = _Tensor.ones_like
tensor_ns_like = _Tensor.ns_like
