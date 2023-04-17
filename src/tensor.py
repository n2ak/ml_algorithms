from __future__ import annotations
import numpy as np


class Tensor:
    import src.activation as act
    import src.ops as ops
    import src.loss as loss

    _is_leaf = True
    _grad = None
    from .grad import GradFn
    grad_fn: GradFn | None = None
    requires_grad = False

    def __init__(self, data: np.ndarray, requires_grad=False) -> None:
        super().__init__()
        self._init(requires_grad)
        self.data = data

    def requires_grad_(self):
        self.requires_grad = True
        return self

    def set_grad_fn(self, grad_fn):
        assert self.grad_fn is None, str(self.grad_fn)
        self.grad_fn = grad_fn

    @property
    def grad(self) -> Tensor: return self._grad
    @property
    def shape(self) -> Tensor: return self.data.shape
    @property
    def size(self) -> Tensor: return self.data.size
    @property
    def T(self) -> Tensor: return self.data.T

    def reshape(self, *shape):
        self.data.reshape(*shape)
        return self

    def set_grad(self, grad):
        self._grad = grad
        assert tuple(self.shape) == tuple(
            self._grad.shape), f"{tuple(self.shape)},{tuple(self._grad.shape)}"

    def zero_grad(self):
        # self._grad *=0
        self._grad = np.zeros_like(self._grad)

    def _init(self, requires_grad):
        self._is_leaf = True
        self._grad = None
        self.grad_fn = None
        self.requires_grad = requires_grad

    @classmethod
    def array(cls, arr: Tensor | np.ndarray | list, dtype=np.float32, requires_grad=False) -> Tensor:
        arr = cls(np.array(arr, dtype=dtype))
        arr._init(requires_grad)
        return arr

    @staticmethod
    def ns_like(x, n): return Tensor.ns(x.shape, n)
    @staticmethod
    def zeros_like(x): return Tensor.ns_like(x.shape, 0)
    @staticmethod
    def ones_like(x): return Tensor.ns_like(x.shape, 1)
    @staticmethod
    def ns(shape, k): return tensor(np.zeros(shape)+k)
    @staticmethod
    def ones(shape): return Tensor.ns(shape, 1)
    @staticmethod
    def zeros(shape): return Tensor.ns(shape, 0)

    def numpy(self): return self.data

    __array__ = numpy

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
        if not isinstance(gradient, Tensor):
            gradient = Tensor.array(gradient)
        assert self.can_calculatebackward(gradient)
        self.grad_fn.calculate(gradient)

    def is_scalar(self):
        # return np.isscalar(self)
        return self.shape == ()

    __add__ = ops.add
    __sub__ = ops.sub
    __mul__ = ops.mul
    __pow__ = ops.pow
    __truediv__ = ops.truediv
    __matmul__ = ops.matmul
    __neg__ = ops.neg
    mean = ops.mean
    sum = ops.sum
    __rmul__ = __mul__
    __radd__ = __add__
    __rsub__ = __sub__
    __iadd__ = __add__
    __isub__ = __sub__
    __imul__ = __mul__

    # ------------activations--------------
    relu = act.relu
    sigmoid = act.sigmoid
    softmax = act.softmax
    log_softmax = act.log_softmax
    # ------------loss--------------
    negative_log_likelihood = loss.negative_log_likelihood
    cross_entropy = loss.cross_entropy
    sequential = ops.sequential
    biased = ops.biased
    linear = ops.linear
    # ------------$--------------
    unique = lambda self, *args, **kwargs: np.unique(self, *args, **kwargs)

    def torch(self):
        import torch
        return torch.tensor(self.numpy(), requires_grad=self.requires_grad)

    @classmethod
    def rand(cls, *args): return tensor(np.random.rand(*args))

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


tensor = Tensor.array
