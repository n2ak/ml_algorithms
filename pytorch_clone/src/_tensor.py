import numpy as np
from .ops import _TensorOps


def is_tensor(x):
    return isinstance(x, Tensor)


class Tensor(_TensorOps):
    @property
    def T(self): return Tensor(self.data.T)
    @property
    def size(self): return self.data.size
    @property
    def ndim(self): return self.data.ndim
    @property
    def dtype(self): return self.data.dtype

    def __init__(self, data, requires_grad=False) -> None:
        if not isinstance(data, np.ndarray):
            data = np.array(data)
        assert not isinstance(data, Tensor)
        self.data = data
        self.make_require_grad(requires_grad)

    def make_require_grad(self, val: bool):
        self.gradient = None
        if val:
            def accumulate_grad(grad):
                assert isinstance(grad, np.ndarray)
                assert grad.shape == self.shape, \
                    f"Expected gradient shape to match tensor shape,but found {self.shape}=/={grad.shape}"
                if self.gradient is None:
                    self.gradient = np.asarray(0).astype(grad.dtype)
                self.gradient = self.gradient + grad
            from src.grad_utils import setup_backward_func
            setup_backward_func(accumulate_grad, "AccumulateGrad")
            self._set_backward_fn(accumulate_grad, )
        self.requires_grad = val

    # def accumulate_grad(self, grad):
    #     assert isinstance(grad, np.ndarray)
    #     assert grad.shape == self.shape, \
    #         f"Expected gradient shape to match tensor shape,but found {self.shape}=/={grad.shape}"
    #     if self.gradient is None:
    #         self.gradient = np.asarray(0).astype(grad.dtype)
    #     self.gradient = self.gradient + grad

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

    def _backward(self, gradient):
        assert self.requires_grad
        assert isinstance(gradient, np.ndarray)
        # print(self.__backward._fn_name, gradient)
        assert self.shape == gradient.shape, f"Expected gradient of shape {gradient.shape} to match the tensor's shape {self.shape}"
        self.__backward(gradient)

    def backward(self, gradient=1):
        if isinstance(gradient, int):
            gradient = np.array(gradient)
        from .grad_utils import grad_off
        with grad_off():
            self._backward(gradient)

    def __repr__(self) -> str:
        return self.data.__repr__()

    def __getitem__(self, item):
        return Tensor(self.data[item])

    def _set_backward_fn(self, backward):
        self.__backward = backward
