import numpy as np
from .core import AddGradFn

from .utils import cross_entropy, is_scalar, linear, log_softmax, negative_log_likelihood, relu, sequential, sigmoid, softmax


class Tensor(np.ndarray):
    def __init__(self) -> None:
        super().__init__()

        self.grad = None
        self.grad_fn = None
    def array(arr,dtype=np.float32):
        arr = np.array(arr,dtype=dtype)
        return arr.view(Tensor)

    def can_calculatebackward(self):
        if self.grad_fn is None:
            raise ""
        if not self.is_scalar():
            raise "Cannot calculate gradient of a non-scalar"

    def backward(self,gradient):
        assert self.can_calculatebackward(gradient,self)
    
    def is_scalar(self):
        return is_scalar(self)
    def __add__(self,other):
        a = Tensor.array(np.add(self,other))
        a.grad_fn = AddGradFn
        return a
    @classmethod
    def rand(cls,*args): return Tensor.array(np.random.rand(*args))
    @classmethod
    def zeros(cls,*args): return Tensor.array(np.zeros(*args))


    # @classmethod
    
def bind():
    def _bind(cls,func):
        return setattr(cls,func.__name__,lambda self,*args,**kwargs: func(self,*args,**kwargs))
    methods = [
        np.exp,
        np.log,
        np.unique,
        linear,
        relu,
        sigmoid,
        sequential,
        softmax,
        log_softmax,
        negative_log_likelihood,
        cross_entropy,
    ]
    print("Bound to numpy")
    for method in methods:
        _bind(Tensor,method)
bind()


