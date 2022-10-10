import numpy as np
from .core import AddGradFn

from .utils import is_scalar, relu

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

    def relu(self): return relu(self)
    def exp(self): return Tensor.array(np.exp(self))
