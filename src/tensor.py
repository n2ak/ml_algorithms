from typing import List
import numpy as np
from .grad import AddGradFn, ExpGradFn, GradFn, IdentityGradFn, MeanGradFn, MulGradFn, SubGradFn, SumGradFn

from .utils import biased, cross_entropy, is_scalar, linear, log_softmax, negative_log_likelihood, relu, sequential, sigmoid, softmax


class Tensor(np.ndarray):
    _is_leaf = True
    _grad = None
    grad_fn: GradFn = None
    requires_grad = False
    def __init__(self,requires_grad =False) -> None:
        super().__init__()
        
        self._init(requires_grad)
    @property
    def grad(self):
        return self._grad
    def set_grad(self,grad):
        self._grad = self._grad or 0
        self._grad = grad
    def _init(self,requires_grad):
        self._is_leaf = True
        self._grad = None
        self.grad_fn: GradFn = None
        self.requires_grad = requires_grad
    def array(arr,dtype=np.float32,requires_grad=False):
        arr = np.array(arr,dtype=dtype).view(Tensor)
        arr._init(requires_grad)
        return arr
    @staticmethod
    def zeros_like(tensor):
        return Tensor.array(np.zeros_like(tensor))
    @staticmethod
    def ns_like(tensor,n):
        return Tensor.zeros_like(tensor) + n
    @staticmethod
    def ones_like(tensor):
        return Tensor.ns_like(tensor,1)

    def is_a_leaf(self): return self._is_leaf
    def can_calculatebackward(self,g):
        if self.is_a_leaf() and self.grad_fn is None:
            self.grad_fn = IdentityGradFn([self])
        if self.grad_fn is None:
            raise "No grad function"
        if self.requires_grad == False:
            raise "Tensor must require grad to perform this operation"
        if not self.is_scalar():
            raise "Cannot calculate gradient of a non-scalar"
        return True

    def backward(self,gradient=1):
        assert self.can_calculatebackward(gradient)
        self.grad_fn.calculate()

    def is_scalar(self):
        # return np.isscalar(self)
        return self.shape == ()
    def from_op(self,grad_fn):
        self._is_leaf = False
        self.grad_fn = grad_fn
        self.requires_grad = True

    @classmethod
    def filter(cls,dep):
        for v in dep:
            if isinstance(v,Tensor) and v.requires_grad:
                return True
        return False
    def bin_op(self,other,func,grad_fn,*args,**kwargs):
        a = Tensor.array(func(self,other,*args,**kwargs))
        return a.check_required(grad_fn,[self,other])
    def op(self,func,grad_fn,*args,**kwargs):
        a = Tensor.array(func(self,*args,**kwargs))
        return a.check_required(grad_fn,[self]) 
        
    def __add__(self,other): return self.bin_op(other,np.add,AddGradFn)
    def __sub__(self,other): return self.bin_op(other,np.subtract,SubGradFn)
    def __mul__(self,other): return self.bin_op(other,np.multiply,MulGradFn)
    def __rmul__(self,other): return self.__mul__(other)
    def __radd__(self,other): return self.__add__(other)
    def __rsub__(self,other): return self.__sub__(other)

    # todo
    def __iadd__(self,other): return self.bin_op(other,np.add,AddGradFn)
    def __sub__(self,other): return self.bin_op(other,np.subtract,SubGradFn)
    def __mul__(self,other): return self.bin_op(other,np.multiply,MulGradFn)
    def __rmul__(self,other): return self.__mul__(other)
    def __radd__(self,other): return self.__add__(other)
    def __rsub__(self,other): return self.__sub__(other)


    def numpy(self):
        return self.view(np.ndarray)

    def mean(self,*args,**kwargs): 
        r = Tensor.array(self.numpy().mean(*args,**kwargs))
        return r.check_required(MeanGradFn,[self])
    def sum(self,*args,**kwargs): 
        r = Tensor.array(self.numpy().sum(*args,**kwargs))
        return r.check_required(SumGradFn,[self])

    def check_required(self,grad_fn,dep):
        requires_grad = Tensor.filter(dep)
        if not requires_grad:
            return self
        self.from_op(grad_fn(dep))
        return self

    @classmethod
    def rand(cls,*args): return Tensor.array(np.random.rand(*args))
    @classmethod
    def zeros(cls,*args): return Tensor.array(np.zeros(*args))
    

    # @classmethod
    
def bind():
    def _bind(cls,func,grad_fn):
        def perform(tensor,*args,**kwargs):
            r : Tensor = func(tensor,*args,**kwargs)
            dep = [tensor,*args]
            if not grad_fn: return r
            return r.check_required(grad_fn,dep)
        return setattr(cls,func.__name__,lambda self,*args,**kwargs: perform(self,*args,**kwargs))
    methods = [
        (np.exp,ExpGradFn),
        (np.log,None),
        (np.unique,None),
        (linear,None),
        (relu,None),
        (sigmoid,None),
        (sequential,None),
        (softmax,None),
        (log_softmax,None),
        (negative_log_likelihood,None),
        (cross_entropy,None),

        (biased,None),
    ]
    print("Bound to numpy")
    for method,grad_fn in methods:
        _bind(Tensor,method,grad_fn)
bind()


