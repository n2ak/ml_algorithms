from abc import ABC, abstractmethod
import numpy
numpy.random.seed(1)

def jacobian(function,):
    pass
class AddGradFn():
    pass
class HasForwardAndIsCallable(ABC):
    @abstractmethod
    def forward(self,*args,**kwargs):
        raise "Unimplemented"

    def __call__(self,*args,**kwargs):
        return self.forward(*args,**kwargs)

def _backward(gradient,tensor):
    grad_fn = tensor.grad_fn
    raise "Unimplemented"