from abc import ABC, abstractmethod


def jacobian(function,):
    pass
class AddGradFn():
    pass
class HasForwardAndIsCallable(ABC):
    @abstractmethod
    def forward(self):
        raise "Unimplemented"

    def __call__(self,*args):
        return self.forward(*args)

def _backward(gradient,tensor):
    grad_fn = tensor.grad_fn
    raise "Unimplemented"