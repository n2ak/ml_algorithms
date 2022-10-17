from abc import ABC, abstractmethod
import numpy as np
original_random_state = None
np.random.seed(1)
def seed(s):
    global original_random_state
    original_random_state = np.random.get_state()
    np.random.seed(s)
def unseed():
     np.random.set_state(original_random_state)

seed(1)
def jacobian(function,):
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