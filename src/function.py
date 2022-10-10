from abc import ABC , abstractmethod
from .utils import relu,sigmoid,softmax

class Function(ABC):
    @abstractmethod
    def __call__():
        raise 'Unimplimented'

class RelU(Function):
    def __call__(self,x):
        return relu(x)
class Sigmoid(Function):
    def __call__(self,x):
        return sigmoid(x)
class Softmax(Function):
    def __init__(self,dim=None) -> None:
        super().__init__()
        self.dim = dim or 0
    def __call__(self,x):
        return softmax(x,dim=self.dim)