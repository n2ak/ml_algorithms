from abc import ABC , abstractmethod

from .core import HasForwardAndIsCallable
from .utils import relu,sigmoid,softmax

class Function(HasForwardAndIsCallable,ABC):
    pass

class RelU(Function):
    def forward(self,x):
        return relu(x)
class Sigmoid(Function):
    def forward(self,x):
        return sigmoid(x)
class Softmax(Function):
    def __init__(self,dim=None) -> None:
        super().__init__()
        self.dim = dim or 0
    def forward(self,x):
        return softmax(x,dim=self.dim)