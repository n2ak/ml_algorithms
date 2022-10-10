from abc import ABC , abstractmethod
import utils
class Function(ABC):
    @abstractmethod
    def __call__():
        raise 'Unimplimented'

class RelU(Function):
    def __call__(self,x):
        return utils.relu(x)
class Sigmoid(Function):
    def __call__(self,x):
        return utils.sigmoid(x)
class Softmax(Function):
    def __init__(self,dim=None) -> None:
        super().__init__()
        self.dim = dim or 0
    def __call__(self,x):
        return utils.softmax(x,dim=self.dim)