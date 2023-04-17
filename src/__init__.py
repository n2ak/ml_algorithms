from .core import *
from .function import *
from .layers import *
from .tensor import *
from .utils import *
from .loss.loss import *
from .optimizer import *


class Module(HasForwardAndIsCallable):
    # TODO
    def __init__(self) -> None:
        self.params = None

    def zero_grad(self):
        params = self.get_parameters()
        for p in params:
            p.zero_grad()

    def get_parameters(self) -> list[Tensor]:
        if self.params:
            return self.params
        import inspect
        p = []
        for k, v in inspect.getmembers(self):
            if isinstance(v, Trainable):
                v: Trainable = v
                p.extend(v.get_trainable_params())
        self.params = p
        return self.params
