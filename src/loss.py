from abc import ABC

from .core import HasForwardAndIsCallable


class Loss(HasForwardAndIsCallable,ABC):
    pass
class CrossEntropyLoss(Loss):
    def __call__(self,x):
        # TODO
        return x