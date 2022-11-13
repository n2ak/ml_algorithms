from __future__ import annotations
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from .tensor import Tensor

from abc import ABC
from .core import HasForwardAndIsCallable


class Function(HasForwardAndIsCallable, ABC):
    pass


class RelU(Function):
    def forward(self, x: Tensor):
        return x.relu()


class Sigmoid(Function):
    def forward(self, x: Tensor):
        return x.sigmoid()


class Softmax(Function):
    def __init__(self, dim=None) -> None:
        super().__init__()
        self.dim = dim or 0

    def forward(self, x: Tensor):
        return x.softmax(dim=self.dim)


class LogSoftmax(Function):
    def __init__(self, dim=None) -> None:
        super().__init__()
        self.dim = dim or 0

    def forward(self, x: Tensor):
        return x.log_softmax(dim=self.dim)
