from __future__ import annotations
from abc import ABC, abstractmethod
from typing import List, TYPE_CHECKING
if TYPE_CHECKING:
    from src._tensor import _Tensor


class Optimizer(ABC):
    @abstractmethod
    def __init__(self, params: List[_Tensor]) -> None:
        self.params = params

    @abstractmethod
    def step(self) -> None:
        pass

    def zero_grad(self):
        for p in self.params:
            p.zero_grad()


class _HasForwardAndIsCallable(ABC):

    @abstractmethod
    def forward(self, *args, **kwargs) -> _Tensor:
        raise Exception("Unimplemented")

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


class _Trainable(ABC):
    @abstractmethod
    def get_trainable_params(self) -> List[_Tensor]:
        pass


class Layer(_Trainable, _HasForwardAndIsCallable, ABC):

    @classmethod
    def init_weights(cls, in_, out_, *args, weights: _Tensor | None = None):
        if weights is None:
            weights = _Tensor.rand(in_, out_, *args)
        else:
            assert weights.shape == (in_, out_), "Invalid weights passed"
        weights.requires_grad = True
        return weights


class Function(_HasForwardAndIsCallable, ABC):
    pass


class Loss(_HasForwardAndIsCallable, ABC):
    pass
