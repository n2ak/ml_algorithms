from abc import ABC, abstractmethod
from typing import List
from .tensor import Tensor


class Optimizer(ABC):
    @abstractmethod
    def __init__(self, params: List[Tensor]) -> None:
        self.params = params

    @abstractmethod
    def step(self) -> None:
        pass


class SGD(Optimizer):
    def __init__(self, params, lr=.001) -> None:
        super().__init__(params)
        self.lr = lr

    def step(self) -> None:
        for p in self.params:
            if p.grad is not None:
                p[:] = p - self.lr * p.grad
