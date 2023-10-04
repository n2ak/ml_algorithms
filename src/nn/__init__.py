from __future__ import annotations
from .activation import *
from .layers import *
from .optimizer import *
from .loss import *
from src._base import _HasForwardAndIsCallable, _Trainable
import typing
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from src._tensor import _Tensor


class Module(_HasForwardAndIsCallable):
    # TODO: Module
    def __init__(self) -> None:
        self.params = None
        self._training = True
        self._built = False
        self.forward = self._forward(self.forward)

    def _build(self):
        for l in self._get_trainables():
            l._set_parent_module(self)
        self._built = True

    def _forward(self, forward):
        def h(*args, **kwargs):
            if not self._built:
                self._build()
            return forward(*args, **kwargs)
        return h

    @property
    def is_training(self): return self._training
    def train(self): self._training = True
    def infer(self): self._training = False

    def zero_grad(self):
        params = self.get_parameters()
        for p in params:
            p.zero_grad()

    def _get_trainables(self) -> typing.List[_Trainable]:
        import inspect
        return [l for k, l in inspect.getmembers(self) if isinstance(l, _Trainable)]

    def get_parameters(self) -> typing.List[_Tensor]:
        if self.params:
            return self.params
        self.params = []
        for l in self._get_trainables():
            self.params.extend(l.get_trainable_params())
        return self.params
