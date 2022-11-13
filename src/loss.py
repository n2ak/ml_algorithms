from __future__ import annotations
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from .tensor import Tensor

from abc import ABC
from .core import HasForwardAndIsCallable


class Loss(HasForwardAndIsCallable, ABC):
    pass


class CrossEntropyLoss(Loss):
    def __init__(self, reduction=None, from_logits=False) -> None:
        super().__init__()
        self.reduction = reduction
        self.from_logits = from_logits

    def forward(self, x: Tensor, t):
        return x.cross_entropy(t, dim=-1, reduction=self.reduction, from_logits=self.from_logits)


class NegativeLogLikelihoodLoss(Loss):
    def __init__(self, reduction=None, from_logits=False) -> None:
        super().__init__()
        self.reduction = reduction
        self.from_logits = from_logits

    def forward(self, x: Tensor, t):
        return x.negative_log_likelihood(t, reduction=self.reduction)
