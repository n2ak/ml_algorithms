from __future__ import annotations
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from ..tensor import Tensor


def cross_entropy(x: Tensor, t, dim=0, reduction="none", from_logits=False) -> Tensor:
    # if from_logits:
    #     TODO : from_logits is True
    #     t = t.flatten().astype(np.int32)
    x = x.log_softmax(dim)
    x = x.negative_log_likelihood(t, reduction=reduction)
    return x


def negative_log_likelihood(x: Tensor, t, reduction="none") -> Tensor:
    from src import Tensor
    if reduction != "none":
        assert False, f"{reduction=} not supported"
    assert len(x.shape) == 2 and len(t.shape) == 1, f"{x.shape} , {t.shape}"
    t = t.numpy().astype(int)
    y = Tensor.zeros((len(t), x.shape[-1]))
    y[list(range(len(t))), t] = -1
    res = (x*y).sum(axis=1)
    return res
