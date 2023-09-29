from __future__ import annotations
from typing import TYPE_CHECKING
from src._base import Loss
if TYPE_CHECKING:
    from src._tensor import _Tensor
from src.grad.utils import register_grad_fn
from src.grad.grad import CrossEntropyGradFn, NLLGradFn
from src.utils import _printed, as_layer

from functools import partial
printed_loss = _printed(type="loss")
as_loss_layer = partial(as_layer, module_name=__name__, base=(Loss,))


@printed_loss
@as_loss_layer("CrossEntropyLoss")
@register_grad_fn(CrossEntropyGradFn)
def cross_entropy(x: _Tensor, t, dim=-1, reduction="mean") -> _Tensor:
    # if from_logits:
    #     TODO : from_logits is True
    #     t = t.flatten().astype(np.int32)
    x = x.log_softmax(dim)
    x = x.nll(t, reduction=reduction)
    return x


@printed_loss
@as_loss_layer("NLLLoss")
@register_grad_fn(NLLGradFn)
def negative_log_likelihood(x: _Tensor, t, reduction="mean") -> _Tensor:
    from src._tensor import tensor_zeros
    t = t.numpy().astype(int)
    y = tensor_zeros((len(t), x.shape[-1]))
    y.data[list(range(len(t))), t] = -1
    res = (x*y).sum(axis=1)
    if reduction == "mean":
        res = res.mean()
    elif reduction == "sum":
        res = res.sum()
    return res


@printed_loss
@as_loss_layer("MSELoss")
# @register_grad_fn(MSEGradFn)
def mse(x: _Tensor, t) -> _Tensor:
    batch_size = x.shape[0]
    res = ((x - t)**2).sum() / batch_size
    return res
