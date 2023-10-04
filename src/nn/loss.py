from __future__ import annotations
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from src._tensor import _Tensor
from src.grad.utils import register_grad_fn
from src.grad.grad import CrossEntropyGradFn, NLLGradFn
from src.utils import as_loss_layer, printed_loss


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
    from src import tensor
    t = t.numpy().astype(int)
    y = tensor.zeros((len(t), x.shape[-1]))
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
