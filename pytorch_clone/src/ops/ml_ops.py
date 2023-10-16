from __future__ import annotations
from src.utils import printed_ml_ops, as_layer
# TODO ,MaximumGradFn
import numpy as np
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from src._tensor import _Tensor

from src.grad.utils import register_grad, _pass_gradient


@printed_ml_ops
def flatten(x: _Tensor, start_dim=0, end_dim=-1):
    shape = x.shape
    if end_dim < 0:
        end_dim = len(shape)+end_dim
    end_dim = end_dim+1
    new_shape = * \
        shape[:start_dim], np.prod(shape[start_dim:end_dim]), *shape[end_dim:]
    return x.reshape(new_shape)


@printed_ml_ops
@register_grad()
def reshape(x, shape):
    def backward(gradient):
        gradient = gradient.reshape(shape=x.shape)
        _pass_gradient(x, gradient)
    # No copy
    t = x.copy()
    t.data = t.data.reshape(*shape)
    return t, backward


@printed_ml_ops
def linear(a: _Tensor, w: _Tensor, b: _Tensor) -> _Tensor:
    """
    returns a*w+b
    """
    assert a.shape[-1] == w.shape[0], f"{a.shape}@{b.shape}"
    assert b is None or b.shape[-1] == w.shape[-1]
    res = (a @ w).biased(b)
    return res


@printed_ml_ops
def biased(x: _Tensor, bias: _Tensor = None) -> _Tensor:
    if bias is not None:
        # assert tuple(x.shape) == tuple(bias.shape), f"{x.shape} != {bias.shape}"
        x += bias
    return x


@printed_ml_ops
def sequential(x: _Tensor, layers) -> _Tensor:
    for layer in layers:
        x = layer(x)
    return x


@printed_ml_ops
def _conv2d_output_shape(x: _Tensor, out_, ks, p=0, s=1, d=0):
    b, _, w, h = tuple(x.shape)
    s1, s2 = s if isinstance(s, tuple) else (s, s)
    p1, p2 = p if isinstance(p, tuple) else (p, p)
    d1, d2 = d if isinstance(d, tuple) else (d, d)
    ks1, ks2 = ks
    from math import ceil
    # w,h = (w-ks1+p1+s1)/s1,(h-ks2+p2+s2)/s2
    # w = ceil(w) if w - int(w) < .5 else ceil(w)+1
    # h = ceil(h) if h - int(h) < .5 else ceil(h)+1

    w = (w+2*p1-d1*(ks1-1)-1)//s1 + 1
    h = (h+2*p2-d2*(ks2-1)-1)//s2 + 1
    out_shape = b, out_, w, h
    return out_shape


@printed_ml_ops
@register_grad(True)
def conv2d(
    x: _Tensor,
    weight: _Tensor,
):
    def backward(gradient):
        raise NotImplementedError()

    kernel_size = weight.shape[-2:]
    use_torch = True
    input_unf = unfold(x.numpy(), kernel_size,
                       use_torch=use_torch).requires_grad_(x.requires_grad)
    weight = weight.reshape(shape=(weight.shape[0], -1)).T
    res = (input_unf@weight)
    res.data = res.data.transpose(0, 2, 1)
    # TODO into a function
    s = int(np.sqrt(res.shape[-1]))
    conputed_output_shpae = (s, s)
    res = fold(res.numpy(), conputed_output_shpae,
               (1, 1), use_torch=use_torch).requires_grad_(res.requires_grad)
    return res, backward


@printed_ml_ops
def unfold(input, kernel_size, pad=0, stride=1, use_torch=True):
    if not use_torch:
        import numpy as np
        N, C, H, W = input.shape
        filter_h, filter_w = kernel_size
        out_h = (H + 2*pad - filter_h)//stride + 1
        out_w = (W + 2*pad - filter_w)//stride + 1

        img = np.pad(input, [(0, 0), (0, 0),
                             (pad, pad), (pad, pad)], 'constant')
        col = np.zeros((N, C, filter_h, filter_w, out_h, out_w))

        for y in range(filter_h):
            y_max = y + stride*out_h
            for x in range(filter_w):
                x_max = x + stride*out_w
                col[:, :, y, x, :, :] = img[:, :,
                                            y:y_max:stride, x:x_max:stride]

        col = col.transpose(0, 4, 5, 1, 2, 3).reshape(N, out_h*out_w, -1)
        from src import tensor
        return tensor.from_numpy(col)
    else:
        import torch
        from src import tensor
        x = input
        x = torch.tensor(x)
        x = torch.nn.functional.unfold(x, kernel_size)
        x = tensor.from_numpy(x.numpy().transpose(0, 2, 1))
    return x


@printed_ml_ops
def fold(col, shape, kernel_size, pad=0, stride=1, use_torch=True):
    if not use_torch:
        filter_h, filter_w = kernel_size
        H, W = shape
        N = col.shape[0]
        C = col.shape[1]

        out_h = (H + 2*pad - filter_h)//stride + 1
        out_w = (W + 2*pad - filter_w)//stride + 1
        col = col.reshape(N, out_h, out_w, C, filter_h,
                          filter_w).transpose(0, 3, 4, 5, 1, 2)

        img = np.zeros((N, C, H + 2*pad + stride - 1, W + 2*pad + stride - 1))
        for y in range(filter_h):
            y_max = y + stride*out_h
            for x in range(filter_w):
                x_max = x + stride*out_w
                img[:, :, y:y_max:stride, x:x_max:stride] += col[:, :, y, x, :, :]

        res = img[:, :, pad:H + pad, pad:W + pad]
        from src import tensor
        return tensor.from_numpy(res)
    else:
        import torch
        from src import tensor
        x = col
        x = torch.tensor(x)
        x = torch.nn.functional.fold(x, shape, kernel_size)
        x = tensor.from_numpy(x.numpy())
    return x


@printed_ml_ops
@register_grad()
@as_layer(name="Dropout", include_training=True)
def dropout(x, rate, training=True):
    if training is False:
        return x
    x = x*np.random.binomial(1, rate, x.shape)
    return x


@printed_ml_ops
def unsqueeze(x, dim):
    new_shape = list(x.shape)
    for d in _to_iter(dim):
        new_shape.insert(d, 1)
    res = x.reshape(shape=new_shape)
    return res


def _to_iter(shape):
    if isinstance(shape, (list, tuple)):
        return shape
    return [shape]


@printed_ml_ops
def squeeze(x, dim=None):
    if dim is None:
        new_shape = tuple(filter(lambda d: d != 1, x.shape))
    else:
        dim = _to_iter(dim)
        new_shape = list(x.shape)
        for d in sorted(dim, reverse=True):
            assert x.shape[d] == 1, f"The dim={d} is not equal to one"
            new_shape.pop(d)
    return x.reshape(shape=new_shape)


@printed_ml_ops
@register_grad()
def select(x, args):
    def backward(gradient):
        aa = tensor.ones_like(x)*0
        aa.data[args] = gradient.data
        _pass_gradient(x, aa)

    from src import tensor
    xx = tensor.from_numpy(x.data[args]).requires_grad_(x.requires_grad)
    return xx, backward


@printed_ml_ops
def copy_slice(x, slice, y):
    from src.grad import grad_off

    def backward(gradient):
        raise NotImplementedError()
    with grad_off():
        x = x.copy()
        x.data[slice] = y.copy().data
        backward._fn_name = "CopySliceBackward"
    x._backward = backward
