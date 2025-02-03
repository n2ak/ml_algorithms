
from __future__ import annotations
import numba
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from ._tensor import Tensor
import numpy as np
from .grad_utils import differentiable_function, pass_gradient


def _bin_op(func, x, y):
    from ._tensor import Tensor
    return Tensor(func(data_or_self(x), data_or_self(y)))


def _unary_op(func, x, **kwargs):
    from ._tensor import Tensor
    return Tensor(func(x.data, **kwargs)).requires_grad_(x.requires_grad)


def flatten(x: Tensor, start_dim=0, end_dim=-1):
    shape = x.shape
    if end_dim < 0:
        end_dim = len(shape)+end_dim
    end_dim = end_dim+1
    new_shape = * \
        shape[:start_dim], np.prod(shape[start_dim:end_dim]), *shape[end_dim:]
    return x.reshape(new_shape)


@differentiable_function()
def matmul(x: Tensor, other) -> Tensor:
    def backward(gradient: Tensor):
        pass_gradient(x, gradient @ other.data.T)
        pass_gradient(other, x.data.T @ gradient)
    return _bin_op(np.matmul, x, other), backward


@differentiable_function()
def reshape(x, *shape):
    def backward(gradient: Tensor):
        gradient = gradient.reshape(*x.shape)
        pass_gradient(x, gradient)
    # No copy
    t = x.copy()
    t.data = t.data.reshape(*shape)
    return t, backward


def linear(x, w, b):
    assert x.shape[1] == w.shape[0]
    res = (x@w)
    if b is not None:
        res = res + b
    return res


@differentiable_function()
def relu(tensor):
    x = tensor.copy()
    mask = x.data < 0
    x.data[mask] = 0

    def backward(gradient):
        gradient[mask] = 0
        pass_gradient(tensor, gradient)
    return x, backward


def sigmoid(t):
    return 1 / ((-t).exp() + 1)


def softmax(x, dim: int = -1):
    m = x - x.data.max(axis=dim, keepdims=True)
    e = m.exp()
    ss = e.sum(axis=dim, keepdim=True)
    return e/ss
    # avoids overflow , https://timvieira.github.io/blog/post/2014/02/11/exp-normalize-trick/
    x = (x - x.data.max()).exp()
    x = x/x.sum(axis=dim, keepdim=True)
    return x


def log_softmax(x, dim=-1):
    # https://stackoverflow.com/questions/61567597/how-is-log-softmax-implemented-to-compute-its-value-and-gradient-with-better
    # return softmax(x, dim=dim).log()
    new_x = x - x.data.max(axis=dim, keepdims=True)
    res = new_x - new_x.exp().sum(axis=dim, keepdim=True).log()
    return res


def tanh(x):
    a, b = x.exp(), (-x).exp()
    res = (a-b)/(a+b)
    return res


@differentiable_function(3)
def conv2d_slow(
    input,
    weight,
    bias,
    padding=(0, 0),
    stride=(1, 1),
):
    # TODO: add strides
    from ._tensor import Tensor
    (N, C, H, W) = input.shape
    pad1, pad2 = padding
    padded = input
    if padding != (0, 0):
        padded = Tensor(np.pad(input, [
                        (0, 0), (0, 0), (pad1, pad1), (pad2, pad2)], 'constant', constant_values=(0, 0)))
    stride1, stride2 = stride
    (F, C, KH, KW) = weight.shape
    OH = int(((H+pad1*2)-KH) / stride1 + 1)
    OW = int(((W+pad2*2)-KW) / stride2 + 1)
    out = Tensor(np.empty((N, F, OH, OW), dtype=np.float32))

    # todo: subs padding
    def backward(gradient):
        conv2d_backward_slow(gradient, padded, weight, bias,
                             padding, stride, OH, OW, KH, KW, F)
    for i in range(OH):
        for j in range(OW):
            h_index = i*stride1
            w_index = j*stride2
            k = padded.data[:, :, h_index:h_index+KH, w_index:w_index+KW]
            for ki, kernel in enumerate(weight.data):
                res = (kernel*k).sum((1, 2, 3)) + bias.data[ki]
                out.data[..., ki, i, j] = res
    return out, backward


# @numba.njit()
def helper(
    padded: np.ndarray,
    weight: np.ndarray,
    gradient: np.ndarray,
    dpadded: np.ndarray,
    dweight: np.ndarray,
    db: np.ndarray,
    F: int,
    OH: int,
    OW: int,
    stride1: int,
    stride2: int,
    KH: int,
    KW: int,
):
    for i in range(OH):
        for j in range(OW):
            h_index = i*stride1
            w_index = j*stride2
            p = padded[:, :, h_index:h_index+KH, w_index:w_index+KW]
            for k in range(F):
                db[k] = db[k] + (1 * gradient[:, k, i, j]).sum()
                dfilter = gradient[:, k, i, j]
                dres = np.ascontiguousarray(dfilter).reshape(-1, 1, 1, 1)
                dp = weight[k] * dres
                dweight[k] += (p * dres).sum(0)
                # assert dres.shape == p.shape , (dres.shape,p.shape)
                assert dp.shape == p.shape, (dp.shape, p.shape)
                dpadded[:, :, h_index:h_index+KH, w_index:w_index+KW] += dp


def conv2d_backward_slow(gradient, padded, weight, bias, pad, stride, OH, OW, KH, KW, F):
    pad1, pad2 = pad

    def end(pad1):
        if pad1 == 0:
            return None
        return -pad1

    dpadded = np.zeros_like(padded.data)
    dweight = np.zeros_like(weight.data)
    db = np.zeros_like(bias.data)
    helper(padded.data, weight.data, gradient, dpadded,
           dweight, db, F, OH, OW, stride[0], stride[1], KH, KW)

    dx = dpadded[..., pad1:end(pad1), pad2:end(pad2)]
    pass_gradient(padded, dx)
    pass_gradient(weight, dweight)
    pass_gradient(bias, db)


def im2col(A, B, skip):
    # https://stackoverflow.com/questions/30109068/implement-matlabs-im2col-sliding-in-python
    # Parameters
    batch, D, M, N = A.shape
    col_extent = N - B[1] + 1
    row_extent = M - B[0] + 1

    # Get batch block indices
    batch_idx = np.arange(batch)[:, None, None] * D * M * N

    # Get Starting block indices
    start_idx = np.arange(B[0])[None, :, None]*N + np.arange(B[1])

    # Generate Depth indeces
    didx = M*N*np.arange(D)
    start_idx = (didx[None, :, None]+start_idx.ravel()
                 ).reshape((-1, B[0], B[1]))

    # Get offsetted indices across the height and width of input array
    offset_idx = np.arange(row_extent)[None, :, None]*N + np.arange(col_extent)

    # Get all actual indices & index into input array for final output
    act_idx = (batch_idx +
               start_idx.ravel()[None, :, None] +
               offset_idx[:, ::skip[0], ::skip[1]].ravel())

    out = np.take(A, act_idx)
    return out


@differentiable_function(3)
def conv2d(
    input,
    weight,
    bias,
    padding=(0, 0),
    stride=(1, 1),
):
    # TODO: add strides
    (N, C, H, W) = input.shape
    pad1, pad2 = padding
    padded = input
    if padding != (0, 0):
        padded = (np.pad(input, [
                        (0, 0), (0, 0), (pad1, pad1), (pad2, pad2)], 'constant', constant_values=(0, 0)))
    stride1, stride2 = stride
    (F, _, KH, KW) = weight.shape
    OH = int(((H+pad1*2)-KH) / stride1 + 1)
    OW = int(((W+pad2*2)-KW) / stride2 + 1)
    # out = (np.empty((N, F, OH, OW), dtype=np.float32))

    def backward(gradient):
        conv2d_backward_slow(gradient, padded, weight, bias,
                             padding, stride, OH, OW, KH, KW, F)
    col = im2col(padded.data, (KH, KW), skip=stride)
    out = weight.data.reshape(
        weight.shape[0], -1) @ col + bias.data[..., None]
    out = out.reshape(N, F, OH, OW)
    from ._tensor import Tensor
    return Tensor(out), backward


@differentiable_function(3)
def conv2d_fast(
    input,
    weight,
    bias,
    padding=(0, 0),
    stride=(1, 1),
):
    # TODO: add strides
    pad1, pad2 = padding
    padded = input
    if padding != (0, 0):
        padded = (np.pad(input, [
                        (0, 0), (0, 0), (pad1, pad1), (pad2, pad2)], 'constant', constant_values=(0, 0)))
    (_, _, KH, KW) = weight.shape
    pad1, pad2 = padding
    (N, _, H, W) = input.shape
    (F, _, KH, KW) = weight.shape
    stride1, stride2 = stride
    OH = int(((H+pad1*2)-KH) / stride1 + 1)
    OW = int(((W+pad2*2)-KW) / stride2 + 1)
    # out = (np.empty((N, F, OH, OW), dtype=np.float32))

    def backward(gradient):
        import torch
        import torch.nn.grad
        grad_input = torch.nn.grad.conv2d_input(
            tuple(input.shape), torch.from_numpy(weight.data), torch.from_numpy(gradient))
        grad_weight = torch.nn.grad.conv2d_weight(
            torch.from_numpy(input.data), tuple(weight.shape), torch.from_numpy(gradient))

        pass_gradient(input, grad_input.numpy())
        pass_gradient(weight, grad_weight.numpy())
        pass_gradient(bias, np.zeros_like(bias.data))
        # pass_gradient(bias, db)
    out = weight.data.reshape(
        weight.shape[0], -1) @ im2col(padded.data, (KH, KW), skip=stride) + bias.data.reshape(1, -1, 1)
    out = out.reshape(N, F, OH, OW)
    from ._tensor import Tensor
    return Tensor(out), backward


@differentiable_function(2)
def add(x, other):
    res = _bin_op(np.add, x, other)

    def backward(gradient):
        add_backward(gradient, x, other)
    return res, backward


def add_backward(gradient, x, other):
    from ._tensor import Tensor
    for var in [x, other]:
        new_grad = gradient
        if isinstance(var, Tensor) and var.shape != new_grad.shape:
            new_grad = correct_shape(var.data, new_grad)
        pass_gradient(var, new_grad)


@differentiable_function(2)
def sub(x, other):
    def backward(gradient):
        from ._tensor import Tensor

        new_grad = gradient
        for var, M in zip([x, other], [1, -1]):
            if isinstance(gradient, Tensor) and isinstance(var, Tensor) and var.shape != gradient.shape:
                new_grad = correct_shape(var.data, gradient)
            pass_gradient(var, M*new_grad)
    return _bin_op(np.subtract, x, other), backward


def data_or_self(x):
    from ._tensor import Tensor
    return x.data if isinstance(x, Tensor) else x


@differentiable_function(2)
def mul(x, other):
    def backward(gradient):
        pass_gradient(x, data_or_self(other) * gradient)
        pass_gradient(other, data_or_self(x) * gradient)
    return _bin_op(np.multiply, x, other), backward


@differentiable_function(2)
def pow(x, other):
    def backward(gradient):
        pass_gradient(x, other * (x ** (other-1)) * gradient)
        pass_gradient(other, (x ** other)*x.log() * gradient)
    return _bin_op(np.power, x, other), backward


def correct_shape(origin, gradient: np.ndarray):
    assert isinstance(gradient, np.ndarray)
    summing = origin.size < gradient.size
    if summing:
        if len(origin.shape) == 0:
            return gradient.sum()
        summ = []
        for i in range(0, len(gradient.shape)):
            dim1 = origin.shape[len(origin.shape) - i - 1]
            dim2 = gradient.shape[len(gradient.shape) - i-1]
            if len(origin.shape) - i - 1 < 0 or dim1 != dim2:
                # NOTE: when dim1 = 1
                index = len(gradient.shape) - i-1
                summ.append(index)
        grad = gradient
        if len(summ):
            grad = gradient.sum(axis=tuple(summ))
            grad = grad.reshape(*origin.shape)
    else:
        o = np.broadcast(origin.data, gradient)
        gradient.data = np.broadcast_to(gradient.data, o.shape)
        grad = gradient
    assert grad.shape == origin.shape, f"did {summing=},{grad.shape} != {origin.shape}"
    return grad


@differentiable_function(2)
def truediv(x, other):
    def backward(gradient):
        gradient2 = -1*(data_or_self(x)/(data_or_self(other)**2))*gradient
        if isinstance(gradient2, np.ndarray) and isinstance(other, np.ndarray) and other.shape != gradient2.shape:
            gradient2 = correct_shape(other.data, gradient2)
        pass_gradient(x, 1/data_or_self(other) * gradient)
        pass_gradient(other, gradient2)
    return _bin_op(np.divide, x, other), backward


@differentiable_function(2)
def rtruediv(x, other):
    x, other = other, x

    def backward(gradient):
        gradient2 = -1*(data_or_self(x)/(data_or_self(other)**2))*gradient
        from ._tensor import Tensor
        if isinstance(gradient2, Tensor) and isinstance(other, Tensor) and other.shape != gradient2.shape:
            gradient2 = correct_shape(other.data, gradient2)
        pass_gradient(x, np.array(1/data_or_self(other) * gradient), )
        pass_gradient(other, gradient2, )
    return _bin_op(np.divide, x, other), backward


def neg(x):
    return mul(x, -1)


@differentiable_function()
def mean(x, axis=None):
    def backward(gradient):
        if (axis is not None) and (gradient.shape != () and gradient.shape != x.shape):
            gradient_shape = list(x.shape)
            gradient_shape[axis] = 1
            gradient = gradient.reshape(shape=gradient_shape)
        lg = np.ones(x.shape) / x.size

        pass_gradient(x, np.array(lg * gradient))
    return _unary_op(np.mean, x, axis=axis), backward


@differentiable_function()
def sum(x, axis=None, keepdim=False):
    res = _unary_op(np.sum, x, axis=axis, keepdims=keepdim)

    def backward(gradient):
        if (axis is not None) and (gradient.shape != () and gradient.shape != x.shape):
            gradient_shape = list(x.shape)
            gradient_shape[axis] = 1
            gradient = gradient.reshape(*gradient_shape)
        pass_gradient(x, np.ones(x.shape) * gradient)
    return res, backward


@differentiable_function()
def exp(x):
    res = _unary_op(np.exp, x)

    def backward(gradient):
        pass_gradient(x, res.data * gradient)
    return res, backward


@differentiable_function()
def log(x):
    def backward(gradient):
        pass_gradient(x, 1/x.data * gradient)
    return _unary_op(np.log, x), backward


class ops:
    add = add
    sub = sub
    mul = mul
    pow = pow
    truediv = truediv
    rtruediv = rtruediv
    matmul = matmul
    neg = neg
    mean = mean
    sum = sum
    log = log
    exp = exp
    tanh = tanh
    relu = relu
    sigmoid = sigmoid
    softmax = softmax
    log_softmax = log_softmax
    linear = linear
    conv2d = conv2d
    conv2d_slow = conv2d_slow
    conv2d_fast = conv2d_fast
    flatten = flatten
    reshape = reshape
    dropout = None  # dropout
    squeeze = None  # squeeze
    unsqueeze = None  # unsqueeze
