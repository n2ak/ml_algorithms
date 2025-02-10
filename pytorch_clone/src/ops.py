
from __future__ import annotations
import numba
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from ._tensor import Tensor
import numpy as np
from .grad_utils import differentiable_function


def _bin_op(func, x, y) -> Tensor:
    from ._tensor import Tensor
    return Tensor(func(data_or_self(x), data_or_self(y)))


def _unary_op(func, x, **kwargs) -> Tensor:
    from ._tensor import Tensor
    return Tensor(func(x.data, **kwargs)).requires_grad_(x.requires_grad)


def flatten(x: Tensor, start_dim=0, end_dim=-1) -> Tensor:
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
        return gradient @ other.data.T, x.data.T @ gradient
    return _bin_op(np.matmul, x, other), backward


@differentiable_function(1)
def reshape(x, *shape) -> Tensor:
    def backward(gradient: Tensor):
        gradient = gradient.reshape(*x.shape)
        return gradient
    # No copy
    t = x.copy()
    t.data = t.data.reshape(*shape)
    return t, backward


def linear(x, w, b) -> Tensor:
    assert x.shape[1] == w.shape[0]
    res = (x@w)
    if b is not None:
        res = res + b
    return res


@differentiable_function()
def relu(tensor) -> Tensor:
    x = tensor.copy()
    mask = x.data < 0
    x.data[mask] = 0

    def backward(gradient):
        gradient[mask] = 0
        return gradient
    return x, backward


@differentiable_function()
def sigmoid(t) -> Tensor:
    def backward(gradient):
        # dsig(x) = sig(x) * (1 - sig(x))
        local_g = res.data
        local_g = local_g * (1 - local_g)
        return local_g * gradient
    res = 1 / ((-t).exp() + 1)
    return res, backward


@differentiable_function(1)
def softmax(x: Tensor, dim: int = -1) -> Tensor:
    def backward(gradient):
        local = (1-res.data)*res.data
        return local * gradient
    m = x - x.data.max(axis=dim, keepdims=True)
    e = m.exp()
    res = e/e.sum(dim=dim, keepdim=True)
    return res, backward


def log_softmax(x, dim=-1) -> Tensor:
    # https://stackoverflow.com/questions/61567597/how-is-log-softmax-implemented-to-compute-its-value-and-gradient-with-better
    # return x.softmax(dim=dim).log()
    new_x = x - x.data.max(axis=dim, keepdims=True)
    res = new_x - new_x.exp().sum(dim=dim, keepdim=True).log()
    return res


def tanh(x) -> Tensor:
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
    """conv2d slow forward/slow backward"""

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
        return conv2d_backward_slow(gradient, padded, weight, bias,
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
    return dx, dweight, db


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
) -> Tensor:
    """conv2d fast forward/slow backward"""
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
        return conv2d_backward_slow(gradient, padded, weight, bias,
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
) -> Tensor:
    """fast conv2d"""
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

        return grad_input.numpy(), grad_weight.numpy(), np.zeros_like(bias.data)
        # pass_gradient(bias, db)
    out = weight.data.reshape(
        weight.shape[0], -1) @ im2col(padded.data, (KH, KW), skip=stride) + bias.data.reshape(1, -1, 1)
    out = out.reshape(N, F, OH, OW)
    from ._tensor import Tensor
    return Tensor(out), backward


@differentiable_function(2)
def add(x, other) -> Tensor:
    res = _bin_op(np.add, x, other)

    def backward(gradient):
        return add_backward(gradient, x, other)
    return res, backward


def add_backward(gradient, x, other):
    from ._tensor import Tensor
    grads = []
    for var in [x, other]:
        new_grad = gradient
        if isinstance(var, Tensor) and var.shape != new_grad.shape:
            new_grad = correct_shape(var.data, new_grad)
        grads.append(new_grad)
    return tuple(grads)


@differentiable_function(2)
def sub(x, other) -> Tensor:
    def backward(gradient):
        from ._tensor import Tensor
        grads = []
        new_grad = gradient
        for var, M in zip([x, other], [1, -1]):
            if isinstance(gradient, Tensor) and isinstance(var, Tensor) and var.shape != gradient.shape:
                new_grad = correct_shape(var.data, gradient)
            grads.append(M*new_grad)
        return tuple(grads)
    return _bin_op(np.subtract, x, other), backward


def data_or_self(x):
    from ._tensor import Tensor
    return x.data if isinstance(x, Tensor) else x


@differentiable_function(2)
def mul(x, other) -> Tensor:
    def backward(gradient):
        return data_or_self(other) * gradient, data_or_self(x) * gradient
    return _bin_op(np.multiply, x, other), backward


@differentiable_function(2)
def pow(x, other) -> Tensor:
    def backward(gradient):
        a = data_or_self(x)
        b = data_or_self(other)
        dx = b * (a ** (b-1)) * gradient
        dother = (a ** b)*np.log(a) * gradient
        return dx, dother
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
def truediv(x, other) -> Tensor:
    def backward(gradient):
        gradient2 = -1*(data_or_self(x)/(data_or_self(other)**2))*gradient
        if isinstance(gradient2, np.ndarray) and isinstance(other, np.ndarray) and other.shape != gradient2.shape:
            gradient2 = correct_shape(other.data, gradient2)
        dx = 1/data_or_self(other) * gradient
        dother = gradient2
        return dx, dother
    return _bin_op(np.divide, x, other), backward


@differentiable_function(2)
def rtruediv(x, other) -> Tensor:
    x, other = other, x

    def backward(gradient):
        gradient2 = -1*(data_or_self(x)/(data_or_self(other)**2))*gradient
        from ._tensor import Tensor
        if isinstance(gradient2, Tensor) and isinstance(other, Tensor) and other.shape != gradient2.shape:
            gradient2 = correct_shape(other.data, gradient2)
        dx = np.array(1/data_or_self(other) * gradient)
        dother = gradient2
        return dx, dother
    return _bin_op(np.divide, x, other), backward


def neg(x) -> Tensor:
    return mul(x, -1)


@differentiable_function()
def mean(x: Tensor, dim=None, keepdim=False) -> Tensor:
    gradient_shape = list(x.shape)

    def backward(gradient):
        if (dim is not None) and (gradient.shape != () and gradient.shape != x.shape):
            gradient_shape[dim] = 1
            gradient = gradient.reshape(*gradient_shape)
        size = x.size if dim is None else x.shape[dim]
        lg = np.ones(x.shape) / size
        return np.array(lg * gradient)
    return _unary_op(np.mean, x, axis=dim, keepdims=keepdim), backward


@differentiable_function()
def sum(x, dim=None, keepdim=False) -> Tensor:
    res = _unary_op(np.sum, x, axis=dim, keepdims=keepdim)

    def backward(gradient):
        if (dim is not None) and (gradient.shape != () and gradient.shape != x.shape):
            gradient_shape = list(x.shape)
            gradient_shape[dim] = 1
            gradient = gradient.reshape(*gradient_shape)
        return np.ones(x.shape) * gradient
    return res, backward


@differentiable_function()
def exp(x) -> Tensor:
    res = _unary_op(np.exp, x)

    def backward(gradient):
        return res.data * gradient
    return res, backward


@differentiable_function()
def log(x) -> Tensor:
    def backward(gradient):
        return 1/x.data * gradient
    return _unary_op(np.log, x), backward


class unary_ops:
    neg = neg
    log = log
    exp = exp
    tanh = tanh
    relu = relu
    sigmoid = sigmoid
    softmax = softmax
    log_softmax = log_softmax


class reduction_ops:
    mean = mean
    sum = sum


class bin_ops:
    add = add
    sub = sub
    mul = mul
    pow = pow
    truediv = truediv
    rtruediv = rtruediv
    matmul = matmul


class other_ops:
    conv2d = conv2d
    conv2d_slow = conv2d_slow
    conv2d_fast = conv2d_fast
    flatten = flatten
    reshape = reshape
    linear = linear
    dropout = None  # dropout
    squeeze = None  # squeeze
    unsqueeze = None  # unsqueeze


class _TensorOps(unary_ops, reduction_ops, bin_ops, other_ops):
    __add__ = add
    __sub__ = sub
    __mul__ = mul
    __pow__ = pow
    __div__ = truediv
    __truediv__ = truediv
    __rtruediv__ = rtruediv
    __matmul__ = matmul
    __neg__ = neg
    __rmul__ = __mul__
    __radd__ = __add__
    __rsub__ = __sub__
    __iadd__ = __add__
    __isub__ = __sub__
    __imul__ = __mul__
