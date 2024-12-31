
from __future__ import annotations
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from ._tensor import Tensor
import numpy as np
from .grad_utils import differentiable_function, _pass_gradient


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
        _pass_gradient(x, gradient @ other.data.T)
        _pass_gradient(other, x.data.T @ gradient)
    return _bin_op(np.matmul, x, other), backward


@differentiable_function()
def reshape(x, shape):
    def backward(gradient: Tensor):
        gradient = gradient.reshape(*x.shape)
        _pass_gradient(x, gradient)
    # No copy
    t = x.copy()
    t.data = t.data.reshape(*shape)
    return t, backward


def linear(x, w, b):
    res = (x@w)
    if b is not None:
        res = res + b
    return res


@differentiable_function()
def relu(x):
    x = x.copy()
    mask = x.data < 0
    x.data[mask] = 0

    def backward(gradient):
        gradient[mask] = 0
        _pass_gradient(x, gradient)
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
    new_x = x-x.data.max(axis=dim, keepdims=True)
    res = new_x - new_x.exp().sum(axis=dim, keepdim=True).log()
    return res


def tanh(x):
    a, b = x.exp(), (-x).exp()
    res = (a-b)/(a+b)
    return res


@differentiable_function(3)
def conv2d(
    input,
    weight,
    bias,
    out_channels,
    padding=(0, 0),
    stride=(1, 1),
):
    # TODO: add strides
    from ._tensor import Tensor
    N = input.shape[0]
    (N, C, H, W) = input.shape
    pad1, pad2 = padding
    padded = Tensor.from_numpy(
        np.pad(input, [(0, 0), (0, 0), (pad1, pad1), (pad2, pad2)], 'constant', constant_values=(0, 0)))
    stride1, stride2 = stride
    (F, C, KH, KW) = weight.shape
    OH = int(((H+pad1*2)-KH) / stride1 + 1)
    OW = int(((W+pad2*2)-KW) / stride2 + 1)
    out = Tensor.from_numpy(
        np.empty((N, out_channels, OH, OW), dtype=np.float32))

    def backward(gradient):
        dpadded = np.zeros_like(padded)
        dweight = np.zeros_like(weight)
        db = np.zeros_like(bias)
        for i in range(OH):
            for j in range(OW):
                h_index = i*stride1
                w_index = j*stride2
                p = padded.data[:, :, h_index:h_index+KH, w_index:w_index+KW]
                for k in range(F):
                    db[k] = db[k] + (1 * gradient.data[:, k, i, j]).sum()
                    dfilter = gradient.data[:, k, i, j]
                    dres = dfilter.reshape(-1, 1, 1, 1)
                    dp = weight.data[k] * dres
                    dweight[k] += (p * dres).sum(0)
                    # assert dres.shape == p.shape , (dres.shape,p.shape)
                    assert dp.shape == p.shape, (dp.shape, p.shape)
                    dpadded[:, :, h_index:h_index+KH, w_index:w_index+KW] += dp

        def end(pad1):
            if pad1 == 0:
                return None
            return -pad1
        dx = dpadded[..., pad1:end(pad1), pad2:end(pad2)]
        _pass_gradient(input, Tensor.from_numpy(dx))
        _pass_gradient(weight, Tensor.from_numpy(dweight))
        _pass_gradient(bias, Tensor.from_numpy(db))
    # todo: subs padding

    for i in range(OH):
        for j in range(OW):
            h_index = i*stride1
            w_index = j*stride2
            k = padded.data[:, :, h_index:h_index+KH, w_index:w_index+KW]
            for ki, kernel in enumerate(weight.data):
                res = (kernel*k).sum((1, 2, 3)) + bias.data[ki]
                out.data[..., ki, i, j] = res

    return out, backward


@differentiable_function(2)
def add(x, other):

    res = _bin_op(np.add, x, other)

    def backward(gradient):
        from ._tensor import Tensor
        new_grad = gradient
        for var in [x, other]:
            if isinstance(gradient, Tensor) and isinstance(var, Tensor) and var.shape != gradient.shape:
                new_grad = correct_shape(var.data, gradient)
            _pass_gradient(var, new_grad)
    return res, backward


@differentiable_function(2)
def sub(x, other):
    def backward(gradient):
        from ._tensor import Tensor

        new_grad = gradient
        for var, M in zip([x, other], [1, -1]):
            if isinstance(gradient, Tensor) and isinstance(var, Tensor) and var.shape != gradient.shape:
                new_grad = correct_shape(var.data, gradient)
            _pass_gradient(var, M*new_grad)
    return _bin_op(np.subtract, x, other), backward


def data_or_self(x):
    from ._tensor import Tensor
    return x.data if isinstance(x, Tensor) else x


@differentiable_function(2)
def mul(x, other):
    def backward(gradient):
        _pass_gradient(x, data_or_self(other) * gradient)
        _pass_gradient(other, data_or_self(x) * gradient)
    return _bin_op(np.multiply, x, other), backward


@differentiable_function(2)
def pow(x, other):
    def backward(gradient):
        _pass_gradient(x, other * (x ** (other-1)) * gradient)
        _pass_gradient(other, (x ** other)*x.log() * gradient)
    return _bin_op(np.power, x, other), backward


def correct_shape(origin, gradient):
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
            grad = grad.reshape(shape=origin.shape)
    else:
        import numpy as np
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
        _pass_gradient(x, 1/data_or_self(other) * gradient)
        _pass_gradient(other, gradient2)
    return _bin_op(np.divide, x, other), backward


@differentiable_function(2)
def rtruediv(x, other):
    x, other = other, x

    def backward(gradient):
        gradient2 = -1*(data_or_self(x)/(data_or_self(other)**2))*gradient
        from ._tensor import Tensor
        if isinstance(gradient2, Tensor) and isinstance(other, Tensor) and other.shape != gradient2.shape:
            gradient2 = correct_shape(other.data, gradient2)
        _pass_gradient(x, np.array(1/data_or_self(other) * gradient), )
        _pass_gradient(other, gradient2, )
    return _bin_op(np.divide, x, other), backward


def neg(x):
    return mul(x, -1)


@differentiable_function()
def mean(x, axis=None):
    def backward(gradient):
        print("mean")
        if (axis is not None) and (gradient.shape != () and gradient.shape != x.shape):
            gradient_shape = list(x.shape)
            gradient_shape[axis] = 1
            gradient = gradient.reshape(shape=gradient_shape)
        lg = np.ones(x.shape) / x.size

        _pass_gradient(x, np.array(lg * gradient))
    return _unary_op(np.mean, x, axis=axis), backward


@differentiable_function()
def sum(x, axis=None, keepdim=False):
    res = _unary_op(np.sum, x, axis=axis, keepdims=keepdim)

    def backward(gradient):
        if (axis is not None) and (gradient.shape != () and gradient.shape != x.shape):
            gradient_shape = list(x.shape)
            gradient_shape[axis] = 1
            gradient = gradient.reshape(*gradient_shape)
        _pass_gradient(x, np.ones(x.shape) * gradient)
    return res, backward


@differentiable_function()
def exp(x):
    res = _unary_op(np.exp, x)

    def backward(gradient):
        _pass_gradient(x, res.data * gradient)
    return res, backward


@differentiable_function()
def log(x):
    def backward(gradient):
        _pass_gradient(x, 1/x.data * gradient)
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
    flatten = flatten
    reshape = reshape
    dropout = None  # dropout
    squeeze = None  # squeeze
    unsqueeze = None  # unsqueeze
