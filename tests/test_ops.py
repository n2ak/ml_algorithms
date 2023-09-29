from collections import namedtuple
from src.ops import *
import numpy as np
import torch
np.random.seed(0)
torch.manual_seed(0)


def cast_vars(vars):
    from src._tensor import tensor
    ours = []
    torchs = []
    for var in vars:
        if isinstance(var, np.ndarray):
            v1 = tensor(var.copy())
            v2 = torch.from_numpy(var)
        else:
            v1 = v2 = var
        ours.append(v1)
        torchs.append(v2)
    return ours, torchs


def assert_close(var1, var2, atol=1e-6):
    var1 = var1.numpy()
    var2 = var2.numpy()
    print(var1, var2)
    return np.allclose(var1, var2, atol=atol)


def __test(ops):
    for our_op, torch_op, vars in ops:
        ours, torchs = cast_vars(vars)
        val1 = our_op(*ours)
        val2 = torch_op(*torchs)
        assert assert_close(val1, val2), f"Op: {our_op.__name__}"


def test_binary_ops():
    ops = [
        (add, (lambda x, b: x+b), (np.random.random((2, 3, 4)), np.random.random((3, 4)))),
        (sub, (lambda x, b: x-b), (np.random.random((2, 3, 4)), np.random.random((3, 4)))),
        (mul, (lambda x, b: x*b), (np.random.random((2, 3, 4)), np.random.random((3, 4)))),
        (pow, (lambda x, b: x**b), (np.random.random((2, 3, 4)), 3)),
        (truediv, (lambda x, b: x/b),
         (np.random.random((2, 3, 4)), np.random.random((3, 4)))),
        (rtruediv, (lambda x, b: b/x),
         (np.random.random((2, 3, 4)), np.random.random((3, 4)))),
        (matmul, (lambda x, b: x@b),
         (np.random.random((2, 3, 4)), np.random.random((4, 3)))),
        (neg, (lambda x: -x), (np.random.random((2, 3, 4)),)),
    ]
    __test(ops)


def test_unary_ops():
    ops = [
        (exp, torch.exp, (np.random.random((2, 3, 4)),)),
        (log, torch.log, (np.random.random((2, 3, 4)),)),
        (flatten, torch.flatten, (np.random.random((2, 3, 4),),)),
    ]
    __test(ops)


def test_other_ops():
    Params = namedtuple("Params", ["axis"])
    ops = [
        (mean, torch.mean, (np.random.random((2, 3, 4)),), Params(axis=2)),
        (sum, torch.sum, (np.random.random((2, 3, 4)),), Params(axis=2))
    ]
    for our_op, torch_op, vars, kwargs in ops:
        ours, torchs = cast_vars(vars)
        for v in ours:
            print(v.data)
        val1 = our_op(*ours, **kwargs._asdict())
        val2 = torch_op(*torchs, **kwargs._asdict())
        assert assert_close(
            val1, val2), f"Op: {our_op.__name__},kwargs: {(kwargs)}"

# def test_activatio=
