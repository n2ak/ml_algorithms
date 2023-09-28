from src import *


def equal(a, b, t=1e-3, print_ok=False):
    import numpy as np
    assert tuple(a.shape) == tuple(
        b.shape), f"a: {tuple(a.shape)} , b: {tuple(b.shape)}"
    if print_ok:
        print("--------result---------")
        print("a", a)
        print("b", b)
    mean = np.abs((a-b).mean())
    if print_ok:
        print("mean", mean)
        print("-----------------------")
    return mean <= t


def comp(x, type1, type2, args1=[], args2=[], print_ok=False):
    import torch
    # print(type(x))
    a = type1(torch.tensor(x), *args1)
    b = type2(tensor(x), *args2)
    assert equal(a, b, print_ok=print_ok)


def proc(func, *args):
    import torch
    res1 = func(_init=torch.from_numpy, *args)
    from src._tensor import from_numpy
    res2 = func(_init=from_numpy, *args)
    if not isinstance(res1, tuple):
        res1 = (res1,)
    if not isinstance(res2, tuple):
        res2 = (res2,)
    for i, r1, r2 in zip(range(len(res1)), res1, res2):
        assert np.allclose(r1, r2), f"result {i+1} : {r1} != {r2}"
