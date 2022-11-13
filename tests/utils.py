from src import Tensor


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
    a = type1(torch.tensor(x), *args1)
    b = type2(Tensor.array(x), *args2)
    assert equal(a, b, print_ok=print_ok)
