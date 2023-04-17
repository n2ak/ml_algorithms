from tests.utils import equal
from .. import *
import pytest
import torch


def test_dense():
    x = tensor([3, 4, 5])
    w = tensor(
        [[6, 5],
         [2, 8],
         [4, 4]])
    expected = tensor([46, 67])
    layer = Dense.from_weights(w)
    res = layer(x)
    assert res.shape == (w.shape[-1],)
    assert equal(expected, res)


def test_dense2():
    x = np.random.rand(2).astype(np.float32)
    in_, out_ = 2, 3
    a = torch.nn.Linear(in_, out_, bias=False)
    b = Dense.from_weights(a.weight.detach().numpy().T, bias=False)
    a = a(torch.tensor(x))
    b = b(tensor(x))
    equal(a.detach().numpy(), b, print_ok=True)


def test_sequential1():
    x = Tensor.rand(10)
    model = Sequential()
    assert equal(x, model(x))


def test_sequential2():
    in_ = 100
    out_ = 30
    num_layers = 10

    dims = [(in_, 100)]
    for _ in range(num_layers):
        dims.append((dims[-1][-1], np.random.randint(10, 100)))

    dims.append((dims[-1][-1], out_))

    def init_layer(dim):
        l = torch.nn.Linear(*dim, bias=False)
        d = Dense.from_weights(l.weight.detach().numpy().T, bias=False)
        return l, d

    model1 = []
    model2 = []
    for dim in dims:
        l, d = init_layer(dim)
        model1.append(l)
        model2.append(d)

    x = np.random.rand(in_).astype(np.float32)

    model1 = torch.nn.Sequential(*model1)
    model2 = Sequential(model2)
    res1 = model1(torch.tensor(x))
    res2 = model2(tensor(x))

    assert equal(res1.detach().numpy(), res2, print_ok=True)
