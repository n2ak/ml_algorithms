from tests.utils import comp, equal
import torch
import numpy as np
import pytest
from src import *


def equal_grad(a, b, t=1e-3):
    # assert len(args) % 2 == 0
    # for args in range(0,len(args),2)
    #     assert args
    # print(type(a), type(b))
    assert tuple((a.grad).shape) == tuple((b.grad).shape)
    np.testing.assert_allclose(a.grad, b.grad)
    # assert (a.grad - b.grad).sum() <= t


def init(nums=None, shape1=(), shape2=()):
    a_, b_ = nums or (np.random.rand(*shape1), np.random.rand(*shape2))
    a1 = torch.tensor(a_, requires_grad=True)
    b1 = torch.tensor(b_, requires_grad=True)

    a2 = tensor(a_, requires_grad=True)
    b2 = tensor(b_)
    return a1, b1, a2, b2


def bin_op_test(func, nums=None):
    a_1, b_1, a_2, b_2 = init(nums=nums)

    def h(a1, b1, a2, b2):
        res1 = func(a1, b1)
        res2 = func(a2, b2)
        res1.backward()
        res2.backward()
        equal_grad(a1, a2)
        equal_grad(b1, b2)
    h(a_1, b_1, a_2, b_2)
    h(b_1, a_1, b_2, a_2)


def op_test(func):
    a1, _, a2, _ = init()
    res1 = func(a1)
    res2 = func(a2)
    res1.backward()
    res2.backward()
    equal_grad(a1, a2)


def com_op(op):
    a1, b1, a2, b2 = init()
    (op(op(op(a1, b1), b1), a1)).backward()
    (op(op(op(a2, b2), b2), a2)).backward()
    equal_grad(a1, a2)
    equal_grad(b1, b2)
# ------------------------------------------------------------------


def test_add():
    bin_op_test(lambda x, y: x+y)


def test_sub():
    bin_op_test(lambda x, y: x-y)


def test_mul():
    bin_op_test(lambda x, y: x+y)


def test_div():
    bin_op_test(lambda x, y: x/y)  # ,nums=(3.0,2.0))


def test_pow():
    bin_op_test(lambda x, y: x**y)


def test_mul2():
    op_test(lambda x: x*100)


def test_mul3():
    op_test(lambda x: -1 * x)


def test_add2():
    com_op(lambda x, y: x+y)


def test_sub2():
    com_op(lambda x, y: x-y)


@pytest.mark.skip("Why ?")
def test_mul4():
    com_op(lambda x, y: x*y)


def test_exp():
    a1, _, a2, _ = init()
    res1 = a1.exp()
    res2 = a2.exp()
    res1.backward()
    res2.backward()
    equal_grad(a1, a2)


def test_mean():
    a = torch.rand(3, 2)
    a.requires_grad = True
    a.mean().backward()

    b = tensor(a.detach().numpy(), requires_grad=True)
    x = b.mean()
    x.backward()

    equal_grad(a, b)


def test_mean2():
    a = torch.rand(3, 2)
    a.requires_grad = True
    a1 = a.mean(dim=0)
    a1 = a1.mean(dim=0)
    a1.backward()

    b = tensor(a.detach().numpy(), requires_grad=True)
    b1 = b.mean(axis=0)
    b1 = b1.mean(axis=0)
    b1.backward()

    equal_grad(a, b)


def test_sum():
    a = torch.rand(3, 2)
    a.requires_grad = True
    a.sum().backward()

    b = tensor(a.detach().numpy(), requires_grad=True)
    x = b.sum()
    x.backward()

    equal_grad(a, b)


def test_sum2():
    a = torch.rand(3, 2)
    a.requires_grad = True
    a1 = a.sum(dim=0)
    a1 = a1.sum(dim=0)
    a1.backward()

    b = tensor(a.detach().numpy(), requires_grad=True)
    b1 = b.sum(axis=0)
    b1 = b1.sum(axis=0)
    b1.backward()

    equal_grad(a, b)


def test_sum3():
    a = torch.rand(3, 2)
    a.requires_grad = True
    a1 = a.sum(dim=-1)
    a1 = a1.sum()
    a1.backward()

    b = tensor(a.detach().numpy(), requires_grad=True)
    b1 = b.sum(axis=-1)
    b1 = b1.sum()
    b1.backward()

    equal_grad(a, b)


def test_sum4():
    a = torch.rand(3, 2)
    a.requires_grad = True
    a1 = a.sum(dim=-1, keepdim=True)
    a1 = a1.mean()
    a1.backward()

    b = tensor(a.detach().numpy(), requires_grad=True)
    b1 = b.sum(axis=-1, keepdims=True)
    b1 = b1.mean()
    b1.backward()

    equal_grad(a, b)


def test_softmax():
    torch.manual_seed(1)
    a = torch.rand(3, 2)
    a.requires_grad = True
    torch.softmax(a, dim=-1).sum().backward()

    b = tensor(a.detach().numpy(), requires_grad=True)
    f = b.softmax(dim=-1).sum()
    f.backward()
    from src.grad.viz import plot_graph
    print(b)
    plot_graph(f)
    equal_grad(a, b)


def test_cross_entropy():
    torch.manual_seed(1)
    a = torch.rand(3, 2)
    c = torch.randint(0, 2, size=(3,))
    a.requires_grad = True
    res1 = torch.nn.functional.cross_entropy(a, c)
    res1.backward()

    b = tensor(a.detach().numpy(), requires_grad=True)
    d = tensor(c.detach().numpy())
    b.cross_entropy(d)
    res2 = b.cross_entropy(d)
    res2.backward()

    # print("1",torch.nn.functional.cross_entropy(a, c),torch.nn.functional.nll_loss(a.log_softmax(-1),c))
    # print("2", b.cross_entropy(d), b.log_softmax(-1).nll(d))
    equal_grad(a, b)


# @pytest.mark.skip("Broken")
def test_matmul():
    x = Tensor.rand(4, 3).requires_grad_()
    w = Tensor.rand(3, 2).requires_grad_()
    print(x.shape, w.shape, (x @ w))
    r = (x @ w)
    r = r.sum()
    r = r.backward()

    x2 = x.torch()
    w2 = w.torch()
    r2 = (x2 @ w2).sum().backward()

    equal_grad(x, x2)
    equal_grad(w, w2)


@pytest.mark.skip("Needs test_matmul to pass")
def test_2():
    x = Tensor.rand(100).requires_grad_()
    w = Tensor.rand(100, 10)
    b = Tensor.zeros(10).requires_grad_()

    (x @ w + b).sum().backward()

    assert False


@pytest.mark.skip("Needs other tests to pass")
def test_():
    x = Tensor.rand(100).requires_grad_()
    linear = torch.nn.Linear(100, 10)
    linear(torch.from_numpy(x.numpy())).sum().backward()
    w = linear.weight.detach().numpy()
    dense = Dense.from_weights(w.T)
    r = dense(x)
    r = r.sum().backward()

    equal_grad(dense.weights, linear.weight)
    equal_grad(dense.bias, linear.bias)
    # assert False


# def test_complex():
