import torch
import torch.nn.functional as F
from tests._utils import randn, randint, check


def test_unary():
    for i, op in enumerate([
        lambda x:x.log(),
        lambda x: -x,
        lambda x:x.exp(),
        lambda x:x.tanh(),
        lambda x:x.relu(),
        lambda x:x.sigmoid(),
        lambda x:x/10,
        lambda x:x.softmax(-1),
        # lambda x:x.log_softmax(-1),
    ]):
        print(i+1)
        check(op, randn(3, 3),)


def test_binary():
    for i, op in enumerate([
        lambda x, b: x + b,
        lambda x, b: x - b,
        lambda x, b: x * b,
        lambda x, b: x ** b,
        lambda x, b: x / b,
        lambda x, b: b / x,
        lambda x, b: x @ b,
    ]):
        print(i+1)
        check(op, randn(10, 10), randn(10, 10),)


def test_reduction():
    from src.ops import reduction_ops
    for op, t_op in [
        (reduction_ops.sum, torch.sum),
        (reduction_ops.mean, torch.mean),
    ]:
        for keepdim in [False, True]:
            print(op.__name__, "keepdim:", keepdim)
            check(op, randn(10, 10), torch_func=t_op)
            check(lambda x: op(x, dim=1), randn(10, 10),
                  torch_func=lambda x: t_op(x, dim=1))
            check(lambda x: op(
                x, dim=0, keepdim=keepdim), randn(10, 10),
                torch_func=lambda x: t_op(x, dim=0, keepdim=keepdim))


def test_other():
    from src.ops import other_ops
    for i, func in enumerate([
        lambda x: x.flatten(),
        lambda x: x.flatten(1),
        lambda x: x.flatten(0, -2),
        lambda x: x.reshape(-1),
        lambda x: x.reshape(2, -1),
        lambda x: x.reshape(1, -1, 2)
    ]):
        print(i+1, func)
        check(func, randn(4, 8, 28))

    check(other_ops.linear, randn(3, 73), randn(73, 18), randn(18,),
          torch_func=lambda x, w, b: x@w+b)
    # check(other_ops.dropout, randt(3, 73), randt(73, 18), randt(18,))
    # check(other_ops.squeeze, randt(3, 73), randt(73, 18), randt(18,))
    # check(other_ops.unsqueeze, randt(3, 73), randt(73, 18), randt(18,))

    for conv in [
        other_ops.conv2d,
        # other_ops.conv2d_slow,
        # other_ops.conv2d_fast,
    ]:
        print(conv)
        input, weight, bias = randn(
            1, 1, 28, 28), randn(10, 1, 3, 3), randn(10,)
        check(conv, input, weight, bias, torch_func=lambda i, w,
              b: F.conv2d(i, w, b))  # , atol=1e-3, rtol=1e-3)


def test_losses():
    from src import loss
    x, t = randn(9, 5), randint(0, 4, (9), requires_grad=False)
    x = x.copy().log_softmax(-1).requires_grad_()
    check(loss.negative_log_likelihood, x, t,
          torch_func=lambda x, b: F.nll_loss(x, b.long()))

    x, t = randn(9, 5), randint(0, 4, (9), requires_grad=False)
    check(loss.cross_entropy, x, t,
          torch_func=lambda x, b: F.cross_entropy(x, b.long()))


def test_complex():
    for func, inputs in [
        ((lambda x, b, c: (x+b)*c), (randn(10, 10), randn(10, 10), randn(10, 10))),
        ((lambda x, b, c: (x/b)**c), (randn(10, 10), randn(10, 10), randn(10, 10))),
        ((lambda x: x.log().exp()), (randn(10, 10),)),
    ]:
        check(func, *inputs)  # , atol=1e-3, rtol=1e-3)
