import numpy as np
from src.grad_utils import grad_check
# myPath = os.path.dirname(os.path.abspath(__file__))
# sys.path.insert(0, myPath + '/../src')


from src import Tensor


def randn(*shape, requires_grad=True) -> Tensor:
    t = Tensor.randn(*shape).relu()+1
    return t.requires_grad_(val=requires_grad)


def randint(min, max, shape, requires_grad=True) -> Tensor:
    t = Tensor(np.random.randint(min, max, shape))
    return t.requires_grad_(val=requires_grad)


def test_unary():
    from src import unary_ops
    for op in [
        unary_ops.neg,
        unary_ops.log,
        unary_ops.exp,
        unary_ops.tanh,
        unary_ops.relu,
        unary_ops.sigmoid,
        # unary_ops.softmax,
        # unary_ops.log_softmax,
    ]:
        print(op.__name__)
        grad_check(op, randn(10, 10),)


def test_binary():
    from src import bin_ops
    for op in [
        bin_ops.add,
        bin_ops.sub,
        bin_ops.mul,
        bin_ops.pow,
        bin_ops.truediv,
        bin_ops.rtruediv,
        bin_ops.matmul,
    ]:
        print(op.__name__)
        grad_check(op, randn(10, 10), randn(10, 10))


def test_binary():
    from src import reduction_ops
    for op in [
        reduction_ops.sum,
        reduction_ops.mean,
    ]:
        for keepdim in [False, True]:
            print(op.__name__, "keepdim:", keepdim)
            grad_check(lambda x: op(x), randn(10, 10))
            grad_check(lambda x: op(x, axis=1), randn(10, 10))
            grad_check(lambda x: op(x, axis=0, keepdim=keepdim), randn(10, 10))


def test_other():
    from src import other_ops
    for i, func in enumerate([
        lambda x: other_ops.flatten(x),
        lambda x: other_ops.flatten(x, 1),
        lambda x: other_ops.flatten(x, 0, -2),
        lambda x: other_ops.reshape(x, -1),
        lambda x: other_ops.reshape(x, 2, -1),
        lambda x: other_ops.reshape(x, 1, -1, 2)
    ]):
        print(i, func)
        grad_check(func, randn(4, 8, 28))

    grad_check(other_ops.linear, randn(3, 73), randn(73, 18), randn(18,))
    # grad_check(other_ops.dropout, randt(3, 73), randt(73, 18), randt(18,))
    # grad_check(other_ops.squeeze, randt(3, 73), randt(73, 18), randt(18,))
    # grad_check(other_ops.unsqueeze, randt(3, 73), randt(73, 18), randt(18,))

    for conv in [
        other_ops.conv2d,
        # other_ops.conv2d_slow,
        # other_ops.conv2d_fast,
    ]:
        print(conv)
        input, weight, bias = randn(
            1, 1, 28, 28), randn(10, 1, 3, 3), randn(10,)
        grad_check(conv, input, weight, bias)  # , atol=1e-3, rtol=1e-3)


def test_losses():
    from src import loss
    x, t = randn(9, 5), randint(0, 4, (9), requires_grad=False)
    x = x.copy().log_softmax(-1).requires_grad_()
    grad_check(loss.negative_log_likelihood, x, t)

    x, t = randn(9, 5), randint(0, 4, (9), requires_grad=False)
    grad_check(loss.cross_entropy, x, t, h=1e-8)
