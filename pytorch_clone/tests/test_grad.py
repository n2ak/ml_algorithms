from collections import namedtuple
from src.ops import *
import numpy as np
import torch
from .utils import *

np.random.seed(0)
torch.manual_seed(0)


def __test(ops, named=True):
    for i, (our_op, torch_op, vars, kwargs) in enumerate(ops):
        print(f"{i})- Op", our_op.__name__, kwargs)
        ours, torchs = cast_vars(vars)
        if kwargs is None:
            kwargs = dict()
        if not named:
            kwargs = kwargs.values()
            val1 = our_op(*ours, *kwargs)
            val2 = torch_op(*torchs, *kwargs)
        else:
            val1 = our_op(*ours, **kwargs)
            val2 = torch_op(*torchs, **kwargs)

        c1 = val1.sum()
        # c1.grad_fn.print_graph()
        c1.backward(print_ok=False)
        c2 = val2.sum()
        c2.backward()
        for o, t in zip(ours, torchs):
            if isinstance(t, torch.Tensor):
                assert_close(o, t)


def test_funcs_ops():
    def func1(a, b, c, d):
        res = (((a+b)*c)-d)/a
        return res

    def func2(a, b, c, d):
        res = a  # .sum(axis=-1, keepdim=True)
        res = a/a.sum()
        return res
    vars = (
        np.random.random((2)),
        np.random.random((2, 4)),
        np.random.random((2, 4)),
        np.random.random((2, 4)),
    )
    for func in [
        func2,
    ]:
        ours, torchs = cast_vars(vars)
        c = func(*ours)
        d = c.mean()
        d.backward(print_ok=False)
        print(c.grad)
        print(d.grad)
        func(*torchs).mean().backward()
        for o, t in zip(ours, torchs):
            assert_close(o, t)
        # assert False


def test_binary_ops():
    ops = [
        (add, (lambda x, b: x+b),
         (np.random.random((2, 3, 4)), np.random.random((3, 4))), None),
        (add, (lambda x, b: x+b),
         (np.random.random((2, 3, 4)), 2), None),
        (sub, (lambda x, b: x-b),
         (np.random.random((2, 3, 4)), np.random.random((2, 3, 4))), None),
        (mul, (lambda x, b: x*b),
         (np.random.random((2, 3, 4)), np.random.random((2, 3, 4))), None),
        (pow, (lambda x, b: x**b), (np.random.random((2, 3, 4)), 3), None),
        (truediv, (lambda x, b: x/b),
         (np.random.random((2, 3, 4)), np.random.random((2, 3, 4))), None),
        (rtruediv, (lambda x, b: b/x),
         (np.random.random((2, 3, 4)), np.random.random((2, 3, 4))), None),
        (matmul, (lambda x, b: x@b),
         (np.random.random((4, 2)), np.random.random((2, 3))), None),
        (neg, (lambda x: -x), (np.random.random((2, 3, 4)),), None),
        (pow, (lambda x, b: x**b), (np.random.random((2, 3, 4)), 2), None),
    ]
    __test(ops)


def test_unary_ops():
    ops = [
        (exp, torch.exp, (np.random.random((2, 3, 4)),), None),
        (log, torch.log, (np.random.random((2, 3, 4)),), None),
        (flatten, torch.flatten, (np.random.random((2, 3, 4)),), None),
        (flatten, torch.flatten, (np.random.random(
            (2, 3, 4, 5)),), {"start_dim": 1, "end_dim": -2}),
    ]
    __test(ops, named=True)


def test_other_ops():
    ops = [
        (mean, torch.mean, (np.random.random((2, 3, 4)),), None),
        # (mean, torch.mean, (np.random.random((2, 3, 4)),), Params(axis=1)),
        (sum, torch.sum, (np.random.random((2, 3, 4)),), None),
        (sum, torch.sum, (np.random.random((2, 3, 4)),),
         {"axis": 2, "keepdim": True}),
        (unsqueeze, torch.unsqueeze,
         (np.random.random((2, 4)),), {"dim": 1}),
        (squeeze, torch.squeeze, (np.random.random((2, 1, 1, 1, 4)),), None),
        (squeeze, torch.squeeze,
         (np.random.random((2, 1, 1, 4)),), {"dim": 1}),
        (squeeze, torch.squeeze, (np.random.random(
            (2, 1, 1, 1, 4)),), {"dim": (1, 3)}),
    ]
    __test(ops)


def test_activations():
    from src.nn.activation import softmax, log_softmax, sigmoid, relu, tanh

    ops = [
        (relu, torch.relu, (np.random.random((2, 3))*10-5,), None),
        (softmax, torch.softmax, (np.random.random((2, 3)),), {"dim": -1}),
        (log_softmax, torch.log_softmax,
         (np.random.random((2, 3, 4)),), {"dim": -1}),
        (sigmoid, torch.sigmoid, (np.random.random((2, 3, 4)),), None),
        (tanh, torch.tanh, (np.random.random((2, 3, 4)),), None),
    ]
    __test(ops)


def test_loss():
    from src.nn import CrossEntropyLoss, NLLLoss, MSELoss

    def func(x, w, b, argmax=True):
        res = x @ w  # + b
        return res.log_softmax(-1) if argmax else res

    ops = [
        # TODO
        # (CrossEntropyLoss(), torch.nn.CrossEntropyLoss(),
        #  (
        #      np.random.random((100, 10, 3)),
        #      np.random.random((3, 10)),
        #      np.random.randint(0, 9, (100, 10))
        # ), False),
        (CrossEntropyLoss(), torch.nn.CrossEntropyLoss(),
         (
             np.random.random((100, 3)),
             np.random.random((3, 10)),
             np.random.random((10)),
             np.random.randint(0, 9, (100))
        ), False),
        (NLLLoss(), torch.nn.NLLLoss(),
         (
             np.random.random((100, 3)),
             np.random.random((3, 10)),
             np.random.random((10)),
             np.random.randint(0, 9, (100))
        ), True),
        # (MSELoss(), torch.nn.MSELoss(),
        #  (
        #      np.random.random((100, 3)),
        #      np.random.random((3)),
        #      np.random.random((1)),
        #      np.random.random((100))
        # ), False),
    ]

    for ourLoss, torchLoss, vars, argmax in ops:
        print("Op", ourLoss.__class__.__name__)
        x, w, b, y = vars
        [our_x, our_y], [torch_x, torch_y] = cast_vars(
            [x, y], requires_grad=False)

        [our_w, our_b], [torch_w, torch_b] = cast_vars([w, b])
        our_pred, our_true = func(
            our_x,
            our_w,
            our_b,
            argmax=argmax), our_y
        torch_pred, torch_true = func(
            torch_x,
            torch_w,
            torch_b,
            argmax=argmax), torch_y.long()

        assert np.allclose(our_pred, torch_pred.detach())
        loss1 = ourLoss(our_pred, our_true)
        loss2 = torchLoss(torch_pred, torch_true)
        # print("ddd", our_pred.shape, our_pred.shape)
        assert np.allclose(loss1, loss2.detach()), f"{loss1} != {loss2}"
        loss1.backward()
        loss2.backward()
        # assert False
        assert_close(our_w, torch_w)
        assert_close(our_b, torch_b)


def assert_shape(o, t):
    oshape, tshape = tuple(o.shape), tuple(t.shape)
    assert oshape == tshape, f"\nExpected shape: {tshape}\nFound: {oshape}"


def test_select():
    from src import nn, tensor
    import torch
    o = tensor.from_numpy(np.random.randint(
        0, 10, (10, 20, 30, 10))).requires_grad_()
    t = o.torch().requires_grad_()
    ores = o[1, 10:14, :, 9].mean()
    ores.backward()
    tres = t[1, 10:14, :, 9].mean()
    tres.backward()
    print(o.grad.shape, t.grad.shape)
    assert_close(o, t)


def test_copy_slice():
    from src import nn, tensor
    import torch
    o1 = tensor.from_numpy(np.random.randint(
        0, 10, (10, 20, 30, 10))).requires_grad_()
    o2 = tensor.from_numpy(np.random.randint(
        0, 10, (4, 30, 8))).requires_grad_()
    o3 = tensor.from_numpy(np.random.randint(
        0, 10, (10, 30, 4))).requires_grad_()

    t1 = o1.torch().requires_grad_(False)
    t2 = o2.torch()
    t3 = o3.torch()

    print("----")
    print(o1._backward._fn_name)
    o1[1, 10:14, :, 2:] = o2
    print(o1._backward._fn_name)
    o1[1, 10:, :, 4:8] = o3
    print(o1._backward._fn_name)
    o1.sum().backward()

    print("----")
    print(t1.grad_fn)
    t1[1, 10:14, :, 2:] = t2
    print(t1.grad_fn)
    t1[1, 10:, :, 4:8] = t3
    print(t1.grad_fn)
    t1.sum().backward()

    # assert_close(o1, t1)
    assert False


def test_lstm():
    import torch
    import numpy as np
    from src import nn, tensor
    input_size, hidden_size = 10, 3
    torch_lstm = torch.nn.LSTM(
        input_size, hidden_size, batch_first=True, bias=False)
    our_lstm = nn.LSTM(input_size, hidden_size)
    for _ in torch_lstm.all_weights:
        for index, __ in enumerate(_):
            for i, gate in enumerate([
                our_lstm.input_gate,
                our_lstm.forget_gate,
                our_lstm.cell_gate,
                our_lstm.output_gate,
            ]):
                gate.weights[index] = tensor.from_numpy(
                    __[i*hidden_size:(i+1)*hidden_size, :].T.detach().numpy()).requires_grad_()

    X = tensor.from_numpy(np.random.randint(0, 10, (2, 10, input_size)))
    Y = tensor.from_numpy(np.random.randint(0, 10, (2,)))

    toutput, (thn, tcn) = torch_lstm(X.torch())
    ooutput, (ohn, ocn) = our_lstm(X)

    assert_shape(ooutput, toutput)
    assert_shape(ohn, thn)
    assert_shape(ocn, tcn)

    assert_close(ooutput, toutput)
