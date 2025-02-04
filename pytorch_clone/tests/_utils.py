import torch
from src import Tensor
import numpy as np
from src.grad_utils import grad_check


def randn(*shape, requires_grad=True, seed=0) -> Tensor:
    np.random.seed(seed)
    t = Tensor.randn(*shape).relu()+1  # for log
    return t.requires_grad_(val=requires_grad)


def randint(min, max, shape, requires_grad=True, seed=0) -> Tensor:
    np.random.seed(seed)
    t = Tensor(np.random.randint(min, max, shape))
    return t.requires_grad_(val=requires_grad)


def compare_values(outputs, torch_outputs, rtol=1e-05, atol=1e-08):
    if isinstance(outputs, Tensor):
        assert isinstance(torch_outputs, torch.Tensor)
        outputs = [outputs]
        torch_outputs = [torch_outputs]
    assert len(outputs) == len(torch_outputs)
    for o, to in zip(outputs, torch_outputs):
        o = o.numpy()
        to = to.numpy()
        assert o.shape == to.shape
        if not np.allclose(o, to, rtol=rtol, atol=atol):
            print("Nm", o.flatten()[:10])
            print("An", to.flatten()[:10])
            assert False


def check_outputs(func, *inputs: Tensor, torch_func=None):
    if torch_func is None:
        torch_func = func
    torch_inputs = [torch.from_numpy(t.numpy()) for t in inputs]
    outputs = func(*inputs)
    torch_outputs = torch_func(*torch_inputs)
    compare_values(outputs, torch_outputs)


def check(func, *inputs, torch_func=None):
    check_outputs(func, *inputs, torch_func=torch_func)
    grad_check(func, *inputs,)
