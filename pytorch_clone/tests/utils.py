import torch
import numpy as np


def cast_vars(vars, tdtype=torch.float, requires_grad=True):
    from src import tensor
    ours = []
    torchs = []
    for var in vars:
        if isinstance(var, np.ndarray):
            v1 = tensor.from_numpy(var.copy()).astype(
                np.float32).requires_grad_(requires_grad)
            v2 = torch.from_numpy(var).type(
                tdtype).requires_grad_(requires_grad)
        else:
            v1 = v2 = var

        ours.append(v1)
        torchs.append(v2)
    return ours, torchs


def assert_close(var1, var2: torch.Tensor, atol=1e-6):
    print(np.abs(var1-var2.detach().numpy()).max())
    assert np.allclose(var1, var2.detach(),
                       atol=atol), f"Expected: \n{var2},\nFound: \n{var1}"
    # print(var1.grad.data.sum(), var2.grad.sum())
    if var2.grad is None and var1.grad is None:
        return
    assert var1.grad is not None, "var1.grad is None"
    assert var2.grad is not None, "var2.grad is None"
    assert np.allclose(var1.grad, var2.grad,
                       atol=atol), f"Expected Gradient: \n{var2.grad},\nFound Gradient: \n{var1.grad}"
