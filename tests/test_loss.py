from tests.utils import comp
from src import *
import pytest


def test_crossentropyloss():
    import torch
    x = torch.rand(30, 10)
    t = torch.randint(1, 10, size=(30,))
    reductions = "none",  # "sum", "mean"
    for reduction in reductions:
        comp(x,
             torch.nn.CrossEntropyLoss(reduction=reduction),
             CrossEntropyLoss(reduction=reduction),
             args1=[torch.tensor(t)],
             args2=[tensor(t.numpy(), dtype=np.int32)],
             # print_ok=True
             )


@pytest.mark.skip("nll not implimented for multi-dim targets")
def test_crossentropyloss2():
    import torch
    x = torch.randn(30, 10, 10)
    t = torch.randint(1, 10, size=(30, 10))
    reductions = "none", "sum", "mean"
    for reduction in reductions:
        comp(x,
             torch.nn.CrossEntropyLoss(reduction=reduction),
             CrossEntropyLoss(reduction=reduction),
             args1=[torch.tensor(t)],
             args2=[tensor(t.numpy(), dtype=np.int32)],
             # print_ok=True
             )


def test_NegativeLogLikelihoodLoss():
    import torch
    torch.seed = 1
    x = torch.rand(3, 3).type(torch.FloatTensor)
    t = torch.randint(0, 3, size=(3,))
    reductions = "none",  # "sum", "mean"
    for reduction in list(reductions)[:]:
        comp(x,
             torch.nn.NLLLoss(reduction=reduction),
             NegativeLogLikelihoodLoss(reduction=reduction),
             args1=[torch.tensor(t.numpy())],
             args2=[tensor(t, dtype=np.int32)],
             # print_ok=True
             )
