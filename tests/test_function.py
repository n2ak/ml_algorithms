from tests.utils import comp, equal
from .. import *
import torch
import numpy as np
import pytest


def test_relu():
    x = np.random.randn(10, 3, 2)
    comp(x, torch.nn.ReLU(), RelU())


def test_sigmoid():
    x = np.random.randn(10, 3, 2)
    comp(x, torch.nn.Sigmoid(), Sigmoid())


def test_softmax():
    x = np.random.randint(3, size=(1, 10, 4, 2)).astype(np.float32)
    for dim in range(len(x.shape)):
        comp(x, torch.nn.Softmax(dim=dim), Softmax(dim=dim), print_ok=True)


def test_log_softmax():
    x = np.random.randint(3, size=(1, 10, 4, 2)).astype(np.float32)
    for dim in range(len(x.shape)):
        comp(x, torch.nn.LogSoftmax(dim=dim),
             LogSoftmax(dim=dim), print_ok=True)
