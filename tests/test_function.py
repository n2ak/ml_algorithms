from tests.utils import equal
from .. import *
import torch 
import numpy as np

def test_relu():
    x = np.random.randn(10,3,2)
    comp(x,torch.nn.ReLU(),RelU())
    
def test_sigmoid():
    x = np.random.randn(10,3,2)
    comp(x,torch.nn.Sigmoid(),Sigmoid())
import pytest
@pytest.mark.skip(reason="")
def test_softmax():
    x = np.random.randint(10,size=(10,29,30)).astype(np.float32)
    for dim in range(len(x.shape)):
        comp(x,torch.nn.Softmax(dim=dim),Softmax(dim=dim),print_ok=False)

def test_softmax():
    x = np.random.randint(10,size=(10,29,30)).astype(np.float32)
    comp(x,torch.nn.CrossEntropyLoss(),CrossEntropyLoss(),print_ok=False)


def comp(x,class1,class2,print_ok=False):
    a = class1(torch.tensor(x))
    b = class2(Tensor.array(x))
    assert equal(a,b,print_ok=print_ok)