from tests.utils import comp, equal
from .. import *
import torch 
import numpy as np
import pytest

def equal_grad(a,b,t=1e-3):
    # assert len(args) % 2 == 0
    # for args in range(0,len(args),2)
    #     assert args
    assert tuple(Tensor.array(a.grad).shape) == tuple(Tensor.array(b.grad).shape)
    np.testing.assert_allclose(a.grad,b.grad)
    # assert (a.grad - b.grad).sum() <= t
def init(nums=None,shape1=(),shape2=()):
    a_,b_ = nums or (np.random.rand(*shape1),np.random.rand(*shape2))
    a1 = torch.tensor(a_,requires_grad=True)
    b1 = torch.tensor(b_,requires_grad=True)

    a2 = Tensor.array(a_,requires_grad=True)
    b2 = Tensor.array(b_)
    return a1,b1,a2,b2
def bin_op_test(func,nums=None):
    a_1,b_1,a_2,b_2 = init(nums=nums)
    def h(a1,b1,a2,b2):
        res1 = func(a1,b1)
        res2 = func(a2,b2)
        res1.backward()
        res2.backward()
        equal_grad(a1,a2)
        equal_grad(b1,b2)
    h(a_1,b_1,a_2,b_2)
    h(b_1,a_1,b_2,a_2)
def op_test(func):
    a1,_,a2,_ = init()
    res1 = func(a1)
    res2 = func(a2)
    res1.backward()
    print(a2.requires_grad,res2.requires_grad,)
    res2.backward()
    equal_grad(a1,a2)
def com_op(op):
    a1,b1,a2,b2 = init()
    (op(op(op(a1,b1),b1),a1)).backward()
    (op(op(op(a2,b2),b2),a2)).backward()
    equal_grad(a1,a2)
    equal_grad(b1,b2)
#------------------------------------------------------------------
def test_add():
    bin_op_test(lambda x,y: x+y)
def test_sub():
    bin_op_test(lambda x,y: x-y)
def test_mul():
    bin_op_test(lambda x,y: x+y)
def test_div():
    bin_op_test(lambda x,y: x/y)#,nums=(3.0,2.0))
def test_pow():
    bin_op_test(lambda x,y: x**y)
    
def test_mul2():
    op_test(lambda x: x*100)
def test_mul3():
    op_test(lambda x: -1*x)
@pytest.mark.skip("what operation is this ?")
def test_mul3():
    op_test(lambda x: -x)

def test_add2():
    com_op(lambda x,y: x+y)
def test_sub2():
    com_op(lambda x,y: x-y)
@pytest.mark.skip("Why ?")
def test_mul4():
    com_op(lambda x,y: x*y)

def test_exp():
    a1,_,a2,_ = init()
    res1 = a1.exp()
    res2 = a2.exp()
    res1.backward()
    res2.backward()
    equal_grad(a1,a2)
def test_mean():
    a = torch.rand(3,2)
    a.requires_grad = True
    a.mean().backward()
    
    b= Tensor.array(a.detach().numpy(),requires_grad=True)
    x = b.mean()
    x.backward()

    equal_grad(a,b)
def test_mean2():
    a = torch.rand(3,2)
    a.requires_grad = True
    a1 = a.mean(dim=0)
    a1 = a1.mean(dim=0)
    a1.backward()

    b= Tensor.array(a.detach().numpy(),requires_grad=True)
    b1 = b.mean(axis=0)
    b1 = b1.mean(axis=0)
    print(b1)
    b1.backward()

    equal_grad(a,b)
def test_sum():
    a = torch.rand(3,2)
    a.requires_grad = True
    a.sum().backward()
    
    b= Tensor.array(a.detach().numpy(),requires_grad=True)
    x = b.sum()
    x.backward()

    equal_grad(a,b)
def test_sum2():
    a = torch.rand(3,2)
    a.requires_grad = True
    a1 = a.sum(dim=0)
    a1 = a1.sum(dim=0)
    a1.backward()

    b= Tensor.array(a.detach().numpy(),requires_grad=True)
    b1 = b.sum(axis=0)
    b1 = b1.sum(axis=0)
    print(b1)
    b1.backward()

    equal_grad(a,b)