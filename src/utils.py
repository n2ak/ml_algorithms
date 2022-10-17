import numpy as np

def is_scalar(tensor):
    return np.isscalar(tensor)

def relu(tensor):
    tensor = tensor[:]
    tensor[tensor<0] = 0
    return tensor
    
def sigmoid(tensor):
    return 1 / ((-tensor).exp() + 1)

def softmax(tensor,dim=0):
    #avoids overflow , https://timvieira.github.io/blog/post/2014/02/11/exp-normalize-trick/
    b = tensor.max()
    e = (tensor - b).exp()
    return e/e.sum(axis=dim,keepdims=True)

def log_softmax(tensor,dim=0):
    return tensor.softmax(dim=dim).log()
    
def cross_entropy(x,t,dim=0,reduction="none",from_logits=False):
    # if from_logits:
    #     TODO : from_logits is True
    #     t = t.flatten().astype(np.int32)
    return x.log_softmax(dim).negative_log_likelihood(t,reduction=reduction)

# indent = 0
# def printed(func):
#     def p(*args,**kwargs):
#         print(" "*indent-1,f"{func.__name__}")
#         res = func(*args,**kwargs)
#         print(" "*indent,f"{res}")
#     return p

# @printed
def negative_log_likelihood(x,t,reduction="none"):
    lns = []
    for i,_ in enumerate(x):
        # print("a",i,t[i])
        lns.append(- x[i][t[i]])
    from .tensor import Tensor
    lns = Tensor.array(lns)
    if reduction in [None,"none"]:
        return lns
    elif reduction == "sum":
        return lns.sum()
    elif reduction == "mean":
        return lns.mean()
def conv2d_output_shape(x,out_,ks,p=0,s=1,d=0):
    b,_,w,h = tuple(x.shape)
    s1,s2 = s if isinstance(s,tuple) else (s,s)
    p1,p2 = p if isinstance(p,tuple) else (p,p)
    d1,d2 = d if isinstance(d,tuple) else (d,d)
    ks1,ks2 = ks
    from math import ceil
    # w,h = (w-ks1+p1+s1)/s1,(h-ks2+p2+s2)/s2
    # w = ceil(w) if w - int(w) < .5 else ceil(w)+1
    # h = ceil(h) if h - int(h) < .5 else ceil(h)+1

    w = (w+2*p1-d1*(ks1-1)-1)//s1 + 1
    h = (h+2*p2-d2*(ks2-1)-1)//s2 + 1
    out_shape = b,out_,w,h
    return out_shape
def linear(a,w,b):
    """
    returns a*w+b
    """
    assert a.shape[-1] == w.shape[0]
    res = a @ w
    if b is not None:
        assert b.shape[-1] == w.shape[-1]
    return res.biased(b)

def biased(x,bias=None):
    if bias is not None:
        assert tuple(x.shape) == tuple(bias.shape)
        x += bias
    return x
def sequential(x,layers):
    for layer in layers:
        x = layer(x)
    return x

def conv2d(x,w,b,padding=0,stride=1,dilation=0):
    pass