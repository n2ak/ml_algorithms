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

def linear(a,w,b):
    """
    returns a*w+b
    """
    assert a.shape[-1] == w.shape[0]
    res = a @ w
    if b is not None:
        assert b.shape[-1] == w.shape[-1]
        res += b
    return  res

def sequential(x,layers):
    for layer in layers:
        x = layer(x)
    return x

