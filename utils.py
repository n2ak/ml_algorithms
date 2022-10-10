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
    return tensor.exp()/tensor.exp().sum(axis=dim)
    