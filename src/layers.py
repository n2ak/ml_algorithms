from abc import ABC
from typing import List, overload
from .core import HasForwardAndIsCallable
from .tensor import Tensor


class Layer(HasForwardAndIsCallable,ABC):

    @classmethod
    def init_weights(cls,in_,out_,weights:Tensor=None):
        if weights is None:
            weights = Tensor.rand(in_,out_)
        else:
            assert weights.shape == (in_,out_) ,"Invalid weights passed"
        return weights        

class Dense(Layer):
    def __init__(self,in_,out_,bias:bool=True) -> None:
        super().__init__()
        self.in_ , self.out_ = in_, out_

        self.bias = None if not bias else Tensor.zeros(self.out_)
        self.weights = self.init_weights(self.in_ , self.out_)
        
    def forward(self,x:Tensor):
        return x.linear(self.weights,self.bias)
    @classmethod
    def from_weights(cls,weights,bias=True):
        layer = Dense(*weights.shape,bias=bias)
        layer.weights = weights
        return layer


    
class Sequential(HasForwardAndIsCallable):
    def __init__(self,layers:List[Layer]=[]) -> None:
        super().__init__()

        self.layers = list(layers)
    def add_layer(self,layer:Layer):
        self.layers.append(layer)
    def forward(self,x):
        return x.sequential(self.layers)

    