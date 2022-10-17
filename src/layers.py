from abc import ABC
from typing import List, overload

from torch.nn.modules import padding
from .core import HasForwardAndIsCallable
from .tensor import Tensor


class Layer(HasForwardAndIsCallable,ABC):

    @classmethod
    def init_weights(cls,in_,out_,*args,weights:Tensor=None):
        if weights is None:
            weights = Tensor.rand(in_,out_,*args)
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

class Conv2D(Layer):
    def __init__(self,in_,out_,kernel_size,stride=1,padding=0,dilation=1,groups=1,bias=True) -> None:
        super().__init__()
        self.in_,self.out_ = in_,out_
        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation
        self.padding = padding
        self.groups = groups
        
        self.bias = None if not bias else Tensor.zeros(self.out_)
        # TODO
        self.weights = self.init_weights(self.in_ , self.out_,*kernel_size)
    def forward(self, x):
        return x.conv2d(
            self.weights,
            self.bias,
            stride=self.stride,
            padding=self.padding,
            groups=self.dilation,
            )
    
class Sequential(HasForwardAndIsCallable):
    def __init__(self,layers:List[Layer]=[]) -> None:
        super().__init__()
        self.layers = list(layers)
    def add_layer(self,layer:Layer):
        self.layers.append(layer)
    def forward(self,x):
        return x.sequential(self.layers)

    