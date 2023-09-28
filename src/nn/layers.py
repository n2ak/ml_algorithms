from __future__ import annotations
from src._base import Layer
from typing import List
from src.types import *
from src import initialization
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from src._tensor import _Tensor


class Dense(Layer):

    def __init__(self, in_, out_, bias: bool = True) -> None:
        super().__init__()
        self.in_, self.out_ = in_, out_

        self.init_weights(self.in_, self.out_, bias)

    def forward(self, x: _Tensor) -> _Tensor:
        res = x.linear(self.weights, self.bias)
        return res

    @ classmethod
    def from_weights(cls, weights, bias=True):
        from src._tensor import tensor
        layer = Dense(*weights.shape, bias=bias)
        layer.weights = tensor(weights).requires_grad_()
        return layer

    def get_trainable_params(self) -> List[_Tensor]:
        params = [self.weights]
        if self.bias is not None:
            params.append(self.bias)
        return params

    def init_weights(self, inn, out, bias):
        from src._tensor import tensor
        import numpy as np

        self.weights = initialization.kaiming(
            (inn, out),
            fan_mode=inn,
        ).requires_grad_()

        self.bias = None
        if bias:
            self.bias = initialization.kaiming(
                (out,),
                fan_mode=inn,
            ).requires_grad_()


class Conv2D(Layer):
    def __init__(self, in_, out_, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True) -> None:
        super().__init__()
        self.in_, self.out_ = in_, out_
        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation
        self.padding = padding
        self.groups = groups

        from src._tensor import tensor_zeros
        self.bias = None if not bias else tensor_zeros(self.out_)
        # TODO
        self.weights = self.init_weights(self.in_, self.out_, *kernel_size)

    def forward(self, x: _Tensor) -> _Tensor:
        return x.conv2d(
            self.weights,
            self.bias,
            stride=self.stride,
            padding=self.padding,
            groups=self.dilation,
        )


class Sequential(Layer):
    def __init__(self, layers: List[Layer] = []) -> None:
        super().__init__()
        self.layers = layers

    def add_layer(self, layer: Layer):
        self.layers.append(layer)

    def forward(self, x: _Tensor) -> _Tensor:
        return x.sequential(self.layers)

    def get_trainable_params(self):
        from src._base import _Trainable
        params = []
        for l in self.layers:
            if isinstance(l, _Trainable):
                params.extend(l.get_trainable_params())
        return params
