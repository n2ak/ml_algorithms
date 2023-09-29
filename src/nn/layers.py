from __future__ import annotations
from src._base import Layer, Function
from typing import List
from src.types import *
from src import initialization
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from src._tensor import _Tensor


class Flatten(Function):
    def __init__(self, start_dim=1, end_dim=-1) -> None:
        self.start_dim = start_dim
        self.end_dim = end_dim

    def forward(self, x: _Tensor) -> _Tensor:
        return x.flatten(start_dim=self.start_dim, end_dim=self.end_dim)


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
    def __init__(
        self,
        inn,
        out,
        kernel_size,
    ) -> None:
        super().__init__()

        self.channels = inn
        self.out = out
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)

        self.init_weights(inn, out, kernel_size)

    def forward(self, x: _Tensor) -> _Tensor:
        return x.conv2d(
            self.weights,
            bias=self.bias,
        )

    def init_weights(
        self,
        channels,
        out,
        kernel_size,
    ):
        from src._tensor import tensor
        import numpy as np
        self.weights = tensor(np.random.uniform(
            size=(out, channels, *kernel_size))).requires_grad_()
        self.bias = tensor(np.random.uniform(size=(out,))).requires_grad_()


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
