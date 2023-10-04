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
        res = x.flatten(start_dim=self.start_dim, end_dim=self.end_dim)
        return res


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
        from src import tensor
        layer = Dense(*weights.shape, bias=bias)
        layer.weights = tensor.from_numpy(weights).requires_grad_()
        return layer

    def init_weights(self, inn, out, bias):
        from src import tensor
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
        padding,
        stride=1,
    ) -> None:
        super().__init__()

        self.channels = inn
        self.out = out
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)
        if isinstance(stride, int):
            stride = (stride, stride)

        self.init_weights(inn, out, kernel_size, stride)

        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

    def forward(self, x: _Tensor) -> _Tensor:
        from src import tensor
        b, c, d1, d2 = x.shape
        output_shape = (b, self.out, d1, d2)
        output = tensor.zeros(output_shape).requires_grad_()
        x.conv2d(
            kernels=self.weights,
            bias=self.bias,
            output=output,
            padding=self.padding,
            stride=self.stride
        )
        return output

    def init_weights(
        self,
        channels,
        out,
        kernel_size,
        stride,
    ):
        from src import tensor
        import numpy as np
        self.weights = tensor.from_numpy(
            np.random.uniform(size=(out, channels, *kernel_size)),
        ).requires_grad_()
        self.bias = tensor.from_numpy(
            np.random.uniform(size=(out,))).requires_grad_()


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

# class _LSTM_LAYER:


class LSTM_GATE(Layer):
    def __init__(
        self,
        input_size,
        hidden_size,
        bias=False
    ):
        self.init_weights(input_size, hidden_size, bias=bias)

    def forward(self, x, h) -> _Tensor:
        x = (x @ self.weights[0] + h @ self.weights[1])
        if self.bias is not None:
            x = self.bias[0] + self.bias[1]
        return x

    def init_weights(self, input_size, hidden_size, bias=False):
        self.weights = [
            initialization.kaiming((input_size, hidden_size), hidden_size),
            initialization.kaiming((hidden_size, hidden_size), hidden_size),
        ]
        self.bias = None
        if bias:
            self.bias = [
                initialization.kaiming((hidden_size), hidden_size),
                initialization.kaiming((hidden_size), hidden_size),
            ]

    def get_trainable_params(self) -> List[_Tensor]:
        return


class LSTM(Layer):
    def __init__(
        self,
        input_size,
        hidden_size,
        bias=False
    ) -> None:
        self.input_gate = LSTM_GATE(input_size, hidden_size, bias=bias)
        self.forget_gate = LSTM_GATE(input_size, hidden_size, bias=bias)
        self.cell_gate = LSTM_GATE(input_size, hidden_size, bias=bias)
        self.output_gate = LSTM_GATE(input_size, hidden_size, bias=bias)
        self.input_size = input_size
        self.hidden_size = hidden_size

    def forward(self, X) -> _Tensor:
        N, T, C = X.shape
        from src import tensor
        ct = tensor.zeros((N, self.hidden_size))
        ht = tensor.zeros((N, self.hidden_size))
        os = tensor.zeros((N, T, self.hidden_size))
        for i in range(T):
            xt = tensor.from_numpy(X.data[:, i, :])  # ,100 10,100
            ht_1 = ht
            ct_1 = ct

            it = self.input_gate(xt, ht_1).sigmoid()
            ft = self.forget_gate(xt, ht_1).sigmoid()
            gt = self.cell_gate(xt, ht_1).tanh()
            ot = self.output_gate(xt, ht_1).sigmoid()

            ct = ft * ct_1 + it * gt
            ht = ot * ct.tanh()
            os.data[:, i, :] = ot
        return os, (ht.unsqueeze(0), ct.unsqueeze(0))
