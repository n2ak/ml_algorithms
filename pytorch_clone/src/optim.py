import dataclasses
import numpy as np
from ._tensor import Tensor
import typing


class BaseOptim:
    _initialized = False

    def __init__(self, params: typing.Generator[Tensor, None, None]) -> None:
        self.params = params

    def initialize(self):
        if not self._initialized:
            self.params = list(self.params)
            assert len(self.params) > 0
            self.init()
            self._initialized = True

    def step(self):
        self.initialize()
        self._step()

    def zero_grad(self):
        self.initialize()
        for p in self.params:
            p.gradient = None


class SGD(BaseOptim):

    def __init__(self, params, lr, momentum=0, dampening=0, weight_decay=0, nesterov=False) -> None:
        super().__init__(params)
        self.lr = lr
        self.momentum = momentum
        self.dampening = dampening
        self.weight_decay = weight_decay
        self.nesterov = nesterov
        self.t = 0

    def init(self):
        @dataclasses.dataclass
        class State:
            b: np.ndarray
        self.state: list[State] = [] if self.momentum == 0 else [
            State(0) for _ in range(len(self.params))]

    def _step(self):
        lr = self.lr
        momentum = self.momentum
        weight_decay = self.weight_decay
        dampening = self.dampening
        nesterov = self.nesterov
        t = self.t
        for i, p in enumerate(self.params):
            assert p.requires_grad
            g = p.gradient
            if weight_decay != 0:
                g = g + weight_decay * p.data
            if momentum != 0:
                state = self.state[i]
                if t > 1:
                    state.b = momentum * state.b + (1-dampening) * g
                else:
                    state.b = g
                if nesterov:
                    g = g + momentum * state.b
                else:
                    g = state.b
            p.data -= lr * g
        self.t += 1


class Adam(BaseOptim):

    def __init__(self, params, lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False) -> None:
        super().__init__(params)
        self.lr = lr
        self.weight_decay = weight_decay
        self.amsgrad = amsgrad
        self.eps = eps
        self.betas = betas
        self.t = 0

    def init(self):
        @dataclasses.dataclass
        class State:
            m: float  # | np.ndarray
            v: float  # | np.ndarray
            vhatmax: float  # | np.ndarray
        self.state: list[State] = [State(0, 0, 0)
                                   for _ in range(len(self.params))]

    def _step(self):
        lr = self.lr
        t = self.t
        weight_decay = self.weight_decay
        amsgrad = self.amsgrad
        beta1, beta2 = self.betas
        eps = self.eps
        for state, p in zip(self.state, self.params):
            assert p.requires_grad
            g = p.gradient
            if weight_decay != 0:
                g = g + weight_decay * p.data
            state.m = beta1 * state.m + (1-beta1) * g
            state.v = beta2 * state.v + (1-beta2) * (g**2)
            mhat = state.m/(1/beta1)
            vhat = state.v/(1/beta2)
            if amsgrad:
                state.vhatmax = np.maximum(state.vhatmax, vhat)
                p.data -= lr * mhat / (np.sqrt(state.vhatmax) + eps)
            else:
                p.data -= lr * mhat / (np.sqrt(vhat) + eps)
        self.t += 1
