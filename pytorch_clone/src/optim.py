from ._tensor import Tensor


class BaseOptim:
    def __init__(self, params: list[Tensor]) -> None:
        self.params = params

    def step(self):
        raise NotImplementedError("")

    def zero_grad(self):
        for p in self.params:
            p.gradient = None


class SGD(BaseOptim):
    def __init__(self, params, lr=1e-3, momentum=.9) -> None:
        super().__init__(params)
        self.lr = lr
        self.momentum = momentum

    def step(self):
        for p in self.params:
            assert p.requires_grad
            p.data -= self.lr * p.gradient
