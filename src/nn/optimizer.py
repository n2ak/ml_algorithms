from src._base import Optimizer


class SGD(Optimizer):
    def __init__(self, params, lr=.001, momentum=0) -> None:
        super().__init__(params)
        self.lr = lr
        self.momentum = momentum

    def step(self) -> None:
        for p in self.params:
            if p.grad is not None:
                p.data[:] = p - self.lr * p.grad


class Adam:
    # TODO: Adam
    pass
