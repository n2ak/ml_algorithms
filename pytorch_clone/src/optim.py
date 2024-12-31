class SGD:
    def __init__(self, params, lr=1e-3) -> None:
        self.params = params
        self.lr = lr

    def step(self):
        for p in self.params:
            p.data -= self.lr * p.gradient

    def zero_grad(self):
        for p in self.params:
            p.gradient = None
