from src import *


class MyNet(Module):
    def __init__(self) -> None:
        super().__init__()
        self.layer1 = Dense(10, 100)
        self.layer2 = Dense(100, 10)

    def forward(self, x) -> _Tensor:
        x = self.layer1(x)
        x = self.layer2(x)
        return x


def test():
    import numpy as np
    X = np.random.randint(0, 100, size=(50, 10))
    y = np.random.randint(0, 100, size=(50))
    Loss = CrossEntropyLoss()
    X = tensor(X)
    y = tensor(y)
    net = MyNet()
    sgd = SGD(net.get_parameters())

    net.zero_grad()
    y_pred: _Tensor = net(X).argmax(axis=-1)
    loss = (y_pred-y)
    loss = loss.mean()
    loss.backward()


# test()
