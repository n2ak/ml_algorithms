
import tqdm
import torch
import numpy as np


if __name__ == "__main__":
    import sys
    import os
    path = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(path, r"..")
    sys.path.insert(0, path)
    from tests.compare import *

    from src import Module
    from src.nn import Dense, SGD, Sequential, Flatten, Conv2D, CrossEntropyLoss
    from src.dataset import load_mnist

    class OurModel(Module):
        def __init__(self, inn, out) -> None:
            super().__init__()

            self.seq = Sequential([
                Conv2D(1, 10, 2),
                Conv2D(10, 10, 2),
                Flatten(),
                Dense(360, out),
            ])

        def forward(self, x):
            x = self.seq(x)
            return x

    class TorchModel(torch.nn.Module):
        def __init__(self, inn, out) -> None:
            super().__init__()
            self.seq = torch.nn.Sequential(
                torch.nn.Conv2d(1, 10, 2),
                torch.nn.Conv2d(10, 10, 2),
                torch.nn.Flatten(),
                torch.nn.Linear(360, out)
            )

        def forward(self, x):
            x = self.seq(x)
            return x

    X, Y = load_mnist()
    n_classes = len(np.unique(Y))
    ourModel = OurModel(4, 3)
    torchModel = TorchModel(1, n_classes)

    ourOptim = SGD(ourModel.get_parameters())
    torchOptim = torch.optim.SGD(torchModel.parameters(), lr=ourOptim.lr)

    ourCEL = CrossEntropyLoss()
    torchCEL = torch.nn.CrossEntropyLoss()

    params1 = ourModel, ourOptim, ourCEL
    params2 = torchModel, torchOptim, torchCEL

    X = X.reshape(X.shape[0], 1, int(X.shape[1]**.5), -1)
    print(X.shape)
    train(X, Y, params1, params2, iterations=1000)
