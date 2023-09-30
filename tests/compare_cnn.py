
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
    from src.nn import Dense, SGD, Sequential, Flatten, Conv2D, CrossEntropyLoss, ReLU
    from src.dataset import load_mnist, load_cifar10

    class OurModel(Module):
        def __init__(self, inn, out, kernel_size, padding) -> None:
            super().__init__()

            self.seq = Sequential([
                Conv2D(inn, 10, kernel_size, padding),
                ReLU(),
                Conv2D(10, 10, kernel_size, padding),
                ReLU(),
                Flatten(),
                Dense(10240, out),
            ])

        def forward(self, x):
            # print("Our")
            for m in self.seq.layers:
                x = m(x)
                # print(x.shape)
            return x

    class TorchModel(torch.nn.Module):
        def __init__(self, inn, out, kernel_size, padding) -> None:
            super().__init__()
            self.seq = torch.nn.ModuleList(
                [torch.nn.Conv2d(inn, 10, kernel_size, padding=padding),
                 torch.nn.ReLU(),
                 torch.nn.Conv2d(10, 10, kernel_size, padding=padding),
                 torch.nn.ReLU(),
                 torch.nn.Flatten(),
                 torch.nn.Linear(10240, out),]
            )

        def forward(self, x):
            # print("Torch")
            for m in self.seq:
                x = m(x)
                # print(x.shape)
            return x

    # X, Y = load_mnist()
    X, Y = load_cifar10(100)
    _, channels, _, _ = X.shape
    n_classes = len(np.unique(Y))
    print(X.shape, Y.shape, n_classes)
    kernel_size = 3
    padding = "same"
    ourModel = OurModel(channels, n_classes, kernel_size, padding)
    torchModel = TorchModel(channels, n_classes, kernel_size, padding)

    ourOptim = SGD(ourModel.get_parameters())
    torchOptim = torch.optim.SGD(torchModel.parameters(), lr=ourOptim.lr)

    ourCEL = CrossEntropyLoss()
    torchCEL = torch.nn.CrossEntropyLoss()

    params1 = ourModel, ourOptim, ourCEL
    params2 = torchModel, torchOptim, torchCEL

    # X = X.reshape(X.shape[0], 1, int(X.shape[1]**.5), -1)
    train(X, Y, params1, params2, iterations=100)
