
import torch
import numpy as np
from src.log import LOGGER
import tqdm


def train(X, Y, params1, params2, iterations=100):
    from tests.compare import train_tick, acc
    ourModel, ourOptim, ourLossFn = params1
    torchModel, torchOptim, torchLossFn = params2

    bar = range(iterations)
    bar = tqdm.tqdm(bar)
    for _ in bar:
        pred, ourLoss = train_tick(
            ourModel,
            ourOptim,
            ourLossFn,
            X,
            Y,
        )
        pred2, torchLoss = train_tick(
            torchModel,
            torchOptim,
            torchLossFn,
            X.torch(),
            Y.torch(),
        )
        ourAcc = acc(pred, Y)
        torchAcc = acc(pred2, Y.torch().long())
        bar.set_description(
            f"OurLoss {ourLoss.data:.4f}, TorchLoss {torchLoss.item():.4f} OurAcc: {ourAcc*100:.1f} %, TorchAcc: {torchAcc.item()*100:.1f} %")
        LOGGER.debug("finished iteration")


if __name__ == "__main__":
    import sys
    import os
    path = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(path, r"..")
    sys.path.insert(0, path)

    from src import nn
    from src.dataset import load_diabetes

    class OurModel(nn.Module):
        def __init__(self, inn, out) -> None:
            super().__init__()

            self.seq = nn.Sequential([
                nn.Dense(inn, 32, bias=False),
                nn.ReLU(),
                nn.Dense(32, 64, bias=False),
                nn.ReLU(),
                nn.Dense(64, out, bias=False),
            ])

        def forward(self, x):
            for m in self.seq.layers:
                x = m(x)
            return x

    class TorchModel(torch.nn.Module):
        def __init__(self, inn, out) -> None:
            super().__init__()
            self.seq = torch.nn.ModuleList([
                torch.nn.Linear(inn, 32, bias=False),
                torch.nn.ReLU(),
                torch.nn.Linear(32, 64, bias=False),
                torch.nn.ReLU(),
                torch.nn.Linear(64, out, bias=False),
            ])

        def forward(self, x):
            for m in self.seq:
                x = m(x)
            return x

    # X, Y = load_mnist()
    X, Y = load_diabetes()
    ourModel = OurModel(10, 1)
    torchModel = TorchModel(10, 1)

    ourOptim = nn.SGD(ourModel.get_parameters())
    torchOptim = torch.optim.SGD(torchModel.parameters(), lr=ourOptim.lr)

    ourCEL = nn.MSELoss()
    torchCEL = torch.nn.MSELoss()

    params1 = ourModel, ourOptim, ourCEL
    params2 = torchModel, torchOptim, torchCEL

    # X = X.reshape(X.shape[0], 1, int(X.shape[1]**.5), -1)
    train(X, Y, params1, params2, iterations=100)
