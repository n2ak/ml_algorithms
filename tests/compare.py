import tqdm
import torch
import numpy as np

np.random.seed(0)
torch.manual_seed(0)


def train_tick(model, optim, loss_fn, X, Y):
    optim.zero_grad()
    x = model(X)  # .relu()  # .softmax(dim=-1)

    loss = loss_fn(x, Y)
    try:
        # plot_graph(loss,figsize=(20,20))
        # loss.grad_fn.print_graph()
        pass
    except:
        pass
    loss.backward()
    optim.step()
    return x.argmax(-1), loss


def acc(pred, true):
    from sklearn.metrics import accuracy_score
    return accuracy_score(true, pred)


def train(X, Y, params1, params2, iterations=100):
    ourModel, ourOptim, ourLossFn = params1
    torchModel, torchOptim, torchLossFn = params2

    bar = range(iterations)
    bar = tqdm.tqdm(bar)
    for _ in bar:
        pred, ourLoss = train_tick(ourModel, ourOptim, ourLossFn, X, Y)
        pred2, torchLoss = train_tick(
            torchModel, torchOptim, torchLossFn, X.torch().float(), Y.torch().long())
        ourAcc = acc(pred, Y)
        torchAcc = acc(pred2, Y.torch().long())
        bar.set_description(
            f"OurLoss {ourLoss.data:.4f}, TorchLoss {torchLoss.item():.4f} OurAcc: {ourAcc*100:.1f} %, TorchAcc: {torchAcc.item()*100:.1f} %")


if __name__ == "__main__":
    import sys
    import os
    path = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(path, r"..")
    sys.path.insert(0, path)

    from src import Module
    from src.nn import Dense, SGD, Sequential, ReLU, CrossEntropyLoss
    from src.dataset import load_iris
    X, Y = load_iris()

    class OurModel(Module):
        def __init__(self, inn, out) -> None:
            super().__init__()

            self.seq = Sequential([
                Dense(inn, 64, bias=True),
                ReLU(),
                Dense(64, out, bias=True)
            ])

        def forward(self, x):
            x = self.seq(x)
            return x

    class TorchModel(torch.nn.Module):
        def __init__(self, inn, out) -> None:
            super().__init__()
            self.seq = torch.nn.Sequential(
                torch.nn.Linear(inn, 64, bias=True),
                torch.nn.ReLU(),
                torch.nn.Linear(64, out, bias=True),
            )

        def forward(self, x):
            x = self.seq(x)
            return x

    ourModel = OurModel(4, 3)
    torchModel = TorchModel(4, 3)

    ourOptim = SGD(ourModel.get_parameters())
    torchOptim = torch.optim.SGD(torchModel.parameters(), lr=ourOptim.lr)

    ourCEL = CrossEntropyLoss()
    torchCEL = torch.nn.CrossEntropyLoss()

    params1 = ourModel, ourOptim, ourCEL
    params2 = torchModel, torchOptim, torchCEL

    train(X, Y, params1, params2, iterations=1000)
