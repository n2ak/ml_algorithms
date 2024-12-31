from ._tensor import Tensor
from .layers import Linear, Module, Sequential
from .loss import cross_entropy


class DNN(Module):
    def __init__(self, inc, outc):
        self.seq = Sequential(
            Linear(inc, 20),
            Linear(20, 20),
            Linear(20, outc),
        )

    def forward(self, x):
        return self.seq(x)


def main():
    N, inc, outc = 100, 10, 2
    net = DNN(inc, outc)
    from .optim import SGD
    optim = SGD(net.trainable_params())
    from sklearn.datasets import make_classification
    X, y = make_classification(N, inc, n_classes=outc)
    optim.zero_grad()
    out = net.forward(Tensor(X))
    loss = cross_entropy(out, Tensor(y))
    loss.backward()
    optim.step()
    return net, X, y


if __name__ == "__main__":
    main()
