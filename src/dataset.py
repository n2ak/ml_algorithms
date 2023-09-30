

def _load_dataset(loader, swap_channel=False):
    from src import tensor
    x, y = loader()
    if swap_channel:
        import numpy as np
        x = np.transpose(x, (0, 3, 1, 2))
    x = tensor(x)
    y = tensor(y)
    return x, y


def load_mnist():
    from sklearn.datasets import load_digits
    return _load_dataset(lambda: load_digits(return_X_y=True))


def load_cifar10(len=None):
    import torchvision
    import os
    path = os.path.join(__file__, "../../datasets")
    train_d = torchvision.datasets.CIFAR10(
        path, train=True, download=True, transform=None)
    # te_d = torchvision.datasets.CIFAR10(path, train=False, download=True, transform=None)
    X, Y = [], []
    import numpy as np
    for x, y in list(train_d)[:len]:
        x, y = np.array(x), np.array(y)
        X.append(x)
        Y.append(y)
    X = np.array(X)
    Y = np.array(Y)
    return _load_dataset(lambda: (X, Y), swap_channel=True)


def load_iris():
    from sklearn.datasets import load_iris
    return _load_dataset(lambda: load_iris(return_X_y=True))
