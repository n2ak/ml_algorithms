

def _load_dataset(loader):
    from src import tensor
    x, y = loader()
    x = tensor(x)
    y = tensor(y)
    return x, y


def load_mnist():
    from sklearn.datasets import load_digits
    return _load_dataset(lambda: load_digits(return_X_y=True))


def load_iris():
    from sklearn.datasets import load_iris
    return _load_dataset(lambda: load_iris(return_X_y=True))
