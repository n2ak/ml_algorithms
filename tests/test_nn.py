

import pytest


@pytest.mark.skip("To be implemented later")
def test_iris():
    from sklearn.datasets import load_iris

    iris = load_iris()
    X = iris.data
    y = iris.target
