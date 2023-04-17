from tests.utils import comp, equal
from src import *
import pytest
import torch


@pytest.mark.skip("broken")
def test_1():
    b = 100
    unseed()

    def randint(low, high):
        a = int(np.random.randint(low, high, size=()))
        return a
    in_ = randint(1, 3)
    out_ = randint(1, 3)
    w = randint(1, 1000)
    h = randint(1, 1000)
    p = randint(1, 10), randint(1, 10)
    s = randint(1, 10), randint(1, 10)
    ks = randint(1, 10), randint(1, 10)

    x = torch.rand(b, in_, w, h)
    res1 = conv2d_output_shape(x, out_, ks, p, s)
    res2 = torch.nn.Conv2d(in_, out_, ks, padding=p, stride=s)(x).shape
    assert res1 == res2
