from .. import *
import numpy as np
import sys
import os

import pytest
myPath = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, myPath + '/../src')


@pytest.mark.skip()
def test_tensor():
    arr = np.random.rand(1, 2, 4, 5)
    v1 = Tensor.array(arr) + 3
    assert v1.grad_fn == AddGradFn
