import sys, os
myPath = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, myPath + '/../src')
import numpy as np
from .. import *

def test_tensor():
    arr = np.random.rand(1,2,4,5)
    v1 = Tensor.array(arr) + 3
    assert v1.grad_fn == AddGradFn