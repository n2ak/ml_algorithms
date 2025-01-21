
def dataloader(n, *arrays, to_tensor=True):
    from ._tensor import Tensor
    import numpy as np
    batch_size = arrays[0].shape[0]
    assert all([arr.shape[0] == batch_size for arr in arrays])
    indices = np.random.permutation(batch_size)
    import itertools
    i = iter(indices)
    piece = list(itertools.islice(i, n))
    while piece:
        res = [arr[piece] for arr in arrays]
        if to_tensor:
            res = [Tensor(r) for r in res]
        yield res
        piece = list(itertools.islice(i, n))
