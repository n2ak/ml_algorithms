from ._tensor import Tensor


def kaiming(
    size,
    fan_mode,
    distribution="uniform",
    fun="relu",
):
    import numpy as np
    gain = get_gain(fun)
    if isinstance(fan_mode, str):
        assert fan_mode in ["fan_in", "fan_out"]
        f_in, f_out = calculate_fans(size)
        fan_mode = f_in if fan_mode == "fan_in" else f_out
    if distribution == "normal":
        std = gain/np.sqrt(fan_mode)
        t = np.random.normal(0, std**2, size)
    elif distribution == "uniform":
        bound = gain * np.sqrt(3/(fan_mode))
        t = np.random.uniform(-bound, bound, size)
    else:
        raise NotImplementedError(f"Unknown distribution: {distribution}")

    return Tensor(t)


def get_gain(fun: str):
    import numpy as np
    # https://pytorch.org/docs/stable/nn.init.html#torch.nn.init.calculate_gain
    fun = fun.lower()
    ones = ["conv1d", "conv2d", "conv3d", "sigmoid", "linear", "identity"]
    if fun in ones:
        return 1
    d = {
        "relu": np.sqrt(2),
        "selu": 3/4,
        "tanh": 5/4
    }
    return d[fun]


def calculate_fans(shape):
    import numpy as np
    prod = np.prod(shape)
    return prod//shape[1], prod//shape[0]
