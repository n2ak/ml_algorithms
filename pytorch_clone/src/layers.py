
import typing
from ._tensor import Tensor
from .initialization import kaiming


class Module:
    initialized = True
    _params = []
    _sub_modules = []

    def __call__(self, *args, **kwargs):
        if not self.initialized:
            self.init(*args, **kwargs)
            self.initialized = True
        self.assert_shape(*args, **kwargs)
        return self.forward(*args, **kwargs)

    def assert_shape(self, *args, **kwargs):
        pass

    def weights_from_torch(self, module):
        import torch
        name = module.__class__.__name__
        if name != self.TORCH_EQUIV_MOD:
            raise ValueError(
                f"Expected '{self.TORCH_EQUIV_MOD}' but found '{name}'")

        from .utils import copy_param

        def copy(our, t_model: torch.nn.Module):
            if (func := getattr(our, "_set_params_from_torch", None)) is not None:
                func(t_model)
            else:
                for k, v in t_model.named_parameters():
                    ttensor: torch.nn.Parameter = v
                    otensor = getattr(our, k, None)
                    if otensor is None:
                        raise ValueError(
                            f"No '{k}' var in {our.__class__.__name__}")
                    copy_param(our, k, ttensor.detach())
        copy(self, module)

    def get_modules(self):
        import inspect
        layers = [l for k, l in inspect.getmembers(
            self) if isinstance(l, Module)]
        return layers

    def trainable_params(self) -> typing.Generator[Tensor, None, None]:

        from collections import deque
        Q = deque([self])
        while len(Q):
            module = Q.popleft()
            if not issubclass(type(module), Module):
                # ignore if it is a function
                continue
            assert module.initialized
            Q.extendleft(module._sub_modules)
            Q.extendleft(module.get_modules())
            for p in module._params:
                assert isinstance(p, Tensor), type(p)
                assert p.requires_grad
                yield p


def flat(arr_or_elem, result: list = []):
    if isinstance(arr_or_elem, (list, tuple, GeneratorExit)):
        for e in arr_or_elem:
            flat(e, result)
    else:
        result.append(arr_or_elem)
    return result


class Sequential(Module):

    def __init__(self, *layers: Module) -> None:
        self._sub_modules = self.layers = layers

    def forward(self, x):
        for l in self.layers:
            x = l(x)
        return x


class LazyModule(Module):
    initialized = False


class Linear(LazyModule):
    initialized = False

    def __init__(self, inc, outc, bias=True) -> None:
        self.p = inc, outc, bias

    def assert_shape(self, x):
        assert x.ndim == 2
        assert x.shape[1] == self.weight.shape[0], (x.shape, self.weight.shape)

    def init(self, x):
        inc, outc, bias = self.p
        if inc is None:
            inc = x.shape[1]
        self.weight = kaiming((inc, outc), fan_mode=inc).requires_grad_()

        self.bias = None
        self._params = [self.weight]
        if bias:
            self.bias = kaiming((outc,), fan_mode=inc).requires_grad_()
            self._params += [self.bias]

    def forward(self, x):
        res = x.linear(self.weight, self.bias)
        return res


def _2d_tuple(x, n=2):
    if isinstance(x, (tuple, list)):
        return tuple(x)
    return tuple([x for _ in range(n)])


class Conv2D(LazyModule):
    initialized = False

    def __init__(self, inc, outc, kernel, bias=True, conv_impl: typing.Literal["slow", "fast_forward", "fast"] = "slow") -> None:
        self.p = inc, outc, kernel, bias, conv_impl

    def init(self, x):
        inc, outc, kernel, bias, conv_impl = self.p
        impls = ["slow", "fast_forward", "fast"]
        assert conv_impl in impls, f"{conv_impl} not in {impls}"
        k1, k2 = _2d_tuple(kernel)
        if inc is None:
            inc = x.shape[1]

        self.weight = kaiming((outc, inc, k1, k2),
                              fan_mode=inc).requires_grad_()
        self.conv_impl = conv_impl
        self.bias = None
        self.inc = inc
        self._params = [self.weight]
        if bias:
            self.bias = kaiming((outc,), fan_mode=inc).requires_grad_()
            self._params += [self.bias]

    def assert_shape(self, x):
        assert (x.ndim == 4), x.shape
        assert (x.shape[1] == self.inc), (self.inc, x.shape)

    def forward(self, x: Tensor):
        d = {
            "slow": x.conv2d_slow,
            "fast_forward": x.conv2d,
            "fast": x.conv2d_fast,
        }
        func = d[self.conv_impl]
        return func(self.weight, self.bias)


class Activation(Module):
    def init(self, x): pass


class ReLU(Activation):
    def forward(self, x):
        return x.relu()
