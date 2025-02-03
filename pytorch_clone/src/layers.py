
import typing
from ._tensor import Tensor
from .initialization import kaiming


class Module:
    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def _params(self):
        raise NotImplementedError(
            f"_params is not implemented for {self.__class__.__name__}")

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

    def trainable_params(self) -> list[Tensor]:
        import inspect
        layers = [l for k, l in inspect.getmembers(
            self) if isinstance(l, Module)]

        def flat(elem, target: list):
            if isinstance(elem, (list, tuple)):
                for e in elem:
                    flat(e, target)
            else:
                target.append(elem)
            return target
        ps = []
        for l in layers:
            ps.extend(l._params())
        return flat(ps, [])


class Sequential(Module):
    def __init__(self, *layers: Module) -> None:
        self.layers = layers

    def forward(self, x):
        for l in self.layers:
            x = l(x)
        return x

    def _params(self):
        return [l._params() for l in self.layers if issubclass(type(l), Module)]


class Linear(Module):
    def __init__(self, inc, outc, bias=True) -> None:
        self.weight = kaiming((inc, outc), fan_mode=inc).requires_grad_()

        self.bias = None
        if bias:
            self.bias = kaiming((outc,), fan_mode=inc).requires_grad_()

    def forward(self, x):
        N, C = x.shape
        assert self.weight.shape[0] == C, f"Channels don't match {self.weight.shape[0]} != {C}"
        res = x.linear(self.weight, self.bias)
        return res

    def _params(self):
        if self.bias is None:
            return [self.weight]
        return [self.weight, self.bias]


def _2d_tuple(x, n=2):
    if isinstance(x, (tuple, list)):
        return tuple(x)
    return tuple([x for _ in range(n)])


class Conv2D(Module):
    def __init__(self, inc, outc, kernel, bias=True, conv_impl: typing.Literal["slow", "fast_forward", "fast"] = "slow") -> None:
        k1, k2 = _2d_tuple(kernel)
        self.weight = kaiming((outc, inc, k1, k2),
                              fan_mode=inc).requires_grad_()
        self.inc = inc
        impls = ["slow", "fast_forward", "fast"]
        assert conv_impl in impls, f"{conv_impl} not in {impls}"
        self.conv_impl = conv_impl
        self.bias = None
        if bias:
            self.bias = kaiming((outc,), fan_mode=inc).requires_grad_()

    def forward(self, x: Tensor):
        assert (x.ndim == 4), x.shape
        assert (x.shape[1] == self.inc), (self.inc, x.shape)
        d = {
            "slow": x.conv2d_slow,
            "fast_forward": x.conv2d,
            "fast": x.conv2d_fast,
        }
        func = d[self.conv_impl]
        return func(self.weight, self.bias)

    def _params(self):
        if self.bias is None:
            return [self.weight]
        return [self.weight, self.bias]


class ReLU(Module):
    def forward(self, x):
        return x.relu()

    def _params(self): return []
