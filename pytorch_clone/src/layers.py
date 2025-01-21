
from ._tensor import Tensor
from typing import Any
from .initialization import kaiming


class Module:
    def __call__(self, *args: Any, **kwds: Any):
        return self.forward(*args, **kwds)

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
    def __init__(self, *layers) -> None:
        self.layers = layers

    def forward(self, x):
        for l in self.layers:
            x = l.forward(x)
        return x

    def _params(self):
        return [l._params() for l in self.layers]


class Linear(Module):
    def __init__(self, inc, outc, bias=False) -> None:
        self.weight = kaiming((inc, outc), fan_mode=inc).requires_grad_()

        self.bias = None
        if bias:
            self.bias = kaiming((outc,), fan_mode=inc).requires_grad_()

    def forward(self, x):
        assert x.ndim == 2
        res = x.linear(self.weight, self.bias)
        return res

    def _params(self):
        if self.bias is None:
            return [self.weight]
        return [self.weight, self.bias]


class ReLU(Module):
    def forward(self, x):
        return x.relu()

    def _params(self): return []
