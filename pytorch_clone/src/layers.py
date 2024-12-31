
from typing import Any
from .initialization import kaiming


class Module:
    def __call__(self, *args: Any, **kwds: Any):
        return self.forward(*args, **kwds)

    def _params(self): return []

    def weights_from_torch(self, module):
        import torch
        name = module.__class__.__name__
        if name != self.TORCH_EQUIV_MOD:
            raise ValueError(
                f"Expected '{self.TORCH_EQUIV_MOD}' but found '{name}'")

        from src.utils import copy_param

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

    def trainable_params(self):
        import inspect
        layers = [l for k, l in inspect.getmembers(
            self) if isinstance(l, Module)]
        ps = []
        for l in layers:
            ps.extend(l._params())
        return ps


class Sequential(Module):
    def __init__(self, *layers) -> None:
        self.layers = layers

    def forward(self, x):
        for l in self.layers:
            x = l(x)
        return x


class Linear(Module):
    def __init__(self, inc, outc, bias=False) -> None:
        self.weight = kaiming(
            (inc, outc),
            fan_mode=inc,
        ).requires_grad_()

        self.bias = None
        if bias:
            self.bias = kaiming(
                (outc,),
                fan_mode=inc,
            ).requires_grad_()

    def forward(self, x):
        res = x.linear(self.weight, self.bias)
        return res
