from __future__ import annotations
import contextlib
from typing import TYPE_CHECKING, Callable, Any, ClassVar

if TYPE_CHECKING:

    from src._tensor import _Tensor
    from src.grad import GradFn

import contextlib

# Note we use file to store gradu status, the called GradAvail object first is not the same as the other ones


class GradAvail:
    _init = True
    _path = "status.txt"

    @classmethod
    def _prev(cls):
        import os
        if not os.path.exists(cls._path):
            return cls._init
        with open(cls._path, "r") as f:
            content = f.read()
            return content == "True"

    @classmethod
    def _set_grad(cls, status):
        status = bool(status)
        with open(cls._path, "wt") as f:
            f.write(str(status))


class grad_off:
    def __enter__(self) -> None:
        self.prev = GradAvail._prev()
        GradAvail._set_grad(False)

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        GradAvail._set_grad(self.prev)


def can_register_grad():
    return GradAvail._prev()


def register_grad_fn(cls: GradFn, reverse=False):

    def decorator_factory(method: Callable[[], _Tensor]):
        import functools

        @functools.wraps(method)
        def new_method(*args, **kwargs) -> _Tensor:
            if can_register_grad() is False:
                return method(*args, **kwargs)
            with grad_off():
                result = method(*args, **kwargs)
            _register_grad_fn_to_tensor(cls, result, reverse, args, kwargs)
            return result  # return the method to be used normally
        return new_method
    return decorator_factory


def _register_grad_fn_to_tensor(cls, result: _Tensor, reverse, args, kwargs):
    from src._tensor import _Tensor
    result.requires_grad = any(
        [var.requires_grad for var in args if isinstance(var, _Tensor)])
    if result.requires_grad:
        if reverse:
            args = list(reversed(args))
        # NOTE  **kwargs
        result.set_grad_fn(cls(args, result=result, **kwargs))
    else:
        pass
