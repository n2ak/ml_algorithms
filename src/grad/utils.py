from __future__ import annotations
import contextlib
from typing import TYPE_CHECKING, Callable

if TYPE_CHECKING:

    from src._tensor import _Tensor
    from src.grad import GradFn

_grad_stack = [True]


@contextlib.contextmanager
def grad_off():
    global _grad_stack
    _grad_stack.append(False)
    yield
    _grad_stack.pop()


def _can_register_grad():
    assert len(_grad_stack)
    return _grad_stack[-1]


def register_grad_fn(cls: GradFn, reverse=False):

    def decorator_factory(method: Callable[[], _Tensor]):
        import functools

        @functools.wraps(method)
        def new_method(*args, **kwargs) -> _Tensor:
            if _can_register_grad() is False:
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
