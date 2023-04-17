from __future__ import annotations
from typing import TYPE_CHECKING, Callable, Any
if TYPE_CHECKING:
    from src.tensor import Tensor


def register_grad_fn(cls):
    import functools

    def decorator_factory(method: Callable[[], Tensor]):
        @functools.wraps(method)
        def new_method(*args, **kwargs) -> Tensor:
            result = method(*args, **kwargs)
            from src import Tensor
            result.requires_grad = any(
                [var.requires_grad for var in args if isinstance(var, Tensor)])
            if result.requires_grad:
                result.set_grad_fn(cls(args, **kwargs))
            else:
                pass
                # print("No", [v.numpy() for v in args])
            return result  # return the method to be used normally
        return new_method
    return decorator_factory
