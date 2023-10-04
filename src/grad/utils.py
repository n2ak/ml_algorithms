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


def _tensor_and_requires_grad(var):
    from src._tensor import _Tensor
    return isinstance(var, _Tensor) and var.requires_grad


def _pass_gradient(var, gradient):
    if _tensor_and_requires_grad(var):
        var.backward(gradient)


def correct_shape(v, gradient):
    summing = v.size < gradient.size
    if summing:
        if len(v.shape) == 0:
            return gradient.sum()
        summ = []
        for i in range(0, len(gradient.shape)):
            dim1 = v.shape[len(v.shape) - i - 1]
            dim2 = gradient.shape[len(gradient.shape) - i-1]
            if len(v.shape) - i - 1 < 0 or dim1 != dim2:
                # NOTE: when dim1 = 1
                index = len(gradient.shape) - i-1
                summ.append(index)
        grad = gradient
        if len(summ):
            grad = gradient.sum(axis=tuple(summ))
            grad = grad.reshape(shape=v.shape)
    else:
        import numpy as np
        o = np.broadcast(v.data, gradient)
        gradient.data = np.broadcast_to(gradient.data, o.shape)
        grad = gradient
    assert grad.shape == v.shape, f"did {summing=},{grad.shape} != {v.shape}"
    return grad


def register_grad(binary=False):
    def register(func):
        import functools

        @functools.wraps(func)
        def dec(*args, **kwargs):
            if _can_register_grad() is False:
                return func(*args, **kwargs)[0]
            requires_grad = args[0].requires_grad
            if binary:
                requires_grad = _tensor_and_requires_grad(
                    args[0]) or _tensor_and_requires_grad(args[1])
            with grad_off():
                res, backward = func(*args, **kwargs)
            if requires_grad:
                res.requires_grad = True
                backward._fn_name = f"{func.__name__.capitalize().replace('_','')}Backward"
                res._backward = backward
            return res
        return dec
    return register


# def register_grad_fn(cls: GradFn, reverse=False):
#     def decorator_factory(method: Callable[[], _Tensor]):
#         import functools

#         @functools.wraps(method)
#         def new_method(*args, **kwargs) -> _Tensor:
#             if _can_register_grad() is False:
#                 return method(*args, **kwargs)
#             with grad_off():
#                 result = method(*args, **kwargs)
#             _register_grad_fn_to_tensor(cls, result, reverse, args, kwargs)
#             return result  # return the method to be used normally
#         return new_method
#     return decorator_factory


# def _register_grad_fn_to_tensor(cls, result: _Tensor, reverse, args, kwargs):
#     from src._tensor import _Tensor
#     result.requires_grad = any(
#         [var.requires_grad for var in args if isinstance(var, _Tensor)])
#     if result.requires_grad:
#         if reverse:
#             args = list(reversed(args))
#         # NOTE  **kwargs
#         result.set_grad_fn(cls(args, result=result, **kwargs))
#     else:
#         pass
