from __future__ import annotations
from typing import TYPE_CHECKING, Callable
if TYPE_CHECKING:
    from src._tensor import _Tensor

import numpy as np

indent = 0
print_ok = False
max_depth = 5
printable = [
    # "loss",
    # "act",
    "back"
    # "unary_ops",
    # "binary_ops",
    # "other_ops",
]


def is_printable(type):
    global indent, max_depth
    return (indent < max_depth) and (print_ok) and ((type in printable) or (type is None))


def _printed(type=None, arg_print=None):
    def helper(func: Callable):
        import functools

        @functools.wraps(func)
        def p(*args, **kwargs):
            if not is_printable(type):
                return func(*args, **kwargs)
            global indent
            if indent == 0:
                print("------------------------------------")
            print(" "*(indent*5), f"{func.__qualname__}", func)
            if arg_print is not None:
                print(" "*(indent*5), f"called with")
                for arg in args:
                    arg_print(indent, arg)
            # for k,arg in kwargs.items():
            #     arg_print(indent,arg)

            indent += 1
            res = func(*args, **kwargs)
            indent += -1
            return res
        return p
    return helper


printed = _printed()


def as_layer(name, module_name, base):
    import sys

    def decorator_factory(method):
        def set_args(self, args):
            self.args = args

        def call(self, args, kwargs):
            kwargs.update(self.args)
            return method(*args, **kwargs, **self.args)
        new_type = type(name, base, {
            "__init__": lambda self, **args: set_args(self, args),
            "forward": lambda self, *args, **kwargs: call(self, args, kwargs)
        })
        setattr(sys.modules[module_name], name, new_type)

        return method
    return decorator_factory


@printed
def is_scalar(tensor: _Tensor) -> bool:
    return np.isscalar(tensor)
