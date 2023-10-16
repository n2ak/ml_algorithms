from __future__ import annotations
from enum import Enum
from functools import partial
from src._base import Layer, Function, Loss
from typing import Callable
from src.log import LOGGER

indent = 0
print_ok = LOGGER.is_debug()
max_depth = 5


class Printables(Enum):
    ACT = 0
    BACK = 1
    LOSS = 2
    ML_OPS = 3
    BASIC_OPS = 4


printable = [
    # Printables.ACT,
    Printables.BACK,
    # Printables.LOSS,
    # Printables.ML_OPS,
    # Printables.BASIC_OPS,
]


def is_printable(type):
    global indent, max_depth
    return (indent < max_depth) and (print_ok) and ((type in printable) or (type is None))


def __printed(type=None, arg_print=None):
    def helper(func: Callable):
        import functools

        @functools.wraps(func)
        def p(*args, **kwargs):
            if not is_printable(type):
                return func(*args, **kwargs)
            global indent
            if indent == 0:
                LOGGER.debug("------------------------------------")
            LOGGER.debug(
                " "*(indent*5),
                f"{func.__qualname__.replace('.<locals>','')}",
                end=(' ' if (arg_print is not None) else "\n")
            )
            if arg_print is not None:
                LOGGER.debug("called with:", level_prefix=False, end=" ")
                for arg in args:
                    arg_print(indent, arg)
                LOGGER.debug(level_prefix=False)
            # for k,arg in kwargs.items():
            #     arg_print(indent,arg)

            indent += 1
            res = func(*args, **kwargs)
            indent += -1
            return res
        return p
    return helper


def arg_printer(indent, arg):
    from src._tensor import _Tensor
    if isinstance(arg, _Tensor):
        LOGGER.debug(arg.shape, level_prefix=False, end=", ")


def as_module(name, module_name, base, include_training=False):
    import sys

    def decorator_factory(method):
        def set_args(self, args):
            self.args = args

        def call(self, args, kwargs):
            kwargs.update(self.args)
            if include_training:
                kwargs["training"] = self._parent_module.is_training
            return method(*args, **kwargs)

        new_type = type(name, base, {
            "__init__": lambda self, **args: set_args(self, args),
            "forward": lambda self, *args, **kwargs: call(self, args, kwargs)
        })
        setattr(sys.modules[module_name], name, new_type)

        return method
    return decorator_factory


# @printed
# def is_scalar(tensor: _Tensor) -> bool:
#     return np.isscalar(tensor)

printed = __printed()
printed_act = __printed(type=Printables.ACT)
printed_ops = __printed(type=Printables.BASIC_OPS)
printed_loss = __printed(type=Printables.LOSS)
printed_ml_ops = __printed(type=Printables.ML_OPS)
printed_grad = __printed(type=Printables.BACK, arg_print=arg_printer)
printed_grad = __printed(type=Printables.BACK, arg_print=arg_printer)


def __helper(base):
    return partial(as_module, module_name=f"src.nn", base=(base,))


as_layer = __helper(Layer)
as_loss_layer = __helper(Loss)
as_activation_layer = __helper(Function)
