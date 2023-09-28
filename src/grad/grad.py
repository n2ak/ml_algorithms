from __future__ import annotations
from abc import ABC, abstractmethod
from typing import List, Tuple, TYPE_CHECKING
from src.grad.utils import grad_off
if TYPE_CHECKING:
    from src._tensor import _Tensor
from src.utils import _printed


def arg_printer(indent, arg):
    from src._tensor import _Tensor
    if isinstance(arg, _Tensor):
        print(" "*(indent*5), arg.shape)


printed_grad = _printed(type="back", arg_print=arg_printer)
# __all__ = [
# ]


class GradFn(ABC):
    def __init__(self, vars, result=None) -> None:
        self.result = result
        from src._tensor import _Tensor
        self.next_functions: List[Tuple[_Tensor, GradFn]] = []
        self._vars = vars
        for var in vars:
            value: GradFn | None = None
            if isinstance(var, _Tensor):
                # TODO if var.requires_grad and var.is_a_leaf() and var.grad_fn is None:
                if var.is_a_leaf() and var.grad_fn is None:
                    value = AccumulateGrad(var, result=var)
                else:
                    value = var.grad_fn
            self.next_functions.append((var, value))

    def calculate(self, gradient):
        with grad_off():
            return self._calculate(gradient)

    @abstractmethod
    def _calculate(self):
        pass

    def print_graph(self, indent=0, remove_acc=False, print_val=False):
        from src._tensor import _Tensor
        if indent == 0:
            print(type(self).__name__)
        indent += 1
        for k, v in self.next_functions:
            if not isinstance(k, _Tensor) or (remove_acc and type(v) is AccumulateGrad):
                continue
            val = ""
            if print_val:
                import numpy as np
                val = f"val={np.array(k)},"
            print(" "*indent*2+"↳", type(v).__name__,
                  val,
                  f"requires_grad={k.requires_grad}")
            v.print_graph(indent+1, remove_acc, print_val)


class AccumulateGrad(GradFn):

    def __init__(self, x: _Tensor, **kwargs) -> None:
        super().__init__([], **kwargs)
        self.x = x

    @printed_grad
    def _calculate(self, gradient):
        from src._tensor import tensor, _Tensor
        if not isinstance(gradient, _Tensor):
            # TODO: had to
            gradient = tensor(gradient, requires_grad=False)

        grad = self.x.grad
        if grad is None:
            grad = tensor(0)
        gradient.requires_grad = False
        gradient.grad_fn = None
        self.x.set_grad(grad + gradient)


class BinaryOpGradFn(GradFn, ABC):
    @abstractmethod
    def _calculate(self):
        assert len(
            self.next_functions) == 2, f"found: {len(self.next_functions)}"

    # @abstractmethod
    # def get_op(self): raise Exception("Not implemented")


class AddGradFn(BinaryOpGradFn):
    def get_op(self): return "+"

    @printed_grad
    def _calculate(self, gradient):
        super()._calculate()
        for v, grad_fn in self.next_functions:
            from src._tensor import _Tensor
            if isinstance(gradient, _Tensor) and v.shape != gradient.shape:
                summ = []
                for i in range(0, len(gradient.shape)):
                    dim1 = v.shape[len(v.shape) - i - 1]
                    dim2 = gradient.shape[len(gradient.shape) - i-1]
                    if len(v.shape) - i - 1 < 0 or dim1 != dim2:
                        # NOTE: when dim1 = 1
                        summ.append(len(gradient.shape) - i-1)
                grad = gradient
                if len(summ):
                    grad = gradient.sum(axis=tuple(summ))
                up(grad_fn, 1, grad)
            else:
                up(grad_fn, 1, gradient)


class SubGradFn(BinaryOpGradFn):
    def get_op(self): return "-"

    @printed_grad
    def _calculate(self, gradient):
        super()._calculate()
        up(self.next_functions[0][1], 1, gradient)
        up(self.next_functions[1][1], -1, gradient)


class MulGradFn(BinaryOpGradFn):
    def get_op(self): return "*"

    @printed_grad
    def _calculate(self, gradient):
        super()._calculate()

        (x, v0), (y, v1) = self.next_functions

        up(v0, y, gradient)
        up(v1, x, gradient)


class PowGradFn(BinaryOpGradFn):
    def get_op(self): return "**"

    @printed_grad
    def _calculate(self, gradient):
        super()._calculate()
        (x, v0), (y, v1) = self.next_functions
        up(v0, y * (x ** (y-1)), gradient)
        up(v1, (x ** y)*x.log(), gradient)


def up(v, k, gradient, matmul=False):
    if v:
        res = k*gradient if not matmul else k@gradient
        v._calculate(res)


class DivGradFn(BinaryOpGradFn):
    def get_op(self): return "/"

    @printed_grad
    def _calculate(self, gradient):
        super()._calculate()

        (x, v0), (y, v1) = self.next_functions
        up(v0, 1/y, gradient)
        up(v1, -1*(x/(y**2)), gradient)


class MatMulGradFn(BinaryOpGradFn):
    def get_op(self): return "@"

    @printed_grad
    def _calculate(self, gradient):
        # TODO: Broken
        super()._calculate()
        # x @ y
        (x, v0), (y, v1) = self.next_functions
        up(v0, gradient, y.T, True)
        up(v1, x.T, gradient, True)

# x = x+0


class IdentityGradFn(AddGradFn):
    pass


class OneOperatorOpGradFn(GradFn, ABC):
    @abstractmethod
    def _calculate(self):
        super()._calculate()
        assert len(self.next_functions) == 1, f"{len(self.next_functions)}"


class ExpGradFn(OneOperatorOpGradFn):
    @printed_grad
    def _calculate(self, gradient):
        super()._calculate()
        k, v = self.next_functions[0]
        up(v, k.exp(), gradient)


def acc(grad, new):
    return grad * new


class LogGradFn(OneOperatorOpGradFn):
    @printed_grad
    def _calculate(self, gradient):
        super()._calculate()
        k, v = self.next_functions[0]
        up(v, 1/k, gradient)


class MeanGradFn(OneOperatorOpGradFn):
    def __init__(self, vars, axis=None, **kwargs) -> None:
        super().__init__(vars, **kwargs)
        self.axis = axis

    @printed_grad
    def _calculate(self, gradient):
        from src._tensor import _Tensor
        super()._calculate()
        k, v = self.next_functions[0]
        if (self.axis is not None) and (gradient.shape != k.shape):
            gradient_shape = list(k.shape)
            gradient_shape[self.axis] = 1
            gradient = gradient.reshape(*gradient_shape)
        from src._tensor import tensor_ns_like
        up(v, tensor_ns_like(k, 1/k.size), gradient)


class SumGradFn(OneOperatorOpGradFn):
    def __init__(self, vars, axis=None, keepdims=False, **kwargs) -> None:
        super().__init__(vars, **kwargs)

        self.vvv = vars
        self.axis = axis

    @printed_grad
    def _calculate(self, gradient):
        from src._tensor import tensor_ones
        super()._calculate()
        k, v = self.next_functions[0]
        if (self.axis is not None) and (gradient.shape != () and gradient.shape != k.shape):
            gradient_shape = list(k.shape)
            gradient_shape[self.axis] = 1
            gradient = gradient.reshape(*gradient_shape)
        up(v, tensor_ones(k.shape), gradient)


class SoftmaxGradFn(OneOperatorOpGradFn):
    def __init__(self, vars, result=None, dim=None) -> None:
        super().__init__(vars, result)
        self.axis = dim

    @printed_grad
    def _calculate(self, gradient):
        k, v = self.next_functions[0]
        up(v, 0, 0)


class CrossEntropyGradFn(BinaryOpGradFn):

    def __init__(self, vars, result=None, dim=None, reduction="", from_logits=False) -> None:
        super().__init__(vars, result)

    @printed_grad
    def _calculate(self, gradient):
        (x, v0), (y, v1) = self.next_functions
        y: _Tensor = y
        n, *_ = x.shape
        dx = x.softmax(dim=1)
        dx.data[list(range(n)), y.astype(int)] -= 1
        dx /= n
        up(v0, dx, gradient)

    def get_op(self): return " CE "

# TODO MaximumGradFn
# TODO ReLUGradFn
# TODO LogsoftmaxGradFn
# TODO NLLLossGradFn


class NLLGradFn(OneOperatorOpGradFn):
    def __init__(self, vars, result=None, target=None, **kwargs) -> None:
        super().__init__(vars, result)
        self.target = target

    @printed_grad
    def _calculate(self, gradient):
        k, v = self.next_functions[0]
        up(v, 0, 0)
