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
    def __init__(self, vars, result: _Tensor = None) -> None:
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
                # var.came_from = self.result.grad_fn
            self.next_functions.append((var, value))

    def calculate(self, gradient, print_ok=False):
        with grad_off():
            print_indent = -1
            if print_ok:
                _print_gradient(gradient, 0, fromm=self)
                print_indent = 1
            return self._calculate(gradient, print_indent=print_indent)

    @abstractmethod
    def _calculate(self, gradient):
        # print(self.__class__.__name__, "recieved gradient",
        #       gradient.shape, "result", self.result.shape)
        from src._tensor import _Tensor
        assert isinstance(gradient, _Tensor)
        self.result._accumulate_grad(gradient)

    def print_graph(self, indent=0, remove_acc=False, print_val=False):
        from src._tensor import _Tensor
        if indent == 0:
            print(type(self).__name__, self.result.grad)
        indent += 1
        for k, v in self.next_functions:
            if (remove_acc and type(v) is AccumulateGrad):
                continue
            if not isinstance(k, _Tensor):
                print(
                    " "*indent*2+"↳", "Constant",
                    k,
                )
                return
            val = ""
            if print_val:
                import numpy as np
                val = f"val={np.array(k)},"
            print(
                " "*indent*2+"↳", type(v).__name__,
                val,
                f"requires_grad={k.requires_grad},",
                f"name: {k.name}",
                # f'grad: {k.grad}',
            )
            v.print_graph(indent+1, remove_acc, print_val)


class AccumulateGrad(GradFn):

    def __init__(self, x: _Tensor, **kwargs) -> None:
        super().__init__([], **kwargs)
        self.x = x

    @printed_grad
    def _calculate(self, gradient, print_indent=-1):
        from src._tensor import tensor, _Tensor
        if not isinstance(gradient, _Tensor):
            # TODO: had to
            gradient = tensor(gradient, requires_grad=False)

        self.x._accumulate_grad(gradient)
        if print_indent >= 0:
            _print_gradient(self.x.grad, print_indent, self)


class BinaryOpGradFn(GradFn, ABC):
    @abstractmethod
    def _calculate(self, gradient):
        super()._calculate(gradient)
        assert len(
            self.next_functions) == 2, f"found: {len(self.next_functions)}"

    # @abstractmethod
    # def get_op(self): raise Exception("Not implemented")


def un_broadcast(v, gradient):
    if len(v.shape) == 0:
        return gradient.sum()
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
    return grad


class AddGradFn(BinaryOpGradFn):
    def get_op(self): return "+"

    @printed_grad
    def _calculate(self, gradient, print_indent=-1):
        super()._calculate(gradient)
        for v, grad_fn in self.next_functions:
            from src._tensor import _Tensor
            if isinstance(gradient, _Tensor) and isinstance(v, _Tensor) and v.shape != gradient.shape:
                un_broadcast(v, gradient)
            else:
                up([v, grad_fn], 1, gradient, print_indent=print_indent)


class SubGradFn(BinaryOpGradFn):
    def get_op(self): return "-"

    @printed_grad
    def _calculate(self, gradient, print_indent=-1):
        super()._calculate(gradient)
        up(self.next_functions[0], 1, gradient, print_indent=print_indent)
        up(self.next_functions[1], -1, gradient, print_indent=print_indent)


class MulGradFn(BinaryOpGradFn):
    def get_op(self): return "*"

    @printed_grad
    def _calculate(self, gradient, print_indent=-1):
        super()._calculate(gradient)

        (x, v0), (y, v1) = self.next_functions

        up(self.next_functions[0], y, gradient, print_indent=print_indent)
        up(self.next_functions[1], x, gradient, print_indent=print_indent)


class PowGradFn(BinaryOpGradFn):
    def get_op(self): return "**"

    @printed_grad
    def _calculate(self, gradient, print_indent=-1):
        super()._calculate(gradient)
        (x, v0), (y, v1) = self.next_functions
        up(self.next_functions[0], y * (x ** (y-1)),
           gradient, print_indent=print_indent)
        up(self.next_functions[1], (x ** y)*x.log(),
           gradient, print_indent=print_indent)


def _print_gradient(gradient, indent, fromm):
    gradient = '\t'*(indent+1) + \
        str(gradient).replace('\n', '\n'+'\t'*(indent+1))
    print(" "*indent*5, fromm.__class__.__name__, "grad\n", gradient)


def up(next: GradFn, mul, gradient, matmul=False, print_indent=-1):
    var, grad_fn = next
    if grad_fn:
        res = mul*gradient if not matmul else mul@gradient
        if print_indent >= 0:
            _print_gradient(res, print_indent, grad_fn)
            print_indent += 1
        # var._accumulate_grad(res)
        grad_fn._calculate(res, print_indent=print_indent)


class DivGradFn(BinaryOpGradFn):
    def get_op(self): return "/"

    @printed_grad
    def _calculate(self, gradient, print_indent=-1):
        super()._calculate(gradient)

        (x, v0), (y, v1) = self.next_functions
        # TODO broadcasting

        gradient2 = -1*(x/(y**2))*gradient
        from src._tensor import _Tensor
        if isinstance(gradient2, _Tensor) and isinstance(y, _Tensor) and y.shape != gradient2.shape:
            gradient2 = un_broadcast(y, gradient2)

        up(self.next_functions[0], 1/y, gradient, print_indent=print_indent)
        up(self.next_functions[1], 1, gradient2, print_indent=print_indent)


class MatMulGradFn(BinaryOpGradFn):
    def get_op(self): return "@"

    @printed_grad
    def _calculate(self, gradient, print_indent=-1):
        # TODO: Broken
        super()._calculate(gradient)
        # x @ y
        (x, v0), (y, v1) = self.next_functions
        up(self.next_functions[0], gradient,
           y.T, True, print_indent=print_indent)
        up(self.next_functions[1], x.T, gradient,
           True, print_indent=print_indent)

# x = x+0


class IdentityGradFn(AddGradFn):
    pass


class UnaryOpGradFn(GradFn, ABC):
    @abstractmethod
    def _calculate(self, gradient):
        super()._calculate(gradient)
        assert len(self.next_functions) == 1, f"{len(self.next_functions)}"


class ExpGradFn(UnaryOpGradFn):
    @printed_grad
    def _calculate(self, gradient, print_indent=-1):
        super()._calculate(gradient)
        k, v = self.next_functions[0]
        up(self.next_functions[0], k.exp(),
           gradient, print_indent=print_indent)


def acc(grad, new):
    return grad * new


class LogGradFn(UnaryOpGradFn):
    @printed_grad
    def _calculate(self, gradient, print_indent=-1):
        super()._calculate(gradient)
        k, v = self.next_functions[0]
        up(self.next_functions[0], 1/k, gradient, print_indent=print_indent)


class MeanGradFn(UnaryOpGradFn):
    def __init__(self, vars, axis=None, **kwargs) -> None:
        super().__init__(vars, **kwargs)
        self.axis = axis

    @printed_grad
    def _calculate(self, gradient, print_indent=-1):
        super()._calculate(gradient)
        k, v = self.next_functions[0]
        # TODO: Broken
        if (self.axis is not None) and (gradient.shape != () and gradient.shape != k.shape):
            gradient_shape = list(k.shape)
            gradient_shape[self.axis] = 1
            gradient = gradient.reshape(*gradient_shape)
        from src._tensor import tensor_ns_like
        up(self.next_functions[0], tensor_ns_like(
            k, 1/k.size), gradient, print_indent=print_indent)


class SumGradFn(UnaryOpGradFn):
    def __init__(self, vars, axis=None, keepdim=False, **kwargs) -> None:
        super().__init__(vars, **kwargs)

        self.vvv = vars
        self.axis = axis
        self.keepdim = keepdim

    @printed_grad
    def _calculate(self, gradient, print_indent=-1):
        from src._tensor import tensor_ones
        k, v = self.next_functions[0]
        if (self.axis is not None) and (gradient.shape != () and gradient.shape != k.shape):
            gradient_shape = list(k.shape)
            gradient_shape[self.axis] = 1
            gradient = gradient.reshape(*gradient_shape)
        super()._calculate(gradient)
        up(self.next_functions[0], tensor_ones(k.shape),
           gradient, print_indent=print_indent)


class SoftmaxGradFn(UnaryOpGradFn):
    def __init__(self, vars, result=None, dim=None) -> None:
        super().__init__(vars, result)
        self.axis = dim

    @printed_grad
    def _calculate(self, gradient, print_indent=-1):
        super()._calculate(gradient)
        k, v = self.next_functions[0]
        up(self.next_functions[0], 0, 0, print_indent=print_indent)


class CrossEntropyGradFn(BinaryOpGradFn):

    def __init__(self, vars, result=None, dim=None, reduction="", from_logits=False) -> None:
        super().__init__(vars, result)

    @printed_grad
    def _calculate(self, gradient, print_indent=-1):
        super()._calculate(gradient)
        (x, v0), (y, v1) = self.next_functions
        y: _Tensor = y
        n, *_ = x.shape
        dx = x.softmax(dim=1)
        dx.data[list(range(n)), y.astype(int)] -= 1
        dx /= n
        up(self.next_functions[0], dx, gradient, print_indent=print_indent)

    def get_op(self): return " CE "

# TODO MaximumGradFn
# TODO ReLUGradFn


class ReLUGradFn(UnaryOpGradFn):
    def __init__(self, vars, result=None, **kwargs) -> None:
        super().__init__(vars, result)

    @printed_grad
    def _calculate(self, gradient, print_indent=-1):
        super()._calculate(gradient)
        k, v = self.next_functions[0]
        import numpy as np
        gradient.data[np.where(k.relu().data == 0)] = 0
        up(self.next_functions[0], 1, gradient, print_indent=print_indent)

# TODO LogsoftmaxGradFn
# TODO NLLLossGradFn


class ReshapeGradFn(UnaryOpGradFn):
    def __init__(self, vars, result=None, **kwargs) -> None:
        super().__init__(vars, result)

    @printed_grad
    def _calculate(self, gradient, print_indent=-1):
        super()._calculate(gradient)
        from src._tensor import tensor_ones_like
        k, v = self.next_functions[0]
        gradient = gradient.reshape(k.shape)
        up(self.next_functions[0], 1, gradient, print_indent=print_indent)


FlattenGradFn = ReshapeGradFn


class NLLGradFn(BinaryOpGradFn):
    def __init__(self, vars, result=None, target=None, **kwargs) -> None:
        super().__init__(vars, result)
        self.result = result

    @printed_grad
    def _calculate(self, gradient, print_indent=-1):
        super()._calculate(gradient)
        # TODO
        [x, v1], [t, v2] = self.next_functions

        from src._tensor import tensor_zeros
        len_ = len(t)
        y = tensor_zeros((x.shape))
        y.data[list(range(len_)), t.numpy().astype(int)] = -(1/len_)
        up(self.next_functions[0], y, gradient, print_indent=print_indent)
