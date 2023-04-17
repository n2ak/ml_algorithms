from __future__ import annotations
from abc import ABC, abstractmethod
from typing import List, Tuple


class GradFn(ABC):
    @abstractmethod
    def __init__(self, vars) -> None:
        from src.tensor import Tensor
        self.next_functions: List[Tuple[Tensor, GradFn]] = []

        for var in vars:
            value: GradFn | None = None
            if isinstance(var, Tensor):
                if var.is_a_leaf() and var.grad_fn is None:
                    value = AccumulateGrad(var)
                else:
                    value = var.grad_fn
            self.next_functions.append((var, value))

    @abstractmethod
    def calculate(self):
        pass

    def print_graph(self, indent=0, remove_acc=False):
        from src import Tensor
        if indent == 0:
            print(type(self).__name__)
        indent += 1
        for k, v in self.next_functions:
            if not isinstance(k, Tensor) or (remove_acc and type(v) is AccumulateGrad):
                continue
            print(" "*indent*2+"↳", type(v).__name__,
                  f"requires_grad={k.requires_grad}")
            v.print_graph(indent+1, remove_acc)


class AccumulateGrad(GradFn):
    from src. tensor import Tensor

    def __init__(self, x: Tensor) -> None:
        super().__init__([])
        self.x = x

    def calculate(self, gradient):
        from src.tensor import tensor
        gradient = tensor(gradient, requires_grad=False)

        grad = self.x.grad
        if grad is None:
            grad = tensor(1)
        self.x.set_grad(grad * gradient)


class BinaryOpGradFn(GradFn, ABC):
    @abstractmethod
    def __init__(self, vars) -> None:
        super().__init__(vars)

    @abstractmethod
    def calculate(self):
        assert len(
            self.next_functions) == 2, f"found: {len(self.next_functions)}"


class AddGradFn(BinaryOpGradFn):
    def __init__(self, vars) -> None:
        super().__init__(vars)

    def calculate(self, gradient):
        super().calculate()
        up(self.next_functions[0][1], 1, gradient)
        up(self.next_functions[1][1], 1, gradient)


class SubGradFn(BinaryOpGradFn):
    def __init__(self, vars) -> None:
        super().__init__(vars)

    def calculate(self, gradient):
        super().calculate()
        up(self.next_functions[0][1], 1, gradient)
        up(self.next_functions[1][1], -1, gradient)


class MulGradFn(BinaryOpGradFn):
    def __init__(self, vars) -> None:
        super().__init__(vars)

    def calculate(self, gradient):
        super().calculate()

        x, v0 = self.next_functions[0]
        y, v1 = self.next_functions[1]

        up(v0, y, gradient)
        up(v1, x, gradient)


class PowGradFn(BinaryOpGradFn):
    def __init__(self, vars) -> None:
        super().__init__(vars)

    def calculate(self, gradient):
        super().calculate()
        x, v0 = self.next_functions[0]
        y, v1 = self.next_functions[1]
        up(v0, y * (x ** (y-1)), gradient)
        up(v1, (x ** y)*x.log(), gradient)


def up(v, k, gradient):
    if v:
        v.calculate(gradient*k)


class DivGradFn(BinaryOpGradFn):
    def __init__(self, vars) -> None:
        super().__init__(vars)

    def calculate(self, gradient):
        super().calculate()

        x, v0 = self.next_functions[0]
        y, v1 = self.next_functions[1]
        up(v0, 1/y, gradient)
        up(v1, -1*(x/(y**2)), gradient)


class MatMulGradFn(BinaryOpGradFn):
    def __init__(self, vars) -> None:
        super().__init__(vars)

    def calculate(self, gradient):
        super().calculate()
        # x @ y
        x, v0 = self.next_functions[0]
        y, v1 = self.next_functions[1]

        print("yo", x.shape, y.T.shape)
        gradient = 1
        up(v0, y.T, gradient)
        up(v1, x.T, gradient)

# x = x+0


class IdentityGradFn(AddGradFn):
    def __init__(self, vars) -> None:
        super().__init__(vars)


class OneOperatorOpGradFn(GradFn, ABC):
    @abstractmethod
    def __init__(self, vars) -> None:
        super().__init__(vars)

    @abstractmethod
    def calculate(self):
        super().calculate()
        assert len(self.next_functions) == 1, f"{len(self.next_functions)}"


class ExpGradFn(OneOperatorOpGradFn):
    def __init__(self, vars) -> None:
        super().__init__(vars)

    def calculate(self, gradient):
        super().calculate()
        k, v = self.next_functions[0]
        up(v, k.exp(), gradient)


def acc(grad, new):
    return grad * new


class LogGradFn(OneOperatorOpGradFn):
    def __init__(self, vars) -> None:
        super().__init__(vars)

    def calculate(self, gradient):
        super().calculate()
        k, v = self.next_functions[0]
        up(v, 1/k, gradient)


class MeanGradFn(OneOperatorOpGradFn):
    def __init__(self, vars, axis) -> None:
        super().__init__(vars)

        self.axis = axis

    def calculate(self, gradient):
        from src.tensor import Tensor
        super().calculate()
        k, v = self.next_functions[0]
        if (self.axis is not None) and (gradient.shape != k.shape):
            gradient_shape = list(k.shape)
            gradient_shape[self.axis] = 1
            gradient = gradient.reshape(*gradient_shape)
        up(v, Tensor.ns_like(k, 1/k.size), gradient)


class SumGradFn(OneOperatorOpGradFn):
    def __init__(self, vars, axis) -> None:
        super().__init__(vars)

        self.vvv = vars
        self.axis = axis

    def calculate(self, gradient):
        from src.tensor import Tensor
        super().calculate()
        k, v = self.next_functions[0]
        if (self.axis is not None) and (gradient.shape != k.shape):
            gradient_shape = list(k.shape)
            gradient_shape[self.axis] = 1
            gradient = gradient.reshape(*gradient_shape)
        up(v, Tensor.ones(k.shape), gradient)
