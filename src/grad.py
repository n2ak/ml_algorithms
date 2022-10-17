from abc import ABC, abstractmethod
from typing import List



class AccumulateGrad():
    def __init__(self,tensor) -> None:
        self.tensor = tensor
    def calculate(self,gradient):
        from .tensor import Tensor 
        grad = self.tensor.grad or Tensor.array(0)
        self.tensor.set_grad(grad + gradient)
        
class GradFn(ABC):
    
    @abstractmethod
    def __init__(self,vars) -> None:
        super().__init__()
        self.next_functions = []
        from .tensor import Tensor
        for var in vars:
            value = None
            if isinstance(var,Tensor):
                if var.is_a_leaf() and var.grad_fn is None:
                    value = AccumulateGrad(var)
                else:
                    value = var.grad_fn
            self.next_functions.append((var,value))

    @abstractmethod
    def calculate(self,*args,**kwargs):
        pass
    
class BinaryOpGradFn(GradFn,ABC):
    @abstractmethod
    def __init__(self,vars) -> None:
        super().__init__(vars)
    @abstractmethod
    def calculate(self, *args, **kwargs):
        super().calculate(*args,**kwargs)
        assert len(self.next_functions) == 2, f"{len(self.next_functions)}"

class AddGradFn(BinaryOpGradFn):
    def __init__(self,vars) -> None:
        super().__init__(vars)
    
    def calculate(self, *args, **kwargs):
        super().calculate(*args,**kwargs)
        def h(v,k):
            if v : v.calculate(k)
        h(self.next_functions[0][1],1)
        h(self.next_functions[1][1],1)


class SubGradFn(BinaryOpGradFn):
    def __init__(self,vars) -> None:
        super().__init__(vars)
    
    def calculate(self, *args, **kwargs):
        super().calculate(*args,**kwargs)
        def h(v,k):
            if v : v.calculate(k)
        h(self.next_functions[0][1],1)
        h(self.next_functions[1][1],-1)

class MulGradFn(BinaryOpGradFn):
    def __init__(self,vars) -> None:
        super().__init__(vars)
    
    def calculate(self, *args, **kwargs):
        super().calculate(*args,**kwargs)
        def h(v,k):
            if v : v.calculate(k)

        k0,v0 = self.next_functions[0]
        k1,v1 = self.next_functions[1]

        h(v0,k1)
        h(v1,k0)

class IdentityGradFn(AddGradFn):
    def __init__(self,vars) -> None:
        print("vars",vars)
        super().__init__(vars)

class OneOperatorOpGradFn(GradFn,ABC):
    @abstractmethod
    def __init__(self,vars) -> None:
        super().__init__(vars)
    @abstractmethod
    def calculate(self, *args, **kwargs):
        super().calculate(*args,**kwargs)
        assert len(self.next_functions) == 1 , f"{len(self.next_functions)}"

class ExpGradFn(OneOperatorOpGradFn):
    def __init__(self,vars) -> None:
        super().__init__(vars)
    
    def calculate(self, *args, **kwargs):
        super().calculate(*args,**kwargs)
        k,v = self.next_functions[0]
        print(k,v)
        v.calculate(k.exp())
class MeanGradFn(OneOperatorOpGradFn):
    def __init__(self,vars) -> None:
        super().__init__(vars)
    
    def calculate(self, *args, **kwargs):
        super().calculate(*args,**kwargs)
        k,v = self.next_functions[0]
        from .tensor import Tensor
        v.calculate(Tensor.ns_like(k,1/k.size))
class SumGradFn(OneOperatorOpGradFn):
    def __init__(self,vars) -> None:
        super().__init__(vars)
    
    def calculate(self, *args, **kwargs):
        super().calculate(*args,**kwargs)
        k,v = self.next_functions[0]
        from .tensor import Tensor
        v.calculate(Tensor.ones_like(k))