from src._base import _HasForwardAndIsCallable, _Trainable
from src._tensor import tensor, _Tensor


class Module(_HasForwardAndIsCallable):
    # TODO: Module
    def __init__(self) -> None:
        self.params = None

    def zero_grad(self):
        params = self.get_parameters()
        for p in params:
            p.zero_grad()

    def get_parameters(self) -> list[_Tensor]:
        if self.params:
            return self.params
        import inspect
        p = []
        for k, v in inspect.getmembers(self):
            if isinstance(v, _Trainable):
                v: _Trainable = v
                p.extend(v.get_trainable_params())
        self.params = p
        return self.params
