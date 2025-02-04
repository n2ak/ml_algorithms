import contextlib
import numpy as np

_grad_stack = [True]


@contextlib.contextmanager
def grad_off():
    global _grad_stack
    _grad_stack.append(False)
    yield
    _grad_stack.pop()


def is_grad_off(): return not _grad_stack[-1]


def _compare(n_grads, a_grads, rtol, atol):
    """
    Params:
        n_grads: numerical gradients
        a_grads: analytical? gradients
    """
    assert len(n_grads) == len(a_grads) != 0
    for i, ng, ag in zip(range(len(n_grads)), n_grads, a_grads):
        assert ng.shape == ag.shape
        if not np.allclose(ng, ag, rtol=rtol, atol=atol):
            print("Nm", ng.flatten()[:10])
            print("An", ag.flatten()[:10])
            raise ValueError(f"Input at {i}")


def grad_check(func, *inputs, h=1e-6, rtol=1e-05, atol=1e-08):
    """compare the gradient given by the autograd to the numerical gradient"""
    with grad_off():
        grads = numerical_grads(func, inputs, h)
    res = func(*inputs)
    res.backward(np.ones(res.shape))
    _compare(grads, [t.gradient for t in filter(
        _tensor_and_requires_grad, inputs)], rtol, atol)


def numerical_grads(func, inputs, h=1e-6):
    """Numerically compute the gradient of the ouputs of a function wtt its parameters"""
    assert isinstance(inputs, tuple)
    grads = []
    # for every input
    for i in range(len(inputs)):
        if not _tensor_and_requires_grad(inputs[i]):
            continue
        it = np.nditer(inputs[i].data, ["multi_index"], ["readwrite"])
        g = np.zeros_like(inputs[i].data)

        def perform(func, index, h):
            inputs[i].data[index] += h
            res = func(*inputs)
            inputs[i].data[index] -= h
            return res
        # traverse the input element by element
        while not it.finished:
            index = it.multi_index
            # compute the limit (f(x+h)-f(x-h)) / (h - (-h))
            res1 = perform(func, index, h)
            res2 = perform(func, index, -h)
            g[index] = ((res1-res2).sum() / (2*h)).data
            assert np.isfinite(g[index])
            it.iternext()
        grads.append(g)
    return grads


def _tensor_and_requires_grad(var):
    from ._tensor import Tensor
    return isinstance(var, Tensor) and var.requires_grad


def pass_gradient(var, gradient):
    import numpy as np
    assert isinstance(
        gradient, np.ndarray), f"Expcected gradient to be np.ndarray but found {type(gradient)}"
    if _tensor_and_requires_grad(var):
        # gradient's shape must mach the tensor's
        # assert var.shape == gradient.shape, f"Expected gradient of shape {gradient.shape} to match the tensor's shape {var.shape}"
        var._backward(gradient)


def _set_backward_fn(res, backward, func):
    if not hasattr(backward, "_fn_name"):
        backward._fn_name = f"{func.__name__.capitalize().replace('_','')}Backward"
    res._backward = backward


def differentiable_function(n_grad_args=None):
    """
    decorator for a differentiable function that returns a result and a backward
    function
    """
    # assert isinstance(n_grad_args, int)
    def register(func):
        import functools

        @functools.wraps(func)
        def dec(*args, **kwargs):
            if is_grad_off():
                return func(*args, **kwargs)[0]
            with grad_off():
                res, backward = func(*args, **kwargs)
            args_grad = list(
                map(_tensor_and_requires_grad, args[:n_grad_args]))
            res.requires_grad = any(args_grad)
            # print(res.requires_grad, args_grad, backward)
            if res.requires_grad:
                _set_backward_fn(res, backward, func)
            return res
        return dec
    return register
