import functools
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
            print("Gradient mismatch")
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


def try_pass_gradient(var, gradient):
    import numpy as np
    assert isinstance(
        gradient, np.ndarray), f"Expcected gradient to be np.ndarray but found {type(gradient)}"
    if _tensor_and_requires_grad(var):
        var._backward(gradient)


def wrap_backward(old_backward, inputs):
    @functools.wraps(old_backward)
    def new_backward(gradient):
        n_inputs = len(inputs)
        new_gradients = old_backward(gradient)
        if n_inputs == 1:
            assert isinstance(new_gradients, np.ndarray), old_backward._fn_name
            new_gradients = (new_gradients,)
        assert isinstance(
            new_gradients, tuple), f"Expected output of {old_backward._fn_name} to be tuple of len={n_inputs} but found {type(new_gradients)}"
        assert len(new_gradients) == len(
            inputs), f"{old_backward._fn_name} doenst output the same number of" +\
            f"gradients as its inputs, expected {n_inputs} but found {len(new_gradients)}"
        for g, inp in zip(new_gradients, inputs):
            try_pass_gradient(inp, g)
    return new_backward


def capitalize(s: str):
    return s[0].capitalize() + s[1:]


def setup_backward_func(backward, name):
    if not hasattr(backward, "_fn_name"):
        setattr(backward, "_fn_name",
                f"{capitalize(name.replace('_', ''))}Backward")


def differentiable_function(n_grad_args=None):
    """
    decorator for a differentiable function that returns a result and a backward
    function
    """
    # assert isinstance(n_grad_args, int)
    def register(func):

        @functools.wraps(func)
        def dec(*args, **kwargs):
            if is_grad_off():
                return func(*args, **kwargs)[0]
            with grad_off():
                res, backward = func(*args, **kwargs)
            setup_backward_func(backward, func.__name__)
            backward = wrap_backward(backward, args[:n_grad_args])
            args_grad = list(
                map(_tensor_and_requires_grad, args[:n_grad_args]))
            from src import Tensor
            res: Tensor = res
            res.requires_grad = any(args_grad)
            # print(res.requires_grad, args_grad, backward)
            if res.requires_grad:
                res._set_backward_fn(backward)

            return res
        return dec
    return register
