from __future__ import annotations
import graphviz
from typing import TYPE_CHECKING, Callable
if TYPE_CHECKING:
    from src._tensor import _Tensor

import os
os.environ["PATH"] += os.pathsep + 'C:/Program Files/Graphviz/bin/'


def draw_graph(var: _Tensor, filename):

    graph = graphviz.Digraph(format="jpg",
                             graph_attr={'rankdir': "TB"})

    def node(var, name=None, label=None, shape="box"):
        from src._tensor import _Tensor
        is_tensor = isinstance(var, _Tensor)
        if not is_tensor:
            shape = "diamond"
        name = name or str(id(var))
        # if var and (not is_tensor):
        #     name += str(random.randint(0,999999))

        grad_shape = "None"
        if var is not None and var.grad is not None:
            grad_shape = var.grad.shape
        label = label or (
            f"shape: {var.shape}{f' | grad: {grad_shape}' if is_tensor else ''}")
        graph.node(name=name, label=label, shape=shape)

    def edge(f, t, tail_name=None, head_name=None, label=None, **kwargs):
        if f:
            node(f, **kwargs)
            tail_name = str(id(f))
            head_name = str(id(t))
        graph.edge(tail_name=tail_name, head_name=head_name, label=label)

    def binary_op(var: _Tensor):
        (next_var1, grad_fn1), (next_var2, grad_fn2) = var.grad_fn.next_functions

        op = var.grad_fn.get_op()
        operation = f"{id(op)}{next_var1}{next_var2}"

        node(None, name=operation, label=f"{op}", shape="ellipse")
        edge(None, None, tail_name=str(id(next_var1)), head_name=operation)
        edge(None, None, tail_name=str(id(next_var2)), head_name=operation)
        edge(None, None, tail_name=operation, head_name=str(id(var)))

        walk(next_var1)
        walk(next_var2)

    def one_op(var: _Tensor):
        next_var, grad_fn = var.grad_fn.next_functions[0]
        op = var.grad_fn.__class__.__name__
        edge(next_var, var, label=op)
        walk(next_var)

    def walk(var):
        node(var)
        from src._tensor import _Tensor
        if isinstance(var, _Tensor):
            if var.grad_fn is not None:
                if isinstance(var.grad_fn, UnaryOpGradFn):
                    one_op(var)
                if isinstance(var.grad_fn, BinaryOpGradFn):
                    binary_op(var)

    walk(var)
    return graph.render(filename=filename, cleanup=True)


def plot_graph(var, filename=None, delete_file=True, figsize=None):
    import matplotlib.pyplot as plt
    import random
    filename = filename or random.randint(0, 999999)
    filename = draw_graph(var, str(filename))
    im = plt.imread(filename)
    if delete_file:
        import os
        os.remove(filename)
    if figsize is not None:
        plt.figure(figsize=figsize)
    plt.imshow(im)
    plt.axis("off")
    plt.show()
