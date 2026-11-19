from typing import Union
import numpy as np

class Tensor:
    def __init__(self, data: np.ndarray, _children=(), _op=""):
        self.data = data
        self.grad: Union[np.ndarray, None] = np.zeros_like(self.data) 
        self.backward_fn = lambda: None
        self._op = _op
        self._prev = set(_children)

    def __add__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        res = Tensor(self.data + other.data, _children=(self, other), _op="+")

        def _backward_fn():
            if res.grad is None:
                return
            
            self.grad += res.grad
            other.grad += res.grad

        res.backward_fn = _backward_fn
        return res

    def backward(self, grad=None):
        # use post-order DFS to build the topological order
        topo = []
        visited = set()

        def build_topo(node):
            if node not in visited:
                visited.add(node)
                for parent in node._prev:
                    build_topo(parent)
                topo.append(node)

        build_topo(self)

        # initialize grad at the output tensor
        if grad is None:
            if self.data.size != 1:
                 raise ValueError
            self.grad = np.ones_like(self.data, dtype=np.float64)
        else:
            self.grad = np.asarray(grad, dtype=np.float64).reshape(self.data.shape)

        # Traverse in reverse topological order
        for node in reversed(topo):
            node.backward_fn()


    def __repr__(self):
        return f"Tensor(data={self.data}, grad={self.grad})"


if __name__ == "__main__":
    pass