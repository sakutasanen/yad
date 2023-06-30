import numpy as np
from typing import Tuple
from .gradient_function import GradientFunction

# TODO: Only Tensors of floating point and complex dtype can require gradients

class Tensor:
    data:          np.array
    shape:         Tuple[int]
    requires_grad: bool
    grad:          np.array
    grad_fn:       GradientFunction

    def __init__(self, data, requires_grad=False): # TODO: dtype?
        if isinstance(data, (int, float)):
            self.data  = data
            self.shape = 1 # TODO
        else:
            self.data  = np.array(data)
            self.shape = self.data.shape

        self.requires_grad = requires_grad
        self.grad          = 0 if self.requires_grad else None # TODO: It would be nice if initial value is None
        self.grad_fn       = None
    
    def __add__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)

        out = Tensor(self.data + other.data, requires_grad=self.requires_grad or other.requires_grad)

        def _grad_fn():
            if self.requires_grad:
                self.grad += out.grad
            
            if other.requires_grad:
                other.grad += out.grad
        
        if out.requires_grad:
            out.grad_fn = GradientFunction(_grad_fn, (self.grad_fn, other.grad_fn))

        return out
    
    def __radd__(self, other):
        return self + other
    
    def __neg__(self):
        return self * -1

    def __sub__(self, other):
        return self + (-other)

    def __rsub__(self, other):
        return other + (-self)
    
    def __mul__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)

        out = Tensor(self.data * other.data, requires_grad=self.requires_grad or other.requires_grad)

        def _grad_fn():
            if self.requires_grad:
                self.grad += other.data * out.grad

            if other.requires_grad:
                other.grad += self.data * out.grad
        
        if out.requires_grad:
            out.grad_fn = GradientFunction(_grad_fn, (self.grad_fn, other.grad_fn))

        return out

    def __rmul__(self, other):
        return self * other

    def __matmul__(self, other):
        assert isinstance(other, Tensor), "Operands of matrix multiplication must be Tensor objects"

        len_shape       = len(self.shape)
        len_other_shape = len(other.shape)

        assert len_shape <= 2 or len_other_shape <= 2, "Maximum supported dimensions of the matrix multiplication operands is 2"

        out = Tensor(self.data @ other.data, requires_grad=self.requires_grad or other.requires_grad)

        def _grad_fn():
            if self.requires_grad:
                self.grad = out.grad @ other.data.T

            if other.requires_grad:
                other.grad = self.data.T @ out.grad

        if out.requires_grad:
            out.grad_fn = GradientFunction(_grad_fn, (self.grad_fn, other.grad_fn))

        return out

    # TODO: Implement div

    def __pow__(self, exponent):
        assert isinstance(exponent, (int, float)), "Only int/float powers are supported"

        out = Tensor(self.data**exponent, requires_grad=self.requires_grad)

        def _grad_fn():
            self.grad += exponent * self.data**(exponent-1) * out.grad
        
        if out.requires_grad:
            out.grad_fn = GradientFunction(_grad_fn, (self.grad_fn,))

        return out
    
    def __repr__(self):
        return f'Tensor(data={self.data}, grad={self.grad}, requires_grad={self.requires_grad})'

    def __getitem__(self, idx):
        return self.data[idx]

    # TODO: Implement equality comparison

    def sum(self, axis=0):
        out = Tensor(np.sum(self.data, axis=axis), requires_grad=self.requires_grad)

        def _grad_fn():
            self.grad += out.grad
        
        if out.requires_grad:
            out.grad_fn = GradientFunction(_grad_fn, next_functions=(self.grad_fn,))

        return out

    def backward(self):
        # Topological order all of the children in the graph
        topo    = []
        visited = set()

        def build_topo(grad_fn):
            if grad_fn not in visited:
                visited.add(grad_fn)

                if grad_fn is None:
                    return

                for next_grad_fn in grad_fn.next_functions:
                    build_topo(next_grad_fn)

                topo.append(grad_fn)

        build_topo(self.grad_fn)

        # Go one variable at a time and apply the chain rule to get its gradient
        self.grad = np.ones_like(self.data)

        # TODO: Assign gradients to "zero" here?

        for grad_fn in reversed(topo):
            grad_fn()
