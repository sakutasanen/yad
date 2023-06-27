import numpy as np

class Tensor:

    def __init__(self, data, requires_grad=False, ancestors=()):
        if isinstance(data, (int, float)):
            self.data  = data
            self.shape = 1
        else:
            self.data  = np.array(data)
            self.shape = self.data.shape

        self.requires_grad = requires_grad
        self.grad          = 0 if self.requires_grad else None
        self.ancestors     = set(ancestors)
        self._backward     = lambda: None
    
    def __add__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)

        out = Tensor(self.data + other.data, requires_grad=self.requires_grad or other.requires_grad, ancestors=(self, other))

        def _backward():
            if self.requires_grad:
                self.grad += out.grad
            
            if other.requires_grad:
                other.grad += out.grad
        
        out._backward = _backward

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

        out = Tensor(self.data * other.data, requires_grad=self.requires_grad or other.requires_grad, ancestors=(self, other))

        def _backward():
            if self.requires_grad:
                self.grad  += other.data * out.grad

            if other.requires_grad:
                other.grad += self.data * out.grad
        
        if out.requires_grad:
            out._backward = _backward

        return out

    def __rmul__(self, other):
        return self * other

    def __pow__(self, exponent):
        assert isinstance(exponent, (int, float)), "Only int/float powers are supported"

        out = Tensor(self.data**exponent, requires_grad=self.requires_grad, ancestors=(self,))

        def _backward():
            self.grad += exponent * self.data**(exponent-1) * out.grad
        
        if out.requires_grad:
            out._backward = _backward

        return out
    
    def __repr__(self):
        return f'Tensor(data={self.data}, grad={self.grad}, requires_grad={self.requires_grad})'

    def __getitem__(self, idx):
        return self.data[idx]

    # TODO: Implement equality comparison

    def sum(self, axis=0):
        out = Tensor(np.sum(self.data, axis=axis), requires_grad=self.requires_grad, ancestors=(self,))

        def _backward():
            self.grad += out.grad
        
        if out.requires_grad:
            out._backward = _backward

        return out

    def backward(self):
        # Topological order all of the children in the graph
        topo    = []
        visited = set()

        def build_topo(v):
            if v not in visited:
                visited.add(v)

                for ancestor in v.ancestors:
                    build_topo(ancestor)

                topo.append(v)

        build_topo(self)

        # Go one variable at a time and apply the chain rule to get its gradient
        self.grad = np.ones_like(self.data)

        for v in reversed(topo):
            v._backward()
