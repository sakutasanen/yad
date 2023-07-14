import numpy as np
from ..tensor import Tensor
from .module import Module

class Linear(Module):
    def __init__(self, in_features, out_features, name=""):
        """
        Linear layer applies a linear transformation to the input: y = x W.T + b
        input x [*, in_features]
        learnable weights W [out_features, in_features]
        bias b [out_features]
        output y [*, out_features]

        Args:
            in_features: The number of input features.
            out_features: The number of output features.
            name: The name of the Linear layer (optional).
        """
        super().__init__()
        self.in_features  = in_features
        self.out_features = out_features
        self.name         = name

        # Initialize weight and bias with random values
        bound = self.in_features ** -0.5

        self.weight = Tensor(np.random.uniform(-bound, bound, (out_features, in_features)), requires_grad=True)
        self.bias   = Tensor(np.random.uniform(-bound, bound, (out_features, )),            requires_grad=True)

    def forward(self, x):
        """
        Perform a forward pass through the Linear layer.

        Args:
            inp: The input tensor.

        Returns:
            The result of the forward pass.
        """

        return (x @ self.weight.T + self.bias)