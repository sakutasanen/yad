class Optimizer:
    def __init__(self, parameters, lr):
        """
        Base class for optimization algorithms.

        Args:
            parameters: Model parameters to optimize.
            lr: Learning rate for the optimizer.
        """
        self.parameters = list(parameters)
        self.lr         = lr

    def step(self):
        """
        Perform a single optimization step.
        This method should be implemented by subclasses.
        """
        raise NotImplementedError

    def zero_grad(self):
        """
        Set gradients of all parameters to zero.
        """
        for p in self.parameters:
            p.grad *= 0.0