from .optimizer import Optimizer

class SGD(Optimizer):
    def __init__(self, parameters, lr, weight_decay=0):
        """
        Stochastic Gradient Descent (SGD) optimizer.

        Args:
            parameters: Model parameters to optimize.
            lr: Learning rate for the optimizer.
            weight_decay: L2 regularization weight decay factor.
        """
        super().__init__(parameters, lr)
        self.weight_decay = weight_decay

    def step(self):
        """
        Perform a single optimization step using SGD.
        """
        for p in self.parameters:
            if self.weight_decay:
                p.data -= self.lr * (p.grad + p.data * self.weight_decay)
            else:
                p.data -= self.lr * p.grad