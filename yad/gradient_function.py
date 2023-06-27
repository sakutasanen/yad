class GradientFunction:

    def __init__(self, function, next_functions=()):
        self.function       = function
        self.next_functions = set(next_functions)

    def __call__(self):
        self.function()