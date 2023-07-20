"""Functional interface"""

def log(input):
    return input.log()

def relu(input):
    return input.relu()

def sigmoid(input):
    return input.sigmoid()

def binary_cross_entropy(input, target):
    return -(target * input.log() + (1 - target) * (1 - input).log()).sum() / target.shape[0]