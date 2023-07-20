import yad
import numpy as np

def test_multiplication():
    x1 = yad.Tensor([4], requires_grad=True)
    x2 = yad.Tensor([1], requires_grad=True)

    y = x1 * x2

    y.backward()

    assert y[0]    == 4
    assert x1.grad == 1
    assert x2.grad == 4

def test_division():
    x = yad.Tensor([1], requires_grad=True)

    y = x / 5

    y.backward()

    assert y[0]   == 0.2
    assert x.grad == 0.2

def test_pow():
    x = yad.Tensor([2], requires_grad=True)

    y = x**2

    y.backward()

    assert y[0]   == 4
    assert x.grad == 4

def test_scalar_expression():
    x = yad.Tensor(2, requires_grad=True)

    y = (x**2 + 1)**2

    y.backward()

    assert y.data == 25
    assert x.grad == 40

def test_vector_expression():
    x = yad.Tensor([2, 1, 1], requires_grad=False)
    b = yad.Tensor(3,         requires_grad=True)

    y = x**2 - b

    y.backward()

    assert x.grad is None
    assert np.all(y.data == np.array([1, -2, -2]))
    assert np.all(b.grad == np.array([-1, -1, -1]))

# TODO: Remove .data and numpy references