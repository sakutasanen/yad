import yad
import torch
import numpy as np

def test_sum():
    m_data = [1.0, 2.0, 3.0, 4.0]

    m = yad.Tensor(m_data, requires_grad=True)
    o = m.sum()

    o.backward()

    m_t = torch.tensor(m_data, requires_grad=True)
    o_t = m_t.sum()

    o_t.backward()

    assert (o.data == o_t.detach().numpy()).all()
    assert (m.grad == m_t.grad.numpy()).all()

def test_log():
    m_data = [1.0, 2.0]

    m = yad.Tensor(m_data, requires_grad=True)
    o = m.log()

    o.backward()

    m_t = torch.tensor(m_data, requires_grad=True)
    o_t = m_t.log()

    o_t.backward(gradient=torch.tensor([1.0, 1.0]))

    assert np.allclose(o.data, o_t.detach().numpy())
    assert np.allclose(m.grad, m_t.grad.numpy())

def test_relu():
    m_data = [-1.0, 2.0]

    m = yad.Tensor(m_data, requires_grad=True)
    o = m.relu()

    o.backward()

    m_t = torch.tensor(m_data, requires_grad=True)
    o_t = m_t.relu()

    o_t.backward(gradient=torch.tensor([1.0, 1.0]))

    assert (o.data == o_t.detach().numpy()).all()
    assert (m.grad == m_t.grad.numpy()).all()

def test_sigmoid():
    m_data = [0.2, 0.7]

    m = yad.Tensor(m_data, requires_grad=True)
    o = m.sigmoid()

    o.backward()

    m_t = torch.tensor(m_data, requires_grad=True)
    o_t = m_t.sigmoid()

    o_t.backward(gradient=torch.tensor([1.0, 1.0]))

    assert np.allclose(o.data, o_t.detach().numpy())
    assert np.allclose(m.grad, m_t.grad.numpy())