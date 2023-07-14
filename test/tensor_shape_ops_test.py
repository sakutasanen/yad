import yad
import torch

def test_transpose():
    m_data = [[1.0, 2.0], [3.0,  4.0]]

    m = yad.Tensor(m_data, requires_grad=True)
    o = (m**2).T

    o.backward()

    m_t = torch.tensor(m_data, requires_grad=True)
    o_t = (m_t**2).T

    o_t.backward(gradient=torch.tensor([[1.0, 1.0], [1.0, 1.0]]))

    assert (o.data == o_t.detach().numpy()).all()
    assert (m.grad == m_t.grad.numpy()).all()