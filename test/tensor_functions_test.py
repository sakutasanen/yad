import yad
import torch

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
