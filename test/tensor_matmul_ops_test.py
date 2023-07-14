import yad
import torch

def test_matrix_vector_mul():
    m_data = [[1.0, 2.0], [1.0, 2.0]]
    v_data = [[1.0], [1.0]]

    m = yad.Tensor(m_data)
    v = yad.Tensor(v_data, requires_grad=True)

    o = m @ v

    o.backward()

    m_t = torch.tensor(m_data)
    v_t = torch.tensor(v_data, requires_grad=True)

    o_t = m_t @ v_t

    o_t.backward(gradient=torch.tensor([[1.0], [1.0]]))

    assert (o.data == o_t.detach().numpy()).all()
    assert (v.grad == v_t.grad.numpy()).all()

def test_matrix_matrix_mul():
    m1_data = [[1.0, 2.0], [1.0, 2.0]]
    m2_data = [[1.0, 2.0], [1.0, 2.0]]

    m1 = yad.Tensor(m1_data, requires_grad=True)
    m2 = yad.Tensor(m2_data, requires_grad=True)

    o = m1 @ m2

    o.backward()

    m1_t = torch.tensor(m1_data, requires_grad=True)
    m2_t = torch.tensor(m2_data, requires_grad=True)

    o_t = m1_t @ m2_t

    o_t.backward(gradient=torch.tensor([[1.0, 1.0], [1.0, 1.0]]))

    assert (o.data  == o_t.detach().numpy()).all()
    assert (m1.grad == m1_t.grad.numpy()).all()
    assert (m2.grad == m2_t.grad.numpy()).all()