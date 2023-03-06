import unittest
from tinygrad.tensor import Tensor

class TestExample(unittest.TestCase):
  def test_example_readme(self):
    x = Tensor.eye(3, requires_grad=True)
    y = Tensor([[2.0,0,-2.0]], requires_grad=True)
    z = y.matmul(x).sum()
    z.backward()

    print(x.grad.numpy())  # dz/dx
    print(y.grad.numpy())  # dz/dy

  def test_example_matmul(self):
    x = Tensor.eye(256, requires_grad=True)
    y = Tensor.eye(256, requires_grad=True)
    z = y.matmul(x).sum()
    z.backward()

    print(x.grad.numpy())  # dz/dx
    print(y.grad.numpy())  # dz/dy

if __name__ == '__main__':
  unittest.main()
