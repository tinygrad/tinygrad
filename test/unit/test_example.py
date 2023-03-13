import unittest
from tinygrad.tensor import Tensor

class TestExample(unittest.TestCase):
  def _test_example_readme(self, device):
    x = Tensor.eye(3, device=device, requires_grad=True)
    y = Tensor([[2.0,0,-2.0]], device=device, requires_grad=True)
    z = y.matmul(x).sum()
    z.backward()

    print(x.grad.numpy())  # dz/dx
    print(y.grad.numpy())  # dz/dy

    assert x.grad.device == device
    assert y.grad.device == device

  def _test_example_matmul(self, device):
    x = Tensor.eye(64, device=device, requires_grad=True)
    y = Tensor.eye(64, device=device, requires_grad=True)
    z = y.matmul(x).sum()
    z.backward()

    print(x.grad.numpy())  # dz/dx
    print(y.grad.numpy())  # dz/dy

    assert x.grad.device == device
    assert y.grad.device == device

  def test_example_readme_cpu(self): self._test_example_readme("CPU")
  def test_example_readme_gpu(self): self._test_example_readme("GPU")
  def test_example_readme_torch(self): self._test_example_readme("TORCH")
  def test_example_readme_llvm(self): self._test_example_readme("LLVM")

  def test_example_matmul_cpu(self): self._test_example_matmul("CPU")
  def test_example_matmul_gpu(self): self._test_example_matmul("GPU")
  def test_example_matmul_torch(self): self._test_example_matmul("TORCH")
  def test_example_matmul_llvm(self): self._test_example_matmul("LLVM")

if __name__ == '__main__':
  unittest.main()
