import unittest
import numpy as np
import torch
import tinygrad.frontend.torch # noqa: F401

class TestDiagonalViews(unittest.TestCase):
  def _check(self, *shape, dtype=torch.float32):
    a = torch.randn(*shape, dtype=dtype)
    a_tiny = a.tiny()

    linalg_torch  = torch.linalg.diagonal(a)
    linalg_tiny = torch.linalg.diagonal(a_tiny)
    diagonal_tiny = torch.diagonal(a_tiny, dim1=-2, dim2=-1) # linalg.diagonal is alias for diagonal with dim1=-2, dim2=-1
    np.testing.assert_equal(linalg_tiny.cpu().numpy(), linalg_torch.numpy())
    np.testing.assert_equal(diagonal_tiny[-1].cpu().numpy(), linalg_tiny[-1].cpu().numpy()) 
    np.testing.assert_equal(linalg_tiny[-1].cpu().numpy(), linalg_torch[-1].numpy()) # row access is enough to trigger the bug

  def test_cube(self): self._check(3, 3, 3)
  def test_rectangular_last_dims(self): self._check(4, 5, 6)
  def test_high_dimensional(self): self._check(2, 3, 4, 5)

if __name__ == "__main__":
  unittest.main()