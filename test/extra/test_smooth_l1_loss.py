from tinygrad.tensor import Tensor
from extra.training import smooth_l1_loss
import unittest
import torch
from torch.nn.functional import smooth_l1_loss as torch_smooth_l1_loss
import numpy as np

class TestSmoothL1Loss(unittest.TestCase):
  def test_simple(self):
    a = np.array([2, 0, 0.2], dtype=np.float32)
    b = np.array([0.5, 0.1, 45.9], dtype=np.int32)
    self.assertAlmostEqual(self._torch_smooth_l1_loss(a, b), self._tinygrad_smooth_l1_loss(a, b), places=5)

  def test_2d(self):
    a = np.random.standard_normal((10, 10)).astype(np.float32)
    b = np.random.standard_normal((10, 10)).astype(np.float32)
    self.assertAlmostEqual(self._torch_smooth_l1_loss(a, b), self._tinygrad_smooth_l1_loss(a, b), places=5)

  def test_3d(self):
    a = np.random.standard_normal((5, 10, 10)).astype(np.float32)
    b = np.random.standard_normal((5, 10, 10)).astype(np.float32)
    self.assertAlmostEqual(self._torch_smooth_l1_loss(a, b), self._tinygrad_smooth_l1_loss(a, b), places=5)

  def test_beta(self):
    a = np.random.standard_normal((5, 10, 10)).astype(np.float32)
    b = np.random.standard_normal((5, 10, 10)).astype(np.float32)
    for beta in [1, 1.5, 3]:
      with self.subTest(beta=beta):
        self.assertAlmostEqual(self._torch_smooth_l1_loss(a, b, beta=beta), self._tinygrad_smooth_l1_loss(a, b, beta=beta), places=5)

  def test_reduction(self):
    a = np.random.standard_normal((5, 10, 10)).astype(np.float32)
    b = np.random.standard_normal((5, 10, 10)).astype(np.float32)
    for reduction in ['mean', 'sum']:
      with self.subTest(reduction=reduction):
        self.assertAlmostEqual(self._torch_smooth_l1_loss(a, b, reduction=reduction), self._tinygrad_smooth_l1_loss(a, b, reduction=reduction), places=3)

  def _torch_smooth_l1_loss(self, a: np.ndarray, b: np.ndarray, beta=1, reduction='mean') -> float:
    return torch_smooth_l1_loss(torch.Tensor(a), torch.Tensor(b), beta=beta, reduction=reduction).item()

  def _tinygrad_smooth_l1_loss(self, a: np.ndarray, b: np.ndarray, beta=1, reduction='mean') -> float:
    return float(smooth_l1_loss(Tensor(a), Tensor(b), beta=beta, reduction=reduction).numpy())

if __name__ == '__main__':
  unittest.main()
