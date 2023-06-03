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

  def _torch_smooth_l1_loss(self, a: np.ndarray, b: np.ndarray) -> float:
    return torch_smooth_l1_loss(torch.Tensor(a), torch.Tensor(b), beta=1.0, reduction='mean').item()

  def _tinygrad_smooth_l1_loss(self, a: np.ndarray, b: np.ndarray) -> float:
    return float(smooth_l1_loss(Tensor(a), Tensor(b)).numpy())

if __name__ == '__main__':
  unittest.main()
