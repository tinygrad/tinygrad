from tinygrad.tensor import Tensor
from extra.training import focal_loss
import unittest
import torch
from torchvision.ops import sigmoid_focal_loss as torch_sigmoid_focal_loss
import numpy as np
import random

class TestFocalLoss(unittest.TestCase):
  def test_simple(self):
    a = np.array([2, 0, 0.2], dtype=np.float32)
    b = np.array([1, 0, 0], dtype=np.int32)
    self.assertAlmostEqual(self._torch_sigmoid_focal_loss(a, b), self._tinygrad_sigmoid_focal_loss(a, b), places=5)

  def test_2d(self):
    a = np.random.standard_normal((10, 10)).astype(np.float32)
    b = self._random_targets_np_array((10, 10))
    self.assertAlmostEqual(self._torch_sigmoid_focal_loss(a, b), self._tinygrad_sigmoid_focal_loss(a, b), places=5)

  def test_3d(self):
    a = np.random.standard_normal((5, 10, 10)).astype(np.float32)
    b = self._random_targets_np_array((5, 10, 10))
    self.assertAlmostEqual(self._torch_sigmoid_focal_loss(a, b), self._tinygrad_sigmoid_focal_loss(a, b), places=5)

  def test_gamma(self):
    a = np.random.standard_normal((5, 10, 10)).astype(np.float32)
    b = self._random_targets_np_array((5, 10, 10))
    for gamma in [1, 2, 3, 4]:
      with self.subTest(gamma=gamma):
        self.assertAlmostEqual(self._torch_sigmoid_focal_loss(a, b, gamma=gamma), self._tinygrad_sigmoid_focal_loss(a, b, gamma=gamma), places=5)

  def test_alpha(self):
    a = np.random.standard_normal((10, 10)).astype(np.float32)
    b = np.zeros((10, 10), dtype=np.int32)
    for row in range(10): b[row, random.randint(0,9)] = 1
    for alpha in [0, 0.25, 0.5]:
      with self.subTest(alpha=alpha):
        self.assertAlmostEqual(self._torch_sigmoid_focal_loss(a, b, alpha=alpha), self._tinygrad_sigmoid_focal_loss(a, b, alpha=alpha), places=5)

  def _random_targets_np_array(self, shape) -> np.ndarray:
    ret = np.zeros(shape, dtype=np.int32)
    if len(shape) == 1:
      ret[np.random.randint(0, shape[0])] = 1
    else:
      rand_indices = np.expand_dims(np.random.randint(0, shape[-1], shape[:-1]), axis=-1)
      np.put_along_axis(ret, rand_indices, 1, axis=-1)
    return ret

  def _torch_sigmoid_focal_loss(self, a: np.ndarray, b: np.ndarray, alpha=0.1, gamma=1) -> float:
    return torch_sigmoid_focal_loss(torch.Tensor(a), torch.Tensor(b), alpha=alpha, gamma=gamma, reduction='mean').item()

  def _tinygrad_sigmoid_focal_loss(self, a: np.ndarray, b: np.ndarray, alpha=0.1, gamma=1) -> float:
    return float(focal_loss(Tensor(a).sigmoid(), Tensor(b), alpha=alpha, gamma=gamma).numpy())

if __name__ == '__main__':
  unittest.main()
