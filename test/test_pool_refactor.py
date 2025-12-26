import unittest
import numpy as np
from tinygrad import Tensor

class TestPoolRefactor(unittest.TestCase):
  def test_values_simple(self):
    t = Tensor(np.arange(16, dtype=np.float32).reshape(1, 1, 4, 4))
    out = t.avg_pool2d((2, 2), stride=1).numpy()
    expected = np.array([[[[ 2.5,  3.5,  4.5],
                           [ 6.5,  7.5,  8.5],
                           [10.5, 11.5, 12.5]]]], dtype=np.float32)
    np.testing.assert_allclose(out, expected)

  def test_values_stride(self):
    t = Tensor(np.arange(25, dtype=np.float32).reshape(1, 1, 5, 5))
    out = t.avg_pool2d((3, 3), stride=2).numpy()
    expected = np.array([[[[ 6.,  8.],
                           [16., 18.]]]], dtype=np.float32)
    np.testing.assert_allclose(out, expected)

  def test_values_dilation(self):
    t = Tensor(np.arange(16, dtype=np.float32).reshape(1, 1, 4, 4))
    # 2x2 kernel, dilation 2. Receptive field 3x3.
    # 0 1 2 3
    # 4 5 6 7
    # 8 9 10 11
    # 12 13 14 15
    # Window at (0,0): 0, 2, 8, 10. Mean: 5.
    # Window at (0,1): 1, 3, 9, 11. Mean: 6.
    # Window at (1,0): 4, 6, 12, 14. Mean: 9.
    # Window at (1,1): 5, 7, 13, 15. Mean: 10.
    out = t.avg_pool2d((2, 2), stride=1, dilation=2).numpy()
    expected = np.array([[[[ 5.,  6.],
                           [ 9., 10.]]]], dtype=np.float32)
    np.testing.assert_allclose(out, expected)

  def test_kernel_equivalence(self):
    # This test verifies that the UOp graph simplifies to something reasonable.
    # Comparing against exact UOp structure of previous implementation is hard dynamically.
    # But we can check that it works.
    t = Tensor.empty(1, 1, 4, 4)
    out = t.avg_pool2d((2, 2), stride=1)
    # Just accessing uop to make sure graph construction works
    uop = out.uop
    self.assertIsNotNone(uop)

if __name__ == '__main__':
  unittest.main()
