import numpy as np
import unittest
from tinygrad import Tensor, UOp
from tinygrad.uop.ops import AxisType

class TestFold(unittest.TestCase):
  def test_reduce_add(self):
    a = Tensor.randn(10, 10).realize()
    a_red = a.sum(axis=1)
    np.testing.assert_allclose(a_red.numpy(), a.numpy().sum(axis=1), atol=1e-6)

  def test_fold_add(self):
    a = Tensor.randn(10, 10).realize()
    init = Tensor.zeros(10, 1).contiguous()
    a_red = (init+a).fold(init).reshape(10)
    np.testing.assert_allclose(a_red.numpy(), a.numpy().sum(axis=1), atol=1e-6)

  #@unittest.skip("no outer fold yet")
  def test_fold_matmul(self):
    vec = Tensor.randn(1, 10).realize()
    mats = Tensor.randn(3, 10, 10).realize()
    np_mats = mats.numpy()
    np_ref = ((vec.numpy() @ np_mats[0]) @ np_mats[1]) @ np_mats[2]

    i = UOp.range(3, -1, AxisType.OUTER)
    out = (vec @ mats[i]).contiguous().fold(vec, i)

    np.testing.assert_allclose(out.numpy(), np_ref, atol=1e-6)

if __name__ == '__main__':
  unittest.main()
