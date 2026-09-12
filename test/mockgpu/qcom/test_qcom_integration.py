import math, unittest
from typing import cast

from tinygrad import Tensor, dtypes
from tinygrad.helpers import DEV


@unittest.skipUnless(DEV.target("QCOM").interface == "MOCK", "requires DEV=MOCK+QCOM:IR3")
class TestQCOMA630Integration(unittest.TestCase):
  def test_dispatch_layouts(self):
    for size in (1, 3, 17, 65, 257):
      self.assertEqual(Tensor.full((size,), 1.0, device="QCOM").realize().tolist(), [1.0] * size)

  def test_matrix_multiply(self):
    a = Tensor([[1., 2., 3.], [4., 5., 6.]], device="QCOM")
    b = Tensor([[7., 8.], [9., 10.], [11., 12.]], device="QCOM")
    self.assertEqual((a @ b).realize().tolist(), [[58., 64.], [139., 154.]])

  def test_tiled_matrix_multiply_float_and_half(self):
    n = 64
    for dtype in (dtypes.float, dtypes.half):
      lhs = Tensor.ones(n, n, dtype=dtype, device="QCOM").contiguous()
      rhs = Tensor.eye(n, dtype=dtype).to("QCOM").clone()
      self.assertEqual((lhs @ rhs).realize().tolist(), [[1.] * n for _ in range(n)])

  def test_reduction_shaped_matmul(self):
    n = 64
    lhs = Tensor.ones(1, n, device="QCOM").contiguous()
    rhs = Tensor.eye(n).to("QCOM").clone()
    self.assertEqual((lhs @ rhs).realize().tolist(), [[1.] * n])

  def test_random_convolution_and_sfu_select(self):
    rnd = cast(list[float], Tensor.rand(64, device="QCOM").realize().tolist())
    self.assertEqual(len(rnd), 64)
    self.assertTrue(all(0.0 <= x < 1.0 for x in rnd))

    conv = Tensor.ones(1, 1, 8, 8, device="QCOM").conv2d(Tensor.ones(2, 1, 3, 3, device="QCOM")).realize()
    self.assertEqual(conv.tolist(), [[[[9.] * 6 for _ in range(6)] for _ in range(2)]])

    vals = cast(list[float], Tensor([[-3.0, 2.0], [3.0, 4.0]], device="QCOM").sum(axis=1).elu().realize().tolist())
    self.assertAlmostEqual(vals[0], math.expm1(-1.0), places=5)
    self.assertEqual(vals[1], 7.0)


if __name__ == "__main__": unittest.main()
