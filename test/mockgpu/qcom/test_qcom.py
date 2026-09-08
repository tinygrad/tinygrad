import math, unittest
from tinygrad import Tensor
from tinygrad.helpers import DEV

@unittest.skipUnless(DEV.target("QCOM").interface == "MOCK", "requires DEV=MOCK+QCOM:IR3")
class TestQCOMEmulator(unittest.TestCase):
  def test_fill_across_dispatch_layouts(self):
    for size in (1, 3, 17, 65, 257):
      self.assertEqual(Tensor.full((size,), 1.0, device="QCOM").realize().tolist(), [1.0] * size)

  def test_float_add(self):
    a = Tensor([1.0, 2.0, 3.0], device="QCOM")
    b = Tensor([4.0, 5.0, 6.0], device="QCOM")
    self.assertEqual((a + b).realize().tolist(), [5.0, 7.0, 9.0])

  def test_integer_add(self):
    a = Tensor(list(range(17)), device="QCOM")
    self.assertEqual((a + 1).realize().tolist(), list(range(1, 18)))

  def test_matrix_multiply(self):
    a = Tensor([[1., 2., 3.], [4., 5., 6.]], device="QCOM")
    b = Tensor([[7., 8.], [9., 10.], [11., 12.]], device="QCOM")
    self.assertEqual((a @ b).realize().tolist(), [[58., 64.], [139., 154.]])

  def test_tiled_matrix_multiply(self):
    n = 64
    out = Tensor.ones(n, n, device="QCOM").contiguous() @ Tensor.eye(n).to("QCOM").clone()
    self.assertEqual(out.realize().tolist(), [[1.] * n for _ in range(n)])

  def test_shared_memory_reduction(self):
    n = 64
    out = Tensor.ones(1, n, device="QCOM").contiguous() @ Tensor.eye(n).to("QCOM").clone()
    self.assertEqual(out.realize().tolist(), [[1.] * n])

  def test_sfu_and_select(self):
    out = Tensor([[-3.0, 2.0], [3.0, 4.0]], device="QCOM").sum(axis=1).elu().realize()
    self.assertAlmostEqual(out.tolist()[0], math.expm1(-1.0), places=5)
    self.assertEqual(out.tolist()[1], 7.0)

if __name__ == "__main__": unittest.main()
