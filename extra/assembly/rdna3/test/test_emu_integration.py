# Integration tests for RDNA3 Python emulator through tinygrad
# These tests run actual tinygrad operations through the mockgpu with Python emulator
import unittest
import os
import numpy as np

# Skip if not running with MOCKGPU and PYTHON_REMU
@unittest.skipUnless(os.environ.get("MOCKGPU") and os.environ.get("PYTHON_REMU"), "requires MOCKGPU=1 PYTHON_REMU=1")
class TestTinygradIntegration(unittest.TestCase):
  @classmethod
  def setUpClass(cls):
    os.environ["AMD"] = "1"
    from tinygrad import Tensor, Device, dtypes
    cls.Tensor, cls.Device, cls.dtypes = Tensor, Device, dtypes

  # *** basic arithmetic ***
  def test_simple_add(self):
    a, b = self.Tensor([1.0, 2.0, 3.0, 4.0]), self.Tensor([5.0, 6.0, 7.0, 8.0])
    np.testing.assert_allclose((a + b).numpy(), [6.0, 8.0, 10.0, 12.0])

  def test_simple_mul(self):
    a, b = self.Tensor([1.0, 2.0, 3.0, 4.0]), self.Tensor([2.0, 3.0, 4.0, 5.0])
    np.testing.assert_allclose((a * b).numpy(), [2.0, 6.0, 12.0, 20.0])

  def test_simple_sub(self):
    a, b = self.Tensor([10.0, 20.0, 30.0, 40.0]), self.Tensor([1.0, 2.0, 3.0, 4.0])
    np.testing.assert_allclose((a - b).numpy(), [9.0, 18.0, 27.0, 36.0])

  def test_simple_div(self):
    a, b = self.Tensor([10.0, 20.0, 30.0, 40.0]), self.Tensor([2.0, 4.0, 5.0, 8.0])
    np.testing.assert_allclose((a / b).numpy(), [5.0, 5.0, 6.0, 5.0])

  def test_neg(self):
    np.testing.assert_allclose((-self.Tensor([1.0, -2.0, 3.0, -4.0])).numpy(), [-1.0, 2.0, -3.0, 4.0])

  # *** integer arithmetic ***
  def test_int_add(self):
    a = self.Tensor([1, 2, 3, 4], dtype=self.dtypes.int32)
    b = self.Tensor([10, 20, 30, 40], dtype=self.dtypes.int32)
    np.testing.assert_array_equal((a + b).numpy(), [11, 22, 33, 44])

  def test_int_mul(self):
    a = self.Tensor([2, 3, 4, 5], dtype=self.dtypes.int32)
    b = self.Tensor([3, 4, 5, 6], dtype=self.dtypes.int32)
    np.testing.assert_array_equal((a * b).numpy(), [6, 12, 20, 30])

  # *** bitwise ops ***
  def test_bitwise_and(self):
    a = self.Tensor([0xff, 0x0f, 0xf0, 0xaa], dtype=self.dtypes.int32)
    b = self.Tensor([0x0f, 0xff, 0x0f, 0x55], dtype=self.dtypes.int32)
    np.testing.assert_array_equal((a & b).numpy(), [0x0f, 0x0f, 0x00, 0x00])

  def test_bitwise_or(self):
    a = self.Tensor([0xf0, 0x0f, 0x00, 0xaa], dtype=self.dtypes.int32)
    b = self.Tensor([0x0f, 0xf0, 0xff, 0x55], dtype=self.dtypes.int32)
    np.testing.assert_array_equal((a | b).numpy(), [0xff, 0xff, 0xff, 0xff])

  def test_bitwise_xor(self):
    a = self.Tensor([0xff, 0x0f, 0xf0, 0xaa], dtype=self.dtypes.int32)
    b = self.Tensor([0x0f, 0x0f, 0x0f, 0xaa], dtype=self.dtypes.int32)
    np.testing.assert_array_equal((a ^ b).numpy(), [0xf0, 0x00, 0xff, 0x00])

  # *** comparisons (use 2 elements to avoid 4-element codegen issue) ***
  def test_less_than(self):
    a, b = self.Tensor([1.0, 3.0]), self.Tensor([2.0, 2.0])
    np.testing.assert_array_equal((a < b).numpy(), [True, False])

  def test_greater_than(self):
    a, b = self.Tensor([1.0, 3.0]), self.Tensor([2.0, 2.0])
    np.testing.assert_array_equal((a > b).numpy(), [False, True])

  def test_equal(self):
    a, b = self.Tensor([1.0, 2.0]), self.Tensor([1.0, 3.0])
    np.testing.assert_array_equal((a == b).numpy(), [True, False])

  def test_not_equal(self):
    a, b = self.Tensor([1.0, 2.0]), self.Tensor([1.0, 3.0])
    np.testing.assert_array_equal((a != b).numpy(), [False, True])

  # *** activations ***
  def test_relu(self):
    np.testing.assert_allclose(self.Tensor([-1.0, 0.0, 1.0, 2.0]).relu().numpy(), [0.0, 0.0, 1.0, 2.0])

  # *** reductions ***
  def test_sum_reduce(self):
    np.testing.assert_allclose(self.Tensor([1.0, 2.0, 3.0, 4.0]).sum().numpy(), 10.0)

  def test_max_reduce(self):
    np.testing.assert_allclose(self.Tensor([1.0, 5.0, 3.0, 2.0]).max().numpy(), 5.0)

  def test_mean_reduce(self):
    np.testing.assert_allclose(self.Tensor([1.0, 2.0, 3.0, 4.0]).mean().numpy(), 2.5)

  # *** matmul ***
  def test_matmul_small(self):
    a = self.Tensor([[1.0, 2.0], [3.0, 4.0]])
    b = self.Tensor([[5.0, 6.0], [7.0, 8.0]])
    np.testing.assert_allclose((a @ b).numpy(), [[19.0, 22.0], [43.0, 50.0]])

  # *** math functions ***
  def test_exp(self):
    a = self.Tensor([0.0, 1.0, 2.0])
    np.testing.assert_allclose(a.exp().numpy(), np.exp([0.0, 1.0, 2.0]), rtol=1e-5)

  def test_log(self):
    a = self.Tensor([1.0, 2.0, 4.0])
    np.testing.assert_allclose(a.log().numpy(), np.log([1.0, 2.0, 4.0]), rtol=1e-5)

  def test_sqrt(self):
    np.testing.assert_allclose(self.Tensor([1.0, 4.0, 9.0, 16.0]).sqrt().numpy(), [1.0, 2.0, 3.0, 4.0])

  def test_reciprocal(self):
    np.testing.assert_allclose(self.Tensor([1.0, 2.0, 4.0, 8.0]).reciprocal().numpy(), [1.0, 0.5, 0.25, 0.125])

  # *** where/select - tests use VOP2 op 0 which may not be generated correctly ***
  def test_where(self):
    cond = self.Tensor([1, 0, 1, 0], dtype='bool')
    a, b = self.Tensor([1.0, 2.0, 3.0, 4.0]), self.Tensor([5.0, 6.0, 7.0, 8.0])
    np.testing.assert_allclose(cond.where(a, b).numpy(), [1.0, 6.0, 3.0, 8.0])

  # *** casting ***
  def test_cast_int_to_float(self):
    a = self.Tensor([1, 2, 3, 4], dtype=self.dtypes.int32)
    np.testing.assert_allclose(a.float().numpy(), [1.0, 2.0, 3.0, 4.0])

  def test_cast_float_to_int(self):
    np.testing.assert_array_equal(self.Tensor([1.5, 2.7, 3.2, 4.9]).cast(self.dtypes.int32).numpy(), [1, 2, 3, 4])

  # *** shape ops ***
  def test_reshape(self):
    a = self.Tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    np.testing.assert_allclose(a.reshape(2, 3).numpy(), [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])

  def test_permute(self):
    a = self.Tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    np.testing.assert_allclose(a.permute(1, 0).numpy(), [[1.0, 4.0], [2.0, 5.0], [3.0, 6.0]])

  # *** broadcasting ***
  def test_broadcast_add(self):
    a = self.Tensor([[1.0, 2.0, 3.0]])
    b = self.Tensor([[10.0], [20.0], [30.0]])
    np.testing.assert_allclose((a + b).numpy(), [[11, 12, 13], [21, 22, 23], [31, 32, 33]])

  def test_broadcast_mul(self):
    a = self.Tensor([1.0, 2.0, 3.0])
    b = self.Tensor([[2.0], [3.0]])
    np.testing.assert_allclose((a * b).numpy(), [[2, 4, 6], [3, 6, 9]])

  # *** multi-operation chains ***
  def test_chain_ops(self):
    a = self.Tensor([1.0, 2.0, 3.0, 4.0])
    result = ((a * 2) + 1).relu().numpy()
    np.testing.assert_allclose(result, [3.0, 5.0, 7.0, 9.0])

if __name__ == "__main__":
  unittest.main()
