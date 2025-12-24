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
    # Force AMD device with mockgpu
    os.environ["AMD"] = "1"
    from tinygrad import Tensor, Device
    cls.Tensor = Tensor
    cls.Device = Device

  def test_simple_add(self):
    """Test simple tensor addition."""
    a = self.Tensor([1.0, 2.0, 3.0, 4.0])
    b = self.Tensor([5.0, 6.0, 7.0, 8.0])
    c = (a + b).numpy()
    np.testing.assert_allclose(c, [6.0, 8.0, 10.0, 12.0])

  def test_simple_mul(self):
    """Test simple tensor multiplication."""
    a = self.Tensor([1.0, 2.0, 3.0, 4.0])
    b = self.Tensor([2.0, 3.0, 4.0, 5.0])
    c = (a * b).numpy()
    np.testing.assert_allclose(c, [2.0, 6.0, 12.0, 20.0])

  def test_simple_sub(self):
    """Test simple tensor subtraction."""
    a = self.Tensor([10.0, 20.0, 30.0, 40.0])
    b = self.Tensor([1.0, 2.0, 3.0, 4.0])
    c = (a - b).numpy()
    np.testing.assert_allclose(c, [9.0, 18.0, 27.0, 36.0])

  def test_neg(self):
    """Test tensor negation."""
    a = self.Tensor([1.0, -2.0, 3.0, -4.0])
    c = (-a).numpy()
    np.testing.assert_allclose(c, [-1.0, 2.0, -3.0, 4.0])

  def test_relu(self):
    """Test ReLU activation."""
    a = self.Tensor([-1.0, 0.0, 1.0, 2.0])
    c = a.relu().numpy()
    np.testing.assert_allclose(c, [0.0, 0.0, 1.0, 2.0])

  def test_sum_reduce(self):
    """Test sum reduction."""
    a = self.Tensor([1.0, 2.0, 3.0, 4.0])
    c = a.sum().numpy()
    np.testing.assert_allclose(c, 10.0)

  def test_max_reduce(self):
    """Test max reduction."""
    a = self.Tensor([1.0, 5.0, 3.0, 2.0])
    c = a.max().numpy()
    np.testing.assert_allclose(c, 5.0)

  def test_matmul_small(self):
    """Test small matrix multiplication."""
    a = self.Tensor([[1.0, 2.0], [3.0, 4.0]])
    b = self.Tensor([[5.0, 6.0], [7.0, 8.0]])
    c = (a @ b).numpy()
    expected = np.array([[19.0, 22.0], [43.0, 50.0]])
    np.testing.assert_allclose(c, expected)

  def test_exp(self):
    """Test exponential."""
    a = self.Tensor([0.0, 1.0, 2.0])
    c = a.exp().numpy()
    np.testing.assert_allclose(c, np.exp([0.0, 1.0, 2.0]), rtol=1e-5)

  def test_log(self):
    """Test logarithm."""
    a = self.Tensor([1.0, 2.0, 4.0])
    c = a.log().numpy()
    np.testing.assert_allclose(c, np.log([1.0, 2.0, 4.0]), rtol=1e-5)

  def test_sqrt(self):
    """Test square root."""
    a = self.Tensor([1.0, 4.0, 9.0, 16.0])
    c = a.sqrt().numpy()
    np.testing.assert_allclose(c, [1.0, 2.0, 3.0, 4.0])

  def test_reciprocal(self):
    """Test reciprocal."""
    a = self.Tensor([1.0, 2.0, 4.0, 8.0])
    c = a.reciprocal().numpy()
    np.testing.assert_allclose(c, [1.0, 0.5, 0.25, 0.125])

  def test_where(self):
    """Test where/select."""
    cond = self.Tensor([1, 0, 1, 0], dtype='bool')
    a = self.Tensor([1.0, 2.0, 3.0, 4.0])
    b = self.Tensor([5.0, 6.0, 7.0, 8.0])
    c = cond.where(a, b).numpy()
    np.testing.assert_allclose(c, [1.0, 6.0, 3.0, 8.0])

  def test_cast_int_to_float(self):
    """Test casting int to float."""
    from tinygrad import dtypes
    a = self.Tensor([1, 2, 3, 4], dtype=dtypes.int32)
    c = a.float().numpy()
    np.testing.assert_allclose(c, [1.0, 2.0, 3.0, 4.0])

  def test_cast_float_to_int(self):
    """Test casting float to int."""
    from tinygrad import dtypes
    a = self.Tensor([1.5, 2.7, 3.2, 4.9])
    c = a.cast(dtypes.int32).numpy()
    np.testing.assert_array_equal(c, [1, 2, 3, 4])

  def test_reshape(self):
    """Test reshape."""
    a = self.Tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    c = a.reshape(2, 3).numpy()
    expected = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    np.testing.assert_allclose(c, expected)

  def test_permute(self):
    """Test permute/transpose."""
    a = self.Tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    c = a.permute(1, 0).numpy()
    expected = np.array([[1.0, 4.0], [2.0, 5.0], [3.0, 6.0]])
    np.testing.assert_allclose(c, expected)

if __name__ == "__main__":
  unittest.main()
