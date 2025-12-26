#!/usr/bin/env python3
"""Small unit tests to debug RDNA3 renderer issues."""
import unittest
import os
os.environ["AMD"] = "1"
os.environ["AMD_RDNA"] = "1"

from tinygrad import Tensor, Device, dtypes
from tinygrad.helpers import getenv

@unittest.skipUnless(getenv("AMD", 0) and getenv("AMD_RDNA", 0), "AMD RDNA only")
class TestRDNAIDiv(unittest.TestCase):
  """Test integer division edge cases."""

  def test_idiv_simple(self):
    """Basic integer division."""
    a = Tensor([10, 20, 30, 40], dtype=dtypes.int32)
    b = Tensor([2, 4, 5, 8], dtype=dtypes.int32)
    result = (a // b).numpy()
    expected = [5, 5, 6, 5]
    self.assertEqual(list(result), expected)

  def test_idiv_by_constant(self):
    """Division by compile-time constant (uses fast_idiv pattern)."""
    a = Tensor([10, 20, 30, 40], dtype=dtypes.int32)
    result = (a // 3).numpy()
    expected = [3, 6, 10, 13]
    self.assertEqual(list(result), expected)

  def test_idiv_large_values(self):
    """Division with larger values that might overflow float rcp."""
    a = Tensor([1000000, 2000000, 123456789], dtype=dtypes.int32)
    b = Tensor([1000, 500, 12345], dtype=dtypes.int32)
    result = (a // b).numpy()
    expected = [1000, 4000, 10000]
    self.assertEqual(list(result), expected)

  def test_mod_simple(self):
    """Basic modulo operation."""
    a = Tensor([10, 20, 31, 47], dtype=dtypes.int32)
    b = Tensor([3, 7, 5, 8], dtype=dtypes.int32)
    result = (a % b).numpy()
    expected = [1, 6, 1, 7]
    self.assertEqual(list(result), expected)

  def test_mod_by_constant(self):
    """Modulo by constant."""
    a = Tensor([10, 20, 31, 47], dtype=dtypes.int32)
    result = (a % 7).numpy()
    expected = [3, 6, 3, 5]
    self.assertEqual(list(result), expected)

  def test_idiv_signed_negative(self):
    """Signed division with negative values."""
    a = Tensor([-10, 10, -20, 20], dtype=dtypes.int32)
    b = Tensor([3, -3, 7, -7], dtype=dtypes.int32)
    result = (a // b).numpy()
    expected = [-4, -4, -3, -3]  # Python-style floor division
    self.assertEqual(list(result), expected)

  def test_mod_signed_negative(self):
    """Signed modulo with negative values."""
    a = Tensor([-10, 10, -20, 20], dtype=dtypes.int32)
    b = Tensor([3, 3, 7, 7], dtype=dtypes.int32)
    result = (a % b).numpy()
    # Note: Python has different mod semantics than C
    # Python: -10 % 3 = 2, C: -10 % 3 = -1
    # Check what tinygrad does
    print(f"Signed mod result: {list(result)}")


@unittest.skipUnless(getenv("AMD", 0) and getenv("AMD_RDNA", 0), "AMD RDNA only")
class TestRDNAConditionalAccess(unittest.TestCase):
  """Test conditional memory access patterns."""

  def test_where_simple(self):
    """Basic WHERE operation."""
    cond = Tensor([1, 0, 1, 0], dtype=dtypes.int32)
    a = Tensor([10, 20, 30, 40], dtype=dtypes.float32)
    b = Tensor([100, 200, 300, 400], dtype=dtypes.float32)
    result = cond.where(a, b).numpy()
    expected = [10.0, 200.0, 30.0, 400.0]
    self.assertEqual(list(result), expected)

  def test_masked_load_with_invalid_indices(self):
    """Test that invalid indices with mask=False don't cause faults."""
    # Create a small buffer
    buf = Tensor([1.0, 2.0, 3.0, 4.0], dtype=dtypes.float32)
    # Create indices where some are out of bounds
    indices = Tensor([0, 1, 100, 2], dtype=dtypes.int32)  # 100 is out of bounds
    # Create mask that disables the out-of-bounds access
    mask = Tensor([1, 1, 0, 1], dtype=dtypes.int32)
    # The masked gather should not fault on index 100 since mask is 0
    # This requires proper conditional load handling
    result = buf[indices.where(mask, 0)].numpy()
    expected = [1.0, 2.0, 1.0, 3.0]  # masked lane uses index 0
    self.assertEqual(list(result), expected)


@unittest.skipUnless(getenv("AMD", 0) and getenv("AMD_RDNA", 0), "AMD RDNA only")
class TestRDNALoops(unittest.TestCase):
  """Test loop and range computations."""

  def test_sum_reduce(self):
    """Simple sum reduction (uses loop)."""
    a = Tensor([1.0, 2.0, 3.0, 4.0], dtype=dtypes.float32)
    result = a.sum().numpy()
    self.assertAlmostEqual(result, 10.0, places=5)

  def test_sum_reduce_2d(self):
    """2D sum reduction."""
    a = Tensor([[1.0, 2.0], [3.0, 4.0]], dtype=dtypes.float32)
    result = a.sum().numpy()
    self.assertAlmostEqual(result, 10.0, places=5)

  def test_sum_axis_0(self):
    """Sum along axis 0."""
    a = Tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=dtypes.float32)
    result = a.sum(axis=0).numpy()
    expected = [5.0, 7.0, 9.0]
    for r, e in zip(result, expected):
      self.assertAlmostEqual(r, e, places=5)

  def test_sum_axis_1(self):
    """Sum along axis 1."""
    a = Tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=dtypes.float32)
    result = a.sum(axis=1).numpy()
    expected = [6.0, 15.0]
    for r, e in zip(result, expected):
      self.assertAlmostEqual(r, e, places=5)


@unittest.skipUnless(getenv("AMD", 0) and getenv("AMD_RDNA", 0), "AMD RDNA only")
class TestRDNAMatmul(unittest.TestCase):
  """Test matrix multiplication patterns."""

  def test_matmul_2x2(self):
    """Simple 2x2 matmul."""
    a = Tensor([[1.0, 2.0], [3.0, 4.0]], dtype=dtypes.float32)
    b = Tensor([[5.0, 6.0], [7.0, 8.0]], dtype=dtypes.float32)
    result = (a @ b).numpy()
    expected = [[19.0, 22.0], [43.0, 50.0]]
    for i in range(2):
      for j in range(2):
        self.assertAlmostEqual(result[i][j], expected[i][j], places=4)

  def test_matmul_4x4(self):
    """4x4 matmul."""
    a = Tensor.ones(4, 4, dtype=dtypes.float32)
    b = Tensor.ones(4, 4, dtype=dtypes.float32) * 2.0
    result = (a @ b).numpy()
    expected = 8.0  # Each element is 4 * 2 = 8
    for i in range(4):
      for j in range(4):
        self.assertAlmostEqual(result[i][j], expected, places=4)

  def test_matmul_with_backward(self):
    """Matmul backward pass."""
    a = Tensor([[1.0, 2.0], [3.0, 4.0]], dtype=dtypes.float32, requires_grad=True)
    b = Tensor([[5.0, 6.0], [7.0, 8.0]], dtype=dtypes.float32, requires_grad=True)
    c = (a @ b).sum()
    c.backward()
    # Just check it completes without hang
    a_grad = a.grad.numpy()
    b_grad = b.grad.numpy()
    self.assertEqual(a_grad.shape, (2, 2))
    self.assertEqual(b_grad.shape, (2, 2))


@unittest.skipUnless(getenv("AMD", 0) and getenv("AMD_RDNA", 0), "AMD RDNA only")
class TestRDNAConv(unittest.TestCase):
  """Test convolution patterns (where the original bug was found)."""

  def test_conv2d_forward_small(self):
    """Small conv2d forward."""
    x = Tensor.ones(1, 1, 4, 4, dtype=dtypes.float32)
    w = Tensor.ones(1, 1, 3, 3, dtype=dtypes.float32)
    result = x.conv2d(w).numpy()
    self.assertEqual(result.shape, (1, 1, 2, 2))
    # Each output element is sum of 3x3 ones = 9
    for i in range(2):
      for j in range(2):
        self.assertAlmostEqual(result[0, 0, i, j], 9.0, places=4)

  def test_conv2d_backward_simple(self):
    """Conv2d backward pass - simple case."""
    x = Tensor.ones(1, 1, 4, 4, dtype=dtypes.float32, requires_grad=True)
    w = Tensor.ones(1, 1, 3, 3, dtype=dtypes.float32, requires_grad=True)
    y = x.conv2d(w).sum()
    y.backward()
    x_grad = x.grad.numpy()
    w_grad = w.grad.numpy()
    self.assertEqual(x_grad.shape, (1, 1, 4, 4))
    self.assertEqual(w_grad.shape, (1, 1, 3, 3))

  def test_conv2d_backward_with_relu(self):
    """Conv2d backward with relu - one layer."""
    x = Tensor.ones(1, 1, 4, 4, dtype=dtypes.float32, requires_grad=True)
    w = Tensor.ones(1, 1, 3, 3, dtype=dtypes.float32, requires_grad=True)
    y = x.conv2d(w).relu().sum()
    y.backward()
    x_grad = x.grad.numpy()
    w_grad = w.grad.numpy()
    self.assertEqual(x_grad.shape, (1, 1, 4, 4))
    self.assertEqual(w_grad.shape, (1, 1, 3, 3))

  def test_two_conv_layers_no_relu(self):
    """Two conv layers without relu."""
    x = Tensor.ones(1, 1, 8, 8, dtype=dtypes.float32, requires_grad=True)
    w1 = Tensor.ones(1, 1, 3, 3, dtype=dtypes.float32, requires_grad=True)
    w2 = Tensor.ones(1, 1, 3, 3, dtype=dtypes.float32, requires_grad=True)
    y = x.conv2d(w1).conv2d(w2).sum()
    y.backward()
    x_grad = x.grad.numpy()
    self.assertEqual(x_grad.shape, (1, 1, 8, 8))

  def test_two_conv_layers_with_relu_backward(self):
    """Two conv layers with relu and backward - the failing case."""
    x = Tensor.ones(1, 1, 8, 8, dtype=dtypes.float32, requires_grad=True)
    w1 = Tensor.ones(1, 1, 3, 3, dtype=dtypes.float32, requires_grad=True)
    w2 = Tensor.ones(1, 1, 3, 3, dtype=dtypes.float32, requires_grad=True)
    y = x.conv2d(w1).relu().conv2d(w2).relu().sum()
    y.backward()
    x_grad = x.grad.numpy()
    self.assertEqual(x_grad.shape, (1, 1, 8, 8))


@unittest.skipUnless(getenv("AMD", 0) and getenv("AMD_RDNA", 0), "AMD RDNA only")
class TestRDNAIndexComputation(unittest.TestCase):
  """Test index computation edge cases that might cause address overflow."""

  def test_reshape_simple(self):
    """Simple reshape - tests index remapping."""
    a = Tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], dtype=dtypes.float32)
    result = a.reshape(2, 3).numpy()
    self.assertEqual(result.shape, (2, 3))

  def test_transpose_2d(self):
    """2D transpose - tests strided access."""
    a = Tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=dtypes.float32)
    result = a.T.numpy()
    expected = [[1.0, 4.0], [2.0, 5.0], [3.0, 6.0]]
    for i in range(3):
      for j in range(2):
        self.assertAlmostEqual(result[i][j], expected[i][j], places=4)

  def test_strided_access_with_idiv(self):
    """Strided access pattern that uses integer division for index computation."""
    # This pattern: access every 3rd element, divide index by 2
    # Creates index computations like: (i // 3) * stride
    a = Tensor.arange(24, dtype=dtypes.float32).reshape(4, 6)
    result = a[::2, ::3].numpy()  # Every 2nd row, every 3rd col
    expected = [[0.0, 3.0], [12.0, 15.0]]
    for i in range(2):
      for j in range(2):
        self.assertAlmostEqual(result[i][j], expected[i][j], places=4)


@unittest.skipUnless(getenv("AMD", 0) and getenv("AMD_RDNA", 0), "AMD RDNA only")
class TestRDNAMultiKernel(unittest.TestCase):
  """Test multi-kernel sequences (checking kernel scheduling)."""

  def test_multi_kernel_sequence(self):
    """Multiple operations that generate separate kernels."""
    a = Tensor.ones(4, 4, dtype=dtypes.float32)
    b = Tensor.ones(4, 4, dtype=dtypes.float32) * 2
    c = (a + b).realize()
    d = (c * 3).realize()
    e = d.sum().realize()
    result = e.numpy()
    self.assertAlmostEqual(result, 144.0, places=4)  # 16 * (1+2) * 3 = 144


if __name__ == "__main__":
  unittest.main(verbosity=2)
