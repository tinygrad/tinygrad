import unittest
import numpy as np
from tinygrad.helpers import getenv
from tinygrad.lazy import Device
from tinygrad.tensor import Tensor, dtypes

# for GPU, cl_khr_fp16 isn't supported (except now we don't need it!)
# for LLVM, it segfaults because it can't link to the casting function
@unittest.skipIf(getenv("CI", "") != "" and Device.DEFAULT in ["LLVM"], "float16 broken in some CI backends")
class TestDtype(unittest.TestCase):
  def _test_to_np(self, a, np_dtype, target):
    print(a)
    na = a.numpy()
    print(na, na.dtype, a.lazydata.realized)
    assert na.dtype == np_dtype
    np.testing.assert_allclose(na, target)

  def test_half_to_np(self): self._test_to_np(Tensor([1,2,3,4], dtype=dtypes.float16), np.float16, [1,2,3,4])
  def test_int8_to_np(self): self._test_to_np(Tensor([1,2,3,4], dtype=dtypes.int8), np.int8, [1,2,3,4])
  def test_uint8_to_np(self): self._test_to_np(Tensor([1,2,3,4], dtype=dtypes.uint8), np.uint8, [1,2,3,4])

  def _test_cast(self, a, target_dtype, target):
    print(a)
    b = a.cast(target_dtype)
    print(b.numpy())
    assert b.dtype == target_dtype
    np.testing.assert_allclose(b.numpy(), target)

  def test_float_to_half(self): self._test_cast(Tensor([1,2,3,4], dtype=dtypes.float32), dtypes.float16, [1,2,3,4])
  def test_float_to_int8(self): self._test_cast(Tensor([1,2,3,4], dtype=dtypes.float32), dtypes.int8, [1,2,3,4])
  def test_float_to_uint8(self): self._test_cast(Tensor([1,2,3,4], dtype=dtypes.float32), dtypes.uint8, [1,2,3,4])

  def test_half_to_float(self): self._test_cast(Tensor([1,2,3,4], dtype=dtypes.float16), dtypes.float32, [1,2,3,4])
  def test_half_to_int8(self): self._test_cast(Tensor([1,2,3,4], dtype=dtypes.float16), dtypes.int8, [1,2,3,4])
  def test_half_to_uint8(self): self._test_cast(Tensor([1,2,3,4], dtype=dtypes.float16), dtypes.uint8, [1,2,3,4])

  def test_int8_to_float(self): self._test_cast(Tensor([1,2,3,4], dtype=dtypes.int8), dtypes.float32, [1,2,3,4])
  def test_int8_to_half(self): self._test_cast(Tensor([1,2,3,4], dtype=dtypes.int8), dtypes.float16, [1,2,3,4])
  def test_int8_to_uint8(self): self._test_cast(Tensor([1,2,3,4], dtype=dtypes.int8), dtypes.uint8, [1,2,3,4])

  def test_uint8_to_float(self): self._test_cast(Tensor([1,2,3,4], dtype=dtypes.uint8), dtypes.float32, [1,2,3,4])
  def test_uint8_to_half(self): self._test_cast(Tensor([1,2,3,4], dtype=dtypes.uint8), dtypes.float16, [1,2,3,4])
  def test_uint8_to_int8(self): self._test_cast(Tensor([1,2,3,4], dtype=dtypes.uint8), dtypes.int8, [1,2,3,4])

  def _test_add(self, a, b, target_dtype, target):
    c = a+b
    print(c.numpy())
    assert c.dtype == target_dtype
    np.testing.assert_allclose(c.numpy(), target)

  def test_half_add(self): self._test_add(Tensor([1,2,3,4], dtype=dtypes.float16), Tensor([1,2,3,4], dtype=dtypes.float16), dtypes.float16, [2,4,6,8])
  def test_int8_add(self): self._test_add(Tensor([1,2,3,4], dtype=dtypes.int8), Tensor([1,2,3,4], dtype=dtypes.int8), dtypes.int8, [2,4,6,8])

  def _test_mul(self, a, b, target_dtype, target):
    c = a*b
    print(c.numpy())
    assert c.dtype == target_dtype
    np.testing.assert_allclose(c.numpy(), target)

  def test_half_mul(self): self._test_mul(Tensor([1,2,3,4], dtype=dtypes.float16), Tensor([1,2,3,4], dtype=dtypes.float16), dtypes.float16, [1,4,9,16])
  def test_int8_mul(self): self._test_mul(Tensor([1,2,3,4], dtype=dtypes.int8), Tensor([1,2,3,4], dtype=dtypes.int8), dtypes.int8, [1,4,9,16])

  def _test_matmul(self, a, b, target_dtype, target):
    c = a@b
    print(c.numpy())
    assert c.dtype == target_dtype
    np.testing.assert_allclose(c.numpy(), target)

  def test_half_matmul(self): self._test_matmul(Tensor([[1,2],[3,4]], dtype=dtypes.float16), Tensor.eye(2, dtype=dtypes.float16), dtypes.float16, [[1,2],[3,4]])
  def test_int8_matmul(self): self._test_matmul(Tensor([[1,2],[3,4]], dtype=dtypes.int8), Tensor.eye(2, dtype=dtypes.int8), dtypes.int8, [[1,2],[3,4]])

  def _test_add_upcast(self, a, b, target_dtype, target):
    c = a+b
    print(c.numpy())
    assert c.dtype == target_dtype
    np.testing.assert_allclose(c.numpy(), target)

  def test_half_add_upcast_float(self): self._test_add_upcast(Tensor([1,2,3,4], dtype=dtypes.float16), Tensor([1,2,3,4], dtype=dtypes.float32), dtypes.float32, [2,4,6,8])
  def test_int8_add_upcast_float(self): self._test_add_upcast(Tensor([1,2,3,4], dtype=dtypes.int8), Tensor([1,2,3,4], dtype=dtypes.float32), dtypes.float32, [2,4,6,8])
  def test_int8_add_upcast_half(self): self._test_add_upcast(Tensor([1,2,3,4], dtype=dtypes.int8), Tensor([1,2,3,4], dtype=dtypes.float16), dtypes.float16, [2,4,6,8])

  def _test_mul_upcast(self, a, b, target_dtype, target):
    c = a*b
    print(c.numpy())
    assert c.dtype == target_dtype
    np.testing.assert_allclose(c.numpy(), target)

  def test_half_mul_upcast_float(self): self._test_mul_upcast(Tensor([1,2,3,4], dtype=dtypes.float16), Tensor([1,2,3,4], dtype=dtypes.float32), dtypes.float32, [1,4,9,16])
  def test_int8_mul_upcast_float(self): self._test_mul_upcast(Tensor([1,2,3,4], dtype=dtypes.int8), Tensor([1,2,3,4], dtype=dtypes.float32), dtypes.float32, [1,4,9,16])
  def test_int8_mul_upcast_half(self): self._test_mul_upcast(Tensor([1,2,3,4], dtype=dtypes.int8), Tensor([1,2,3,4], dtype=dtypes.float16), dtypes.float16, [1,4,9,16])

  def _test_matmul_upcast(self, a, b, target_dtype, target):
    c = a@b
    print(c.numpy())
    assert c.dtype == target_dtype
    np.testing.assert_allclose(c.numpy(), target)

  def test_half_matmul_upcast_float(self): self._test_matmul_upcast(Tensor([[1,2],[3,4]], dtype=dtypes.float16), Tensor.eye(2, dtype=dtypes.float32), dtypes.float32, [[1,2],[3,4]])
  def test_int8_matmul_upcast_float(self): self._test_matmul_upcast(Tensor([[1,2],[3,4]], dtype=dtypes.int8), Tensor.eye(2, dtype=dtypes.float32), dtypes.float32, [[1,2],[3,4]])
  def test_int8_matmul_upcast_half(self): self._test_matmul_upcast(Tensor([[1,2],[3,4]], dtype=dtypes.int8), Tensor.eye(2, dtype=dtypes.float16), dtypes.float16, [[1,2],[3,4]])

if __name__ == '__main__':
  unittest.main()
