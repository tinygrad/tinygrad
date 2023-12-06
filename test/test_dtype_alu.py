import unittest
from tinygrad import Tensor, dtypes, Device
import operator
import numpy as np
from hypothesis import given, strategies as st, settings

from tinygrad.helpers import CI, getenv

settings.register_profile("my_profile", max_examples=200, deadline=None)
settings.load_profile("my_profile")
print(settings.default)

def skipUnlessFP16Supported(): return unittest.skip("GPU requires cl_khr_fp16") if Device.DEFAULT == "GPU" and CI else unittest.skip("CUDACPU architecture is sm_35 but we need at least sm_70 to run fp16 ALUs") if getenv("CUDACPU") else lambda _x: None

dtypes_float = (dtypes.float32, dtypes.float16)
dtypes_int = (dtypes.int8, dtypes.int16, dtypes.int32, dtypes.int64, dtypes.uint8, dtypes.uint16, dtypes.uint32, dtypes.uint64)
dtypes_bool = (dtypes.bool,)
# TODO: lt and eq should cast in tensor before we can test them, this is a seperate project
binary_operations = (operator.add, operator.sub, operator.mul, operator.truediv, (Tensor.maximum, np.maximum), (Tensor.xor, np.bitwise_xor)) #, operator.lt, operator.eq)
unary_operations = ((Tensor.exp, np.exp), (Tensor.log, np.log), operator.neg, (Tensor.sin, np.sin), (Tensor.sqrt, np.sqrt), (Tensor.reciprocal, np.reciprocal))

class ht:
  float32 = st.floats(width=32, allow_subnormal=False)
  float16 = st.floats(width=16, allow_subnormal=False)
  uint8 = st.integers(0, 255)
  uint16 = st.integers(0, 65535)
  uint32 = st.integers(0, 4294967295)
  int8 = st.integers(-128, 127)
  int16 = st.integers(-32768, 32767)
  int32 = st.integers(-2147483648, 2147483647)
  bool = st.booleans()

def universal_test(a, b, dtype, op):
  if not isinstance(op, tuple): op = (op, op)
  tensor_value = (op[0](Tensor([a], dtype=dtype), Tensor([b], dtype=dtype))).numpy()
  numpy_value = op[1](np.array([a]).astype(dtype.np), np.array([b]).astype(dtype.np))
  if dtype in dtypes_float: np.testing.assert_allclose(tensor_value, numpy_value, atol=1e-10)
  else: np.testing.assert_equal(tensor_value, numpy_value)

def universal_test_unary(a, dtype, op):
  if not isinstance(op, tuple): op = (op, op)
  tensor_value = op[0](Tensor([a], dtype=dtype)).numpy()
  numpy_value = op[1](np.array([a]).astype(dtype.np))
  if dtype in dtypes_float: np.testing.assert_allclose(tensor_value, numpy_value, atol=1e-7, rtol=1e-5 if dtype == dtypes.float32 else 1e-2)  # exp and log are approximations
  else: np.testing.assert_equal(tensor_value, numpy_value)

def universal_test_cast(a, in_dtype, dtype):
  tensor_value = Tensor([a], dtype=in_dtype).cast(dtype)
  numpy_value = np.array([a]).astype(dtype.np)
  np.testing.assert_equal(tensor_value, numpy_value)

class TestDTypeALU(unittest.TestCase):
  @given(ht.float32, ht.float32, st.sampled_from(binary_operations))
  def test_float32(self, a, b, op): universal_test(a, b, dtypes.float32, op)

  @skipUnlessFP16Supported()
  @given(ht.float16, ht.float16, st.sampled_from(binary_operations))
  def test_float16(self, a, b, op): universal_test(a, b, dtypes.float16, op)

  @given(ht.float32, st.sampled_from(unary_operations))
  def test_float32_unary(self, a, op): universal_test_unary(a, dtypes.float32, op)

  @skipUnlessFP16Supported()
  @given(ht.float32, st.sampled_from(unary_operations))
  def test_float16_unary(self, a, op): universal_test_unary(a, dtypes.float16, op)

  @given(ht.uint8, ht.uint8, st.sampled_from(binary_operations))
  def test_uint8(self, a, b, op): universal_test(a, b, dtypes.uint8, op)

  @unittest.skipIf(Device.DEFAULT == "TORCH", "no uint16 in torch")
  @given(ht.uint16, ht.uint16, st.sampled_from(binary_operations))
  def test_uint16(self, a, b, op): universal_test(a, b, dtypes.uint16, op)

  @unittest.skipIf(Device.DEFAULT == "TORCH", "no uint32 in torch")
  @given(ht.uint32, ht.uint32, st.sampled_from(binary_operations))
  def test_uint32(self, a, b, op): universal_test(a, b, dtypes.uint32, op)

  @given(ht.int8, ht.int8, st.sampled_from(binary_operations))
  def test_int8(self, a, b, op): universal_test(a, b, dtypes.int8, op)

  @given(ht.int16, ht.int16, st.sampled_from(binary_operations))
  def test_int16(self, a, b, op): universal_test(a, b, dtypes.int16, op)

  @given(ht.int32, ht.int32, st.sampled_from(binary_operations))
  def test_int32(self, a, b, op): universal_test(a, b, dtypes.int32, op)

  @given(ht.bool, ht.bool, st.sampled_from(((operator.add, operator.add), (operator.mul, operator.mul))))
  def test_bool(self, a, b, op): universal_test(a, b, dtypes.bool, op)

  @given(ht.int32, ht.int32, ht.float32, st.sampled_from(binary_operations), st.sampled_from(binary_operations))
  def test_int32_midcast_float(self, a, b, c, op1, op2):
    at, bt, ct = Tensor([a], dtype=dtypes.int32), Tensor([b], dtype=dtypes.int32), Tensor([c], dtype=dtypes.float32)
    an, bn, cn = np.array([a]).astype(np.int32), np.array([b]).astype(np.int32), np.array([c]).astype(np.float32)
    tensor_value = op2[0](op1[0](at, bt).cast(dtypes.float32), ct).numpy()
    numpy_value = op2[1](op1[1](an, bn).astype(np.float32), cn)
    np.testing.assert_almost_equal(tensor_value, numpy_value)

  @given(ht.float32, ht.float32, ht.int32, st.sampled_from(binary_operations), st.sampled_from(binary_operations))
  def test_float_midcast_int32(self, a, b, c, op1, op2):
    at, bt, ct = Tensor([a], dtype=dtypes.float32), Tensor([b], dtype=dtypes.float32), Tensor([c], dtype=dtypes.int32)
    an, bn, cn = np.array([a]).astype(np.float32), np.array([b]).astype(np.float32), np.array([c]).astype(np.int32)
    tensor_value = op2[0](op1[0](at, bt).cast(dtypes.int32), ct).numpy()
    numpy_value = op2[1](op1[1](an, bn).astype(np.int32), cn)
    np.testing.assert_equal(tensor_value, numpy_value)

  @given(ht.float32, st.sampled_from(dtypes_float+dtypes_int+dtypes_bool))
  def test_float_cast(self, a, dtype): universal_test_cast(a, dtypes.float32, dtype)

  @given(ht.int32, st.sampled_from(dtypes_float+dtypes_int+dtypes_bool))
  def test_int32_cast(self, a, dtype): universal_test_cast(a, dtypes.int32, dtype)

if __name__ == '__main__':
  unittest.main()
