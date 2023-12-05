import unittest
from tinygrad import Tensor, dtypes, Device
import operator
import numpy as np
from hypothesis import given, strategies as st, settings

dtypes_float = (dtypes.float32, dtypes.float16)
dtypes_int = (dtypes.int8, dtypes.int16, dtypes.int32, dtypes.int64, dtypes.uint8, dtypes.uint16, dtypes.uint32, dtypes.uint64)
# truediv is broken
binary_operations = (operator.add, operator.sub, operator.mul) #, operator.truediv)

MAX_EXAMPLES = 200

def universal_test(a, b, dtype, op):
  tensor_value = (op(Tensor([a], dtype=dtype), Tensor([b], dtype=dtype))).numpy()
  numpy_value = op(np.array([a]).astype(dtype.np), np.array([b]).astype(dtype.np))
  if dtype in dtypes_float: np.testing.assert_almost_equal(tensor_value, numpy_value)
  else: np.testing.assert_equal(tensor_value, numpy_value)

class TestDTypeALU(unittest.TestCase):
  @settings(max_examples=MAX_EXAMPLES, deadline=None)
  @given(st.floats(width=32, allow_subnormal=False), st.floats(width=32, allow_subnormal=False), st.sampled_from(binary_operations))
  def test_float32(self, a, b, op): universal_test(a, b, dtypes.float32, op)

  @settings(max_examples=MAX_EXAMPLES, deadline=None)
  @given(st.floats(width=16, allow_subnormal=False), st.floats(width=16, allow_subnormal=False), st.sampled_from(binary_operations))
  def test_float16(self, a, b, op): universal_test(a, b, dtypes.float16, op)

  @settings(max_examples=MAX_EXAMPLES, deadline=None)
  @given(st.integers(0, 255), st.integers(0, 255), st.sampled_from(binary_operations))
  def test_uint8(self, a, b, op): universal_test(a, b, dtypes.uint8, op)

  @unittest.skipIf(Device.DEFAULT == "TORCH", "no uint16 in torch")
  @settings(max_examples=MAX_EXAMPLES, deadline=None)
  @given(st.integers(0, 65535), st.integers(0, 65535), st.sampled_from(binary_operations))
  def test_uint16(self, a, b, op): universal_test(a, b, dtypes.uint16, op)

  @unittest.skipIf(Device.DEFAULT == "TORCH", "no uint32 in torch")
  @settings(max_examples=MAX_EXAMPLES, deadline=None)
  @given(st.integers(0, 4294967295), st.integers(0, 4294967295), st.sampled_from(binary_operations))
  def test_uint32(self, a, b, op): universal_test(a, b, dtypes.uint32, op)

  @settings(max_examples=MAX_EXAMPLES, deadline=None)
  @given(st.integers(-128, 127), st.integers(-128, 127), st.sampled_from(binary_operations))
  def test_int8(self, a, b, op): universal_test(a, b, dtypes.int8, op)

  @settings(max_examples=MAX_EXAMPLES, deadline=None)
  @given(st.integers(-32768, 32767), st.integers(-32768, 32767), st.sampled_from(binary_operations))
  def test_int16(self, a, b, op): universal_test(a, b, dtypes.int16, op)

  @settings(max_examples=MAX_EXAMPLES, deadline=None)
  @given(st.integers(-2147483648, 2147483647), st.integers(-2147483648, 2147483647), st.sampled_from(binary_operations))
  def test_int32(self, a, b, op): universal_test(a, b, dtypes.int32, op)

  @settings(max_examples=MAX_EXAMPLES, deadline=None)
  @given(st.integers(-2147483648, 2147483647), st.integers(-2147483648, 2147483647), st.floats(width=32, allow_subnormal=False), st.sampled_from(binary_operations), st.sampled_from(binary_operations))
  def test_int32_midcast_float(self, a, b, c, op1, op2):
    at, bt, ct = Tensor([a], dtype=dtypes.int32), Tensor([b], dtype=dtypes.int32), Tensor([c], dtype=dtypes.float32)
    an, bn, cn = np.array([a]).astype(np.int32), np.array([b]).astype(np.int32), np.array([c]).astype(np.float32)
    tensor_value = op2(op1(at, bt).cast(dtypes.float32), ct).numpy()
    numpy_value = op2(op1(an, bn).astype(np.float32), cn)
    np.testing.assert_almost_equal(tensor_value, numpy_value)

  @settings(max_examples=MAX_EXAMPLES, deadline=None)
  @given(st.floats(width=32, allow_subnormal=False), st.floats(width=32, allow_subnormal=False), st.integers(-2147483648, 2147483647), st.sampled_from(binary_operations), st.sampled_from(binary_operations))
  def test_float_midcast_int32(self, a, b, c, op1, op2):
    at, bt, ct = Tensor([a], dtype=dtypes.float32), Tensor([b], dtype=dtypes.float32), Tensor([c], dtype=dtypes.int32)
    an, bn, cn = np.array([a]).astype(np.float32), np.array([b]).astype(np.float32), np.array([c]).astype(np.int32)
    tensor_value = op2(op1(at, bt).cast(dtypes.int32), ct).numpy()
    numpy_value = op2(op1(an, bn).astype(np.int32), cn)
    np.testing.assert_equal(tensor_value, numpy_value)

if __name__ == '__main__':
  unittest.main()
