import unittest
from tinygrad import Tensor, dtypes
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
  np.testing.assert_equal(tensor_value, numpy_value)

class TestDTypeALU(unittest.TestCase):
  @unittest.skip("flaky")
  @settings(max_examples=MAX_EXAMPLES, deadline=None)
  @given(st.floats(width=32), st.floats(width=32), st.sampled_from(binary_operations))
  def test_float32(self, a, b, op): universal_test(a, b, dtypes.float32, op)

  @unittest.skip("flaky")
  @settings(max_examples=MAX_EXAMPLES, deadline=None)
  @given(st.floats(width=16), st.floats(width=16), st.sampled_from(binary_operations))
  def test_float16(self, a, b, op): universal_test(a, b, dtypes.float16, op)

  @settings(max_examples=MAX_EXAMPLES, deadline=None)
  @given(st.integers(0, 255), st.integers(0, 255), st.sampled_from(binary_operations))
  def test_uint8(self, a, b, op): universal_test(a, b, dtypes.uint8, op)

  @settings(max_examples=MAX_EXAMPLES, deadline=None)
  @given(st.integers(0, 65535), st.integers(0, 65535), st.sampled_from(binary_operations))
  def test_uint16(self, a, b, op): universal_test(a, b, dtypes.uint16, op)

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


if __name__ == '__main__':
  unittest.main()
