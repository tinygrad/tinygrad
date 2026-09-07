"""Generated conversions must preserve tinygrad's numeric rounding contract."""
import numpy as np
import pytest
from tinygrad import Tensor, dtypes


@pytest.mark.parametrize('dtype,values,expected', [
  (np.int32, [19958399, -19958399, 16777219, -16777219], [19958400, -19958400, 16777220, -16777220]),
  (np.uint32, [19958399, 16777219, 33554431, 4294967295], [19958400, 16777220, 33554432, 4294967296]),
])
def test_integer_to_float32_uses_nearest_even(dtype, values, expected):
  # Odd halfway significands must round to the even F32 neighbor, not toward zero.
  result = Tensor(np.array(values, dtype=dtype)).cast(dtypes.float32).numpy()
  np.testing.assert_array_equal(result, np.array(expected, dtype=np.float32))


def test_float_to_integer_still_truncates_toward_zero():
  result = Tensor(np.array([-2.9, -1.5, 0, 1.5, 2.9], dtype=np.float32)).cast(dtypes.int32).numpy()
  np.testing.assert_array_equal(result, [-2, -1, 0, 1, 2])
