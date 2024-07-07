import unittest
from tinygrad import dtypes
from tinygrad.tensor import Tensor
from test.helpers import is_dtype_supported
from test.test_schedule import check_schedule

class TestFastmathSchedule(unittest.TestCase):
  # w/o payne_hanek_reduction (fp16)
  @unittest.skipUnless(is_dtype_supported(dtypes.half), "need half")
  def test_fastmath_sin_fp16_fusion(self):
    a = Tensor.empty(10, dtype=dtypes.float16)
    b = Tensor.empty(10, dtype=dtypes.float16)
    c = a.sin() + b.sin()
    c = c.sin()
    check_schedule(c, 1)
  # w/ payne_hanek_reduction (fp32)
  def test_fastmath_sin_fusion(self):
    a = Tensor.empty(10)
    b = Tensor.empty(10)
    c = a.sin() + b.sin()
    c = c.sin()
    check_schedule(c, 1)

  def test_fastmath_log2_fusion(self):
    a = Tensor.empty(10)
    b = Tensor.empty(10)
    c = a.log2() + b.log2()
    c = c.log2()
    check_schedule(c, 1)

  def test_fastmath_exp2_fusion(self):
    a = Tensor.empty(10)
    b = Tensor.empty(10)
    c = a.exp2() + b.exp2()
    c = c.exp2()
    check_schedule(c, 1)
