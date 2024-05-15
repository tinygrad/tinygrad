import unittest

from tinygrad import Tensor, dtypes, Device
import operator
import numpy as np
from hypothesis import given, strategies as strat, settings
from tinygrad.dtype import DType
from tinygrad.helpers import CI, getenv
from tinygrad.engine.schedule import create_schedule
from tinygrad.engine.realize import run_schedule
from tinygrad.ops import UnaryOps
from test.helpers import is_dtype_supported

settings.register_profile("my_profile", max_examples=200, deadline=None)
settings.load_profile("my_profile")
print(settings.default)

dtypes_float = (dtypes.float32, dtypes.float16)
dtypes_int = (dtypes.int8, dtypes.int16, dtypes.int32, dtypes.int64, dtypes.uint8, dtypes.uint16, dtypes.uint32, dtypes.uint64)
dtypes_bool = (dtypes.bool,)
binary_operations = [operator.add, operator.sub, operator.mul, operator.lt, operator.eq]

# TODO: LLVM comparing with nan is incorrect
if Device.DEFAULT == "LLVM":
  binary_operations.remove(operator.lt)
  binary_operations.remove(operator.eq)

integer_binary_operations = binary_operations + [(Tensor.xor, np.bitwise_xor)]
unary_operations = [(Tensor.exp, np.exp), (Tensor.log, np.log), operator.neg, (Tensor.sin, np.sin),
                    (Tensor.sqrt, np.sqrt), (Tensor.reciprocal, np.reciprocal)]

# TODO: enable this (this is a dtype issue)
#binary_operations.append(operator.truediv)

# TODO: enable mod on Tensor
#binary_operations.append(operator.mod)

# TODO: (a+b)/2 in tensor.py's maximum can overflow. This requires a new implementation of maximum that can be backpropagated
#binary_operations += [(Tensor.maximum, np.maximum)]

# TODO: CUDACPU segfaults on sin
# TODO: METAL sin is flaky for float16
if getenv("CUDACPU") or (getenv("MOCKGPU") and Device.DEFAULT == "NV") or Device.DEFAULT == "METAL": unary_operations.remove((Tensor.sin, np.sin))

class ht:
  float64 = strat.floats(width=64, allow_subnormal=False)
  float32 = strat.floats(width=32, allow_subnormal=False)
  float16 = strat.floats(width=16, allow_subnormal=False)
  uint8 = strat.integers(0, 255)
  uint16 = strat.integers(0, 65535)
  uint32 = strat.integers(0, 2**32-1)
  uint64 = strat.integers(0, 2**64-1)
  int8 = strat.integers(-128, 127)
  int16 = strat.integers(-32768, 32767)
  int32 = strat.integers(-2147483648, 2147483647)
  int64 = strat.integers(-9223372036854775808, 9223372036854775807)
  bool = strat.booleans()

def universal_test(a, b, dtype, op):
  if not isinstance(op, tuple): op = (op, op)
  tensor_value = (op[0](Tensor([a], dtype=dtype), Tensor([b], dtype=dtype))).numpy()
  numpy_value = op[1](np.array([a]).astype(dtype.np), np.array([b]).astype(dtype.np))
  if dtype in dtypes_float: np.testing.assert_allclose(tensor_value, numpy_value, atol=1e-10)
  else: np.testing.assert_equal(tensor_value, numpy_value)

def universal_test_unary(a, dtype, op):
  if not isinstance(op, tuple): op = (op, op)
  out: Tensor = op[0](Tensor([a], dtype=dtype))
  sched = create_schedule([out.lazydata])
  ast = sched[-1].ast[0]
  run_schedule(sched)
  tensor_value = out.numpy()
  numpy_value = op[1](np.array([a]).astype(dtype.np))
  if dtype in dtypes_float:
    np.testing.assert_allclose(tensor_value, numpy_value, atol=1e-3, rtol=1e-2)
  else: np.testing.assert_equal(tensor_value, numpy_value)
  if op[0] != Tensor.reciprocal: # reciprocal is not supported in most backends
    op = [x for x in ast.lazyops if x.op in UnaryOps][0]
    assert op.dtype == dtype

def universal_test_cast(a, in_dtype, dtype):
  tensor_value = Tensor([a], dtype=in_dtype).cast(dtype)
  numpy_value = np.array([a]).astype(dtype.np)
  np.testing.assert_equal(tensor_value.numpy(), numpy_value)

def universal_test_midcast(a, b, c, op1, op2, d1:DType, d2:DType):
  if not isinstance(op1, tuple): op1 = (op1, op1)
  if not isinstance(op2, tuple): op2 = (op2, op2)
  at, bt, ct = Tensor([a], dtype=d1), Tensor([b], dtype=d1), Tensor([c], dtype=d2)
  an, bn, cn = np.array([a]).astype(d1.np), np.array([b]).astype(d1.np), np.array([c]).astype(d2.np)
  tensor_value = op2[0](op1[0](at, bt).cast(d2), ct).numpy()
  numpy_value = op2[1](op1[1](an, bn).astype(d2.np), cn)
  np.testing.assert_allclose(tensor_value, numpy_value, rtol=1e-6 if getenv("PTX") else 1e-7)

class TestDTypeALU(unittest.TestCase):
  @unittest.skipUnless(is_dtype_supported(dtypes.float64, Device.DEFAULT), f"no float64 on {Device.DEFAULT}")
  @given(ht.float64, ht.float64, strat.sampled_from(binary_operations))
  def test_float64(self, a, b, op): universal_test(a, b, dtypes.float64, op)

  @given(ht.float32, ht.float32, strat.sampled_from(binary_operations))
  def test_float32(self, a, b, op): universal_test(a, b, dtypes.float32, op)

  @unittest.skipUnless(is_dtype_supported(dtypes.float16, Device.DEFAULT), f"no float16 on {Device.DEFAULT}")
  @given(ht.float16, ht.float16, strat.sampled_from(binary_operations))
  def test_float16(self, a, b, op): universal_test(a, b, dtypes.float16, op)

  @given(ht.float32, strat.sampled_from(unary_operations))
  def test_float32_unary(self, a, op): universal_test_unary(a, dtypes.float32, op)

  @unittest.skipUnless(is_dtype_supported(dtypes.float16, Device.DEFAULT), f"no float16 on {Device.DEFAULT}")
  @given(ht.float16, strat.sampled_from(unary_operations))
  def test_float16_unary(self, a, op): universal_test_unary(a, dtypes.float16, op)

  @given(ht.uint8, ht.uint8, strat.sampled_from(integer_binary_operations))
  def test_uint8(self, a, b, op): universal_test(a, b, dtypes.uint8, op)

  @unittest.skipUnless(is_dtype_supported(dtypes.uint16, Device.DEFAULT), f"no uint16 on {Device.DEFAULT}")
  @given(ht.uint16, ht.uint16, strat.sampled_from(integer_binary_operations))
  def test_uint16(self, a, b, op): universal_test(a, b, dtypes.uint16, op)

  @unittest.skipUnless(is_dtype_supported(dtypes.uint32, Device.DEFAULT), f"no uint32 on {Device.DEFAULT}")
  @given(ht.uint32, ht.uint32, strat.sampled_from(integer_binary_operations))
  def test_uint32(self, a, b, op): universal_test(a, b, dtypes.uint32, op)

  @unittest.skipUnless(is_dtype_supported(dtypes.uint64, Device.DEFAULT), f"no uint64 on {Device.DEFAULT}")
  @given(ht.uint64, ht.uint64, strat.sampled_from(integer_binary_operations))
  def test_uint64(self, a, b, op): universal_test(a, b, dtypes.uint64, op)

  @given(ht.int8, ht.int8, strat.sampled_from(integer_binary_operations))
  def test_int8(self, a, b, op): universal_test(a, b, dtypes.int8, op)

  @given(ht.int16, ht.int16, strat.sampled_from(integer_binary_operations))
  def test_int16(self, a, b, op): universal_test(a, b, dtypes.int16, op)

  @given(ht.int32, ht.int32, strat.sampled_from(integer_binary_operations))
  def test_int32(self, a, b, op): universal_test(a, b, dtypes.int32, op)

  @given(ht.int64, ht.int64, strat.sampled_from(integer_binary_operations))
  def test_int64(self, a, b, op): universal_test(a, b, dtypes.int64, op)

  @given(ht.bool, ht.bool, strat.sampled_from(((operator.add, operator.add), (operator.mul, operator.mul))))
  def test_bool(self, a, b, op): universal_test(a, b, dtypes.bool, op)

  @given(ht.int32, ht.int32, ht.float32, strat.sampled_from(integer_binary_operations), strat.sampled_from(binary_operations))
  def test_int32_midcast_float(self, a, b, c, op1, op2): universal_test_midcast(a, b, c, op1, op2, dtypes.int32, dtypes.float32)

  # Metal and CUDACPU and HIP behave differently than numpy in CI for overflows
  skip_overflow = CI and (Device.DEFAULT in {"HSA", "AMD", "NV"} or getenv("CUDACPU"))
  @given(strat.floats(width=32, min_value=0, max_value=10.0) if skip_overflow else ht.float32,
         strat.floats(width=32, min_value=0, max_value=10.0) if skip_overflow else ht.float32,
         ht.int32, strat.sampled_from(binary_operations), strat.sampled_from(integer_binary_operations))
  def test_float_midcast_int32(self, a, b, c, op1, op2): universal_test_midcast(a, b, c, op1, op2, dtypes.float32, dtypes.int32)

  @unittest.skip("broken. TODO: fix it")
  @given(ht.float32, strat.sampled_from(dtypes_float+dtypes_int+dtypes_bool))
  def test_float_cast(self, a, dtype): universal_test_cast(a, dtypes.float32, dtype)

  @unittest.skip("broken. TODO: fix it")
  @given(ht.int32, strat.sampled_from(dtypes_float+dtypes_int+dtypes_bool))
  def test_int32_cast(self, a, dtype): universal_test_cast(a, dtypes.int32, dtype)

if __name__ == '__main__':
  unittest.main()
