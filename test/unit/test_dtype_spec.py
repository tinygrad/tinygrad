import unittest, math, struct
from tinygrad.tensor import dtypes
from tinygrad.dtype import DTYPES_DICT, truncate, float_to_fp16, float_to_bf16, _to_np_dtype, least_upper_dtype
from tinygrad.device import is_dtype_supported
from tinygrad.helpers import getenv
from hypothesis import given, settings, strategies as strat
import numpy as np
import torch

settings.register_profile("my_profile", max_examples=50, deadline=None, derandomize=getenv("DERANDOMIZE_CI", False))
settings.load_profile("my_profile")

core_dtypes = list(DTYPES_DICT.values())

FP8E4M3_MAX = 448.0
FP8E5M2_MAX = 57344.0

def u32_to_f32(u): return struct.unpack('f', struct.pack('I', u))[0]
def f32_to_u32(f): return struct.unpack('I', struct.pack('f', f))[0]

class TestHelpers(unittest.TestCase):
  signed_ints = (dtypes.int8, dtypes.int16, dtypes.int32, dtypes.int64)
  uints = (dtypes.uint8, dtypes.uint16, dtypes.uint32, dtypes.uint64)
  floats = (dtypes.float16, dtypes.float32, dtypes.float64)

  @given(strat.sampled_from(signed_ints+uints), strat.integers(min_value=1, max_value=8))
  def test_is_int(self, dtype, amt):
    assert dtypes.is_int(dtype.vec(amt) if amt > 1 else dtype)
    assert not dtypes.is_float(dtype.vec(amt) if amt > 1 else dtype)

  @given(strat.sampled_from(uints), strat.integers(min_value=1, max_value=8))
  def test_is_unsigned_uints(self, dtype, amt):
    assert dtypes.is_unsigned(dtype.vec(amt) if amt > 1 else dtype)

  @given(strat.sampled_from(signed_ints), strat.integers(min_value=1, max_value=8))
  def test_is_unsigned_signed_ints(self, dtype, amt):
    assert not dtypes.is_unsigned(dtype.vec(amt) if amt > 1 else dtype)

  @given(strat.sampled_from(floats), strat.integers(min_value=1, max_value=8))
  def test_is_float(self, dtype, amt):
    assert dtypes.is_float(dtype.vec(amt) if amt > 1 else dtype)
    assert not dtypes.is_int(dtype.vec(amt) if amt > 1 else dtype)
    assert not dtypes.is_unsigned(dtype.vec(amt) if amt > 1 else dtype)

  def test_bf16_is_float(self):
    assert dtypes.is_float(dtypes.bfloat16)

  def test_fp8s_are_float(self):
    assert dtypes.is_float(dtypes.fp8e4m3)
    assert dtypes.is_float(dtypes.fp8e5m2)

  @given(strat.sampled_from([d for d in DTYPES_DICT.values() if dtypes.is_float(d) or dtypes.is_int(d)]), strat.integers(min_value=2, max_value=8))
  def test_scalar(self, dtype, amt):
    assert dtype.vec(amt).scalar() == dtype

  def test_from_py(self):
    assert dtypes.from_py(True) == dtypes.bool
    assert dtypes.from_py(2) == dtypes.default_int
    assert dtypes.from_py(3.0) == dtypes.default_float
    assert dtypes.from_py([]) == dtypes.default_float
    assert dtypes.from_py(()) == dtypes.default_float
    assert dtypes.from_py([True]) == dtypes.bool
    assert dtypes.from_py([True, 2]) == dtypes.default_int
    assert dtypes.from_py([True, 3.0]) == dtypes.default_float
    assert dtypes.from_py([2, 3.0]) == dtypes.default_float
    assert dtypes.from_py([True, 2, 3.0]) == dtypes.default_float
    with self.assertRaises(RuntimeError): dtypes.from_py(None)
    with self.assertRaises(RuntimeError): dtypes.from_py([None])
    with self.assertRaises(RuntimeError): dtypes.from_py({})
    with self.assertRaises(RuntimeError): dtypes.from_py(set())

  def test_dtype_range(self):
    for dt in core_dtypes:
      if dtypes.is_float(dt):
        np.testing.assert_equal(dtypes.min(dt), -math.inf)
        np.testing.assert_equal(dtypes.max(dt), math.inf)
        np.testing.assert_equal(dt.min, -math.inf)
        np.testing.assert_equal(dt.max, math.inf)
      elif dtypes.is_int(dt):
        info = np.iinfo(_to_np_dtype(dt))
        np.testing.assert_equal(dtypes.min(dt), info.min)
        np.testing.assert_equal(dtypes.max(dt), info.max)
        np.testing.assert_equal(dt.min, info.min)
        np.testing.assert_equal(dt.max, info.max)
      else:
        assert dt == dtypes.bool, dt
        np.testing.assert_equal(dtypes.min(dt), False)
        np.testing.assert_equal(dtypes.max(dt), True)
        np.testing.assert_equal(dt.min, False)
        np.testing.assert_equal(dt.max, True)

  def test_dtype_range_vec(self):
    for dt in core_dtypes:
      self.assertEqual(dt.min, dt.vec(4).min)
      self.assertEqual(dt.max, dt.vec(4).max)

  def test_float_to_fp16(self):
    self.assertEqual(float_to_fp16(1), 1)
    self.assertEqual(float_to_fp16(65504), 65504)
    self.assertEqual(float_to_fp16(65519.999), 65504)
    self.assertEqual(float_to_fp16(65520), math.inf)
    self.assertEqual(float_to_fp16(1e-8), 0.0)
    self.assertEqual(float_to_fp16(-65504), -65504)
    self.assertEqual(float_to_fp16(-65519.999), -65504)
    self.assertEqual(float_to_fp16(-65520), -math.inf)
    self.assertTrue(math.isnan(float_to_fp16(math.nan)))

  def test_float_to_bf16(self):
    # TODO: fuzz this better
    max_bf16 = torch.finfo(torch.bfloat16).max
    for a in [1, 1.1, 1234, 23456, -777.777, max_bf16, max_bf16 * 1.00001, -max_bf16, -max_bf16 * 1.00001, math.inf, -math.inf]:
      self.assertEqual(float_to_bf16(a), torch.tensor([a], dtype=torch.bfloat16).item())
    self.assertTrue(math.isnan(float_to_bf16(math.nan)))

  def test_float_to_bf16_nan(self):
    # In f32, NaN = exp 0xFF and mantissa ≠ 0. Quiet-vs-signaling is bit 22 of the mantissa: 1 = qNaN, 0 = sNaN.
    # qNaN(+/-), sNaN(+/-) overflow(+/-)
    patterns = [0x7FC00001, 0xFFC00001, 0x7F800001, 0xFF800001, 0x7FFFFFFF, 0xFFFFFFFF]
    for u in patterns:
      x = u32_to_f32(u)
      y = float_to_bf16(x)
      t = torch.tensor([x], dtype=torch.bfloat16).item()
      self.assertTrue(math.isnan(y))
      self.assertTrue(math.isnan(t))

  def test_float_to_bf16_round(self):
    # round_to_nearest_even
    uppers = [0x3f800000, 0x41230000, 0xC1460000] # 1.0, 10.1875, -12.375
    for upper in uppers:
      base = upper & 0xFFFF0000
      base_f32 = u32_to_f32(base)
      base_f32_round_up = u32_to_f32(base + 0x00010000)

      # low < 0x8000(0.5ULP) -> round down
      x = u32_to_f32(base | 0x00007000)
      self.assertEqual(float_to_bf16(x), base_f32)
      self.assertEqual(torch.tensor([x], dtype=torch.bfloat16).item(), base_f32)

      # low > 0x8000(0.5ULP) -> round up
      x = u32_to_f32(base | 0x0000C000)
      self.assertEqual(float_to_bf16(x), base_f32_round_up)
      self.assertEqual(torch.tensor([x], dtype=torch.bfloat16).item(), base_f32_round_up)

      # low == 0x8000(0.5ULP) and LSB even -> round down
      if ((upper >> 16) & 1) == 0:
        x = u32_to_f32(base | 0x00008000)
        self.assertEqual(float_to_bf16(x), base_f32)
        self.assertEqual(torch.tensor([x], dtype=torch.bfloat16).item(), base_f32)
      # low == 0x8000(0.5ULP) and LSB odd -> round up
      else:
        x = u32_to_f32(base | 0x00008000)
        self.assertEqual(float_to_bf16(x), base_f32_round_up)
        self.assertEqual(torch.tensor([x], dtype=torch.bfloat16).item(), base_f32_round_up)

  def test_float_to_bf16_boundary(self):
    # bf16 max finite: exp=0xFE, faction=0x7F => 0x7F7F0000(f32)
    # bf16 inf(+/-):   exp=0xFF
    base = 0x7F7F0000
    inf_u32 = 0x7F800000

    # low < 0.5ULP
    x = u32_to_f32(base | 0x00007FFF)
    self.assertEqual(f32_to_u32(float_to_bf16(x)), base)
    self.assertEqual(f32_to_u32(torch.tensor([x], dtype=torch.bfloat16).item()), base)

    # low > 0.5ULP -> overflows to +inf
    x = u32_to_f32(base | 0x0000C000)
    self.assertEqual(f32_to_u32(float_to_bf16(x)), inf_u32)
    self.assertEqual(f32_to_u32(torch.tensor([x], dtype=torch.bfloat16).item()), inf_u32)

    # low == 0.5ULP and LSB odd -> overflows to +inf
    x = u32_to_f32(base | 0x00008000)
    self.assertEqual(f32_to_u32(float_to_bf16(x)), inf_u32)
    self.assertEqual(f32_to_u32(torch.tensor([x], dtype=torch.bfloat16).item()), inf_u32)

  @given(strat.floats(width=32, allow_subnormal=True, allow_nan=True, allow_infinity=True))
  def test_truncate_fp8e4m3(self, x):
    if math.isnan(x): np.testing.assert_equal(truncate[dtypes.fp8e4m3](x), x)
    elif math.isinf(x): np.testing.assert_equal(truncate[dtypes.fp8e4m3](x), math.copysign(math.nan, x))
    elif x > FP8E4M3_MAX: np.testing.assert_equal(truncate[dtypes.fp8e4m3](x), FP8E4M3_MAX)
    elif x < -FP8E4M3_MAX: np.testing.assert_equal(truncate[dtypes.fp8e4m3](x), -FP8E4M3_MAX)
    else: np.testing.assert_equal(truncate[dtypes.fp8e4m3](x), torch.tensor(x, dtype=torch.float8_e4m3fn).float().item())

  @given(strat.floats(width=32, allow_subnormal=True, allow_nan=True, allow_infinity=True))
  def test_truncate_fp8e5m2(self, x):
    if math.isnan(x): np.testing.assert_equal(truncate[dtypes.fp8e5m2](x), x)
    elif math.isinf(x): np.testing.assert_equal(truncate[dtypes.fp8e5m2](x), x)
    elif x > FP8E5M2_MAX: np.testing.assert_equal(truncate[dtypes.fp8e5m2](x), FP8E5M2_MAX)
    elif x < -FP8E5M2_MAX: np.testing.assert_equal(truncate[dtypes.fp8e5m2](x), -FP8E5M2_MAX)
    else: np.testing.assert_equal(truncate[dtypes.fp8e5m2](x), torch.tensor(x, dtype=torch.float8_e5m2).float().item())

class TestTypePromotion(unittest.TestCase):
  @given(strat.sampled_from(core_dtypes))
  def test_self_promo_to_self(self, dtype):
    assert least_upper_dtype(dtype) == dtype
    assert least_upper_dtype(dtype, dtype) == dtype
    assert least_upper_dtype(dtype, dtype, dtype) == dtype

  @given(strat.sampled_from(core_dtypes), strat.sampled_from(core_dtypes))
  def test_promo_resulted_higher_than_inputs(self, dtype1, dtype2):
    result = least_upper_dtype(dtype1, dtype2)
    assert not (result < dtype1) and not (result < dtype2)

  def test_dtype_promo(self):
    assert least_upper_dtype(dtypes.bool, dtypes.int8) == dtypes.int8
    assert least_upper_dtype(dtypes.int8, dtypes.uint8) == dtypes.int16
    assert least_upper_dtype(dtypes.uint8, dtypes.int16) == dtypes.int16
    assert least_upper_dtype(dtypes.int16, dtypes.uint16) == dtypes.int32
    assert least_upper_dtype(dtypes.uint16, dtypes.int32) == dtypes.int32
    assert least_upper_dtype(dtypes.int32, dtypes.uint32) == dtypes.int64
    assert least_upper_dtype(dtypes.uint32, dtypes.int64) == dtypes.int64
    # similar to jax but we don't use weak type
    assert least_upper_dtype(dtypes.int64, dtypes.uint64) == dtypes.uint64  # is this correct?
    assert least_upper_dtype(dtypes.float16, dtypes.float32) == dtypes.float32
    assert least_upper_dtype(dtypes.float32, dtypes.float64) == dtypes.float64

    assert least_upper_dtype(dtypes.bool, dtypes.float32) == dtypes.float32
    assert least_upper_dtype(dtypes.bool, dtypes.float64) == dtypes.float64
    assert least_upper_dtype(dtypes.float16, dtypes.int64) == dtypes.float16
    assert least_upper_dtype(dtypes.float16, dtypes.uint64) == dtypes.float16
    assert least_upper_dtype(dtypes.fp8e4m3, dtypes.fp8e5m2) == dtypes.half
    assert least_upper_dtype(dtypes.fp8e4m3, dtypes.bfloat16) == dtypes.bfloat16
    assert least_upper_dtype(dtypes.fp8e5m2, dtypes.bfloat16) == dtypes.bfloat16
    assert least_upper_dtype(dtypes.fp8e4m3, dtypes.float16) == dtypes.float16
    assert least_upper_dtype(dtypes.fp8e5m2, dtypes.float16) == dtypes.float16
    assert least_upper_dtype(dtypes.fp8e4m3, dtypes.int64) == dtypes.fp8e4m3
    assert least_upper_dtype(dtypes.fp8e4m3, dtypes.uint64) == dtypes.fp8e4m3
    assert least_upper_dtype(dtypes.fp8e5m2, dtypes.int64) == dtypes.fp8e5m2
    assert least_upper_dtype(dtypes.fp8e5m2, dtypes.uint64) == dtypes.fp8e5m2

if __name__ == '__main__':
  unittest.main()
