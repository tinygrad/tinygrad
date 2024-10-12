import unittest, operator, subprocess, math
import numpy as np
import torch
from typing import Any, List
from tinygrad.helpers import getenv, DEBUG, CI
from tinygrad.dtype import DType, DTYPES_DICT, ImageDType, PtrDType, least_upper_float, least_upper_dtype, truncate_fp16
from tinygrad import Device, Tensor, dtypes
from tinygrad.tensor import _to_np_dtype
from hypothesis import given, settings, strategies as strat
from test.helpers import is_dtype_supported, rand_for_dtype

settings.register_profile("my_profile", max_examples=200, deadline=None, derandomize=getenv("DERANDOMIZE_CI", False))
settings.load_profile("my_profile")

core_dtypes = list([v for k,v in DTYPES_DICT.items() if k != 'pyint'])
if Device.DEFAULT == "CPU": core_dtypes.remove(dtypes.bfloat16)  # NOTE: this is for teenygrad, don't remove
dtype_ints = [dt for dt in core_dtypes if dtypes.is_int(dt) and is_dtype_supported(dt)]
dtype_floats = [dt for dt in core_dtypes if dtypes.is_float(dt) and is_dtype_supported(dt)]

def get_available_cast_dtypes(dtype: DType) -> List[DType]:
  if not is_dtype_supported(dtype): return []
  # dont cast internal dtypes
  return [v for k, v in DTYPES_DICT.items() if v != dtype and is_dtype_supported(v) and not k.startswith("_") and k != 'pyint']

def _test_to_np(a:Tensor, np_dtype, target):
  if DEBUG >= 2: print(a)
  na = a.numpy()
  if DEBUG >= 2: print(na, na.dtype, a.lazydata.base.realized)
  try:
    assert na.dtype == np_dtype
    np.testing.assert_allclose(na, target)
  except AssertionError as e:
    raise AssertionError(f"\ntensor {a.numpy()} does not match target {target} with np_dtype {np_dtype}") from e

def _assert_eq(tensor:Tensor, target_dtype:DType, target):
  if DEBUG >= 2: print(tensor.numpy())
  try:
    assert tensor.dtype == target_dtype
    np.testing.assert_allclose(tensor.numpy(), target, rtol={dtypes.float16:1e-3, dtypes.bfloat16:1e-2}.get(target_dtype, 1e-7))
  except AssertionError as e:
    raise AssertionError(f"\ntensor {tensor.numpy()} dtype {tensor.dtype} does not match target {target} with dtype {target_dtype}") from e

def _test_op(fxn, target_dtype:DType, target):
  _assert_eq(fxn(), target_dtype, target)
def _test_cast(a:Tensor, target_dtype:DType):
  if a.is_floating_point() and dtypes.is_unsigned(target_dtype):
    # converting negative float to unsigned integer is undefined
    a = a.abs()
  if target_dtype == dtypes.half and Device.DEFAULT == "PYTHON":
    # TODO: struct.pack cannot pack value > 65504 (max of half) into e format
    a = (a > 65504).where(65504, a)
  if CI and Device.DEFAULT == "CLANG" and (target_dtype, a.dtype) in [(dtypes.double, dtypes.half), (dtypes.half, dtypes.double)]:
    # TODO: cast between double and half are broken https://github.com/tinygrad/tinygrad/issues/4084
    return

  _test_op(lambda: a.cast(target_dtype), target_dtype, list(a.numpy().astype(_to_np_dtype(target_dtype))))
def _test_bitcast(a:Tensor, target_dtype:DType, target=None):
  if target_dtype == dtypes.bfloat16: raise unittest.SkipTest("no test for bf16 bitcast yet")
  _test_op(lambda: a.bitcast(target_dtype), target_dtype, target or a.numpy().view(_to_np_dtype(target_dtype)).tolist())

class TestDType(unittest.TestCase):
  DTYPE: Any = None
  DATA: Any = None
  @classmethod
  def setUpClass(cls):
    if not cls.DTYPE or not is_dtype_supported(cls.DTYPE): raise unittest.SkipTest("dtype not supported")
    cls.DATA = rand_for_dtype(cls.DTYPE, 10)
  def setUp(self):
    if self.DTYPE is None: raise unittest.SkipTest("base class")

  def test_to_np(self):
    _test_to_np(Tensor(self.DATA, dtype=self.DTYPE), _to_np_dtype(self.DTYPE), np.array(self.DATA, dtype=_to_np_dtype(self.DTYPE)))

  def test_casts_to(self): list(map(
    lambda dtype: _test_cast(Tensor(self.DATA, dtype=dtype), self.DTYPE),
    get_available_cast_dtypes(self.DTYPE)
  ))
  def test_casts_from(self): list(map(
    lambda dtype: _test_cast(Tensor(self.DATA, dtype=self.DTYPE), dtype),
    get_available_cast_dtypes(self.DTYPE)
  ))

  def test_same_size_ops(self):
    list(map(
      lambda dtype: _test_ops(a_dtype=self.DTYPE, b_dtype=dtype) if dtype.itemsize == self.DTYPE.itemsize else None,
      get_available_cast_dtypes(self.DTYPE)
    ))
  def test_upcast_ops(self):
    list(map(
      lambda dtype: _test_ops(a_dtype=self.DTYPE, b_dtype=dtype) if dtype.itemsize > self.DTYPE.itemsize else None,
      get_available_cast_dtypes(self.DTYPE)
  ))
  def test_upcast_to_ops(self):
    list(map(
      lambda dtype: _test_ops(a_dtype=dtype, b_dtype=self.DTYPE) if dtype.itemsize < self.DTYPE.itemsize else None,
      get_available_cast_dtypes(self.DTYPE)
  ))
  def test_bitcast(self):
    if Device.DEFAULT == "WEBGL": raise unittest.SkipTest("no bitcast in WebGL GLSL")
    if self.DTYPE == dtypes.bool: raise unittest.SkipTest("no bools in bitcast")
    list(map(
      lambda dtype:
        _test_bitcast(Tensor(self.DATA[:8], dtype=self.DTYPE), dtype) if dtype != dtypes.bool else None,
     get_available_cast_dtypes(self.DTYPE)
    ))

  def test_dtypes_fields(self):
    fields = dtypes.fields()
    self.assertTrue(all(isinstance(value, DType) for value in fields.values()))
    self.assertTrue(all(issubclass(_to_np_dtype(value), np.generic) for value in fields.values() if _to_np_dtype(value) is not None))

  def test_resulting_and_init_dtypes_match(self):
    dtypes = list(map(np.dtype, ["bool", "uint8", "int8", "int16", "int32", "int64", "float32", "float64"]))
    data = [1., 2., 0., 0.5, -1.5, 5.25]
    for dt in dtypes:
      arr = np.asarray(data).astype(dt)
      tin = Tensor(arr).numpy()
      tor = torch.as_tensor(arr).detach().numpy()
      assert dt == tin.dtype == tor.dtype, f"dtype mismatch: expected={dt} | tinygrad={tin.dtype} | torch={tor.dtype}"
      np.testing.assert_allclose(tin, tor, atol=1e-6, rtol=1e-3)

  def test_finfo(self):
    if self.DTYPE not in [dtypes.float16, dtypes.bfloat16, dtypes.float32, dtypes.float64]: return
    info = np.finfo(_to_np_dtype(self.DTYPE))
    assert info.bits == self.DTYPE.itemsize*8
    assert info.nexp == dtypes.finfo(self.DTYPE)[0]
    assert info.nmant == dtypes.finfo(self.DTYPE)[1]

def _test_ops(a_dtype:DType, b_dtype:DType, target_dtype=None):
  target_dtype = target_dtype or least_upper_dtype(a_dtype, b_dtype)
  if not is_dtype_supported(a_dtype) or not is_dtype_supported(b_dtype) or not is_dtype_supported(target_dtype): return
  if a_dtype == dtypes.bool or b_dtype == dtypes.bool: return
  _assert_eq(Tensor([1,2,3,4], dtype=a_dtype)+Tensor([1,2,3,4], dtype=b_dtype), target_dtype, [2,4,6,8])
  _assert_eq((Tensor([1], dtype=a_dtype).cast(b_dtype)+Tensor([1], dtype=a_dtype).cast(b_dtype)).cast(a_dtype), a_dtype, [2])
  _assert_eq(Tensor([1,2,3,4], dtype=a_dtype)*Tensor([1,2,3,4], dtype=b_dtype), target_dtype, [1,4,9,16])
  _assert_eq(Tensor([[1,2],[3,4]], dtype=a_dtype)@Tensor.eye(2, dtype=b_dtype), target_dtype, [[1,2],[3,4]])
  _assert_eq(Tensor([1,1,1,1], dtype=a_dtype)+Tensor.ones((4,4), dtype=b_dtype), target_dtype, 2*Tensor.ones(4,4).numpy())

@unittest.skipUnless(is_dtype_supported(dtypes.bfloat16), "bfloat16 not supported")
class TestBFloat16(unittest.TestCase):
  def test_bf16_creation_numpy(self):
    data = [-1, 1, 2]
    t = Tensor(data, dtype=dtypes.bfloat16)
    assert t.dtype == dtypes.bfloat16
    tnp = t.numpy()
    assert tnp.dtype == np.float32
    np.testing.assert_allclose(tnp, np.array(data))

  def test_bf16_ones(self):
    t = Tensor.ones(3, 5, dtype=dtypes.bfloat16)
    assert t.dtype == dtypes.bfloat16
    np.testing.assert_allclose(t.numpy(), np.ones((3, 5)))

  def test_bf16_eye(self):
    t = Tensor.eye(3, dtype=dtypes.bfloat16)
    assert t.dtype == dtypes.bfloat16
    np.testing.assert_allclose(t.numpy(), np.eye(3))

@unittest.skipUnless(is_dtype_supported(dtypes.bfloat16), "bfloat16 not supported")
class TestBFloat16DType(unittest.TestCase):
  def test_bf16_to_float(self):
    _test_cast(Tensor([100000], dtype=dtypes.bfloat16), dtypes.float32)

  def test_float_to_bf16(self):
    _test_cast(Tensor([100000], dtype=dtypes.float32), dtypes.bfloat16)

  def test_bf16(self):
    t = Tensor([10000, -1, -1000, -10000, 20]).cast(dtypes.bfloat16)
    t.realize()
    back = t.cast(dtypes.float32)
    assert tuple(back.numpy().tolist()) == (9984., -1, -1000, -9984, 20)

@unittest.skipUnless(is_dtype_supported(dtypes.bfloat16), "bfloat16 not supported")
class TestBFloat16DTypeCast(unittest.TestCase):
  def test_f16_to_bf16_conversion(self):
    original_tensor = Tensor([1.0, 2.0, 3.0], dtype=dtypes.float16)
    converted_tensor = original_tensor.cast(dtypes.bfloat16)
    self.assertEqual(converted_tensor.dtype, dtypes.bfloat16)
    back_to_float32 = converted_tensor.cast(dtypes.float32)
    original_to_float32 = original_tensor.cast(dtypes.float32)
    np.testing.assert_allclose(back_to_float32.numpy(), original_to_float32.numpy(), rtol=1e-2, atol=1e-3)

  def test_f16_to_bf16_edge_cases(self):
    edge_cases = Tensor([0.0, -0.0, float('inf'), float('-inf'), float('nan')], dtype=dtypes.float16)
    converted = edge_cases.cast(dtypes.bfloat16).cast(dtypes.float32)
    np.testing.assert_equal(converted.numpy(), edge_cases.cast(dtypes.float32).numpy())

  def test_f16_to_bf16_range_precision(self):
    large_value = Tensor([65504.0], dtype=dtypes.float16)  # Max representable in float16
    small_value = Tensor([6.1035e-5], dtype=dtypes.float16)  # Smallest positive normal float16
    large_converted = large_value.cast(dtypes.bfloat16).cast(dtypes.float32)
    small_converted = small_value.cast(dtypes.bfloat16).cast(dtypes.float32)
    np.testing.assert_allclose(large_converted.numpy(), large_value.cast(dtypes.float32).numpy(), rtol=1e-2, atol=1e-3)
    np.testing.assert_equal(small_converted.numpy(), small_value.cast(dtypes.float32).numpy())

  def test_f16_to_bf16_randomized(self):
    np.random.seed(42)  # For reproducibility
    random_values = Tensor(np.random.uniform(-65504, 65504, 1000), dtype=dtypes.float16)
    converted = random_values.cast(dtypes.bfloat16).cast(dtypes.float32)
    np.testing.assert_allclose(converted.numpy(), random_values.cast(dtypes.float32).numpy(), rtol=1e-2, atol=1e-3)

class TestHalfDType(TestDType): DTYPE = dtypes.half

class TestFloatDType(TestDType):
  DTYPE = dtypes.float

  def test_float_to_uint(self):
    _test_op(lambda: Tensor([-0.9, -0.3, 1.2], dtype=dtypes.float32).cast(dtypes.uint32), dtypes.uint32,
             [0, 0, 1])

class TestDoubleDType(TestDType):
  DTYPE = dtypes.double
  @unittest.skipIf((CI and Device.DEFAULT in {"CUDA", "NV"}) or getenv("PTX"), "conversion not supported on CI CUDA and PTX")  # TODO: why not?
  def test_float64_increased_precision(self):
    for func in [
      lambda t: t.exp(),
      lambda t: t.exp2(),
      lambda t: t.log(),
      lambda t: t.log2(),
      lambda t: t.sqrt(),
      lambda t: t.rsqrt(),
      lambda t: t.sin(),
      lambda t: t.cos(),
      lambda t: t.tan(),
      lambda t: t.sigmoid(),
    ]:
      a = [2, 3, 4]
      np.testing.assert_allclose(func(Tensor(a, dtype=self.DTYPE)).numpy(), func(torch.tensor(a, dtype=torch.float64)), rtol=1e-12, atol=1e-12)

  def test_float64_to_float32_cast_inf(self):
    _test_op(lambda: Tensor([3.4e40, 3.4e38, 1, 0], dtype=dtypes.float64).cast(dtypes.float32),
             dtypes.float32, [float('inf'), 3.4e38, 1, 0])


class TestInt8DType(TestDType):
  DTYPE = dtypes.int8
  @unittest.skipIf(getenv("CUDA",0)==1 or getenv("PTX", 0)==1, "cuda saturation works differently")
  def test_int8_to_uint8_negative(self):
    _test_op(lambda: Tensor([-1, -2, -3, -4], dtype=dtypes.int8).cast(dtypes.uint8), dtypes.uint8, [255, 254, 253, 252])

  def test_int8_to_uint16_negative(self):
    _test_op(lambda: Tensor([-1, -2, -3, -4], dtype=dtypes.int8).cast(dtypes.uint16), dtypes.uint16, [2**16-1, 2**16-2, 2**16-3, 2**16-4])

class TestUint8DType(TestDType):
  DTYPE = dtypes.uint8
  @unittest.skipIf(getenv("CUDA",0)==1 or getenv("PTX", 0)==1, "cuda saturation works differently")
  def test_uint8_to_int8_overflow(self):
    _test_op(lambda: Tensor([255, 254, 253, 252], dtype=dtypes.uint8).cast(dtypes.int8), dtypes.int8, [-1, -2, -3, -4])

@unittest.skipIf(Device.DEFAULT == "WEBGL", "No bitcast on WebGL")
class TestBitCast(unittest.TestCase):
  @given(strat.sampled_from(dtype_ints + dtype_floats), strat.sampled_from(dtype_ints + dtype_floats))
  def test_shape_change_bitcast(self, dt1, dt2):
    if dt2 == dtypes.bfloat16: raise unittest.SkipTest("no test for bf16 bitcast yet")
    data = rand_for_dtype(dt1, 32).reshape(2, 2, 8)
    _test_op(lambda: Tensor(data, dtype=dt1).bitcast(dt2), dt2, data.view(_to_np_dtype(dt2)).tolist())

  def test_shape_change_bitcast_exceptions(self):
    with self.assertRaises(RuntimeError):
      # should fail because 3 int8 is 3 bytes but float16 is two and 3 isn't a multiple of 2
      Tensor.empty((3,), dtype=dtypes.int8).bitcast(dtypes.float16)

    with self.assertRaises(RuntimeError):
      # should fail because backprop through bitcast is undefined
      Tensor.empty((4,), dtype=dtypes.int8, requires_grad=True).bitcast(dtypes.float16)

  def test_bitcast_float_to_int32(self):
    a = Tensor([1.,2,3])
    b = a.bitcast(dtypes.int32)
    assert b.numpy()[0] == 0x3f800000

  def test_bitcast_upcasted(self):
    a = Tensor.zeros(100, 4, dtype=dtypes.int32).contiguous() + 0x3f800000
    b = a.bitcast(dtypes.float32)
    assert b.numpy()[0,0] == 1.

class TestInt16DType(TestDType): DTYPE = dtypes.int16

class TestUint16DType(TestDType):
  DTYPE = dtypes.uint16

  def test_uint16_to_int8_overflow(self):
    _test_op(lambda: Tensor([2**16-1, 2**16-2, 1, 0], dtype=dtypes.uint16).cast(dtypes.int8), dtypes.int8, [-1, -2, 1, 0])

class TestInt32DType(TestDType): DTYPE = dtypes.int32
class TestUint32DType(TestDType): DTYPE = dtypes.uint32

class TestInt64DType(TestDType): DTYPE = dtypes.int64
class TestUint64DType(TestDType):
  DTYPE = dtypes.uint64
  def test_uint64_load(self):
    assert Tensor(2**64 - 1, dtype=dtypes.uint64).numpy() == 2**64 - 1

class TestBoolDType(TestDType): DTYPE = dtypes.bool

class TestImageDType(unittest.TestCase):
  def test_image_scalar(self):
    assert dtypes.imagef((10,10)).scalar() == dtypes.float32
    assert dtypes.imageh((10,10)).scalar() == dtypes.float32
  def test_image_vec(self):
    assert dtypes.imagef((10,10)).vec(4) == dtypes.float32.vec(4)
    assert dtypes.imageh((10,10)).vec(4) == dtypes.float32.vec(4)

class TestEqStrDType(unittest.TestCase):
  def test_image_ne(self):
    if ImageDType is None: raise unittest.SkipTest("no ImageDType support")
    assert dtypes.float == dtypes.float32, "float doesn't match?"
    assert dtypes.imagef((1,2,4)) != dtypes.imageh((1,2,4)), "different image dtype doesn't match"
    assert dtypes.imageh((1,2,4)) != dtypes.imageh((1,4,2)), "different shape doesn't match"
    assert dtypes.imageh((1,2,4)) == dtypes.imageh((1,2,4)), "same shape matches"
    assert isinstance(dtypes.imageh((1,2,4)), ImageDType)
  def test_ptr_ne(self):
    if PtrDType is None: raise unittest.SkipTest("no PtrDType support")
    # TODO: is this the wrong behavior?
    assert PtrDType(dtypes.float32) == dtypes.float32
    assert not (PtrDType(dtypes.float32) != dtypes.float32)
    assert PtrDType(dtypes.float32) == PtrDType(dtypes.float32)
    assert not (PtrDType(dtypes.float32) != PtrDType(dtypes.float32))
    #assert PtrDType(dtypes.float32) != dtypes.float32
  def test_strs(self):
    if PtrDType is None: raise unittest.SkipTest("no PtrDType support")
    self.assertEqual(str(dtypes.imagef((1,2,4))), "dtypes.imagef((1, 2, 4))")
    self.assertEqual(str(PtrDType(dtypes.float32)), "PtrDType(dtypes.float)")

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
      elif dtypes.is_int(dt):
        info = np.iinfo(_to_np_dtype(dt))
        np.testing.assert_equal(dtypes.min(dt), info.min)
        np.testing.assert_equal(dtypes.max(dt), info.max)
      else:
        assert dt == dtypes.bool, dt
        np.testing.assert_equal(dtypes.min(dt), False)
        np.testing.assert_equal(dtypes.max(dt), True)

  def test_truncate_fp16(self):
    self.assertEqual(truncate_fp16(1), 1)
    self.assertEqual(truncate_fp16(65504), 65504)
    self.assertEqual(truncate_fp16(65519.999), 65504)
    self.assertEqual(truncate_fp16(65520), math.inf)

class TestTypeSpec(unittest.TestCase):
  def setUp(self):
    self.old_default_int, self.old_default_float = dtypes.default_int, dtypes.default_float
  def tearDown(self):
    dtypes.default_int, dtypes.default_float = self.old_default_int, self.old_default_float

  def test_set_dtype_default(self):
    for default_int in [dtypes.int8, dtypes.int16, dtypes.int32, dtypes.int64]:
      dtypes.default_int = default_int
      assert dtypes.default_int == default_int

    for default_float in [dtypes.float16, dtypes.bfloat16, dtypes.float32, dtypes.float64]:
      dtypes.default_float = default_float
      assert dtypes.default_float == default_float

  def test_env_set_default_float(self):
    # check default
    subprocess.run(['python3 -c "from tinygrad import dtypes; assert dtypes.default_float == dtypes.float"'],
                    shell=True, check=True)
    # check change
    subprocess.run(['DEFAULT_FLOAT=HALF python3 -c "from tinygrad import dtypes; assert dtypes.default_float == dtypes.half"'],
                    shell=True, check=True)
    # check invalid
    with self.assertRaises(subprocess.CalledProcessError):
      subprocess.run(['DEFAULT_FLOAT=INT32 python3 -c "from tinygrad import dtypes"'],
                      shell=True, check=True)

    with self.assertRaises(subprocess.CalledProcessError):
      subprocess.run(['DEFAULT_FLOAT=TYPO python3 -c "from tinygrad import dtypes"'],
                      shell=True, check=True)

  def test_dtype_str_arg(self):
    n = np.random.normal(0, 1, (10, 10)).astype(np.float32)
    tested = 0
    for dtype_str, dtype in [
      ("bool", dtypes.bool), ("int8", dtypes.int8), ("int", dtypes.int), ("uint32", dtypes.uint32), ("float32", dtypes.float32)]:
      np.testing.assert_equal(Tensor(n, dtype=dtype_str).numpy(), Tensor(n, dtype=dtype).numpy())
      np.testing.assert_equal(Tensor(n).cast(dtype_str).numpy(), Tensor(n).cast(dtype).numpy())
      if dtype.itemsize == 4:
        np.testing.assert_equal(Tensor(n).bitcast(dtype_str).numpy(), Tensor(n).bitcast(dtype).numpy())
        tested += 1
    assert tested == 3

    with self.assertRaises(AttributeError): Tensor([1, 2, 3], dtype="nonexistdtype")
    with self.assertRaises(AttributeError): Tensor([1, 2, 3], dtype="")

    np.testing.assert_equal(Tensor(n).sum(acc_dtype="int16").numpy(), Tensor(n).sum(acc_dtype=dtypes.int16).numpy())

  @given(strat.sampled_from(dtype_ints), strat.sampled_from(dtype_floats))
  def test_creation(self, default_int, default_float):
    dtypes.default_int, dtypes.default_float = default_int, default_float
    _assert_eq(Tensor(True), dtypes.bool, True)
    _assert_eq(Tensor(None), dtypes.default_float, [])
    _assert_eq(Tensor(2), dtypes.default_int, 2)
    _assert_eq(Tensor(2.34), dtypes.default_float, 2.34)
    _assert_eq(Tensor([]), dtypes.default_float, [])
    _assert_eq(Tensor([1]), dtypes.default_int, [1])
    _assert_eq(Tensor([1.1]), dtypes.default_float, [1.1])

    _assert_eq(Tensor.eye(0), dtypes.default_float, np.eye(0))
    _assert_eq(Tensor.eye(3), dtypes.default_float, np.eye(3))
    _assert_eq(Tensor.eye(3, dtype=dtypes.int64), dtypes.int64, np.eye(3))
    if is_dtype_supported(dtypes.float16):
      _assert_eq(Tensor.eye(3, dtype=dtypes.float16), dtypes.float16, np.eye(3))

  @given(strat.sampled_from(dtype_ints), strat.sampled_from(dtype_floats))
  def test_full(self, default_int, default_float):
    dtypes.default_int, dtypes.default_float = default_int, default_float

    _assert_eq(Tensor.zeros((2, 3)), dtypes.default_float, np.zeros((2, 3)))
    _assert_eq(Tensor.zeros((2, 3), dtype=dtypes.int64), dtypes.int64, np.zeros((2, 3)))
    if is_dtype_supported(dtypes.float16):
      _assert_eq(Tensor.zeros((2, 3), dtype=dtypes.float16), dtypes.float16, np.zeros((2, 3)))

    _assert_eq(Tensor.ones((2, 3)), dtypes.default_float, np.ones((2, 3)))
    _assert_eq(Tensor.ones((2, 3), dtype=dtypes.int64), dtypes.int64, np.ones((2, 3)))
    if is_dtype_supported(dtypes.float16):
      _assert_eq(Tensor.ones((2, 3), dtype=dtypes.float16), dtypes.float16, np.ones((2, 3)))

    _assert_eq(Tensor.full((2, 3), 3.0), dtypes.default_float, np.full((2, 3), 3.0))
    _assert_eq(Tensor.full((2, 3), 3), dtypes.default_int, np.full((2, 3), 3))
    _assert_eq(Tensor.full((2, 3), True), dtypes.bool, np.full((2, 3), True))
    _assert_eq(Tensor.full((2, 3), 3, dtype=dtypes.int64), dtypes.int64, np.full((2, 3), 3))
    _assert_eq(Tensor.full((2, 3), 3.0, dtype=dtypes.int64), dtypes.int64, np.full((2, 3), 3))
    if is_dtype_supported(dtypes.float16):
      _assert_eq(Tensor.full((2, 3), 3, dtype=dtypes.float16), dtypes.float16, np.full((2, 3), 3))
      _assert_eq(Tensor.full((2, 3), 3.0, dtype=dtypes.float16), dtypes.float16, np.full((2, 3), 3))

  @given(strat.sampled_from(dtype_ints), strat.sampled_from(dtype_floats))
  def test_reduce_0d_default(self, default_int, default_float):
    dtypes.default_int, dtypes.default_float = default_int, default_float
    _assert_eq(Tensor.ones((2,3,0)).sum(2), dtypes.default_float, np.zeros((2, 3)))
    # TODO: what should this one be?
    # _assert_eq(Tensor.ones((2,3,0), dtype=dtypes.default_int).sum(2), dtypes.default_int, np.zeros((2, 3)))
    _assert_eq(Tensor.ones((2,3,0), dtype=dtypes.int32).sum(2), dtypes.int32, np.zeros((2, 3)))

  @given(strat.sampled_from(dtype_ints), strat.sampled_from(dtype_floats))
  def test_arange(self, default_int, default_float):
    dtypes.default_int, dtypes.default_float = default_int, default_float

    _assert_eq(Tensor.arange(5), dtypes.default_int, np.arange(5))
    _assert_eq(Tensor.arange(120), dtypes.default_int, np.arange(120))
    _assert_eq(Tensor.arange(5.0), dtypes.default_float, np.arange(5))
    _assert_eq(Tensor.arange(5, dtype=dtypes.int16), dtypes.int16, np.arange(5))
    _assert_eq(Tensor.arange(5, dtype=dtypes.int64), dtypes.int64, np.arange(5))
    if is_dtype_supported(dtypes.float16):
      _assert_eq(Tensor.arange(5, dtype=dtypes.float16), dtypes.float16, np.arange(5))
    _assert_eq(Tensor.arange(3, 9, 0.7), dtypes.default_float, np.arange(3, 9, 0.7))
    _assert_eq(Tensor.arange(3, 8.5, 3), dtypes.default_float, np.arange(3, 8.5, 3))
    # stop-start and step have different signs
    _assert_eq(Tensor.arange(3, 5, -2), dtypes.default_int, np.arange(3, 5, -2))
    _assert_eq(Tensor.arange(5.0, 3.0), dtypes.default_float, np.arange(5.0, 3.0))

  @given(strat.sampled_from(core_dtypes), strat.sampled_from([operator.gt, operator.ge, operator.le, operator.lt, operator.eq, operator.ne]))
  def test_bool_ops(self, dtype, op):
    assert op(Tensor.ones(4, 4, dtype=dtype), Tensor.ones(4, 4, dtype=dtype)).dtype == dtypes.bool

  @given(strat.sampled_from(core_dtypes), strat.sampled_from(dtype_ints), strat.sampled_from(dtype_floats))
  def test_functions_return_index(self, dtype, default_int, default_float):
    dtypes.default_int, dtypes.default_float = default_int, default_float
    assert Tensor([0, 1], dtype=dtype).argmax().dtype == dtypes.int32
    assert Tensor([0, 1], dtype=dtype).argmin().dtype == dtypes.int32
    assert Tensor([0, 1], dtype=dtype).multinomial().dtype == dtypes.int32

  @given(strat.sampled_from(core_dtypes), strat.sampled_from(dtype_ints))
  def test_tensor_indexing_returns_same_dtype(self, data_dtype, indices_dtype):
    X_data =  Tensor.ones(60000, 1, 28, 28, dtype=data_dtype)
    indices =  Tensor.randint(512, high=X_data.shape[0]).cast(indices_dtype)
    assert X_data[indices].dtype == X_data.dtype

  @given(strat.sampled_from(core_dtypes), strat.sampled_from(dtype_ints))
  def test_gather_returns_same_dtype(self, data_dtype, indices_dtype):
    X_data = Tensor([[1, 0], [0, 1]], dtype=data_dtype)
    indices = Tensor([[0, 0], [1, 0]], dtype=indices_dtype)
    assert X_data.gather(0, indices).dtype == X_data.dtype
    assert X_data.gather(1, indices).dtype == X_data.dtype

  @given(strat.sampled_from(dtype_floats), strat.sampled_from(dtype_floats))
  def test_attention_returns_same_dtype(self, data_dtype, default_float):
    dtypes.default_float = default_float
    query = Tensor.rand(32, 8, 128, 64, dtype=data_dtype)
    key = Tensor.rand(32, 8, 128, 64, dtype=data_dtype)
    value = Tensor.rand(32, 8, 128, 64, dtype=data_dtype)
    mask = (Tensor.rand(32, 8, 128, 128) < 0.5)
    assert query.scaled_dot_product_attention(key, value, is_causal=True).dtype == data_dtype
    assert query.scaled_dot_product_attention(key, value, is_causal=True, dropout_p=0.3).dtype == data_dtype
    assert query.scaled_dot_product_attention(key, value, is_causal=False).dtype == data_dtype
    assert query.scaled_dot_product_attention(key, value, attn_mask=mask).dtype == data_dtype

class TestTypePromotion(unittest.TestCase):
  @given(strat.sampled_from(core_dtypes))
  def test_self_promo_to_self(self, dtype):
    assert least_upper_dtype(dtype) == dtype
    assert least_upper_dtype(dtype, dtype) == dtype
    assert least_upper_dtype(dtype, dtype, dtype) == dtype

  @given(strat.sampled_from(core_dtypes), strat.sampled_from(core_dtypes))
  def test_promo_resulted_higher_than_inputs(self, dtype1, dtype2):
    result = least_upper_dtype(dtype1, dtype2)
    assert result >= dtype1 and result >= dtype2

  def test_dtype_promo(self):
    assert least_upper_dtype(dtypes.bool, dtypes.int8) == dtypes.int8
    assert least_upper_dtype(dtypes.int8, dtypes.uint8) == dtypes.int16
    assert least_upper_dtype(dtypes.uint8, dtypes.int16) == dtypes.int16
    assert least_upper_dtype(dtypes.int16, dtypes.uint16) == dtypes.int32
    assert least_upper_dtype(dtypes.uint16, dtypes.int32) == dtypes.int32
    assert least_upper_dtype(dtypes.int32, dtypes.uint32) == dtypes.int64
    assert least_upper_dtype(dtypes.uint32, dtypes.int64) == dtypes.int64
    # similar to jax but we don't use weak type
    assert least_upper_dtype(dtypes.int64, dtypes.uint64) == dtypes.float16
    assert least_upper_dtype(dtypes.float16, dtypes.float32) == dtypes.float32
    assert least_upper_dtype(dtypes.float32, dtypes.float64) == dtypes.float64

    assert least_upper_dtype(dtypes.bool, dtypes.float32) == dtypes.float32
    assert least_upper_dtype(dtypes.bool, dtypes.float64) == dtypes.float64
    assert least_upper_dtype(dtypes.float16, dtypes.int64) == dtypes.float16
    assert least_upper_dtype(dtypes.float16, dtypes.uint64) == dtypes.float16

  @given(strat.sampled_from(dtype_floats))
  def test_float_to_float(self, dt):
    assert least_upper_float(dt) == dt

class TestAutoCastType(unittest.TestCase):
  def setUp(self):
    self.old_default_int, self.old_default_float = dtypes.default_int, dtypes.default_float
  def tearDown(self):
    dtypes.default_int, dtypes.default_float = self.old_default_int, self.old_default_float

  @given(strat.sampled_from([d for d in core_dtypes if dtypes.is_int(d) and is_dtype_supported(d)]))
  def test_int_to_float_unary_func(self, dtype):
    for func in [
      lambda t: t.exp(),
      lambda t: t.exp2(),
      lambda t: t.log(),
      lambda t: t.log2(),
      lambda t: t.sqrt(),
      lambda t: t.rsqrt(),
      lambda t: t.sin(),
      lambda t: t.cos(),
      lambda t: t.tan(),
      lambda t: t.sigmoid(),
    ]:
      a = [2, 3, 4]
      # float16 can have larger precision errors
      np.testing.assert_allclose(func(Tensor(a, dtype=dtype)).numpy(), func(torch.tensor(a)), rtol=1e-3, atol=1e-3)

  @given(strat.sampled_from(core_dtypes))
  def test_broadcast_scalar(self, dt):
    assert (Tensor.ones(4, 4, dtype=dt) + 2.3).dtype == (dt if dtypes.is_float(dt) else dtypes.default_float)
    assert (Tensor.ones(4, 4, dtype=dt) + 2).dtype == (dt if dtypes.is_float(dt) or dtypes.is_int(dt) else dtypes.default_int)
    if Device.DEFAULT != "WEBGPU" and dt != dtypes.bool:
      assert (Tensor.ones(4, 4, dtype=dt) + True).dtype == dt

  def test_sum(self):
    assert (Tensor([0, 1], dtype=dtypes.bool)).sum().dtype == dtypes.int32
    assert (Tensor([0, 1], dtype=dtypes.int8)).sum().dtype == dtypes.int32
    assert (Tensor([0, 1], dtype=dtypes.int16)).sum().dtype == dtypes.int32
    assert (Tensor([0, 1], dtype=dtypes.int32)).sum().dtype == dtypes.int32
    assert (Tensor([0, 1], dtype=dtypes.int64)).sum().dtype == dtypes.int64
    assert (Tensor([0, 1], dtype=dtypes.uint8)).sum().dtype == dtypes.uint32
    assert (Tensor([0, 1], dtype=dtypes.uint16)).sum().dtype == dtypes.uint32
    assert (Tensor([0, 1], dtype=dtypes.uint32)).sum().dtype == dtypes.uint32
    assert (Tensor([0, 1], dtype=dtypes.uint64)).sum().dtype == dtypes.uint64
    assert (Tensor([0, 1], dtype=dtypes.float16)).sum().dtype == dtypes.float16
    #assert (Tensor([0, 1], dtype=dtypes.bfloat16)).sum().dtype == dtypes.bfloat16
    assert (Tensor([0, 1], dtype=dtypes.float32)).sum().dtype == dtypes.float32
    assert (Tensor([0, 1], dtype=dtypes.float64)).sum().dtype == dtypes.float64

  @unittest.skipUnless(is_dtype_supported(dtypes.float16), "need float16")
  def test_sum_acc_dtype(self):
    t = Tensor([40000, 40000], dtype=dtypes.float16)
    # default float16 sum returns in float16, overflowed in this case
    assert t.sum().dtype == dtypes.float16
    assert math.isinf(t.sum().numpy().item())
    # specifiying acc_dtype and it's not downcasted
    assert t.sum(acc_dtype=dtypes.float32).dtype == dtypes.float32
    np.testing.assert_allclose(t.sum(acc_dtype=dtypes.float32).numpy(), 80000)

  def test_prod_acc_dtype(self):
    t = Tensor([100, 200], dtype=dtypes.int32)
    assert t.prod().dtype == dtypes.int32
    np.testing.assert_allclose(t.prod().numpy(), 20000)
    assert t.prod(acc_dtype=dtypes.float32).dtype == dtypes.float32
    np.testing.assert_allclose(t.prod(acc_dtype=dtypes.float32).numpy(), 20000)

  def test_mean(self):
    assert (Tensor([0, 1], dtype=dtypes.bool)).mean().dtype == dtypes.float32
    assert (Tensor([0, 1], dtype=dtypes.int8)).mean().dtype == dtypes.float32
    assert (Tensor([0, 1], dtype=dtypes.int16)).mean().dtype == dtypes.float32
    assert (Tensor([0, 1], dtype=dtypes.int32)).mean().dtype == dtypes.float32
    assert (Tensor([0, 1], dtype=dtypes.int64)).mean().dtype == dtypes.float32
    assert (Tensor([0, 1], dtype=dtypes.uint8)).mean().dtype == dtypes.float32
    assert (Tensor([0, 1], dtype=dtypes.uint16)).mean().dtype == dtypes.float32
    assert (Tensor([0, 1], dtype=dtypes.uint32)).mean().dtype == dtypes.float32
    assert (Tensor([0, 1], dtype=dtypes.uint64)).mean().dtype == dtypes.float32
    assert (Tensor([0, 1], dtype=dtypes.float16)).mean().dtype == dtypes.float16
    #assert (Tensor([0, 1], dtype=dtypes.bfloat16)).mean().dtype == dtypes.bfloat16
    assert (Tensor([0, 1], dtype=dtypes.float32)).mean().dtype == dtypes.float32
    assert (Tensor([0, 1], dtype=dtypes.float64)).mean().dtype == dtypes.float64

  def test_cumsum(self):
    assert (Tensor([0, 1], dtype=dtypes.bool)).cumsum(0).dtype == dtypes.int32
    assert (Tensor([0, 1], dtype=dtypes.int8)).cumsum(0).dtype == dtypes.int32
    assert (Tensor([0, 1], dtype=dtypes.int16)).cumsum(0).dtype == dtypes.int32
    assert (Tensor([0, 1], dtype=dtypes.int32)).cumsum(0).dtype == dtypes.int32
    assert (Tensor([0, 1], dtype=dtypes.int64)).cumsum(0).dtype == dtypes.int64
    assert (Tensor([0, 1], dtype=dtypes.uint8)).cumsum(0).dtype == dtypes.uint32
    assert (Tensor([0, 1], dtype=dtypes.uint16)).cumsum(0).dtype == dtypes.uint32
    assert (Tensor([0, 1], dtype=dtypes.uint32)).cumsum(0).dtype == dtypes.uint32
    assert (Tensor([0, 1], dtype=dtypes.uint64)).cumsum(0).dtype == dtypes.uint64
    assert (Tensor([0, 1], dtype=dtypes.float16)).cumsum(0).dtype == dtypes.float16
    #assert (Tensor([0, 1], dtype=dtypes.bfloat16)).cumsum(0).dtype == dtypes.bfloat16
    assert (Tensor([0, 1], dtype=dtypes.float32)).cumsum(0).dtype == dtypes.float32
    assert (Tensor([0, 1], dtype=dtypes.float64)).cumsum(0).dtype == dtypes.float64

  @given(strat.sampled_from(core_dtypes), strat.sampled_from(core_dtypes), strat.sampled_from(core_dtypes))
  def test_matmul(self, dt1, dt2, acc_dt):
    t1 = Tensor([0, 1], dtype=dt1)
    t2 = Tensor([0, 1], dtype=dt2)
    assert (t1 @ t2).dtype == least_upper_dtype(dt1, dt2)
    # if acc_dtype is specified, return in acc_dtype
    assert (t1.matmul(t2, acc_dtype=acc_dt).dtype == acc_dt)

  @staticmethod
  def check_where_alternate_input_other(input_, other, data_type):
    assert (Tensor([True, False]).where(input_, other)).dtype == data_type
    assert (Tensor([True, False]).where(other, input_)).dtype == data_type

  @given(strat.sampled_from(core_dtypes), strat.sampled_from(core_dtypes))
  def test_where_no_scalar(self, dt1, dt2):
    self.check_where_alternate_input_other(Tensor(2, dtype=dt1), Tensor(3, dtype=dt2), least_upper_dtype(dt1, dt2))

  @given(strat.sampled_from(core_dtypes))
  def test_where_one_scalar(self, dt):
    t = Tensor(2, dtype=dt)
    self.check_where_alternate_input_other(t, 3.2, (dt if dtypes.is_float(dt) else dtypes.default_float))
    self.check_where_alternate_input_other(t, 3, (dt if dtypes.is_float(dt) or dtypes.is_int(dt) else dtypes.default_int))
    self.check_where_alternate_input_other(t, True, dt)

  def test_where_two_scalars(self):
    self.check_where_alternate_input_other(3.1, 3.2, dtypes.default_float)
    self.check_where_alternate_input_other(3.1, 3, dtypes.default_float)
    self.check_where_alternate_input_other(3.1, True, dtypes.default_float)
    self.check_where_alternate_input_other(3, 2, dtypes.default_int)
    self.check_where_alternate_input_other(3, True, dtypes.default_int)
    self.check_where_alternate_input_other(False, True, dtypes.bool)

  @given(strat.sampled_from(core_dtypes), strat.sampled_from(core_dtypes))
  def test_maximum(self, dt1, dt2):
    assert Tensor([0, 1, 2], dtype=dt1).maximum(Tensor([2, 0, 5], dtype=dt2)).dtype == least_upper_dtype(dt1, dt2)

  @given(strat.sampled_from(core_dtypes))
  def test_maximum_const(self, dt):
    assert Tensor([1, 2], dtype=dt).maximum(3.1).dtype == (dt if dtypes.is_float(dt) else dtypes.default_float)
    assert Tensor([1, 2], dtype=dt).maximum(3).dtype == (dt if dtypes.is_float(dt) or dtypes.is_int(dt) else dtypes.default_int)
    assert Tensor([1, 2], dtype=dt).maximum(True).dtype == dt

  def test_div(self):
    assert (Tensor([1, 2], dtype=dtypes.int32) / Tensor([2, 2], dtype=dtypes.int32)).dtype == dtypes.default_float
    assert (Tensor([1, 2], dtype=dtypes.int16) / Tensor([2, 2], dtype=dtypes.int32)).dtype == dtypes.default_float
    assert (Tensor([1, 2], dtype=dtypes.float32) / Tensor([2, 2], dtype=dtypes.float16)).dtype == dtypes.float32
    assert (Tensor([1, 2], dtype=dtypes.int32) / Tensor([2, 2], dtype=dtypes.float16)).dtype == dtypes.float16

  def test_div_const(self):
    assert (Tensor([1, 2], dtype=dtypes.int32) / 2).dtype == dtypes.default_float
    assert (Tensor([1, 2], dtype=dtypes.int32) / 2.0).dtype == dtypes.default_float
    assert (Tensor([1, 2], dtype=dtypes.float16) / 2).dtype == dtypes.float16
    assert (Tensor([1, 2], dtype=dtypes.float16) / 2.0).dtype == dtypes.float16

  def test_gradient_dtype(self):
    old_default_float = dtypes.default_float

    for default_dtype in [dtypes.float16, dtypes.bfloat16, dtypes.float32, dtypes.float64]:
      if not is_dtype_supported(default_dtype): continue
      dtypes.default_float = default_dtype
      for dtype in [dtypes.float16, dtypes.bfloat16, dtypes.float32, dtypes.float64]:
        if not is_dtype_supported(dtype): continue
        if DEBUG >= 2:
          print(f"testing {default_dtype=}, {dtype=}")
        a = Tensor([1, 2, 3], dtype=dtype, requires_grad=True)
        b = (a * 5).sum()
        b.backward()  # if there is dtype mismatch, lazy should assert
        assert a.grad.dtype == a.dtype
        np.testing.assert_allclose(a.grad.numpy(), [5, 5, 5])

    dtypes.default_float = old_default_float

  @unittest.skipUnless(is_dtype_supported(dtypes.half), "need half")
  def test_backward_sum_acc_dtype(self):
    # test acc of sum in the backward is upcasted to float
    t = Tensor([5, -5], dtype=dtypes.half, requires_grad=True)
    t.reshape(2, 1).expand(2, 10001).max().backward()
    np.testing.assert_allclose(t.grad.numpy(), [1, 0])

  @unittest.skipIf(Device.DEFAULT=="PYTHON", "very slow")
  @unittest.skipUnless(is_dtype_supported(dtypes.half), "need half")
  def test_mean_half_precision_underflow(self):
    N = 10000
    x = 0.001
    t = Tensor([[x]], dtype=dtypes.half, requires_grad=True).expand(N, N).contiguous()
    np.testing.assert_allclose(t.mean(axis=1).numpy(), np.array([x] * N, dtype=np.float16), rtol=1e-3)

  @unittest.skipUnless(is_dtype_supported(dtypes.half), "need half")
  def test_mean_half_precision_overflow(self):
    N = 256
    t = Tensor([60000] * N*N, dtype=dtypes.half, requires_grad=True).reshape(N, N)
    np.testing.assert_allclose(t.mean().numpy(), 60000)
    t.square().mean().backward()
    np.testing.assert_allclose(t.grad.numpy().flatten(), [60000 * 2 / (N*N)] * N*N)

  @unittest.skipUnless(is_dtype_supported(dtypes.half), "need half")
  def test_softmax_dtype(self):
    data = [1, 2, 3]
    t = Tensor(data, dtype=dtypes.half)
    tt = torch.tensor(data, dtype=torch.half)

    out = t.softmax(0)
    self.assertEqual(out.dtype, dtypes.half)
    np.testing.assert_allclose(out.numpy(), tt.softmax(0).numpy(), rtol=1e-3)
    out = t.softmax(0, dtype=dtypes.float)
    self.assertEqual(out.dtype, dtypes.float)
    np.testing.assert_allclose(out.numpy(), tt.softmax(0, dtype=torch.float).numpy(), rtol=1e-3)
    out = t.log_softmax(0)
    self.assertEqual(out.dtype, dtypes.half)
    np.testing.assert_allclose(out.numpy(), tt.log_softmax(0).numpy(), rtol=1e-3)
    out = t.log_softmax(0, dtype=dtypes.float)
    self.assertEqual(out.dtype, dtypes.float)
    np.testing.assert_allclose(out.numpy(), tt.log_softmax(0, dtype=torch.float).numpy(), rtol=1e-3)

class TestImplicitFunctionTypeChange(unittest.TestCase):
  def test_functions(self):
    result = []
    for func in [
      lambda t: t.exp(),
      lambda t: t.exp2(),
      lambda t: t.log(),
      lambda t: t.log2(),
      lambda t: t.sqrt(),
      lambda t: t.sin(),
    ]:
      t = func(Tensor([4.0, 3.0])).max() == func(Tensor([4.0, 3.0]))
      result.append(t.numpy().sum())
    assert all(result)

class TestTensorMethod(unittest.TestCase):
  @given(strat.sampled_from(core_dtypes))
  def test_abs_diff(self, dt):
    if dt == dtypes.bool or not is_dtype_supported(dt): return
    a, b = Tensor([2], dtype=dt), Tensor([1], dtype=dt)
    ret = (a - b).abs()
    np.testing.assert_allclose(ret.numpy(), np.abs(a.numpy()-b.numpy()))

if __name__ == '__main__':
  unittest.main()
