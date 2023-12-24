import warnings
import unittest
import numpy as np
import torch
import operator
from tinygrad.helpers import CI, getenv, DEBUG, OSX, temp
from tinygrad.dtype import DType, DTYPES_DICT, ImageDType, PtrDType, least_upper_float, least_upper_dtype
from tinygrad import Device
from tinygrad.tensor import Tensor, dtypes
from typing import Any, Set
from hypothesis import given, settings, strategies as st

core_dtypes = list(DTYPES_DICT.values())
floats = [dt for dt in core_dtypes if dtypes.is_float(dt)]
def is_dtype_supported(dtype: DType, device: str = Device.DEFAULT):
  if dtype == dtypes.bfloat16: return False # numpy doesn't support bf16, tested separately in TestBFloat16DType
  if device in ["WEBGPU", "WEBGL"]: return dtype in [dtypes.float, dtypes.int32, dtypes.uint32]
  if device == "TORCH": return dtype not in [dtypes.uint16, dtypes.uint32, dtypes.uint64]
  # for CI GPU, cl_khr_fp16 isn't supported
  # for CI LLVM, it segfaults because it can't link to the casting function
  # CUDA in CI uses CUDACPU that does not support half
  if dtype == dtypes.half: return not (CI and device in ["GPU", "LLVM", "CUDA"])
  if dtype == dtypes.float64: return device != "METAL" and not (OSX and device == "GPU")
  return True

def get_available_cast_dtypes(dtype: DType) -> Set[DType]:
  if not is_dtype_supported(dtype): return []
  return set([v for k, v in DTYPES_DICT.items() if v != dtype and is_dtype_supported(v) and not k.startswith("_")]) # dont cast internal dtypes

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
    np.testing.assert_allclose(tensor.numpy(), target)
  except AssertionError as e:
    raise AssertionError(f"\ntensor {tensor.numpy()} dtype {tensor.dtype} does not match target {target} with dtype {target_dtype}") from e

def _test_op(fxn, target_dtype:DType, target):
  _assert_eq(fxn(), target_dtype, target)
def _test_cast(a:Tensor, target_dtype:DType):
  _test_op(lambda: a.cast(target_dtype), target_dtype, list(a.numpy().astype(target_dtype.np)))
def _test_bitcast(a:Tensor, target_dtype:DType, target=None):
  _test_op(lambda: a.bitcast(target_dtype), target_dtype, target or a.numpy().view(target_dtype.np).tolist())

class TestDType(unittest.TestCase):
  DTYPE: Any = None
  DATA: Any = None
  @classmethod
  def setUpClass(cls):
    if not cls.DTYPE or not is_dtype_supported(cls.DTYPE): raise unittest.SkipTest("dtype not supported")
    if dtypes.is_int(cls.DTYPE): cls.DATA = np.random.randint(0, 100, size=10, dtype=cls.DTYPE.np).tolist()
    elif cls.DTYPE == dtypes.bool: cls.DATA = np.random.choice([True, False], size=10).tolist()
    else: cls.DATA = np.random.uniform(0, 1, size=10).tolist()
  def setUp(self):
    if self.DTYPE is None: raise unittest.SkipTest("base class")

  def test_to_np(self): _test_to_np(Tensor(self.DATA, dtype=self.DTYPE), self.DTYPE.np, np.array(self.DATA, dtype=self.DTYPE.np))

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
        _test_bitcast(Tensor(self.DATA, dtype=self.DTYPE), dtype) if dtype.itemsize == self.DTYPE.itemsize and dtype != dtypes.bool else None,
     get_available_cast_dtypes(self.DTYPE)
    ))
  def test_load_downcast_ops(self):
    from functools import reduce
    def sz(x): return x.itemsize
    def is_int(*x): return all(dtypes.is_int(_x) for _x in x)
    def is_uint(*x): return all(dtypes.is_unsigned(_x) for _x in x)
    def get_test_data(dt):
      test_data = [0 if x < 0 and is_uint(dt) else x for x in [10000, -23222332, 1, 1000, 0, 100001, 20]]
      return tuple([x if is_int(dt) else float(x) for x in test_data])
    inf_f = float('inf')
    expected_half = [((dtypes.half, t), (10000.0, 0, 1.0, 1000.0, 0.0, inf_f, 20.0)) for t in (dtypes.uint, dtypes.ulong)]
    expected_bool = [(dtypes.bool, (True, True, True, True, False, True, True)),]
    expected_bool += [((dtypes.bool, t), (True, False, True, True, False, True, True)) for t in (dtypes.uint, dtypes.ulong)]
    # Those looks good
    expected = [(dtypes.uchar, (255, 0, 1, 255, 0, 255, 20)), (dtypes.char, (127, -128, 1, 127, 0, 127, 20)),
                (dtypes.half,(10000.0, -inf_f, 1.0, 1000.0, 0.0, inf_f, 20.0)), (dtypes.ushort, (10000, 0, 1, 1000, 0, 65535, 20)),
                (dtypes.short, (10000, -32768, 1, 1000, 0, 32767, 20)), (dtypes.uint, (10000, 0, 1, 1000, 0, 100001, 20))]
    expected += expected_half + expected_bool
    # Those int-to-int conversion are likely just the truncation, but at least they are consistent across backends
    truncations = {(sz(dtypes.char), 0, 0):(16, -60, 1, -24, 0, -95, 20), (sz(dtypes.char), 1, 0):(16, 196, 1, 232, 0, 161, 20),
                   (sz(dtypes.char), 0, 1):(16, 0, 1, -24, 0, -95, 20), (sz(dtypes.char), 1, 1):(16, 0, 1, 232, 0, 161, 20),
                   (sz(dtypes.short), 0, 0):(10000, -22588, 1, 1000, 0, -31071, 20), (sz(dtypes.short), 1, 0):(10000, 42948, 1, 1000, 0, 34465, 20),
                   (sz(dtypes.short), 0, 1):(10000, 0, 1, 1000, 0, -31071, 20), (sz(dtypes.short), 1, 1):(10000, 0, 1, 1000, 0, 34465, 20),
                   (sz(dtypes.int), 1, 0):(10000, 4271744964, 1, 1000, 0, 100001, 20)}
    # Those are broken in some backends, but not in others or broken in a different ways, tuples are (dest, soruce)
    dt = dtypes
    def from_T2Achar(T): return ((dt.char, T), (dt.uchar, T))
    def from_T2Ashort(T): return ((dt.short, T), (dt.ushort, T))
    def from_TstoTs(T1, T2): return reduce(lambda x, y: x+y, [tt(t) for t in (T1) for tt in T2])
    broken = {
      'TORCH':from_TstoTs((dt.half, dt.float, dt.double), (from_T2Achar,)) + ((dt.short, dt.float),) + ((dt.short, dt.double),),
      'CPU':from_TstoTs((dt.half, dt.float, dt.double), (from_T2Achar,)) + from_TstoTs((dt.float, dt.double), (from_T2Ashort,)) +
            ((dt.uint, dt.double),),
      'CLANG':from_TstoTs((dt.float, dt.double), (from_T2Achar, from_T2Ashort)) + from_T2Achar(dt.half) + ((dt.uint, dt.double),),
      'METAL':from_T2Achar(dt.half),
      'CUDA':from_TstoTs((dt.float, dt.double), (from_T2Achar, from_T2Ashort)),
      # OpenCL is weird, it works with zero errors on some devices, but not on the others
      'GPU':from_TstoTs((dt.float, dt.double), (from_T2Achar, from_T2Ashort)) + ((dt.uint, dt.double),)
    }
    load_types = [x for x in get_available_cast_dtypes(self.DTYPE) if sz(x) > sz(self.DTYPE) and is_int(x)]
    bitcast_to_types = dict([(x, [y for y in get_available_cast_dtypes(x) if y != self.DTYPE and sz(y) == sz(x)]) for x in load_types])
    for t1, t2 in [(t1, t2) for t1, t2s in bitcast_to_types.items() for t2 in t2s if not is_int(t2) or sz(t2) >= sz(dtypes.int)]:
      with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        t = Tensor([get_test_data(t2)], dtype=t2)
        tin = tuple(t.numpy().tolist()[0])
        _t = t.bitcast(t1).realize().bitcast(t2).cast(self.DTYPE).numpy().tolist()
      akey = (sz(self.DTYPE), is_uint(self.DTYPE), is_uint(t2))
      expect_an = [((self.DTYPE, t2), truncations[akey])] if is_int(self.DTYPE, t2) and akey in truncations else []
      expect = dict(expected + expect_an)
      def get_expect():
        return tuple([int(x) if is_int(self.DTYPE) else x for x in expect.get((self.DTYPE, t2), expect.get(self.DTYPE, get_test_data(t2)))])
      def match_expect(r): return len(r) == 1 and tuple(r[0]) == get_expect()
      br_expect = (self.DTYPE, t2) in broken.get(Device.DEFAULT, [])
      test_descr = f'{Device.DEFAULT}: {tin} {t2}-b->{t1}->r-b->{t2}-c->{self.DTYPE} = {tuple(_t[0])}'
      if match_expect(_t):
        assert not br_expect, test_descr + ' is not broken'
        if DEBUG > 0: print(f'Passed: {test_descr}')
        continue
      test_descr += f'\nExpected: {get_expect()}'
      assert br_expect, test_descr
      if DEBUG > 0 and br_expect: print(f'Still broken: {test_descr}')

  def test_dtypes_fields(self):
    fields = dtypes.fields()
    self.assertTrue(all(isinstance(value, DType) for value in fields.values()))
    self.assertTrue(all(issubclass(value.np, np.generic) for value in fields.values() if value.np is not None))

def _test_ops(a_dtype:DType, b_dtype:DType, target_dtype=None):
  target_dtype = target_dtype or least_upper_dtype(a_dtype, b_dtype)
  if not is_dtype_supported(a_dtype) or not is_dtype_supported(b_dtype) or not is_dtype_supported(target_dtype): return
  if a_dtype == dtypes.bool or b_dtype == dtypes.bool: return
  _assert_eq(Tensor([1,2,3,4], dtype=a_dtype)+Tensor([1,2,3,4], dtype=b_dtype), target_dtype, [2,4,6,8])
  _assert_eq(Tensor([1,2,3,4], dtype=a_dtype)*Tensor([1,2,3,4], dtype=b_dtype), target_dtype, [1,4,9,16])
  _assert_eq(Tensor([[1,2],[3,4]], dtype=a_dtype)@Tensor.eye(2, dtype=b_dtype), target_dtype, [[1,2],[3,4]])
  _assert_eq(Tensor([1,1,1,1], dtype=a_dtype)+Tensor.ones((4,4), dtype=b_dtype), target_dtype, 2*Tensor.ones(4,4).numpy())

@unittest.skipUnless(Device.DEFAULT in ["LLVM", "TORCH"], "bfloat16 not supported")
class TestBFloat16DType(unittest.TestCase):
  def test_bf16_to_float(self):
    with self.assertRaises(AssertionError):
      _test_cast(Tensor([100000], dtype=dtypes.bfloat16), dtypes.float32)

  def test_float_to_bf16(self):
    with self.assertRaises(AssertionError):
      _test_cast(Tensor([100000], dtype=dtypes.float32), dtypes.bfloat16)

  # torch.tensor([10000, -1, -1000, -10000, 20]).type(torch.bfloat16)

  def test_bf16(self):
    t = Tensor([10000, -1, -1000, -10000, 20]).cast(dtypes.bfloat16)
    t.realize()
    back = t.cast(dtypes.float32)
    assert tuple(back.numpy().tolist()) == (9984., -1, -1000, -9984, 20)

  def test_bf16_disk_write_read(self):
    t = Tensor([10000, -1, -1000, -10000, 20]).cast(dtypes.float32)
    t.to(f"disk:{temp('f32')}").realize()

    # hack to "cast" f32 -> bf16
    with open(temp('f32'), "rb") as f: dat = f.read()
    adat = b''.join([dat[i+2:i+4] for i in range(0, len(dat), 4)])
    with open(temp('bf16'), "wb") as f: f.write(adat)

    t = Tensor.empty(5, dtype=dtypes.bfloat16, device=f"disk:{temp('bf16')}").llvm().realize()
    back = t.cast(dtypes.float32)
    assert tuple(back.numpy().tolist()) == (9984., -1, -1000, -9984, 20)

class TestHalfDtype(TestDType): DTYPE = dtypes.half

class TestFloatDType(TestDType): DTYPE = dtypes.float

class TestDoubleDtype(TestDType): DTYPE = dtypes.double

class TestInt8Dtype(TestDType):
  DTYPE = dtypes.int8
  @unittest.skipIf(getenv("CUDA",0)==1 or getenv("PTX", 0)==1, "cuda saturation works differently")
  def test_int8_to_uint8_negative(self):
    _test_op(lambda: Tensor([-1, -2, -3, -4], dtype=dtypes.int8).cast(dtypes.uint8), dtypes.uint8, [255, 254, 253, 252])

class TestUint8Dtype(TestDType):
  DTYPE = dtypes.uint8
  @unittest.skipIf(getenv("CUDA",0)==1 or getenv("PTX", 0)==1, "cuda saturation works differently")
  def test_uint8_to_int8_overflow(self):
    _test_op(lambda: Tensor([255, 254, 253, 252], dtype=dtypes.uint8).cast(dtypes.int8), dtypes.int8, [-1, -2, -3, -4])

@unittest.skipIf(Device.DEFAULT == "WEBGL", "No bitcast on WebGL")
class TestBitCast(unittest.TestCase):
  def test_shape_change_bitcast(self):
    with self.assertRaises(AssertionError):
      _test_bitcast(Tensor([100000], dtype=dtypes.float32), dtypes.uint8, [100000])

  def test_bitcast_float_to_int32(self):
    a = Tensor([1.,2,3])
    b = a.bitcast(dtypes.int32)
    assert b.numpy()[0] == 0x3f800000

  def test_bitcast_upcasted(self):
    a = Tensor.zeros(100, 4, dtype=dtypes.int32).contiguous() + 0x3f800000
    b = a.bitcast(dtypes.float32)
    assert b.numpy()[0,0] == 1.

class TestInt16Dtype(TestDType): DTYPE = dtypes.int16
class TestUint16Dtype(TestDType): DTYPE = dtypes.uint16

class TestInt32Dtype(TestDType): DTYPE = dtypes.int32
class TestUint32Dtype(TestDType): DTYPE = dtypes.uint32

class TestInt64Dtype(TestDType): DTYPE = dtypes.int64
class TestUint64Dtype(TestDType): DTYPE = dtypes.uint64

class TestBoolDtype(TestDType): DTYPE = dtypes.bool

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
    #assert PtrDType(dtypes.float32) == PtrDType(dtypes.float32)
    #assert PtrDType(dtypes.float32) != dtypes.float32
  def test_strs(self):
    if PtrDType is None: raise unittest.SkipTest("no PtrDType support")
    self.assertEqual(str(dtypes.imagef((1,2,4))), "dtypes.imagef((1, 2, 4))")
    self.assertEqual(str(PtrDType(dtypes.float32)), "ptr.dtypes.float")

class TestHelpers(unittest.TestCase):
  signed_ints = (dtypes.int8, dtypes.int16, dtypes.int32, dtypes.int64)
  uints = (dtypes.uint8, dtypes.uint16, dtypes.uint32, dtypes.uint64)
  floats = (dtypes.float16, dtypes.float32, dtypes.float64)

  @given(st.sampled_from(signed_ints+uints), st.integers(min_value=1, max_value=8))
  def test_is_int(self, dtype, amt):
    assert dtypes.is_int(dtype.vec(amt) if amt > 1 else dtype)
    assert not dtypes.is_float(dtype.vec(amt) if amt > 1 else dtype)

  @given(st.sampled_from(uints), st.integers(min_value=1, max_value=8))
  def test_is_unsigned_uints(self, dtype, amt):
    assert dtypes.is_unsigned(dtype.vec(amt) if amt > 1 else dtype)

  @given(st.sampled_from(signed_ints), st.integers(min_value=1, max_value=8))
  def test_is_unsigned_signed_ints(self, dtype, amt):
    assert not dtypes.is_unsigned(dtype.vec(amt) if amt > 1 else dtype)

  @given(st.sampled_from(floats), st.integers(min_value=1, max_value=8))
  def test_is_float(self, dtype, amt):
    assert dtypes.is_float(dtype.vec(amt) if amt > 1 else dtype)
    assert not dtypes.is_int(dtype.vec(amt) if amt > 1 else dtype)
    assert not dtypes.is_unsigned(dtype.vec(amt) if amt > 1 else dtype)

  def test_bf16_is_float(self):
    assert dtypes.is_float(dtypes.bfloat16)

  @given(st.sampled_from([d for d in DTYPES_DICT.values() if dtypes.is_float(d) or dtypes.is_int(d)]), st.integers(min_value=2, max_value=8))
  def test_scalar(self, dtype, amt):
    assert dtype.vec(amt).scalar() == dtype

class TestTypeSpec(unittest.TestCase):
  def setUp(self):
    self.old_default_int, self.old_default_float = dtypes.default_int, dtypes.default_float
  def tearDown(self):
    dtypes.default_int, dtypes.default_float = self.old_default_int, self.old_default_float

  def test_set_dtype_default(self):
    dtypes.default_int = dtypes.int16
    assert dtypes.default_int == dtypes.int16
    dtypes.default_int = dtypes.int64
    assert dtypes.default_int == dtypes.int64
    dtypes.default_int = dtypes.int32
    assert dtypes.default_int == dtypes.int32
    dtypes.default_float = dtypes.float16
    assert dtypes.default_float == dtypes.float16
    dtypes.default_float = dtypes.float64
    assert dtypes.default_float == dtypes.float64

  @given(st.sampled_from([dtypes.int8,dtypes.int16,dtypes.int32,dtypes.int64]), st.sampled_from([dtypes.float16,dtypes.float32,dtypes.float64]))
  def test_creation(self, default_int, default_float):
    dtypes.default_int, dtypes.default_float = default_int, default_float
    assert Tensor(True).dtype == dtypes.bool
    assert Tensor(None).dtype == dtypes.default_float
    assert Tensor(2).dtype == dtypes.default_int
    assert Tensor(2.34).dtype == dtypes.default_float
    assert Tensor([]).dtype == dtypes.default_float
    assert Tensor([1]).dtype == dtypes.default_int
    assert Tensor([1.1]).dtype == dtypes.default_float
    assert Tensor([0,1], dtype=dtypes.bfloat16).dtype == dtypes.bfloat16

    assert Tensor.eye(0).dtype == dtypes.default_float
    assert Tensor.eye(3).dtype == dtypes.default_float
    assert Tensor.eye(3, dtype=dtypes.float16).dtype == dtypes.float16
    assert Tensor.eye(3, dtype=dtypes.int64).dtype == dtypes.int64


  @given(st.sampled_from([dtypes.int8,dtypes.int16,dtypes.int32,dtypes.int64]), st.sampled_from([dtypes.float16,dtypes.float32,dtypes.float64]))
  def test_full(self, default_int, default_float):
    dtypes.default_int, dtypes.default_float = default_int, default_float

    assert Tensor.ones([2,3]).dtype == dtypes.default_float
    assert Tensor.zeros([2,3]).dtype == dtypes.default_float
    assert Tensor.full([2,3], 3.3).dtype == dtypes.default_float
    assert Tensor.full([2,3], 3).dtype == dtypes.default_int
    assert Tensor.full([2,3], True).dtype == dtypes.bool

    assert Tensor.zeros(3, 3).dtype == dtypes.default_float
    assert Tensor.zeros(3, 3, dtype=dtypes.float16).dtype == dtypes.float16
    assert Tensor.zeros(3, 3, dtype=dtypes.int64).dtype == dtypes.int64

    assert Tensor.ones(3, 3).dtype == dtypes.default_float
    assert Tensor.ones(3, 3, dtype=dtypes.float16).dtype == dtypes.float16
    assert Tensor.ones(3, 3, dtype=dtypes.int64).dtype == dtypes.int64

    assert Tensor.full((3, 3), 3).dtype == dtypes.default_int
    assert Tensor.full((3, 3), 3.0).dtype == dtypes.default_float
    assert Tensor.full((3, 3), 3, dtype=dtypes.float16).dtype == dtypes.float16
    assert Tensor.full((3, 3), 3, dtype=dtypes.int64).dtype == dtypes.int64

  def test_reduce_0d_default(self):
    assert Tensor.ones([2,3,0]).sum(2).dtype ==  dtypes.default_float
    # assert Tensor.ones([2,3,0], dtype=dtypes.int).sum(2).dtype == dtypes.int  # requires reduceop acc fix

  @given(st.sampled_from([dtypes.int8,dtypes.int16,dtypes.int32,dtypes.int64]), st.sampled_from([dtypes.float16,dtypes.float32,dtypes.float64]))
  def test_arange(self, default_int, default_float):
    dtypes.default_int, dtypes.default_float = default_int, default_float

    assert Tensor.arange(5).dtype == dtypes.default_int
    assert Tensor.arange(5.0).dtype == dtypes.default_float
    assert Tensor.arange(5, dtype=dtypes.int16).dtype == dtypes.int16
    assert Tensor.arange(5, dtype=dtypes.int64).dtype == dtypes.int64
    assert Tensor.arange(5, dtype=dtypes.float16).dtype == dtypes.float16
    assert Tensor.arange(3, 9, 0.7).dtype == dtypes.default_float
    assert Tensor.arange(3, 8.5, 3).dtype == dtypes.default_float

  @unittest.skipIf(Device.DEFAULT == "WEBGPU", "WEBGPU doesn't follow the bool ops spec")
  @given(st.sampled_from(core_dtypes), st.sampled_from([operator.gt, operator.ge, operator.le, operator.lt, operator.eq, operator.ne]))
  def test_bool_ops(self, dtype, op):
    assert op(Tensor.rand(4, 4, dtype=dtype), Tensor.rand(4, 4, dtype=dtype)).dtype == dtypes.bool

  @given(st.sampled_from(core_dtypes),
         st.sampled_from([dtypes.int8,dtypes.int16,dtypes.int32,dtypes.int64]), st.sampled_from([dtypes.float16,dtypes.float32,dtypes.float64]))
  def test_functions_return_index(self, dtype, default_int, default_float):
    dtypes.default_int, dtypes.default_float = default_int, default_float
    assert Tensor([0, 1], dtype=dtype).argmax().dtype == dtypes.default_int
    assert Tensor([0, 1], dtype=dtype).argmin().dtype == dtypes.default_int
    assert Tensor([0, 1], dtype=dtype).multinomial().dtype == dtypes.default_int

class TestTypePromotion(unittest.TestCase):
  @given(st.sampled_from(core_dtypes))
  def test_self_promo_to_self(self, dtype):
    assert least_upper_dtype(dtype) == dtype
    assert least_upper_dtype(dtype, dtype) == dtype
    assert least_upper_dtype(dtype, dtype, dtype) == dtype

  @given(st.sampled_from(core_dtypes), st.sampled_from(core_dtypes))
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

  @given(st.sampled_from(floats))
  def test_float_to_float(self, dt):
    assert least_upper_float(dt) == dt

class TestAutoCastType(unittest.TestCase):
  def setUp(self):
    self.old_default_int, self.old_default_float = dtypes.default_int, dtypes.default_float
  def tearDown(self):
    dtypes.default_int, dtypes.default_float = self.old_default_int, self.old_default_float

  @given(st.sampled_from([d for d in DTYPES_DICT.values() if dtypes.is_int(d) and is_dtype_supported(d)]))
  @settings(deadline=None)
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

  @given(st.sampled_from([dtypes.float16,dtypes.float32,dtypes.float64]))
  def test_broadcast_float(self, default_float):
    dtypes.default_float = default_float
    assert (Tensor.rand(4, 4, dtype=dtypes.bool) + 2.3).dtype == dtypes.default_float
    assert (Tensor.rand(4, 4, dtype=dtypes.int) + 2.3).dtype == dtypes.default_float
    assert (Tensor.rand(4, 4, dtype=dtypes.int8) + 2.3).dtype == dtypes.default_float
    assert (Tensor.rand(4, 4, dtype=dtypes.uint64) + 2.3).dtype == dtypes.default_float
    assert (Tensor.rand(4, 4, dtype=dtypes.float16) + 2.3).dtype == dtypes.float16
    assert (Tensor.rand(4, 4, dtype=dtypes.bfloat16) + 2.3).dtype == dtypes.bfloat16
    assert (Tensor.rand(4, 4, dtype=dtypes.float32) + 2.3).dtype == dtypes.float32
    assert (Tensor.rand(4, 4, dtype=dtypes.float64) + 2.3).dtype == dtypes.float64

  @given(st.sampled_from([dtypes.int8,dtypes.int16,dtypes.int32,dtypes.int64]))
  def test_broadcast_int(self, default_int):
    dtypes.default_int = default_int
    assert (Tensor.rand(4, 4, dtype=dtypes.bool) + 2).dtype == dtypes.default_int
    assert (Tensor.rand(4, 4, dtype=dtypes.int) + 2).dtype == dtypes.int
    assert (Tensor.rand(4, 4, dtype=dtypes.int8) + 2).dtype == dtypes.int8
    assert (Tensor.rand(4, 4, dtype=dtypes.uint64) + 2).dtype == dtypes.uint64
    assert (Tensor.rand(4, 4, dtype=dtypes.float16) + 2).dtype == dtypes.float16
    assert (Tensor.rand(4, 4, dtype=dtypes.bfloat16) + 2).dtype == dtypes.bfloat16
    assert (Tensor.rand(4, 4, dtype=dtypes.float32) + 2).dtype == dtypes.float32
    assert (Tensor.rand(4, 4, dtype=dtypes.float64) + 2).dtype == dtypes.float64

  def test_broadcast_bool(self):
    if Device.DEFAULT != "WEBGPU":
      assert (Tensor([0, 1], dtype=dtypes.bool) + True).dtype == dtypes.bool
    assert (Tensor([0, 1], dtype=dtypes.int) + True).dtype == dtypes.int32
    assert (Tensor([0, 1], dtype=dtypes.int8) + True).dtype == dtypes.int8
    assert (Tensor([0, 1], dtype=dtypes.uint64) + True).dtype == dtypes.uint64
    assert (Tensor([0, 1], dtype=dtypes.float16) + True).dtype == dtypes.float16
    assert (Tensor([0, 1], dtype=dtypes.bfloat16) + True).dtype == dtypes.bfloat16
    assert (Tensor([0, 1], dtype=dtypes.float32) + True).dtype == dtypes.float32
    assert (Tensor([0, 1], dtype=dtypes.float64) + True).dtype == dtypes.float64

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
    assert (Tensor([0, 1], dtype=dtypes.bfloat16)).sum().dtype == dtypes.bfloat16
    assert (Tensor([0, 1], dtype=dtypes.float32)).sum().dtype == dtypes.float32
    assert (Tensor([0, 1], dtype=dtypes.float64)).sum().dtype == dtypes.float64

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
    assert (Tensor([0, 1], dtype=dtypes.bfloat16)).cumsum(0).dtype == dtypes.bfloat16
    assert (Tensor([0, 1], dtype=dtypes.float32)).cumsum(0).dtype == dtypes.float32
    assert (Tensor([0, 1], dtype=dtypes.float64)).cumsum(0).dtype == dtypes.float64

  @given(st.sampled_from(core_dtypes), st.sampled_from(core_dtypes))
  @settings(deadline=None)
  def test_matmul(self, dt1, dt2):
    assert (Tensor([0, 1], dtype=dt1) @ Tensor([0, 1], dtype=dt2)).dtype == least_upper_dtype(dt1, dt2)

if __name__ == '__main__':
  unittest.main()
