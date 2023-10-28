import unittest
import numpy as np
from tinygrad.helpers import CI, DTYPES_DICT, getenv, DType, DEBUG, ImageDType, PtrDType
from tinygrad.ops import Device
from tinygrad.tensor import Tensor, dtypes
from typing import List, Optional
from extra.utils import OSX, temp

def is_dtype_supported(dtype: DType):
  # for GPU, cl_khr_fp16 isn't supported (except now we don't need it!)
  if dtype == dtypes.half: return not (CI and Device.DEFAULT in ["GPU", "LLVM"]) and Device.DEFAULT != "WEBGPU" and getenv("CUDACPU") != 1
  if dtype in [dtypes.double, dtypes.float64]: return Device.DEFAULT not in ["WEBGPU", "METAL"] and not OSX
  if dtype == dtypes.bfloat16: return False # TODO return Device.DEFAULT == "LLVM"
  if dtype in [dtypes.int8, dtypes.uint8]: return Device.DEFAULT not in ["WEBGPU"]
  if dtype in [dtypes.int16, dtypes.uint16]: return Device.DEFAULT not in ["WEBGPU", "TORCH"]
  if dtype == dtypes.uint32: return Device.DEFAULT not in ["TORCH"]
  if dtype in [dtypes.int64, dtypes.uint64]: return Device.DEFAULT not in ["WEBGPU", "TORCH"]
  if dtype == dtypes.bool: return Device.DEFAULT not in ["WEBGPU"]
  return True

def get_cast_dtypes(dtype: DType) -> List[DType]: return [v for k, v in DTYPES_DICT.items() if v != dtype and is_dtype_supported(v) and not k.startswith("_")] # dont cast internal dtypes

def _test_to_np(a:Tensor, np_dtype, target):
  if DEBUG >= 2: print(a)
  na = a.numpy()
  if DEBUG >= 2: print(na, na.dtype, a.lazydata.realized)
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

def _test_op(fxn, target_dtype:DType, target): _assert_eq(fxn(), target_dtype, target)
def _test_cast(a:Tensor, target_dtype:DType, target): _test_op(lambda: a.cast(target_dtype), target_dtype, target)

def _test_bitcast(a:Tensor, target_dtype:DType, target): _test_op(lambda: a.bitcast(target_dtype), target_dtype, target)

# tests no-op casts from source_dtype to target_dtypes
def _test_casts_from(tensor_contents:List, source_dtype:DType, target_contents:Optional[List]=None):
  target_dtypes = get_cast_dtypes(source_dtype)
  list(map(
    lambda t_dtype: _test_cast(Tensor(tensor_contents, dtype=source_dtype), t_dtype, target_contents or np.array(tensor_contents, dtype=source_dtype.np).astype(t_dtype.np).tolist()),
    target_dtypes
  ))
# tests no-op casts from source_dtypes to target_dtype
def _test_casts_to(tensor_contents:List, target_dtype:DType, target_contents:Optional[List]=None):
  source_dtypes = get_cast_dtypes(target_dtype)
  list(map(
    lambda s_dtype: _test_cast(Tensor(tensor_contents, dtype=s_dtype), target_dtype, target_contents or np.array(tensor_contents, dtype=s_dtype.np).astype(target_dtype.np).tolist()),
    source_dtypes
  ))

def _test_ops(a_dtype:DType, b_dtype:DType, target_dtype:DType):
  if not is_dtype_supported(a_dtype) or not is_dtype_supported(b_dtype) or not is_dtype_supported(target_dtype): raise unittest.SkipTest("dtype not supported")
  _assert_eq(Tensor([1,2,3,4], dtype=a_dtype)+Tensor([1,2,3,4], dtype=b_dtype), target_dtype, [2,4,6,8])
  _assert_eq(Tensor([1,2,3,4], dtype=a_dtype)*Tensor([1,2,3,4], dtype=b_dtype), target_dtype, [1,4,9,16])
  _assert_eq(Tensor([[1,2],[3,4]], dtype=a_dtype)@Tensor.eye(2, dtype=b_dtype), target_dtype, [[1,2],[3,4]])
  _assert_eq(Tensor([1,1,1,1], dtype=a_dtype)+Tensor.ones((4,4), dtype=b_dtype), target_dtype, 2*Tensor.ones(4,4).numpy())

class TestBFloat16DType(unittest.TestCase):
  def setUp(self):
    if not is_dtype_supported(dtypes.bfloat16): raise unittest.SkipTest("bfloat16 not supported")
  def test_bf16_to_float(self):
    with self.assertRaises(AssertionError):
      _test_cast(Tensor([100000], dtype=dtypes.bfloat16), dtypes.float32, [100000])

  def test_float_to_bf16(self):
    with self.assertRaises(AssertionError):
      _test_cast(Tensor([100000], dtype=dtypes.float32), dtypes.bfloat16, [100000])

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
    dat = open(temp('f32'), "rb").read()
    adat = b''.join([dat[i+2:i+4] for i in range(0, len(dat), 4)])
    with open(temp('bf16'), "wb") as f: f.write(adat)

    t = Tensor.empty(5, dtype=dtypes.bfloat16, device=f"disk:{temp('bf16')}").llvm().realize()
    back = t.cast(dtypes.float32)
    assert tuple(back.numpy().tolist()) == (9984., -1, -1000, -9984, 20)

class TestHalfDtype(unittest.TestCase):
  def setUp(self):
    if not is_dtype_supported(dtypes.float16): raise unittest.SkipTest("float16 not supported")
  def test_float16_to_np(self): _test_to_np(Tensor([1,2,3,4], dtype=dtypes.float16), np.float16, [1,2,3,4])
  def test_casts_to_half(self): _test_casts_to([1,2,3,4], target_dtype=dtypes.float16)
  def test_casts_from_half(self): _test_casts_from([1,2,3,4], source_dtype=dtypes.float16)
  def test_half_upcast_ops(self): _test_ops(a_dtype=dtypes.float16, b_dtype=dtypes.float32, target_dtype=dtypes.float32)
  def test_upcast_to_half_ops(self): _test_ops(a_dtype=dtypes.int8, b_dtype=dtypes.float16, target_dtype=dtypes.float16)

class TestFloat32Dtype(unittest.TestCase):
  def test_float32_to_np(self): _test_to_np(Tensor([1,2,3,4], dtype=dtypes.float32), np.float32, [1,2,3,4])
  def test_casts_to_float32(self): _test_casts_to([1,2,3,4], target_dtype=dtypes.float32)
  def test_casts_from_float32(self): _test_casts_from([1,2,3,4], source_dtype=dtypes.float32)
  def test_float32_upcast_ops(self): _test_ops(a_dtype=dtypes.float32, b_dtype=dtypes.float32, target_dtype=dtypes.float32)
  def test_upcast_to_float32_ops(self): _test_ops(a_dtype=dtypes.int8, b_dtype=dtypes.float32, target_dtype=dtypes.float32)

class TestDoubleDtype(unittest.TestCase):
  def setUp(self):
    if not is_dtype_supported(dtypes.double): raise unittest.SkipTest("double not supported")
  def test_double_to_np(self): _test_to_np(Tensor([1,2,3,4], dtype=dtypes.double), np.double, [1,2,3,4])
  def test_casts_to_double(self): _test_casts_to([1,2,3,4], target_dtype=dtypes.double)
  def test_casts_from_double(self): _test_casts_from([1,2,3,4], source_dtype=dtypes.double)
  def test_double_upcast_ops(self): _test_ops(a_dtype=dtypes.double, b_dtype=dtypes.double, target_dtype=dtypes.double)
  def test_upcast_to_double_ops(self): _test_ops(a_dtype=dtypes.int8, b_dtype=dtypes.double, target_dtype=dtypes.double)

class TestInt8Dtype(unittest.TestCase):
  def setUp(self):
    if not is_dtype_supported(dtypes.int8): raise unittest.SkipTest("int8 not supported")
  def test_int8_to_np(self): _test_to_np(Tensor([1,2,3,4], dtype=dtypes.int8), np.int8, [1,2,3,4])
  def test_casts_to_int8(self): _test_casts_to([1,2,3,4], target_dtype=dtypes.int8)
  def test_casts_from_int8(self): _test_casts_from([1,2,3,4], source_dtype=dtypes.int8)
  def test_int8_upcast_ops(self): _test_ops(a_dtype=dtypes.int8, b_dtype=dtypes.int8, target_dtype=dtypes.int8)
  def test_upcast_to_int8_ops(self): _test_ops(a_dtype=dtypes.int8, b_dtype=dtypes.int8, target_dtype=dtypes.int8)

  @unittest.skipIf(getenv("CUDA",0)==1, "cuda saturation works differently")
  @unittest.skipIf(getenv("PTX",0)==1, "cuda saturation doesn't wrap")
  def test_int8_to_uint8_negative(self): _test_op(lambda: Tensor([-1, -2, -3, -4], dtype=dtypes.int8).cast(dtypes.uint8), dtypes.uint8, [255, 254, 253, 252])

  @unittest.skipIf(getenv("PTX",0)==1, "cuda saturation doesn't wrap")
  def test_uint8_to_int8_overflow(self): _test_op(lambda: Tensor([255, 254, 253, 252], dtype=dtypes.uint8).cast(dtypes.int8), dtypes.int8, [-1, -2, -3, -4])

class TestInt16Dtype(unittest.TestCase):
  def setUp(self):
    if not is_dtype_supported(dtypes.int16): raise unittest.SkipTest("int16 not supported")
  def test_int16_to_np(self): _test_to_np(Tensor([1,2,3,4], dtype=dtypes.int16), np.int16, [1,2,3,4])
  def test_casts_to_int16(self): _test_casts_to([1,2,3,4], target_dtype=dtypes.int16)
  def test_casts_from_int16(self): _test_casts_from([1,2,3,4], source_dtype=dtypes.int16)
  def test_int16_upcast_ops(self): _test_ops(a_dtype=dtypes.int16, b_dtype=dtypes.int16, target_dtype=dtypes.int16)
  def test_upcast_to_int16_ops(self): _test_ops(a_dtype=dtypes.int16, b_dtype=dtypes.int16, target_dtype=dtypes.int16)

class TestInt32Dtype(unittest.TestCase):
  def setUp(self):
    if not is_dtype_supported(dtypes.int32): raise unittest.SkipTest("int32 not supported")
  def test_int32_to_np(self): _test_to_np(Tensor([1,2,3,4], dtype=dtypes.int32), np.int32, [1,2,3,4])
  def test_casts_to_int32(self): _test_casts_to([1,2,3,4], target_dtype=dtypes.int32)
  def test_casts_from_int32(self): _test_casts_from([1,2,3,4], source_dtype=dtypes.int32)
  def test_int32_upcast_ops(self): _test_ops(a_dtype=dtypes.int32, b_dtype=dtypes.int32, target_dtype=dtypes.int32)
  def test_upcast_to_int32_ops(self): _test_ops(a_dtype=dtypes.int32, b_dtype=dtypes.int32, target_dtype=dtypes.int32)

class TestInt64Dtype(unittest.TestCase):
  def setUp(self):
    if not is_dtype_supported(dtypes.int64): raise unittest.SkipTest("int64 not supported")
  def test_int64_to_np(self): _test_to_np(Tensor([1,2,3,4], dtype=dtypes.int64), np.int64, [1,2,3,4])
  def test_casts_to_int64(self): _test_casts_to([1,2,3,4], target_dtype=dtypes.int64)
  def test_casts_from_int64(self): _test_casts_from([1,2,3,4], source_dtype=dtypes.int64)
  def test_int64_upcast_ops(self): _test_ops(a_dtype=dtypes.int64, b_dtype=dtypes.int64, target_dtype=dtypes.int64)
  def test_upcast_to_int64_ops(self): _test_ops(a_dtype=dtypes.int64, b_dtype=dtypes.int64, target_dtype=dtypes.int64)

@unittest.skipIf(Device.DEFAULT == "WEBGPU", "host-shareablity is a requirement for storage buffers, but 'bool' type is not host-shareable")
class TestBoolDtype(unittest.TestCase):
  def setUp(self):
    if not is_dtype_supported(dtypes.bool): raise unittest.SkipTest("bool not supported")
  def test_bool_to_np(self): _test_to_np(Tensor([0,1,1,0], dtype=dtypes.bool), np.bool_, [0,1,1,0])
  def test_casts_to_bool(self): _test_casts_to([0,1,1,0], target_dtype=dtypes.bool)
  def test_casts_from_bool(self): _test_casts_from([0,1,1,0], source_dtype=dtypes.bool)

class TestEqStrDType(unittest.TestCase):
  def test_image_ne(self):
    assert dtypes.float == dtypes.float32, "float doesn't match?"
    assert dtypes.imagef((1,2,4)) != dtypes.imageh((1,2,4)), "different image dtype doesn't match"
    assert dtypes.imageh((1,2,4)) != dtypes.imageh((1,4,2)), "different shape doesn't match"
    assert dtypes.imageh((1,2,4)) == dtypes.imageh((1,2,4)), "same shape matches"
    assert isinstance(dtypes.imageh((1,2,4)), ImageDType)
  def test_ptr_ne(self):
    # TODO: is this the wrong behavior?
    assert PtrDType(dtypes.float32) == dtypes.float32
    #assert PtrDType(dtypes.float32) == PtrDType(dtypes.float32)
    #assert PtrDType(dtypes.float32) != dtypes.float32
  def test_strs(self):
    self.assertEqual(str(dtypes.imagef((1,2,4))), "dtypes.imagef((1, 2, 4))")
    self.assertEqual(str(PtrDType(dtypes.float32)), "ptr.dtypes.float")

@unittest.skipIf(Device.DEFAULT not in {"CPU", "TORCH"}, "only bitcast in CPU and TORCH")
class TestBitCast(unittest.TestCase):
  def test_float32_bitcast_to_int32(self): _test_bitcast(Tensor([1,2,3,4], dtype=dtypes.float32), dtypes.int32, [1065353216, 1073741824, 1077936128, 1082130432])
  @unittest.skipIf(Device.DEFAULT == "TORCH", "no uint32 in torch")
  def test_float32_bitcast_to_uint32(self): _test_bitcast(Tensor([1,2,3,4], dtype=dtypes.float32), dtypes.uint32, [1065353216, 1073741824, 1077936128, 1082130432])
  def test_int32_bitcast_to_float32(self): _test_bitcast(Tensor([1065353216, 1073741824, 1077936128, 1082130432], dtype=dtypes.int32), dtypes.float32, [1.0, 2.0, 3.0, 4.0])

  # NOTE: these are the same as normal casts
  def test_int8_bitcast_to_uint8(self): _test_bitcast(Tensor([-1, -2, -3, -4], dtype=dtypes.int8), dtypes.uint8, [255, 254, 253, 252])
  def test_uint8_bitcast_to_int8(self): _test_bitcast(Tensor([255, 254, 253, 252], dtype=dtypes.uint8), dtypes.int8, [-1, -2, -3, -4])
  @unittest.skipIf(Device.DEFAULT == "TORCH", "no uint64 in torch")
  def test_int64_bitcast_to_uint64(self): _test_bitcast(Tensor([-1, -2, -3, -4], dtype=dtypes.int64), dtypes.uint64, [18446744073709551615, 18446744073709551614, 18446744073709551613, 18446744073709551612])
  @unittest.skipIf(Device.DEFAULT == "TORCH", "no uint64 in torch")
  def test_uint64_bitcast_to_int64(self): _test_bitcast(Tensor([18446744073709551615, 18446744073709551614, 18446744073709551613, 18446744073709551612], dtype=dtypes.uint64), dtypes.int64, [-1, -2, -3, -4])

  def test_shape_change_bitcast(self):
    with self.assertRaises(AssertionError):
      _test_bitcast(Tensor([100000], dtype=dtypes.float32), dtypes.uint8, [100000])

if __name__ == '__main__':
  unittest.main()
