import unittest
import numpy as np
from tinygrad.helpers import getenv, DType, DEBUG, CI
from tinygrad.ops import Device
from tinygrad.tensor import Tensor, dtypes
from typing import List, Optional
from extra.utils import OSX, temp
import copy

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
def _test_casts_from(tensor_contents:List, source_dtype:DType, target_dtypes:List[DType], target_contents:Optional[List]=None):
  if target_contents is None: target_contents = copy.deepcopy(tensor_contents)
  list(map(
    lambda t_dtype: _test_cast(Tensor(tensor_contents, dtype=source_dtype), t_dtype, target_contents),
    target_dtypes
  ))
# tests no-op casts from source_dtypes to target_dtype
def _test_casts_to(tensor_contents:List, source_dtypes:List[DType], target_dtype:DType, target_contents:Optional[List]=None):
  if target_contents is None: target_contents = copy.deepcopy(tensor_contents)
  list(map(
    lambda s_dtype: _test_cast(Tensor(tensor_contents, dtype=s_dtype), target_dtype, target_contents),
    source_dtypes
  ))

def _test_ops(a_dtype:DType, b_dtype:DType, target_dtype:DType):
  _assert_eq(Tensor([1,2,3,4], dtype=a_dtype)+Tensor([1,2,3,4], dtype=b_dtype), target_dtype, [2,4,6,8])
  _assert_eq(Tensor([1,2,3,4], dtype=a_dtype)*Tensor([1,2,3,4], dtype=b_dtype), target_dtype, [1,4,9,16])
  _assert_eq(Tensor([[1,2],[3,4]], dtype=a_dtype)@Tensor.eye(2, dtype=b_dtype), target_dtype, [[1,2],[3,4]])
  _assert_eq(Tensor([1,1,1,1], dtype=a_dtype)+Tensor.ones((4,4), dtype=b_dtype), target_dtype, 2*Tensor.ones(4,4).numpy())

class TestBFloat16DType(unittest.TestCase):
  def test_bf16_to_float(self):
    with self.assertRaises(AssertionError):
      _test_cast(Tensor([100000], dtype=dtypes.bfloat16), dtypes.float32, [100000])

  def test_float_to_bf16(self):
    with self.assertRaises(AssertionError):
      _test_cast(Tensor([100000], dtype=dtypes.float32), dtypes.bfloat16, [100000])

  # torch.tensor([10000, -1, -1000, -10000, 20]).type(torch.bfloat16)

  @unittest.skipIf(Device.DEFAULT not in ["LLVM"], "bf16 only on LLVM")
  def test_bf16(self):
    t = Tensor([10000, -1, -1000, -10000, 20]).cast(dtypes.bfloat16)
    t.realize()
    back = t.cast(dtypes.float32)
    assert tuple(back.numpy().tolist()) == (9984., -1, -1000, -9984, 20)

  @unittest.skipIf(Device.DEFAULT not in ["LLVM"], "bf16 only on LLVM")
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

# for GPU, cl_khr_fp16 isn't supported (except now we don't need it!)
# for LLVM, it segfaults because it can't link to the casting function
@unittest.skipIf((getenv("CI", "") != "" and Device.DEFAULT in ["LLVM"]) or Device.DEFAULT == "WEBGPU", "float16 broken in some CI backends")
class TestHalfDtype(unittest.TestCase):
  def test_float16_to_np(self): _test_to_np(Tensor([1,2,3,4], dtype=dtypes.float16), np.float16, [1,2,3,4])
  def test_casts_to_half(self): _test_casts_to([1,2,3,4], source_dtypes=[dtypes.float32, dtypes.int8, dtypes.uint8], target_dtype=dtypes.float16)
  def test_casts_from_half(self): _test_casts_from([1,2,3,4], source_dtype=dtypes.float16, target_dtypes=[dtypes.int8, dtypes.uint8, dtypes.float32, dtypes.int32, dtypes.int64])
  def test_half_upcast_ops(self): _test_ops(a_dtype=dtypes.float16, b_dtype=dtypes.float32, target_dtype=dtypes.float32)
  def test_upcast_to_half_ops(self): _test_ops(a_dtype=dtypes.int8, b_dtype=dtypes.float16, target_dtype=dtypes.float16)

@unittest.skipIf(Device.DEFAULT in ["WEBGPU", "METAL"] or OSX, "float64 is not supported by some backends")
class TestDoubleDtype(unittest.TestCase):
  def test_float64_to_np(self): _test_to_np(Tensor([1,2,3,4], dtype=dtypes.double), np.double, [1,2,3,4])
  def test_casts_to_float64(self): _test_casts_to([1,2,3,4], source_dtypes=[dtypes.float32, dtypes.int32, dtypes.uint8], target_dtype=dtypes.float64)
  def test_upcast_to_float64_ops(self): _test_ops(a_dtype=dtypes.int8, b_dtype=dtypes.float64, target_dtype=dtypes.float64)

@unittest.skipIf(Device.DEFAULT == "WEBGPU", "webgpu does not support int8")
class TestInt8Dtype(unittest.TestCase):
  def test_int8_to_np(self): _test_to_np(Tensor([1,2,3,4], dtype=dtypes.int8), np.int8, [1,2,3,4])
  def test_uint8_to_np(self): _test_to_np(Tensor([1,2,3,4], dtype=dtypes.uint8), np.uint8, [1,2,3,4])
  def test_int64_to_np(self): _test_to_np(Tensor([1,2,3,4], dtype=dtypes.int64), np.int64, [1,2,3,4])

  def test_casts_to_int8(self): _test_casts_from([1,2,3,4], source_dtype=dtypes.float32, target_dtypes=[dtypes.int8, dtypes.uint8, dtypes.int32, dtypes.int64])
  def test_casts_from_int8(self): _test_casts_from([1,2,3,4], source_dtype=dtypes.int8, target_dtypes=[dtypes.float32, dtypes.uint8, dtypes.int32, dtypes.int64])
  def test_casts_from_uint8(self): _test_casts_from([1,2,3,4], source_dtype=dtypes.uint8, target_dtypes=[dtypes.float32, dtypes.int8, dtypes.int32, dtypes.int64])

  def test_int8_ops(self): _test_ops(a_dtype=dtypes.int8, b_dtype=dtypes.int8, target_dtype=dtypes.int8)
  def test_int64_ops(self): _test_ops(a_dtype=dtypes.int64, b_dtype=dtypes.int64, target_dtype=dtypes.int64)
  def test_int8_upcast_float(self): _test_ops(a_dtype=dtypes.int8, b_dtype=dtypes.float32, target_dtype=dtypes.float32)
  def test_int8_upcast_int64(self): _test_ops(a_dtype=dtypes.int8, b_dtype=dtypes.int64, target_dtype=dtypes.int64)

  @unittest.skipIf(getenv("CUDA",0)==1, "cuda saturation works differently")
  @unittest.skipIf(getenv("PTX",0)==1, "cuda saturation doesn't wrap")
  def test_int8_to_uint8_negative(self): _test_op(lambda: Tensor([-1, -2, -3, -4], dtype=dtypes.int8).cast(dtypes.uint8), dtypes.uint8, [255, 254, 253, 252])

  @unittest.skipIf(getenv("PTX",0)==1, "cuda saturation doesn't wrap")
  def test_uint8_to_int8_overflow(self): _test_op(lambda: Tensor([255, 254, 253, 252], dtype=dtypes.uint8).cast(dtypes.int8), dtypes.int8, [-1, -2, -3, -4])

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

class TestInt32Dtype(unittest.TestCase):
  def test_int32_to_np(self): _test_to_np(Tensor([1,2,3,4], dtype=dtypes.int32), np.int32, [1,2,3,4])

  @unittest.skipIf(Device.DEFAULT == "WEBGPU", "webgpu does not support int64")
  def test_casts_to_int32(self): _test_casts_to([1,2,3,4], source_dtypes=[dtypes.float32, dtypes.int64], target_dtype=dtypes.int32)
  @unittest.skipIf(Device.DEFAULT == "WEBGPU", "webgpu does not support int64")
  def test_casts_from_int32(self): _test_casts_from([1,2,3,4], source_dtype=dtypes.int32, target_dtypes=[dtypes.float32, dtypes.int64])

  def test_int32_ops(self): _test_ops(a_dtype=dtypes.int32, b_dtype=dtypes.int32, target_dtype=dtypes.int32)
  def test_int32_upcast_float32(self): _test_ops(a_dtype=dtypes.int32, b_dtype=dtypes.float32, target_dtype=dtypes.float32)
  @unittest.skipIf(Device.DEFAULT == "WEBGPU", "webgpu does not support int64")
  def test_int32_upcast_int64(self): _test_ops(a_dtype=dtypes.int32, b_dtype=dtypes.int64, target_dtype=dtypes.int64)

class TestBoolDtype(unittest.TestCase):
  def test_casts_from_bool(self): _test_casts_from([0,1,1,0], source_dtype=dtypes.bool, target_dtypes=[dtypes.float32, dtypes.int32])
  def test_casts_to_bool(self): _test_casts_to([0,1,1,0], source_dtypes=[dtypes.float32, dtypes.int32], target_dtype=dtypes.bool)

if __name__ == '__main__':
  unittest.main()
