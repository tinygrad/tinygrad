import unittest
import numpy as np
from tinygrad.helpers import getenv, DType, DEBUG
from tinygrad.lazy import Device
from tinygrad.tensor import Tensor, dtypes
from extra.utils import OSX, temp

def _test_to_np(a:Tensor, np_dtype, target):
  print(a)
  na = a.numpy()
  print(na, na.dtype, a.lazydata.realized)
  assert na.dtype == np_dtype
  np.testing.assert_allclose(na, target)

def _test_op(fxn, target_dtype:DType, target):
  c = fxn()
  if DEBUG >= 2: print(c.numpy())
  assert c.dtype == target_dtype
  np.testing.assert_allclose(c.numpy(), target)

def _test_cast(a:Tensor, target_dtype:DType, target): _test_op(lambda: a.cast(target_dtype), target_dtype, target)
def _test_add(a:Tensor, b:Tensor, target_dtype:DType, target): _test_op(lambda: a+b, target_dtype, target)
def _test_mul(a:Tensor, b:Tensor, target_dtype:DType, target): _test_op(lambda: a*b, target_dtype, target)
def _test_matmul(a:Tensor, b:Tensor, target_dtype:DType, target): _test_op(lambda: a@b, target_dtype, target)
def _test_add_upcast(a:Tensor, b:Tensor, target_dtype:DType, target): _test_op(lambda: a+b, target_dtype, target)
def _test_mul_upcast(a:Tensor, b:Tensor, target_dtype:DType, target): _test_op(lambda: a*b, target_dtype, target)
def _test_matmul_upcast(a:Tensor, b:Tensor, target_dtype:DType, target): _test_op(lambda: a@b, target_dtype, target)


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
  def test_half_to_np(self): _test_to_np(Tensor([1,2,3,4], dtype=dtypes.float16), np.float16, [1,2,3,4])

  def test_half_to_float(self): _test_cast(Tensor([1,2,3,4], dtype=dtypes.float16), dtypes.float32, [1,2,3,4])
  def test_half_to_int8(self): _test_cast(Tensor([1,2,3,4], dtype=dtypes.float16), dtypes.int8, [1,2,3,4])
  def test_half_to_uint8(self): _test_cast(Tensor([1,2,3,4], dtype=dtypes.float16), dtypes.uint8, [1,2,3,4])
  def test_half_to_int32(self): _test_cast(Tensor([1,2,3,4], dtype=dtypes.float16), dtypes.int32, [1,2,3,4])
  def test_half_to_int64(self): _test_cast(Tensor([1,2,3,4], dtype=dtypes.float16), dtypes.int64, [1,2,3,4])

  def test_float_to_half(self): _test_cast(Tensor([1,2,3,4], dtype=dtypes.float32), dtypes.float16, [1,2,3,4])
  def test_int8_to_half(self): _test_cast(Tensor([1,2,3,4], dtype=dtypes.int8), dtypes.float16, [1,2,3,4])
  def test_uint8_to_half(self): _test_cast(Tensor([1,2,3,4], dtype=dtypes.uint8), dtypes.float16, [1,2,3,4])

  def test_half_add(self): _test_add(Tensor([1,2,3,4], dtype=dtypes.float16), Tensor([1,2,3,4], dtype=dtypes.float16), dtypes.float16, [2,4,6,8])
  def test_half_mul(self): _test_mul(Tensor([1,2,3,4], dtype=dtypes.float16), Tensor([1,2,3,4], dtype=dtypes.float16), dtypes.float16, [1,4,9,16])
  def test_half_matmul(self): _test_matmul(Tensor([[1,2],[3,4]], dtype=dtypes.float16), Tensor.eye(2, dtype=dtypes.float16), dtypes.float16, [[1,2],[3,4]])

  def test_half_add_upcast_float(self): _test_add_upcast(Tensor([1,2,3,4], dtype=dtypes.float16), Tensor([1,2,3,4], dtype=dtypes.float32), dtypes.float32, [2,4,6,8])
  def test_int8_add_upcast_half(self): _test_add_upcast(Tensor([1,2,3,4], dtype=dtypes.int8), Tensor([1,2,3,4], dtype=dtypes.float16), dtypes.float16, [2,4,6,8])
  def test_int8_mul_upcast_half(self): _test_mul_upcast(Tensor([1,2,3,4], dtype=dtypes.int8), Tensor([1,2,3,4], dtype=dtypes.float16), dtypes.float16, [1,4,9,16])
  def test_half_mul_upcast_float(self): _test_mul_upcast(Tensor([1,2,3,4], dtype=dtypes.float16), Tensor([1,2,3,4], dtype=dtypes.float32), dtypes.float32, [1,4,9,16])
  def test_half_matmul_upcast_float(self): _test_matmul_upcast(Tensor([[1,2],[3,4]], dtype=dtypes.float16), Tensor.eye(2, dtype=dtypes.float32), dtypes.float32, [[1,2],[3,4]])
  def test_int8_matmul_upcast_half(self): _test_matmul_upcast(Tensor([[1,2],[3,4]], dtype=dtypes.int8), Tensor.eye(2, dtype=dtypes.float16), dtypes.float16, [[1,2],[3,4]])

@unittest.skipIf(Device.DEFAULT == "WEBGPU", "webgpu does not support int8")
class TestInt8Dtype(unittest.TestCase):
  def test_int8_to_np(self): _test_to_np(Tensor([1,2,3,4], dtype=dtypes.int8), np.int8, [1,2,3,4])
  def test_uint8_to_np(self): _test_to_np(Tensor([1,2,3,4], dtype=dtypes.uint8), np.uint8, [1,2,3,4])
  def test_int64_to_np(self): _test_to_np(Tensor([1,2,3,4], dtype=dtypes.int64), np.int64, [1,2,3,4])

  def test_float_to_int8(self): _test_cast(Tensor([1,2,3,4], dtype=dtypes.float32), dtypes.int8, [1,2,3,4])
  def test_float_to_uint8(self): _test_cast(Tensor([1,2,3,4], dtype=dtypes.float32), dtypes.uint8, [1,2,3,4])
  def test_float_to_int64(self): _test_cast(Tensor([1,2,3,4], dtype=dtypes.float32), dtypes.int64, [1,2,3,4])

  def test_int8_to_float(self): _test_cast(Tensor([1,2,3,4], dtype=dtypes.int8), dtypes.float32, [1,2,3,4])
  def test_int8_to_uint8(self): _test_cast(Tensor([1,2,3,4], dtype=dtypes.int8), dtypes.uint8, [1,2,3,4])
  def test_int8_to_int32(self): _test_cast(Tensor([1,2,3,4], dtype=dtypes.int8), dtypes.int32, [1,2,3,4])
  def test_int8_to_int64(self): _test_cast(Tensor([1,2,3,4], dtype=dtypes.int8), dtypes.int64, [1,2,3,4])

  def test_uint8_to_float(self): _test_cast(Tensor([1,2,3,4], dtype=dtypes.uint8), dtypes.float32, [1,2,3,4])
  def test_uint8_to_int8(self): _test_cast(Tensor([1,2,3,4], dtype=dtypes.uint8), dtypes.int8, [1,2,3,4])
  def test_uint8_to_int64(self): _test_cast(Tensor([1,2,3,4], dtype=dtypes.uint8), dtypes.int64, [1,2,3,4])

  def test_int8_add(self): _test_add(Tensor([1,2,3,4], dtype=dtypes.int8), Tensor([1,2,3,4], dtype=dtypes.int8), dtypes.int8, [2,4,6,8])
  def test_int64_add(self): _test_add(Tensor([1,2,3,4], dtype=dtypes.int64),Tensor([1,2,3,4], dtype=dtypes.int64), dtypes.int64, [2,4,6,8])

  def test_int8_mul(self): _test_mul(Tensor([1,2,3,4], dtype=dtypes.int8), Tensor([1,2,3,4], dtype=dtypes.int8), dtypes.int8, [1,4,9,16])
  def test_int64_mul(self): _test_mul(Tensor([1,2,3,4], dtype=dtypes.int64), Tensor([1,2,3,4], dtype=dtypes.int64), dtypes.int64, [1,4,9,16])

  def test_int8_matmul(self): _test_matmul(Tensor([[1,2],[3,4]], dtype=dtypes.int8), Tensor.eye(2, dtype=dtypes.int8), dtypes.int8, [[1,2],[3,4]])
  def test_int64_matmul(self): _test_matmul(Tensor([[1,2],[3,4]], dtype=dtypes.int64), Tensor.eye(2, dtype=dtypes.int64), dtypes.int64, [[1,2],[3,4]])

  def test_int8_add_upcast_float(self): _test_add_upcast(Tensor([1,2,3,4], dtype=dtypes.int8), Tensor([1,2,3,4], dtype=dtypes.float32), dtypes.float32, [2,4,6,8])
  def test_int8_mul_upcast_float(self): _test_mul_upcast(Tensor([1,2,3,4], dtype=dtypes.int8), Tensor([1,2,3,4], dtype=dtypes.float32), dtypes.float32, [1,4,9,16])
  def test_int8_matmul_upcast_float(self): _test_matmul_upcast(Tensor([[1,2],[3,4]], dtype=dtypes.int8), Tensor.eye(2, dtype=dtypes.float32), dtypes.float32, [[1,2],[3,4]])

  def test_int8_add_upcast_int64(self): _test_add_upcast(Tensor([1,2,3,4], dtype=dtypes.int8), Tensor([1,2,3,4], dtype=dtypes.int64), dtypes.int64, [2,4,6,8])
  def test_int8_mul_upcast_int64(self): _test_mul_upcast(Tensor([1,2,3,4], dtype=dtypes.int8), Tensor([1,2,3,4], dtype=dtypes.int64), dtypes.int64, [1,4,9,16])
  def test_int8_matmul_upcast_int64(self): _test_matmul_upcast(Tensor([[1,2],[3,4]], dtype=dtypes.int8), Tensor.eye(2, dtype=dtypes.int64), dtypes.int64, [[1,2],[3,4]])

  @unittest.skipIf(getenv("CUDA",0)==1, "cuda saturation works differently")
  def test_int8_to_uint8_negative(self): _test_op(lambda: Tensor([-1, -2, -3, -4], dtype=dtypes.int8).cast(dtypes.uint8), dtypes.uint8, [255, 254, 253, 252])

  def test_uint8_to_int8_overflow(self): _test_op(lambda: Tensor([255, 254, 253, 252], dtype=dtypes.uint8).cast(dtypes.int8), dtypes.int8, [-1, -2, -3, -4])

class TestInt32Dtype(unittest.TestCase):
  def test_int32_to_np(self): _test_to_np(Tensor([1,2,3,4], dtype=dtypes.int32), np.int32, [1,2,3,4])

  def test_float_to_int32(self): _test_cast(Tensor([1,2,3,4], dtype=dtypes.float32), dtypes.int32, [1,2,3,4])
  def test_int64_to_int32(self): _test_cast(Tensor([1,2,3,4], dtype=dtypes.int64), dtypes.int32, [1,2,3,4])

  def test_int32_to_float(self): _test_cast(Tensor([1,2,3,4], dtype=dtypes.int32), dtypes.float32, [1,2,3,4])
  def test_int32_to_int64(self): _test_cast(Tensor([1,2,3,4], dtype=dtypes.int32), dtypes.int64, [1,2,3,4])

  def test_int32_add(self): _test_add(Tensor([1,2,3,4], dtype=dtypes.int32), Tensor([1,2,3,4], dtype=dtypes.int32), dtypes.int32, [2,4,6,8])
  def test_int32_mul(self): _test_mul(Tensor([1,2,3,4], dtype=dtypes.int32), Tensor([1,2,3,4], dtype=dtypes.int32), dtypes.int32, [1,4,9,16])
  def test_int32_matmul(self): _test_matmul(Tensor([[1,2],[3,4]], dtype=dtypes.int32), Tensor.eye(2, dtype=dtypes.int32), dtypes.int32, [[1,2],[3,4]])

  def test_int32_add_upcast_float(self): _test_add_upcast(Tensor([1,2,3,4], dtype=dtypes.int32), Tensor([1,2,3,4], dtype=dtypes.float32), dtypes.float32, [2,4,6,8])
  def test_int32_mul_upcast_float(self): _test_mul_upcast(Tensor([1,2,3,4], dtype=dtypes.int32), Tensor([1,2,3,4], dtype=dtypes.float32), dtypes.float32, [1,4,9,16])
  def test_int32_matmul_upcast_float(self): _test_matmul_upcast(Tensor([[1,2],[3,4]], dtype=dtypes.int32), Tensor.eye(2, dtype=dtypes.float32), dtypes.float32, [[1,2],[3,4]])

  def test_int32_add_upcast_int64(self): _test_add_upcast(Tensor([1,2,3,4], dtype=dtypes.int32), Tensor([1,2,3,4], dtype=dtypes.int64), dtypes.int64, [2,4,6,8])
  def test_int32_mul_upcast_int64(self): _test_mul_upcast(Tensor([1,2,3,4], dtype=dtypes.int32), Tensor([1,2,3,4], dtype=dtypes.int64), dtypes.int64, [1,4,9,16])
  def test_int32_matmul_upcast_int64(self): _test_matmul_upcast(Tensor([[1,2],[3,4]], dtype=dtypes.int32), Tensor.eye(2, dtype=dtypes.int64), dtypes.int64, [[1,2],[3,4]])

if __name__ == '__main__':
  unittest.main()
