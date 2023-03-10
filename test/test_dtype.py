import unittest
import numpy as np
from tinygrad.helpers import getenv
from tinygrad.lazy import Device
from tinygrad.tensor import Tensor, dtypes

# for GPU, cl_khr_fp16 isn't supported
# for LLVM, it segfaults because it can't link to the casting function
@unittest.skipIf(getenv("CI") and Device.DEFAULT in ["GPU", "LLVM"], "float16 broken in some CI backends")
class TestDtype(unittest.TestCase):
  def test_half_to_np(self):
    a = Tensor([1,2,3,4], dtype=dtypes.float16)
    print(a)
    na = a.numpy()
    print(na, na.dtype, a.lazydata.realized)
    assert na.dtype == np.float16

  def test_half_add(self):
    a = Tensor([1,2,3,4], dtype=dtypes.float16)
    b = Tensor([1,2,3,4], dtype=dtypes.float16)
    c = a+b
    print(c.numpy())
    assert c.dtype == dtypes.float16

  def test_upcast_float(self):
    # NOTE: there's no downcasting support
    a = Tensor([1,2,3,4], dtype=dtypes.float16).float()
    print(a)
    na = a.numpy()
    print(na, na.dtype)
    assert na.dtype == np.float32

  def test_half_add_upcast(self):
    a = Tensor([1,2,3,4], dtype=dtypes.float16)
    b = Tensor([1,2,3,4], dtype=dtypes.float32)
    c = a+b
    print(c.numpy())
    assert c.dtype == dtypes.float32

if __name__ == '__main__':
  unittest.main()