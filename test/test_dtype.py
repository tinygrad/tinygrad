import unittest
import numpy as np
from tinygrad.tensor import Tensor, dtypes

class TestDtype(unittest.TestCase):
  def test_half_to_np(self):
    a = Tensor([1,2,3,4], dtype=dtypes.float16)
    np_a = a.numpy()
    print(np_a, np_a.dtype)
    assert np_a.dtype == np.float16

  def test_half_add(self):
    a = Tensor([1,2,3,4], dtype=dtypes.float16)
    b = Tensor([1,2,3,4], dtype=dtypes.float16)
    c = a+b
    print(c.numpy())
    assert c.dtype == dtypes.float16

  def test_half_add_upcast(self):
    a = Tensor([1,2,3,4], dtype=dtypes.float16)
    b = Tensor([1,2,3,4], dtype=dtypes.float32)
    c = a+b
    print(c.numpy())
    assert c.dtype == dtypes.float32

if __name__ == '__main__':
  unittest.main()