import unittest
import numpy as np
from tinygrad.tensor import Tensor
from tinygrad.vmap import vmap

class TestVmap(unittest.TestCase):
  def test_simple_vmap(self):
    @vmap(in_axes=0)
    def my_func(x):
      return x.sum(axis=0) * 2

    x = Tensor.ones(3, 10, 2).contiguous()
    result = my_func(x)
    val = result.realize().numpy()
    
    self.assertEqual(val.shape, (3, 2))
    np.testing.assert_allclose(val, 20)

  def test_broadcast_vmap(self):
    @vmap(in_axes=(0, None))
    def add_fixed(x, y):
      return x + y

    x_batch = Tensor([10, 20, 30]).contiguous()
    y_fixed = Tensor([1]).contiguous()
    
    res = add_fixed(x_batch, y_fixed)
    val = res.realize().numpy()
    
    np.testing.assert_array_equal(val, [[11], [21], [31]])

  def test_tuple_output(self):
    @vmap(in_axes=(0, None))
    def power_and_diff(x, val):
      return x * val, x - val

    x = Tensor([1, 2, 3]).contiguous()
    val = Tensor([10]).contiguous()
    
    out_mul, out_sub = power_and_diff(x, val)
    res_mul = out_mul.realize().numpy()
    res_sub = out_sub.realize().numpy()
    
    self.assertEqual(res_mul[1], 20)
    self.assertEqual(res_sub[1], -8)

if __name__ == "__main__":
  unittest.main()