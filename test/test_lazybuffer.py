#!/usr/bin/env python
import numpy as np
import unittest
from tinygrad.lazy import LazyBuffer, Device
from tinygrad.tensor import Tensor
from tinygrad.shape.symbolic import Variable
from tinygrad.ops import GlobalCounters

class TestLazyBuffer(unittest.TestCase):
  def test_fromcpu_buffer_sharing(self):
    a = np.arange(8)
    assert LazyBuffer.fromCPU(a).realized._buf is a

  def test_fromcpu_shape_tracker(self):
    def helper(a: np.ndarray):
      print(a.shape, a.strides, a.flags.c_contiguous)
      b = LazyBuffer.fromCPU(a).realize()
      assert b.st.contiguous == a.flags.c_contiguous
      assert b.st.shape == a.shape
      np.testing.assert_equal(a, b.toCPU())

    for ndims in range(1, 4):
      a = np.random.randn(*(4,)*ndims).astype(np.float32)
      for stride in [-2, 1, 2]:
        for start in [0, 1]:
          helper(a[(slice(start, None, stride),)*ndims])

  def test_shuffle_pad_ops_cmpeq(self):
    y = Tensor([1]).cat(Tensor([1]).eq(0)).numpy()
    z = Tensor([1, 0]).numpy()
    np.testing.assert_allclose(y, z)

  def test_shuffle_pad_ops_div(self):
    y = Tensor([1]).cat(Tensor([1]).div(Tensor([2.0]))).numpy()
    z = Tensor([1, 0.5]).numpy()
    np.testing.assert_allclose(y, z)

  def test_shuffle_pad_ops_log(self):
    y = Tensor([1]).cat(Tensor([1]).log()).numpy()
    z = Tensor([1, 0]).numpy()
    np.testing.assert_allclose(y, z)

  def test_shuffle_pad_ops_exp(self):
    y = Tensor([1]).cat(Tensor([1]).exp()).numpy()
    z = Tensor([1, np.e]).numpy()
    np.testing.assert_allclose(y, z)

  @unittest.skipUnless(Device.DEFAULT in ["METAL", "CUDA", "GPU"], "Only GPU backends supports cache")
  def test_children_count(self):
    a = Tensor.rand(8,8,8)
    d1 = a.sum((0))
    d2 = a.sum((0)).reshape(32,2)
    assert len(d1.lazydata.op.src[0].children) == 1
    in1 = d1.reshape(16,4)
    d3 = in1.reshape(8,8)
    assert len(d3.lazydata.op.src[0].children) == 2

    GlobalCounters.cache = []
    l = Tensor.rand(8,8)
    r = Tensor.rand(8,8)
    dd = d1 + l
    dd.realize()
    de = d3 + r
    de.realize()
    assert len(GlobalCounters.cache) == 3
    assert GlobalCounters.cache[0][0].name.startswith("r_") # Reduce should not merged 2 times.
    assert GlobalCounters.cache[1][0].name.startswith("E_")
    assert GlobalCounters.cache[2][0].name.startswith("E_")
    GlobalCounters.cache = None

class TestVariableBuffer(unittest.TestCase):
  def test_get_variable_buffers_no_variable(self):
    t = Tensor.rand(2, 3)
    assert t.lazydata.get_variable_buffers() == {}

  def test_get_variable_buffers_one_variable(self):
    v = Variable("v", 1, 10)
    t = Tensor.rand(2, 3).reshape(v, 3)
    buffers = t.lazydata.get_variable_buffers()
    assert len(buffers) == 1 and buffers[v].realize().realized.toCPU() == 2
    v = Variable("v", 1, 10)
    t = Tensor.rand(2, 3).reshape(2, v)
    buffers = t.lazydata.get_variable_buffers()
    assert len(buffers) == 1 and buffers[v].realize().realized.toCPU() == 3

  def test_get_variable_buffers_cat(self):
    v1 = Variable("v1", 1, 10)
    v2 = Variable("v2", 1, 10)
    t1 = Tensor.rand(2, 3).reshape(v1, 3)
    t2 = Tensor.rand(6, 3).reshape(v2, 3)
    t = t1.cat(t2)
    buffers = t.lazydata.get_variable_buffers()
    assert len(buffers) == 2 and buffers[v1].realize().realized.toCPU() == 2 and buffers[v2].realize().realized.toCPU() == 6

if __name__ == "__main__":
  unittest.main()
