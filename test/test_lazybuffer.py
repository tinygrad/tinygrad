#!/usr/bin/env python
import numpy as np
import unittest
from tinygrad.lazy import LazyBuffer

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

if __name__ == "__main__":
  unittest.main()
