#!/usr/bin/env python
import os
import unittest
import numpy as np
os.environ['IMAGE'] = '1'
os.environ['GPU'] = '1'
from tinygrad.tensor import Tensor
from tinygrad.llops.ops_gpu import CLImage

class TestImage(unittest.TestCase):
  def test_create_image(self):
    t = Tensor.ones(128, 128, 1)
    t = t.reshape(128, 32, 4) + 3
    t.realize()
    assert isinstance(t.lazydata.realized._buf, CLImage)
    np.testing.assert_array_equal(t.numpy(), np.ones((128,32,4))*4)

  def test_sum_image(self):
    t1 = Tensor.ones(16, 16, 1).reshape(16, 4, 4) + 3
    t1.realize()
    assert isinstance(t1.lazydata.realized._buf, CLImage)
    t1 = t1.sum()
    t1.realize()
    assert t1.numpy()[0] == 16*4*4*4, f"got {t1.numpy()}"
  
  def test_add_image(self):
    t1 = Tensor.ones(16, 16, 1).reshape(16, 4, 4) + 3
    t2 = Tensor.ones(16, 16, 1).reshape(16, 4, 4) + 4
    t1.realize()
    t2.realize()
    t3 = t1 + t2
    t3.realize()
    assert isinstance(t3.lazydata.realized._buf, CLImage)
    np.testing.assert_array_equal(t3.numpy(), np.ones((16,4,4))*9)

if __name__ == '__main__':
  unittest.main()
