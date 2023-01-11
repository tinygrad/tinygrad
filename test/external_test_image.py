#!/usr/bin/env python
import os
import unittest
os.environ['IMAGE'] = '1'
os.environ['GPU'] = '1'
from tinygrad.tensor import Tensor
from tinygrad.llops.ops_gpu import CLImage

class TestOpt(unittest.TestCase):
  def test_create_image(self):
    t = Tensor.ones(128, 128, 1)
    t = t.reshape(128, 32, 4) + 3
    t.realize()
    assert isinstance(t.lazydata.realized._buf, CLImage)
    print(t.numpy())

if __name__ == '__main__':
  unittest.main()
