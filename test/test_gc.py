#!/usr/bin/env python
import unittest
from tinygrad.tensor import Tensor, GPU

class TestGC(unittest.TestCase):
  gpu = False
  def test_gc(self):
    a = Tensor.zeros(4,4, gpu=self.gpu)
    b = Tensor.zeros(4,4, gpu=self.gpu)
    (a*b).mean().backward()
    assert(Tensor.allocated > 0)
    del a,b
    assert(Tensor.allocated == 0)

  def test_gc_complex(self):
    a = Tensor.zeros(4,4, gpu=self.gpu)
    b = Tensor.zeros(4,4, gpu=self.gpu)
    assert(Tensor.allocated == 2)
    (a*b).mean().backward()
    assert(Tensor.allocated == 4)
    del b
    assert(Tensor.allocated == 2)
    b = Tensor.zeros(4,4, gpu=self.gpu)
    print(Tensor.allocated)
    (a*b).mean().backward()
    print(Tensor.allocated)
    assert(Tensor.allocated == 4)
    del b
    assert(Tensor.allocated == 2)

    

if GPU:
  class TestGCGPU(TestGC):
    gpu = True

if __name__ == '__main__':
  unittest.main()
