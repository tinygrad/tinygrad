#!/usr/bin/env python
import gc
import unittest
from tinygrad.tensor import Tensor, GPU

def tensors_allocated():
  return sum([isinstance(x, Tensor) for x in gc.get_objects()])
    
class TestGC(unittest.TestCase):
  gpu = False

  def test_gc(self):
    a = Tensor.zeros(4,4, gpu=self.gpu)
    b = Tensor.zeros(4,4, gpu=self.gpu)
    (a*b).mean().backward()
    assert(tensors_allocated() > 0)
    del a,b
    assert(tensors_allocated() == 0)

  def test_gc_complex(self):
    a = Tensor.zeros(4,4, gpu=self.gpu)
    b = Tensor.zeros(4,4, gpu=self.gpu)
    assert(tensors_allocated() == 2)
    (a*b).mean().backward()
    assert(tensors_allocated() == 4)
    del b
    assert(tensors_allocated() == 2)
    b = Tensor.zeros(4,4, gpu=self.gpu)
    print(tensors_allocated())
    (a*b).mean().backward()
    print(tensors_allocated())
    assert(tensors_allocated() == 4)
    del b
    assert(tensors_allocated() == 2)

    

if GPU:
  class TestGCGPU(TestGC):
    gpu = True

if __name__ == '__main__':
  unittest.main()
