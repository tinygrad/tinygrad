#!/usr/bin/env python
import gc
import unittest
from tinygrad.tensor import Tensor, ANE, GPU, DeviceTypes

def tensors_allocated():
  return sum([isinstance(x, Tensor) for x in gc.get_objects()])
    
class TestGC(unittest.TestCase):
  device = DeviceTypes.CPU

  def test_gc(self):
    a = Tensor.zeros(4,4, device=self.device)
    b = Tensor.zeros(4,4, device=self.device)
    (a*b).mean().backward()
    assert(tensors_allocated() > 0)
    del a,b
    assert(tensors_allocated() == 0)

  def test_gc_complex(self):
    a = Tensor.zeros(4,4, device=self.device)
    b = Tensor.zeros(4,4, device=self.device)
    assert(tensors_allocated() == 2)
    (a*b).mean().backward()
    assert(tensors_allocated() == 4)
    del b
    assert(tensors_allocated() == 2)
    b = Tensor.zeros(4,4, device=self.device)
    print(tensors_allocated())
    (a*b).mean().backward()
    print(tensors_allocated())
    assert(tensors_allocated() == 4)
    del b
    assert(tensors_allocated() == 2)

@unittest.skipUnless(GPU, "Requires GPU")
class TestGCGPU(TestGC):
  device = DeviceTypes.GPU 

@unittest.skipUnless(ANE, "Requires ANE")
class TestGCANE(TestGC):
  device=DeviceTypes.ANE

if __name__ == '__main__':
  unittest.main()
