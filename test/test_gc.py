#!/usr/bin/env python
import gc
import unittest
import numpy as np
from tinygrad.device import Buffer
from tinygrad.engine.realize import run_schedule
from tinygrad.tensor import Tensor

def tensors_allocated():
  return sum([isinstance(x, Tensor) for x in gc.get_objects()])

def bufs_allocated():
  return sum([isinstance(x, Buffer) for x in gc.get_objects()])

class TestGC(unittest.TestCase):

  def test_gc(self):
    Tensor.manual_seed(0)
    a = Tensor.rand(4, 4, requires_grad=True)
    b = Tensor.zeros(4, 4, requires_grad=True)
    (a*b).mean().backward()
    assert (tensors_allocated() > 0)
    del a,b
    assert (tensors_allocated() == 1) # one for Tensor._device_rng_counters

  def test_gc_complex(self):
    Tensor.manual_seed(0)
    a = Tensor(np.zeros((4, 4), dtype=np.float32), requires_grad=True)
    b = Tensor.rand(4, 4, requires_grad=True)
    assert (tensors_allocated() == 4)
    (a*b).mean().backward()
    assert (tensors_allocated() == 5)
    del b
    assert (tensors_allocated() == 3)
    b = Tensor(np.zeros((4, 4), dtype=np.float32), requires_grad=True)
    print(tensors_allocated())
    (a*b).mean().backward()
    print(tensors_allocated())
    assert (tensors_allocated() == 5)
    del b
    assert (tensors_allocated() == 3)

  def test_schedule_gc(self):
    init = bufs_allocated()
    x = Tensor.ones(256).contiguous().realize()
    y = Tensor.ones(5, 5).contiguous()
    y.schedule()
    del x
    del y
    self.assertEqual(bufs_allocated()-init, 0)

  def test_schedule_gc_with_inputs(self):
    init = bufs_allocated()
    x = Tensor.ones(256).contiguous().realize()
    y = x+Tensor.ones(256).contiguous()
    ys = y.schedule()
    del x
    run_schedule(ys)
    np.testing.assert_equal(y.numpy(), np.full((256,), 2))
    self.assertEqual(bufs_allocated()-init, 1)
    del y
    self.assertEqual(bufs_allocated()-init, 0)

if __name__ == '__main__':
  unittest.main()
