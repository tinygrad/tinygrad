#!/usr/bin/env python
import gc, inspect
import unittest
import numpy as np
from tinygrad.device import Buffer
from tinygrad.engine.realize import run_linear
from tinygrad.uop.ops import UOp
from tinygrad.tensor import Tensor

def _allocations_of_type(t):
  ret = 0
  for x in gc.get_objects():
    try:
      if isinstance(x, t): ret += 1
    except ReferenceError:
      pass
  return ret

def tensors_allocated():
  gc.collect()
  return _allocations_of_type(Tensor)

def bufs_allocated():
  # count Buffer objects that own storage: a realized (or to-be-realized) BUFFER UOp owns one, views are transient and excluded
  gc.collect()
  return sum(1 for x in gc.get_objects() if isinstance(x, Buffer) and x._base is None)

class TestGC(unittest.TestCase):

  def test_gc(self):
    Tensor.manual_seed(0)
    base = tensors_allocated()
    a = Tensor.rand(4, 4)
    b = Tensor.zeros(4, 4)
    (a*b).mean().backward()
    assert (tensors_allocated()-base > 0)
    del a,b
    assert (tensors_allocated()-base == 2) # one for Tensor._device_rng_counters, and one for Tensor._device_seeds
    Tensor.manual_seed(0)

  def test_gc_complex(self):
    Tensor.manual_seed(0)
    base = tensors_allocated()
    a = Tensor(np.zeros((4, 4), dtype=np.float32))
    b = Tensor.rand(4, 4)
    assert (tensors_allocated()-base == 4)
    (a*b).mean().backward()
    assert (tensors_allocated()-base == 6)
    del b
    assert (tensors_allocated()-base == 4)
    b = Tensor(np.zeros((4, 4), dtype=np.float32))
    print(tensors_allocated())
    (a*b).mean().backward()
    print(tensors_allocated())
    assert (tensors_allocated()-base == 6)
    del b
    assert (tensors_allocated()-base == 4)
    Tensor.manual_seed(0)

  def test_schedule_gc(self):
    init = bufs_allocated()
    x = Tensor.ones(256).contiguous().realize()
    y = Tensor.ones(5, 5).contiguous()
    y.schedule_linear()
    del x
    del y
    self.assertEqual(bufs_allocated()-init, 0)

  def test_schedule_gc_with_inputs(self):
    init = bufs_allocated()
    x = Tensor.ones(256).contiguous().realize()
    y = x+Tensor.ones(256).contiguous()
    del x
    run_linear(*y.linear_with_vars())
    self.assertEqual(bufs_allocated()-init, 1)
    del y
    self.assertEqual(bufs_allocated()-init, 0)

  def test_toposort_blocks_gc(self):
    init = bufs_allocated()
    x = Tensor.ones(4,4).contiguous().realize()+1
    self.assertEqual(bufs_allocated()-init, 1)
    # try commenting this part out, it's green!
    x.uop.toposort()
    del x
    if bufs_allocated()-init != 0:
      print(inspect.getclosurevars(UOp.toposort().fget))
      raise AssertionError(f"never gced {[x for x in gc.get_objects() if isinstance(x, Buffer)]}")

  def test_buffer_ownership(self):
    init = bufs_allocated()
    a = Tensor.empty(10)
    # the Buffer object is owned by the BUFFER UOp 1:1, it exists from creation (device memory is still allocated lazily)
    self.assertEqual(bufs_allocated()-init, 1)
    a.realize()
    real_buf = a.uop.buffer
    self.assertIs(a.uop.arg.buffer, real_buf)
    self.assertEqual(bufs_allocated()-init, 1)
    del a.uop
    self.assertEqual(bufs_allocated()-init, 1) # the Buffer object is still held here
    del real_buf
    self.assertEqual(bufs_allocated()-init, 0)

  def test_assign_keeps_buffer(self):
    init = bufs_allocated()
    a = Tensor.full((4,), 1.).contiguous()
    a.realize()
    real_buf = a.uop.buffer
    a.assign(Tensor.full((4,), 2.))
    # assign writes in place: the AFTER still references the same Buffer
    self.assertIs(a.uop.src[0].buffer, real_buf)
    a.realize()
    del a
    self.assertEqual(bufs_allocated()-init, 1) # the Buffer object is still held here
    del real_buf
    self.assertEqual(bufs_allocated()-init, 0)

if __name__ == '__main__':
  unittest.main()
