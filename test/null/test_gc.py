#!/usr/bin/env python
import gc, weakref, contextlib
import unittest
import numpy as np
from tinygrad.engine.realize import run_linear
from tinygrad.tensor import Tensor
from tinygrad.dtype import dtypes, AddrSpace
from tinygrad.uop.ops import UOp

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

@contextlib.contextmanager
def assert_freed(*objects):
  refs = [weakref.ref(obj) for obj in objects]
  del objects
  yield
  gc.collect()
  for ref in refs: assert ref() is None, f"{ref()} was not freed"

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
    x = Tensor.ones(256).contiguous().realize()
    y = Tensor.ones(5, 5).contiguous()
    y.schedule_linear()
    with assert_freed(x.uop.buffer, y.uop.buffer):
      del x, y

  def test_schedule_gc_with_inputs(self):
    x = Tensor.ones(256).contiguous().realize()
    with assert_freed(x.uop.buffer):
      y = x+Tensor.ones(256).contiguous()
      del x
      run_linear(*y.linear_with_vars())
    with assert_freed(y.uop.buffer):
      del y

  def test_toposort_blocks_gc(self):
    x = Tensor.ones(4,4).contiguous().realize()+1
    with assert_freed(x.uop.src[0].buffer):
      x.uop.toposort()
      del x

  def test_buffer_ownership(self):
    a = Tensor.empty(10).realize()
    real_buf = a.uop.buffer
    slot = a.uop.arg.slot
    key = (real_buf, a.uop.dtype, a.uop.device)
    self.assertLess(slot, 0)
    self.assertIs(UOp.buffers[slot], real_buf)
    self.assertIs(UOp.from_buffer(real_buf, a.uop.dtype, a.uop.device), a.uop)
    with assert_freed(real_buf):
      self.assertFalse(real_buf.is_allocated())
      a.realize()
      self.assertIs(a.uop.buffer, real_buf)
      del a.uop # the Buffer object is still held by real_buf
      self.assertNotIn(slot, UOp.buffers)
      self.assertNotIn(key, UOp.buffer_slots)
      del key
      del real_buf

  def test_cyclic_gc_releases_buffer(self):
    a = Tensor.empty(10).realize()
    buf = a.uop.buffer
    key = (buf, a.uop.dtype, a.uop.device)
    slot = a.uop.arg.slot
    uop_ref, buf_ref = weakref.ref(a.uop), weakref.ref(buf)
    holder = [a]
    holder.append(holder)
    del a, holder
    gc.collect()
    self.assertIsNone(uop_ref())
    self.assertNotIn(slot, UOp.buffers)
    self.assertNotIn(key, UOp.buffer_slots)
    del key, buf
    gc.collect()
    self.assertIsNone(buf_ref())

  def test_scratch_buffers_do_not_unbind_globals(self):
    real = UOp.new_buffer("CPU", 4, dtypes.float32)
    real_buf = real.buffer
    for addrspace in (AddrSpace.LOCAL, AddrSpace.REG):
      scratch = UOp.placeholder((4,), dtypes.float32, real.arg.slot, addrspace)
      del scratch
      gc.collect()
      self.assertIs(real.buffer, real_buf)

  def test_assign_keeps_buffer(self):
    a = Tensor.full((4,), 1.).contiguous()
    a.realize()
    real_buf = a.uop.buffer
    with assert_freed(real_buf):
      a.assign(Tensor.full((4,), 2.))
      # assign writes in place: the AFTER still references the same Buffer
      self.assertIs(a.uop.src[0].buffer, real_buf)
      a.realize()
      self.assertIs(a.uop.buffer, real_buf)
      del a
      self.assertTrue(real_buf.is_allocated()) # the Buffer object is still held here
      del real_buf

if __name__ == '__main__':
  unittest.main()
