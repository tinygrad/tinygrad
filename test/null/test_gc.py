#!/usr/bin/env python
import gc, weakref, contextlib
import unittest
from dataclasses import replace
import numpy as np
from tinygrad.engine.realize import run_linear
from tinygrad.tensor import Tensor
from tinygrad.dtype import dtypes, AddrSpace
from tinygrad.uop.ops import UOp, Ops, uop_fields

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
    self.assertIs(UOp.buffers[slot], real_buf)
    with assert_freed(real_buf):
      self.assertFalse(real_buf.is_allocated())
      a.realize()
      self.assertIs(a.uop.buffer, real_buf)
      del a.uop # the Buffer object is still held by real_buf
      self.assertIs(UOp.buffers[slot], real_buf) # and it keeps its slot for as long as it lives
      del real_buf
    self.assertNotIn(slot, UOp.buffers)
    # a slot is never bound to another Buffer
    self.assertGreater(UOp.new_buffer("NULL", 10, dtypes.float32).arg.slot, slot)

  def test_cyclic_gc_releases_buffer(self):
    a = Tensor.empty(10).realize()
    slot, uop_ref, buf_ref = a.uop.arg.slot, weakref.ref(a.uop), weakref.ref(a.uop.buffer)
    holder = [a]
    holder.append(holder)
    del a, holder
    gc.collect()
    self.assertIsNone(uop_ref())
    self.assertIsNone(buf_ref())
    self.assertNotIn(slot, UOp.buffers)

  def test_buffers_of_one_slot_share_ownership(self):
    # rewrites make other BUFFERs of the same slot (volatile, image, another dtype): each one holds the storage, none of them takes it away
    a = UOp.new_buffer("NULL", 4, dtypes.float32)
    buf_ref = weakref.ref(a.buffer)
    b = a.replace(arg=replace(a.arg, volatile=True))
    self.assertIsNot(b, a)
    self.assertIs(b.buffer, a.buffer)
    del b
    self.assertIs(a.buffer, buf_ref())
    b = a.replace(arg=replace(a.arg, volatile=True))
    del a
    self.assertIsNotNone(buf_ref())
    self.assertIs(b.buffer, buf_ref())
    del b
    self.assertIsNone(buf_ref())

  def test_trace_buffers_do_not_hold_storage(self):
    a = UOp.new_buffer("NULL", 4, dtypes.float32)
    b = UOp.from_buffer(a.buffer, dtypes.int32)
    with assert_freed(a.buffer):
      traced = []
      for u in (a, b):
        op, _, arg = uop_fields[u.trace_num]
        self.assertLess(arg.slot, 0)
        traced.append(data:=UOp(op, u.src, arg))
        self.assertIsNone(data.slot_buffer)
        self.assertEqual(uop_fields[data.trace_num][2], arg) # tracing an unbound node keeps its negative slot
      self.assertEqual(traced[0].arg.slot, traced[1].arg.slot)
      del a, b, u

  def test_only_global_buffers_look_up_slots(self):
    # scratch BUFFERs, ALLOCs and PARAMs number their slots on their own: a slot they share with a realized BUFFER is not its storage
    real = UOp.new_buffer("NULL", 4, dtypes.float32)
    buf_ref = weakref.ref(real.buffer)
    others = [UOp.placeholder((4,), dtypes.float32, real.arg.slot, addrspace) for addrspace in (AddrSpace.LOCAL, AddrSpace.REG)]
    others += [real.replace(op=Ops.ALLOC), UOp.param(real.arg.slot, dtypes.float32, 4, "NULL")]
    for u in others:
      self.assertEqual(u.arg.slot, real.arg.slot)
      self.assertIsNone(u.realized)
      with self.assertRaises(AssertionError): _ = u.buffer
    del real
    self.assertIsNone(buf_ref())

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
