#!/usr/bin/env python
import unittest
import numpy as np
from weakref import ref
from tinygrad.helpers import GlobalCounters
from tinygrad.runtime.lib import RawBuffer, LRUAllocator
from tinygrad.helpers import dtypes, prod
from tinygrad.ops import Device
from tinygrad.tensor import Tensor

def check_gc():
  if Device.DEFAULT == "GPU":
    from extra.introspection import print_objects
    assert print_objects() == 0

class FakeDeviceBuffer:
  def __init__(self, sz, dt, device):
    self.id = 1
    self.size = sz
    self.dtype = dt
    self.device = device
  def __del__(self):
    assert self.id == 0, "Should called _do_free() before"

class FakeAllocator(LRUAllocator):
  def _do_alloc(self, size, dtype, device, **kwargs): return FakeDeviceBuffer(size, dtype, device)
  def _do_free(self, buf):
    buf.id -= 1
    assert buf.id == 0, f"Free should be called once, but {buf.id}"
  def __del__(self): # Fake allocator should clear all buffers after each test.
    for v in self.cached_buffers.values():
      for buf, _ in v: self._free_buffer(buf)

FAKE_GLOBAL_ALLOCATOR = None
class FakeBuffer(RawBuffer):
  def __init__(self, size, dtype, device='0'):
    global FAKE_GLOBAL_ALLOCATOR
    super().__init__(size, dtype, allocator=FAKE_GLOBAL_ALLOCATOR, **{'device': device})
    assert self._buf.size == size and self._buf.dtype == dtype and self._buf.device == device, "This allocator requires 100% match of dtype and size."
  @classmethod
  def fromCPU(cls, x:np.ndarray, **kwargs): return cls(prod(x.shape), dtypes.from_np(x.dtype), **kwargs)
  def toCPU(self): return np.empty(self.size, dtype=self.dtype.np)

def alloc(allocator, size, dtype, **kwargs):
  global FAKE_GLOBAL_ALLOCATOR
  FAKE_GLOBAL_ALLOCATOR = allocator
  buf = FakeBuffer(size, dtype, **kwargs)
  assert buf.dtype == dtype and buf.size == size
  FAKE_GLOBAL_ALLOCATOR = None
  return buf

def alloc_free_trace(allocator, size, dtype, **kwargs):
  buf = alloc(allocator, size, dtype, **kwargs)
  return ref(buf._buf)

def cmp_trace_and_buf(buf, trace_ref): return trace_ref and trace_ref() == buf._buf

class TestAllocators(unittest.TestCase):
  def test_lru_allocator_reusage(self):
    mc, mu = GlobalCounters.mem_cached, GlobalCounters.mem_used
    def test():
      lru_allocator = FakeAllocator(2048)
      traced_buf = alloc_free_trace(lru_allocator, 16, dtypes.float32)
      assert GlobalCounters.mem_cached - mc == 16*dtypes.float32.itemsize, "Buffer should be cached"
      for _ in range(32):
        def __test():
          buf = alloc(lru_allocator, 16, dtypes.float32)
          assert cmp_trace_and_buf(buf, traced_buf), "Buffer should be reused"
        __test()

      usedbuf = alloc(lru_allocator, 16, dtypes.float32)
      for _ in range(32):
        def __test():
          buf = alloc(lru_allocator, 16, dtypes.float32)
          assert usedbuf != buf, "Nobody should get used buffer"
        __test()
      assert GlobalCounters.mem_used - mu == 16*dtypes.float32.itemsize, "Only usedbuf is still allocated."
    test()
    check_gc()

  def test_lru_allocator_cache_free(self):
    mc, mu = GlobalCounters.mem_cached, GlobalCounters.mem_used
    def test():
      lru_allocator = FakeAllocator(128)
      refs = []
      for _ in range(32):
        refs.append(alloc_free_trace(lru_allocator, 16, dtypes.float32))
      for sz in range(1, 32):
        alloc_free_trace(lru_allocator, sz, dtypes.float32)
        assert GlobalCounters.mem_used + GlobalCounters.mem_cached - mc - mu <= 128, "Should not allocate on device more than allowed (128)"
      for r in refs: assert r() is None, "All refs should be dead, since buffers were cleared from cache"
    test()
    check_gc()

  def test_lru_allocator_multidevice(self):
    def test():
      lru_allocator = FakeAllocator(256)
      refs=[]
      for i in range(8):
        refs.append(alloc_free_trace(lru_allocator, 16, dtypes.float32, device=str(i)))
      for i in range(64):
        def __test():
          dev = str(i % 8)
          buf = alloc(lru_allocator, 16, dtypes.float32, device=dev)
          assert cmp_trace_and_buf(buf, refs[i%8]), "Buffer should be reused"
        __test()
      for r in refs: assert r() is not None, "All refs should be cached"
    test()
    check_gc()

  @unittest.skip("failing in CI")
  def test_gpu_copyout(self):
    def test():
      from tinygrad.runtime.ops_gpu import CL

      # Allocation to init the allocator.
      tx = Tensor.rand(1)
      tx.realize()
      free_space = CL.cl_allocator.free_space[tx.lazydata.realized._device]

      # Spawning 128mb objects to fill half of free_space
      will_allocate = free_space // 3
      trash_allocation_size = free_space // 2

      def sp():
        trash_buffer = Tensor.rand(trash_allocation_size // 4)
        trash_buffer.realize()
      sp()

      xx = Tensor.rand(will_allocate // 4)
      _ = xx.numpy()
    test()
    check_gc()

if __name__ == "__main__":
  unittest.main()
